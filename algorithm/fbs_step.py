"""
Forward-backward splitting with history and deviations
(Algorithm 1 of Sadeghi, Banert, Giselsson 2024).

Every variable is a list of 4 tensors (u, w, p, q).
The state carried between iterations is the tuple

    (x, y_prev, p_prev, z_prev, u, v, u_prev, v_prev)

where u, v are the deviations of the current iteration.
"""

import torch
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm


def zero_state(shapes, device):
    def zeros():
        return [torch.zeros(s, device=device) for s in shapes]
    return tuple(zeros() for _ in range(8))


def one_step(x, y_prev, p_prev, z_prev, u, v, n, params, C, RA):
    """One iteration of the algorithm (steps 5-8)."""
    a = params.alpha(n)      # constant step size, so alpha_bar = alpha
    lam = params.lam(n)
    cu = params.theta_bar(n) * params.gamma0 * params.beta_bar / params.theta_hat(n)
    blocks = range(len(x))

    y = [x[i] + a * (y_prev[i] - x[i]) + u[i] for i in blocks]
    z = [x[i] + a * (p_prev[i] - x[i]) + a * (z_prev[i] - p_prev[i]) + cu * u[i] + v[i]
         for i in blocks]

    Cy = C(y)
    p = RA(z, Cy)

    x_new = [x[i] + lam * (p[i] - z[i]) + a * lam * (z_prev[i] - p_prev[i])
             for i in blocks]

    residual = torch.sqrt(sum((p[i] - y[i]).pow(2).sum() for i in blocks) + 1e-12)

    return x_new, y, p, z, Cy, residual


def safeguard(u_raw, v_raw, delta, n, params, M_norm_sq, alpha):
    """
    Rescale the raw directions (u_raw, v_raw) into deviations for iteration
    n + 1 that satisfy the safeguarding condition (3):

        (lam + mu) * (theta_tilde/theta_hat ||u||_M^2 + theta_hat/theta ||v||_M^2)
            <= zeta * delta

    The directions are first normalised, then scaled to use a fraction
    alpha^2 of the allowed budget.
    """
    def normalise(blocks):
        norm = torch.sqrt(sum(b.pow(2).sum() for b in blocks)).clamp(min=1e-6)
        return [b / norm for b in blocks]

    u_raw, v_raw = normalise(u_raw), normalise(v_raw)

    lam_mu = params.lam(n + 1) + params.mu(n + 1)
    c_u = lam_mu * params.theta_tilde(n + 1) / params.theta_hat(n + 1)
    c_v = lam_mu * params.theta_hat(n + 1) / params.theta(n + 1)

    Q = c_u * M_norm_sq(u_raw) + c_v * M_norm_sq(v_raw)
    scale = alpha * torch.sqrt(params.zeta * delta / Q.clamp(min=1e-12))

    return [scale * b for b in u_raw], [scale * b for b in v_raw]


def fbs_iteration(n, state, functions, params, direction, alpha, use_safeguard):
    """
    One iteration + choice of the next deviations.

    direction : None (zero deviations) or a function
                (x, p, y, z, u_prev, v_prev, Cy) -> (u_raw, v_raw)
    """
    x, y_prev, p_prev, z_prev, u, v, u_prev, v_prev = state

    x_new, y, p, z, Cy, residual = one_step(
        x, y_prev, p_prev, z_prev, u, v, n, params, functions["C"], functions["RA"])

    if direction is None:
        u_next, v_next = u, v
    else:
        u_raw, v_raw = direction(x_new, p, y, z, u_prev, v_prev, Cy)
        if use_safeguard:
            delta = functions["compute_delta"](p, x, p_prev, z, z_prev, y, y_prev, u, v, n)
            u_next, v_next = safeguard(u_raw, v_raw, delta, n, params,
                                       functions["M_norm_sq"], alpha)
        else:
            u_next = [alpha * b for b in u_raw]
            v_next = [alpha * b for b in v_raw]

    return (x_new, y, p, z, u_next, v_next, u, v), residual


def unroll(functions, params, shapes, T, device, direction=None, alpha=0.99,
           use_safeguard=True, keep_history=False, progress=False):
    """
    Run T iterations from x = 0.

    Returns
        kkt       : KKT residual at every iteration. When gradients are
                    enabled only kkt[-1] carries a graph (training loss).
        residuals : fixed-point residual ||p_n - y_n|| at every iteration
        x_hist    : primal iterates [u, w] (every iteration if keep_history,
                    otherwise only the last one)
    """
    state = zero_state(shapes, device)
    with_grad = torch.is_grad_enabled()
    kkt, residuals, x_hist = [], [], []

    for n in tqdm(range(T), disable=not progress, leave=False):
        args = (n, state, functions, params, direction, alpha, use_safeguard)
        if with_grad:
            # activations are recomputed during backward instead of stored
            state, residual = checkpoint(fbs_iteration, *args, use_reentrant=False)
        else:
            state, residual = fbs_iteration(*args)

        x = state[0]
        with torch.set_grad_enabled(with_grad and n == T - 1):
            kkt.append(functions["kkt_residual_norm"](x))
        residuals.append(residual.detach())
        if keep_history or n == T - 1:
            x_hist.append([b.detach() for b in x[:2]])

    return kkt, residuals, x_hist
