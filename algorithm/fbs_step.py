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

    The directions are first normalised, then scaled to use at most a
    fraction alpha^2 of the allowed budget. The network chooses how much of
    it: a raw direction with root mean square r is shrunk by r / sqrt(r^2 + 1)
    (close to 1 for a large output, close to 0 for a small one), which can
    only make the left-hand side smaller.

    The square roots are written so that their gradient stays finite at 0.
    """
    def normalise(blocks):
        """Unit direction and its shrinking factor in [0, 1)."""
        sq = sum(b.pow(2).sum() for b in blocks)
        size = sum(b.numel() for b in blocks[:2])     # the dual blocks are zero
        norm = torch.sqrt(sq + 1e-12)
        return [b / norm for b in blocks], norm / torch.sqrt(sq + size)

    def safe_sqrt(x):
        positive = x > 0
        return torch.where(positive, torch.sqrt(torch.where(positive, x, torch.ones_like(x))),
                           torch.zeros_like(x))

    (u_raw, shrink_u), (v_raw, shrink_v) = normalise(u_raw), normalise(v_raw)

    lam_mu = params.lam(n + 1) + params.mu(n + 1)
    c_u = lam_mu * params.theta_tilde(n + 1) / params.theta_hat(n + 1)
    c_v = lam_mu * params.theta_hat(n + 1) / params.theta(n + 1)

    Q = c_u * M_norm_sq(u_raw) + c_v * M_norm_sq(v_raw)
    scale = alpha * safe_sqrt(params.zeta * delta / Q.clamp(min=1e-12))

    return ([shrink_u * scale * b for b in u_raw], [shrink_v * scale * b for b in v_raw])


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
           use_safeguard=True, keep_history=False, progress=False, monitor=True):
    """
    Run T iterations from x = 0.

    Returns
        kkt       : KKT residual at every iteration. When gradients are
                    enabled only kkt[-1] carries a graph (training loss).
                    Empty if monitor is False (used to time the iterations).
        residuals : fixed-point residual ||p_n - y_n|| at every iteration
        x_hist    : primal iterates [u, w] (every iteration if keep_history,
                    otherwise only the last one). When gradients are enabled
                    the kept iterates carry a graph, for losses other than the
                    KKT residual.
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
        if monitor:
            with torch.set_grad_enabled(with_grad and n == T - 1):
                kkt.append(functions["kkt_residual_norm"](x))
        residuals.append(residual.detach())
        if keep_history or n == T - 1:
            x_hist.append(list(x[:2]) if with_grad else [b.detach() for b in x[:2]])

    return kkt, residuals, x_hist
