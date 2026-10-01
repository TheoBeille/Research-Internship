"""
Problem setup for TGV2 tomography, written as the monotone inclusion

    0 in A(x) + C(x),      x = (u, w, p, q)

    u : image                      [1, 1, H, W]
    w : auxiliary vector field     [1, 2, H, W]
    p : dual of (grad u - w)       [1, 2, H, W]
    q : dual of E w                [1, 3, H, W]

The operators are built with ODL and wrapped as torch functions.
"""

import numpy as np
import odl
import odl.contrib.torch as odl_torch
import torch
from odl.operator.pspace_ops import ProductSpaceOperator


# ============================================================
# Operators and data
# ============================================================

def _make_ray_transform(space, n_angles):
    """Parallel-beam ray transform, using the first available backend."""
    angles = odl.uniform_partition(0, 2 * np.pi, n_angles)
    detector = odl.uniform_partition(-360, 360, space.shape[0])
    geometry = odl.tomo.Parallel2dGeometry(angles, detector)
    for impl in ("astra_cuda", "astra_cpu", "skimage"):
        try:
            return odl.tomo.RayTransform(space, geometry, impl=impl)
        except Exception as err:
            last_err = err
    raise RuntimeError(f"No usable RayTransform backend found: {last_err}")


def _wrap(odl_operator):
    """ODL operator -> function on torch tensors of shape [1, C, H, W]."""
    layer = odl_torch.OperatorModule(odl_operator)

    def apply(x):
        out = layer(x.cpu()).to(x.device)
        if out.dim() == 5:          # product-space output: [1, 1, C, H, W]
            out = out.squeeze(1)
        if out.dim() == 3:          # product-space input: [1, H, W]
            out = out.unsqueeze(0)
        return out.float()

    return apply


def _norm(blocks):
    return torch.sqrt(sum(b.pow(2).sum() for b in blocks))


def _operator_norm(op, adj, x, n_iter):
    """Largest singular value of `op` by power iteration (x: list of tensors)."""
    for _ in range(n_iter):
        x = adj(op(x))
        x = [b / _norm(x) for b in x]
    return _norm(op(x)).item()


def get_setup(size, n_angles=180, seed=0, noise_level=0.0, device=None,
              phantom_array=None):
    """
    Build the operators and the noisy sinogram for one ground-truth image.

    phantom_array : (size, size) numpy array with values in [0, 1]
    seed          : only used for the noise
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    a = size / 2
    space = odl.uniform_discr([-a, -a], [a, a], [size, size], dtype="float32")
    ray_transform = _make_ray_transform(space, n_angles)

    D = odl.Gradient(space, method="forward", pad_mode="symmetric")
    Dx = odl.PartialDerivative(space, 0, method="forward", pad_mode="symmetric")
    Dy = odl.PartialDerivative(space, 1, method="forward", pad_mode="symmetric")
    E = ProductSpaceOperator([[Dx, 0], [0, Dy], [0.5 * Dy, 0.5 * Dx]])

    grad, gradT = _wrap(D), _wrap(D.adjoint)
    sym_grad, sym_gradT = _wrap(E), _wrap(E.adjoint)
    K_raw, KT_raw = _wrap(ray_transform), _wrap(ray_transform.adjoint)

    # normalise the ray transform so that ||K|| = 1
    norm_K = _operator_norm(lambda x: [K_raw(x[0])], lambda r: [KT_raw(r[0])],
                            [torch.randn(1, 1, size, size, device=device)], n_iter=30)

    def K(u):
        return K_raw(u) / norm_K

    def KT(r):
        return KT_raw(r) / norm_K

    # B(u, w) = (grad u - w, E w),   B^T(p, q) = (grad^T p, -p + E^T q)
    def B(x):
        return [grad(x[0]) - x[1], sym_grad(x[1])]

    def BT(xi):
        return [gradT(xi[0]), -xi[0] + sym_gradT(xi[1])]

    norm_B = _operator_norm(B, BT,
                            [torch.randn(1, 1, size, size, device=device),
                             torch.randn(1, 2, size, size, device=device)], n_iter=60)

    phantom = torch.tensor(np.asarray(phantom_array), dtype=torch.float32,
                           device=device)[None, None]
    data = K(phantom)
    if noise_level > 0:
        g = torch.Generator().manual_seed(int(seed) + 10_000)
        noise = torch.randn(data.shape, generator=g).to(device)
        data = data + noise_level * data.abs().mean() * noise

    return dict(
        device=device,
        K=K, KT=KT, grad=grad, gradT=gradT, E=sym_grad, ET=sym_gradT,
        norm_B=norm_B,
        data=data,                    # noisy sinogram
        initial_state=KT(data),       # back-projection (for display only)
        phantom=phantom,              # ground truth
        space=space, ray_transform=ray_transform,   # ODL objects, used by PDHG
    )


# ============================================================
# Algorithm parameters (Sadeghi, Banert, Giselsson 2024)
# ============================================================

class Params:

    def __init__(self, lam0=0.9, beta_bar=1.0, gamma0=2, alpha1=0.0005,
                 alpha2=0.04, zeta=0.9, size=128):
        self.lam0 = lam0
        self.beta_bar = beta_bar
        self.gamma0 = gamma0       # constant step size
        self.alpha1 = alpha1       # ||p||_inf <= alpha1
        self.alpha2 = alpha2       # ||q||_inf <= alpha2
        self.zeta = zeta           # fraction of the budget the deviations may use
        self.size = size

    def lam(self, n):
        return self.lam0 * (1 + n) ** 0.3

    def mu(self, n):
        return self.lam(n) ** 2 / self.lam0 - self.lam(n)

    def alpha(self, n):
        return (self.lam(n) - self.lam0) / self.lam(n)

    # the four theta sequences are all (constant / lam0) * lam_n^2
    def _scaled(self, c, n):
        return c / self.lam0 * self.lam(n) ** 2

    def theta(self, n):
        return self._scaled(4 - self.gamma0 * self.beta_bar - 2 * self.lam0, n)

    def theta_hat(self, n):
        return self._scaled(2 - self.lam0 * self.gamma0 * self.beta_bar, n)

    def theta_bar(self, n):
        return self._scaled(1 - self.lam0, n)

    def theta_tilde(self, n):
        return self._scaled(self.gamma0 * self.beta_bar, n)


# ============================================================
# TGV2 objective
# ============================================================

def make_objective(setup, params):
    """F(u, w) = 1/2 ||K u - y||^2 + alpha1 ||grad u - w||_1 + alpha2 ||E w||_1"""
    K, grad, E, y = setup["K"], setup["grad"], setup["E"], setup["data"]

    def objective(u, w):
        return (0.5 * (K(u) - y).pow(2).sum()
                + params.alpha1 * (grad(u) - w).abs().sum()
                + params.alpha2 * E(w).abs().sum())

    return objective


# ============================================================
# Functions used by the algorithm
# ============================================================

def build_algo_functions(setup, params, step_safety=0.95):
    """
    Returns the functions of the forward-backward iteration.

    The resolvent is taken in the metric

        M = gamma * [[ I/s , -B^T ],
                     [ -B  ,  I/s ]],        s = step_safety / ||B||,

    which makes p = (M + gamma A)^{-1}(M z - gamma C y) explicit (one
    PDHG-like step). gamma cancels in this step: it only enters the algorithm
    through the theta coefficients, as the product gamma * beta_bar.
    """
    K, KT, y = setup["K"], setup["KT"], setup["data"]
    grad, gradT, E, ET = setup["grad"], setup["gradT"], setup["E"], setup["ET"]
    s = step_safety / setup["norm_B"]

    def B(u, w):
        return grad(u) - w, E(w)

    def BT(p, q):
        return gradT(p), -p + ET(q)

    def dot(a, b):
        return sum((ai * bi).sum() for ai, bi in zip(a, b))

    def C(x):
        """Forward operator: gradient of the data term (acts on u only)."""
        return [KT(K(x[0]) - y)] + [torch.zeros_like(b) for b in x[1:]]

    def resolvent(z, Cy):
        zu, zw, zp, zq = z
        bt_p, bt_q = BT(zp, zq)
        pu = zu - s * (Cy[0] + bt_p)
        pw = zw - s * (Cy[1] + bt_q)
        b_u, b_w = B(2 * pu - zu, 2 * pw - zw)
        pp = torch.clamp(zp + s * b_u, -params.alpha1, params.alpha1)
        pq = torch.clamp(zq + s * b_w, -params.alpha2, params.alpha2)
        return [pu, pw, pp, pq]

    def M_inner(a, b):
        cross = dot(B(a[0], a[1]), b[2:]) + dot(B(b[0], b[1]), a[2:])
        return params.gamma0 * (dot(a, b) / s - cross)

    def M_norm_sq(a):
        return M_inner(a, a)

    def compute_delta(p, x, p_prev, z, z_prev, y_, y_prev, u, v, n):
        """Safeguarding budget l_n, eq. (4) of Sadeghi et al."""
        gamma, beta = params.gamma0, params.beta_bar
        mu, a = params.mu(n), params.alpha(n)
        cu = gamma * beta * params.lam(n) ** 2 / params.theta_hat(n)
        cv = 2 * params.theta_bar(n) / params.theta(n)

        core = [p[i] - x[i] + a * (x[i] - p_prev[i]) + cu * u[i] - cv * v[i]
                for i in range(4)]
        diff_z = [((z[i] - p[i]) - (z_prev[i] - p_prev[i])) / gamma for i in range(4)]
        diff_p = [p[i] - p_prev[i] for i in range(4)]
        diff_py = [(p[i] - y_[i]) - (p_prev[i] - y_prev[i]) for i in range(4)]

        delta = (params.theta(n) / 2 * M_norm_sq(core)
                 + 2 * mu * gamma * M_inner(diff_z, diff_p)
                 + mu * gamma * beta / 2 * M_norm_sq(diff_py))
        return torch.clamp(delta, min=0.0)

    def kkt_residual_norm(x):
        """|| (A + C)(x) ||, zero exactly at a solution."""
        u, w, p, q = x
        r1 = KT(K(u) - y) + gradT(p)
        r2 = -p + ET(q)
        r3 = p - torch.clamp(p + grad(u) - w, -params.alpha1, params.alpha1)
        r4 = q - torch.clamp(q + E(w), -params.alpha2, params.alpha2)
        return _norm([r1, r2, r3, r4])

    tgv_objective = make_objective(setup, params)

    def objective(x):
        return tgv_objective(x[0], x[1])

    return dict(
        C=C,
        RA=resolvent,
        compute_delta=compute_delta,
        M_norm_sq=M_norm_sq,
        kkt_residual_norm=kkt_residual_norm,
        objective=objective,
    )
