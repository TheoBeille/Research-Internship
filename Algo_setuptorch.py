import odl
import numpy as np
import odl.contrib.torch as odl_torch
import torch
from odl.operator.pspace_ops import ProductSpaceOperator
from odl.tomo.geometry.parallel import parallel_beam_geometry

# ============================================================
# SETUP PROBLEM in ODL + Converting in torch  --  TOMOGRAPHY
# ============================================================

def ensure_4d(x):
    """Ensure tensor is [B, C, H, W]."""
    if x.dim() == 5:
        x = x.squeeze(1)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return x.float()


def _make_ray_transform(U, angle_partition, detector_partition):
    """Parallel-beam ray transform, trying the available ASTRA/skimage impls."""
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
    last_err = None
    for impl in ("astra_cuda", "astra_cpu", "skimage"):
        try:
            return odl.tomo.RayTransform(U, geometry, impl=impl)
        except Exception as e:                       
            last_err = e
            continue
    raise RuntimeError(f"No usable RayTransform backend found: {last_err}")


def _power_norm(fwd, adj, in_shape, device, n_iter=30):
    """Largest singular value of `fwd` via power iteration on adj o fwd."""
    x = torch.randn(*in_shape, device=device)
    x = x / x.norm().clamp(min=1e-12)
    for _ in range(n_iter):
        x = adj(fwd(x))
        x = x / x.norm().clamp(min=1e-12)
    return fwd(x).norm().item()


def _norm_B(grad, gradT, E, ET, size, device, n_iter=60):
    """||B|| with B(u,w) = (grad u - w, E w),  B^T(p,q) = (gradT p, -p + ET q)."""
    u = torch.randn(1, 1, size, size, device=device)
    w = torch.randn(1, 2, size, size, device=device)
    n = (u.pow(2).sum() + w.pow(2).sum()).sqrt().clamp(min=1e-12)
    u, w = u / n, w / n
    for _ in range(n_iter):
        p = grad(u) - w
        q = E(w)
        u2 = gradT(p)
        w2 = -p + ET(q)
        n = (u2.pow(2).sum() + w2.pow(2).sum()).sqrt().clamp(min=1e-12)
        u, w = u2 / n, w2 / n
    p = grad(u) - w
    q = E(w)
    return (p.pow(2).sum() + q.pow(2).sum()).sqrt().item()


def get_setup(size, n_angles=180, seed=0, noise_level=0.0, device=None,
              phantom_array=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    a = size / 2
    U = odl.uniform_discr([-a, -a], [a, a], [size, size], dtype='float32')
    angle_partition = odl.uniform_partition(0, 2 * np.pi, n_angles)
    detector_partition = odl.uniform_partition(-360, 360, size)
    K = _make_ray_transform(U, angle_partition, detector_partition)

    if phantom_array is not None:
        assert phantom_array.shape == (size, size), (
            f"phantom_array shape {phantom_array.shape} != ({size}, {size})"
        )
        phantom = phantom_array
    else:
        phantom = odl.phantom.tgv_phantom(U)

    D  = odl.Gradient(U, method='forward', pad_mode='symmetric')
    Dx = odl.PartialDerivative(U, 0, method='forward', pad_mode='symmetric')
    Dy = odl.PartialDerivative(U, 1, method='forward', pad_mode='symmetric')
    V  = D.range

    E = ProductSpaceOperator(
        [
            [Dx, 0],
            [0, Dy],
            [0.5 * Dy, 0.5 * Dx],
        ]
    )

    domain = odl.ProductSpace(U, V)

    # ---- ODL operators wrapped as torch modules
    K_layer     = odl_torch.OperatorModule(K)
    K_adj_layer = odl_torch.OperatorModule(K.adjoint)
    D_layer     = odl_torch.OperatorModule(D)
    D_adj_layer = odl_torch.OperatorModule(D.adjoint)
    E_layer     = odl_torch.OperatorModule(E)
    E_adj_layer = odl_torch.OperatorModule(E.adjoint)

    def odl_op(layer, x):
        dev = x.device
        return layer(x.cpu()).to(dev)

    def K_raw(u):
        return ensure_4d(odl_op(K_layer, u)).float()

    def KT_raw(r):
        return ensure_4d(odl_op(K_adj_layer, r)).float()

    def grad_torch(u):
        return ensure_4d(odl_op(D_layer, u)).float()

    def gradT_torch(p):
        return ensure_4d(odl_op(D_adj_layer, p)).float()

    def E_torch(w):
        return ensure_4d(odl_op(E_layer, w)).float()

    def ET_torch(q):
        return ensure_4d(odl_op(E_adj_layer, q)).float()

    norm_K = _power_norm(K_raw, KT_raw, (1, 1, size, size), device)

    def K_torch(u):
        return K_raw(u) / norm_K

    def KT_torch(r):
        return KT_raw(r) / norm_K

    def A_torch(u):
        out0 = gradT_torch(u[2])
        out1 = -u[2] + ET_torch(u[3])
        out2 = -grad_torch(u[0]) + u[1]
        out3 = -E_torch(u[1])
        return [out0, out1, out2, out3]

    phantom_t = (torch.tensor(np.asarray(phantom), dtype=torch.float32)
                 .unsqueeze(0).unsqueeze(0).to(device))          # [1,1,H,W]
    y = K_torch(phantom_t)                                       # sinogram

    if noise_level > 0:
        g = torch.Generator(device='cpu').manual_seed(int(seed) + 10_000)
        noise = torch.randn(y.shape, generator=g).to(device)
        y = y + noise_level * y.abs().mean() * noise
    init = KT_torch(y)    

    norm_B = _norm_B(grad_torch, gradT_torch, E_torch, ET_torch, size, device)

    return dict(
        space=U,
        domain=domain,
        device=device,
        size=size,
        n_angles=n_angles,
        K=K_torch, KT=KT_torch,
        A=A_torch,
        grad=grad_torch, gradT=gradT_torch, E=E_torch, ET=ET_torch,
        data=y,                       # sinogram (the measured data)
        initial_state=init,           # back-projection (algorithm start, image space)
        phantom=phantom_t,            # clean image (ground truth)
        norm_K=1.0,                   # after normalization
        norm_B=norm_B,
        ray_transform=K,
        K_layer=K_layer, K_adj_layer=K_adj_layer,
        D_layer=D_layer, D_adj_layer=D_adj_layer,
        E_layer=E_layer, E_adj_layer=E_adj_layer,
    )


# ============================================================
# PARAMETERS
# ============================================================

class Params:

    def __init__(self,
                 lam0=0.9,
                 beta_bar=1.0,
                 gamma0=2,
                 alpha1=0.0005,
                 alpha2=0.04,
                 zeta=0.9,
                 size=128,
                 ):
        self.lam0     = lam0
        self.beta_bar = beta_bar
        self.gamma0   = gamma0
        self.alpha1   = alpha1     # ||p||_inf <= alpha1
        self.alpha2   = alpha2     # ||q||_inf <= alpha2
        self.zeta     = zeta
        self.size     = size

    def lam(self, n):
        return self.lam0 * (1 + n) ** 0.3

    def gamma(self, n):
        return self.gamma0

    def mu(self, n):
        l = self.lam(n)
        return (1 / self.lam0) * l**2 - l

    def alpha(self, n):
        l = self.lam(n)
        return 0 if l == 0 else (l - self.lam0) / l

    def alpha_bar(self, n):
        gam = self.gamma(n)
        gam_prev = self.gamma(n - 1) if n > 0 else gam
        if n == 0:
            return 0
        l = self.lam(n)
        return (gam / gam_prev) * (l - self.lam0) / l

    def theta(self, n):
        gam = self.gamma(n)
        return (4 - gam * self.beta_bar - 2 * self.lam0) \
               / self.lam0 * self.lam(n)**2

    def theta_hat(self, n):
        gam = self.gamma(n)
        return (2 - self.lam0 * gam * self.beta_bar) \
               / self.lam0 * self.lam(n)**2

    def theta_bar(self, n):
        return (1 - self.lam0) / self.lam0 * self.lam(n)**2

    def theta_tilde(self, n):
        gam = self.gamma(n)
        return gam * self.beta_bar / self.lam0 * self.lam(n)**2


# ============================================================
# TGV² OBJECTIVE
# ============================================================

def make_objective(setup, params, alpha1=None, alpha2=None):
    K    = setup["K"]        
    grad = setup["grad"]     
    E    = setup["E"]        
    y    = setup["data"]     
    a1 = params.alpha1 if alpha1 is None else alpha1
    a2 = params.alpha2 if alpha2 is None else alpha2

    def parts(u, w):
        data = 0.5 * (K(u) - y).pow(2).sum()
        reg1 = a1 * (grad(u) - w).abs().sum()       
        reg2 = a2 * E(w).abs().sum()                
        return data, reg1, reg2

    def objective(u, w):
        data, reg1, reg2 = parts(u, w)
        return data + reg1 + reg2

    objective.parts = parts
    objective.alpha1 = a1
    objective.alpha2 = a2
    return objective


def objective_history(objective, x_hist):
    return np.asarray([float(objective(x[0], x[1])) for x in x_hist])


# ============================================================
# BUILD ALGORITHM FUNCTIONS (Closed-Form Resolvent via Metric M)
# ============================================================

def build_algo_functions(setup, params, step_safety=0.95):
    device = setup['device']

    K, KT = setup['K'], setup['KT']
    y      = setup['data']                       
    grad_torch  = setup['grad']
    gradT_torch = setup['gradT']
    E_torch     = setup['E']
    ET_torch    = setup['ET']
    A_torch     = setup['A']
    norm_B      = float(setup['norm_B'])

    params.beta_bar = max(params.beta_bar, setup['norm_K'] ** 2)

    def proj_linf_ball(z, alpha):
        return torch.clamp(z, -alpha, alpha)

    def B_op(u, w):
        return grad_torch(u) - w, E_torch(w)

    def Bt_op(p, q):
        return gradT_torch(p), -p + ET_torch(q)

    # ---- Closed-Form Resolvent using Metric M (Chambolle-Pock / PDHG style) ----
    def resolvent_A(z, cy, gamma):

        zu, zw, zp, zq = z
        cy_u, cy_w, cy_p, cy_q = cy

        rho = step_safety
        tau = rho / (gamma * norm_B)
        sigma = rho / (gamma * norm_B)

   
        bt_zp, bt_zq = Bt_op(zp, zq)
        p_u = zu - tau * gamma * cy_u - tau * gamma * bt_zp
        p_w = zw - tau * gamma * cy_w - tau * gamma * bt_zq

        b_zu, b_zw = B_op(zu, zw)
        b_pu, b_pw = B_op(p_u, p_w)

        arg_p = zp - sigma * gamma * b_zu + 2.0 * sigma * gamma * b_pu
        arg_q = zq - sigma * gamma * b_zw + 2.0 * sigma * gamma * b_pw

        p_p = proj_linf_ball(arg_p, params.alpha1)
        p_q = proj_linf_ball(arg_q, params.alpha2)

        return [p_u, p_w, p_p, p_q]

    # ---- Induced Metric M operations for Safeguard (`compute_delta_torch`) ----
    def M_inner(a, b, gamma):
        au, aw, ap, aq = a
        bu, bw, bp, bq = b
        
        rho = step_safety
        tau = rho / (gamma * norm_B)
        sigma = rho / (gamma * norm_B)

        inner_primal = (au * bu).sum() + (aw * bw).sum()
        inner_dual   = (ap * bp).sum() + (aq * bq).sum()

        b_au, b_aw = B_op(au, aw)
        b_bu, b_bw = B_op(bu, bw)

        cross_term1 = (b_au * bp).sum() + (b_aw * bq).sum()
        cross_term2 = (b_bu * ap).sum() + (b_bw * aq).sum()

        return (1.0 / tau) * inner_primal + (1.0 / sigma) * inner_dual - gamma * (cross_term1 + cross_term2)

    def M_norm_sq(vec, gamma):
        return M_inner(vec, vec, gamma)

    def C(x):
        u = x[0]
        data_grad = KT(K(u) - y)
        return [
            data_grad,
            torch.zeros_like(x[1]),
            torch.zeros_like(x[2]),
            torch.zeros_like(x[3]),
        ]

    def kkt_residual(x):
        u, w, p, q = x
        r1 = KT(K(u) - y) + gradT_torch(p)
        r2 = -p + ET_torch(q)
        r3 = p - proj_linf_ball(p + grad_torch(u) - w, params.alpha1)
        r4 = q - proj_linf_ball(q + E_torch(w), params.alpha2)
        return r1, r2, r3, r4

    def kkt_residual_norm(x):
        r1, r2, r3, r4 = kkt_residual(x)
        return torch.sqrt(
            r1.pow(2).sum() + r2.pow(2).sum()
            + r3.pow(2).sum() + r4.pow(2).sum()
        )

    _obj = make_objective(setup, params)

    def objective(x):
        return _obj(x[0], x[1])

    def objective_parts(x):
        return _obj.parts(x[0], x[1])

    def compute_delta_torch(p, x, p_prev, z, z_prev, y_, y_prev, u, v, n):
        gam = params.gamma(n)
        gam_prev = params.gamma(n - 1) if n > 0 else gam
        th   = params.theta(n)
        th_h = params.theta_hat(n)
        th_b = params.theta_bar(n)
        mu_n = params.mu(n)
        a_n  = params.alpha(n)

        core = [p[i] - x[i]
                + a_n * (x[i] - p_prev[i])
                + (gam * params.beta_bar * params.lam(n)**2 / th_h) * u[i]
                - (2 * th_b / th) * v[i]
                for i in range(4)]
        
       
        term1 = (th / 2.0) * M_norm_sq(core, gam)

        diff_z  = [(z[i] - p[i]) / gam - (z_prev[i] - p_prev[i]) / gam_prev
                   for i in range(4)]
        diff_pp = [p[i] - p_prev[i] for i in range(4)]
        
       
        term2   = 2.0 * mu_n * gam * M_inner(diff_z, diff_pp, gam)

        diff_py = [(p[i] - y_[i]) - (p_prev[i] - y_prev[i]) for i in range(4)]
        
        
        term3   = (mu_n * gam * params.beta_bar / 2.0) * M_norm_sq(diff_py, gam)

        result = torch.as_tensor(term1 + term2 + term3,
                                 dtype=torch.float32, device=device)
        return torch.clamp(result, min=0.0)

    return dict(
        RA=resolvent_A,
        C=C,
        A=A_torch,
        K=K, KT=KT,
        grad=grad_torch,
        gradT=gradT_torch,
        E=E_torch,
        ET=ET_torch,
        data=y,
        compute_delta_torch=compute_delta_torch,
        kkt_residual=kkt_residual,
        kkt_residual_norm=kkt_residual_norm,
        objective=objective,
        objective_parts=objective_parts,
    )