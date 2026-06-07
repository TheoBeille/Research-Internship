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


def _make_ray_transform(U, n_angles):
    """Parallel-beam ray transform, trying the available ASTRA/skimage impls."""
    geometry = parallel_beam_geometry(U, num_angles=n_angles)
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


def get_setup(size, n_angles=180, seed=0, noise_level=0.0, device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   
    U = odl.uniform_discr([0, 0], [size, size], [size, size], dtype='float32')

    A = _make_ray_transform(U, n_angles)
    #A=odl.IdentityOperator(U)
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

    # ---- ODL operators wrapped as torch modules -----------------------------
    A_layer     = odl_torch.OperatorModule(A)
    A_adj_layer = odl_torch.OperatorModule(A.adjoint)
    D_layer     = odl_torch.OperatorModule(D)
    D_adj_layer = odl_torch.OperatorModule(D.adjoint)
    E_layer     = odl_torch.OperatorModule(E)
    E_adj_layer = odl_torch.OperatorModule(E.adjoint)

    def odl_op(layer, x):
        dev = x.device
        return layer(x.cpu()).to(dev)

  
    def A_raw(u):
        return ensure_4d(odl_op(A_layer, u)).float()

    def AT_raw(r):
        return ensure_4d(odl_op(A_adj_layer, r)).float()

    
    def grad_torch(u):
        return ensure_4d(odl_op(D_layer, u)).float()

    def gradT_torch(p):
        return ensure_4d(odl_op(D_adj_layer, p)).float()

    def E_torch(w):
        return ensure_4d(odl_op(E_layer, w)).float()

    def ET_torch(q):
        return ensure_4d(odl_op(E_adj_layer, q)).float()

 
    norm_A = _power_norm(A_raw, AT_raw, (1, 1, size, size), device)

    def A_torch(u):
        return A_raw(u) / norm_A

    def AT_torch(r):
        return AT_raw(r) / norm_A


    def K_torch(u):
        out0 = gradT_torch(u[2])
        out1 = -u[2] + ET_torch(u[3])
        out2 = -grad_torch(u[0]) + u[1]
        out3 = -E_torch(u[1])
        return [out0, out1, out2, out3]

    # ---- data: sinogram of the phantom (+ optional noise) -------------------
    phantom_t = (torch.tensor(np.asarray(phantom), dtype=torch.float32)
                 .unsqueeze(0).unsqueeze(0).to(device))          # [1,1,H,W]
    y = A_torch(phantom_t)                                       # sinogram

    if noise_level > 0:
        g = torch.Generator(device='cpu').manual_seed(int(seed) + 10_000)
        noise = torch.randn(y.shape, generator=g).to(device)
        y = y + noise_level * y.abs().mean() * noise
    init = AT_torch(y)    

    norm_B = _norm_B(grad_torch, gradT_torch, E_torch, ET_torch, size, device)

    return dict(
        space=U,
        domain=domain,
        device=device,
        size=size,
        n_angles=n_angles,
        A=A_torch, AT=AT_torch,
        K=K_torch,
        grad=grad_torch, gradT=gradT_torch, E=E_torch, ET=ET_torch,
        data=y,                       # sinogram (the measured data)
        initial_state=init,           # back-projection (algorithm start, image space)
        phantom=phantom_t,            # clean image (ground truth)
        norm_A=1.0,                   # after normalization
        norm_B=norm_B,
        ray_transform=A,
        A_layer=A_layer, A_adj_layer=A_adj_layer,
        D_layer=D_layer, D_adj_layer=D_adj_layer,
        E_layer=E_layer, E_adj_layer=E_adj_layer,
    )


# ============================================================
# PARAMETERS  (identical to the denoising code, see internship note)
# ============================================================

class Params:

    def __init__(self,
                 lam0=0.9,
                 beta_bar=1.0,
                 gamma0=1,
                 alpha1=0.1,
                 alpha2=0.1,
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
# BUILD ALGORITHM FUNCTIONS
# ============================================================

def build_algo_functions(setup, params, gamma_safety=0.85, step_safety=0.95,
                         resolvent_iter=100, resolvent_tol=1e-7):
    device = setup['device']

    A, AT = setup['A'], setup['AT']
    y      = setup['data']                       # sinogram
    grad_torch  = setup['grad']
    gradT_torch = setup['gradT']
    E_torch     = setup['E']
    ET_torch    = setup['ET']
    K_torch     = setup['K']
    norm_B      = float(setup['norm_B'])


    params.beta_bar = max(params.beta_bar, setup['norm_A'] ** 2) 
    coc_max         = 2.0 / (params.lam0 * params.beta_bar)        
    params.gamma0   = gamma_safety * coc_max

    def proj_linf_ball(z, alpha):
        return torch.clamp(z, -alpha, alpha)

   
    def resolvent_A(z, gamma, max_iter=resolvent_iter, tol=resolvent_tol):
        zu, zw, zp, zq = z
       
        u = zu.clone(); w = zw.clone()
        ua = u.clone(); wa = w.clone()                # extrapolated point
        L = 1.0 + (gamma * norm_B) ** 2
        step = step_safety / L
        t_prev = 1.0

        for _ in range(max_iter):
            p = proj_linf_ball(zp + gamma * (grad_torch(ua) - wa), params.alpha1)
            q = proj_linf_ball(zq + gamma * E_torch(wa),           params.alpha2)

            g_u = (ua - zu) + gamma * gradT_torch(p)
            g_w = (wa - zw) + gamma * (-p + ET_torch(q))

            u_new = ua - step * g_u
            w_new = wa - step * g_w

            t_new = 0.5 * (1.0 + (1.0 + 4.0 * t_prev ** 2) ** 0.5)
            beta  = (t_prev - 1.0) / t_new
            ua = u_new + beta * (u_new - u)
            wa = w_new + beta * (w_new - w)

            res = torch.sqrt((u_new - u).pow(2).sum() + (w_new - w).pow(2).sum())
            u, w, t_prev = u_new, w_new, t_new
            if res.item() < tol:
                break

        
        p = proj_linf_ball(zp + gamma * (grad_torch(u) - w), params.alpha1)
        q = proj_linf_ball(zq + gamma * E_torch(w),          params.alpha2)
        return [u, w, p, q]


    def C(x):
        u = x[0]
        data_grad = AT(A(u) - y)
        return [
            data_grad,
            torch.zeros_like(x[1]),
            torch.zeros_like(x[2]),
            torch.zeros_like(x[3]),
        ]


    def kkt_residual(x):
        u, w, p, q = x
        r1 = AT(A(u) - y) + gradT_torch(p)
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
        term1 = th / 2 * sum(c.norm()**2 for c in core)

        diff_z  = [(z[i] - p[i]) / gam - (z_prev[i] - p_prev[i]) / gam_prev
                   for i in range(4)]
        diff_pp = [p[i] - p_prev[i] for i in range(4)]
        term2   = 2 * mu_n * gam * sum(
                      (diff_z[i] * diff_pp[i]).sum() for i in range(4))

        diff_py = [(p[i] - y_[i]) - (p_prev[i] - y_prev[i]) for i in range(4)]
        term3   = (mu_n * gam * params.beta_bar / 2.0) * sum(
                      d.norm()**2 for d in diff_py)

        result = torch.as_tensor(term1 + term2 + term3,
                                 dtype=torch.float32, device=device)
        return torch.clamp(result, min=0.0)

    return dict(
        RA=resolvent_A,
        C=C,
        K=K_torch,
        A=A, AT=AT,
        grad=grad_torch,
        gradT=gradT_torch,
        E=E_torch,
        ET=ET_torch,
        data=y,
        compute_delta_torch=compute_delta_torch,
        kkt_residual=kkt_residual,
        kkt_residual_norm=kkt_residual_norm,
    )