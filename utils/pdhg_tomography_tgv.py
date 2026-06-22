

import numpy as np
import torch
import odl


def _odl_image_to_torch(odl_elem, device):
    """ODL element on the image space U -> torch tensor [1, 1, H, W]."""
    arr = np.asarray(odl_elem)
    return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)


def build_pdhg_problem(setup, params):
    """Assemble the PDHG operator `op` and functionals `f`, `g` for the problem
    defined by `setup` / `params`.  Returns (op, f, g, domain)."""
    U = setup["space"]
    A = setup["ray_transform"]        
    norm_A_const = odl.power_method_opnorm(A, maxiter=30)
    A_n = (1.0 / norm_A_const) * A

    y_np = np.asarray(setup["data"].detach().cpu())[0, 0]
    y_odl = A.range.element(y_np)

    # TGV operators, identical to get_setup.
    G  = odl.Gradient(U, method="forward", pad_mode="symmetric")
    V  = G.range
    Dx = odl.PartialDerivative(U, 0, method="forward", pad_mode="symmetric")
    Dy = odl.PartialDerivative(U, 1, method="forward", pad_mode="symmetric")
    E  = odl.operator.ProductSpaceOperator(
        [[Dx, 0], [0, Dy], [0.5 * Dy, 0.5 * Dx]])
    W  = E.range

    domain = odl.ProductSpace(U, V)


    op = odl.BroadcastOperator(
        A_n * odl.ComponentProjection(domain, 0),
        odl.ReductionOperator(G, odl.ScalingOperator(V, -1)),
        E * odl.ComponentProjection(domain, 1))

    f = odl.solvers.ZeroFunctional(domain)


    data_fit = 0.5 * odl.solvers.L2NormSquared(A.range).translated(y_odl)


    l1_grad = params.alpha1 * odl.solvers.L1Norm(V)
    l1_sym  = params.alpha2 * odl.solvers.L1Norm(W)

    g = odl.solvers.SeparableSum(data_fit, l1_grad, l1_sym)

    return op, f, g, domain


class _ObjectiveTracker:

    def __init__(self, obj_fn, device):
        self.obj_fn = obj_fn
        self.device = device
        self.hist = []

    def __call__(self, x):
        u = torch.tensor(np.asarray(x[0]), dtype=torch.float32,
                         device=self.device).unsqueeze(0).unsqueeze(0)
        w = torch.tensor(np.asarray(x[1]), dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        self.hist.append(float(self.obj_fn(u, w)))


def run_pdhg(setup, params, niter=300, verbose=False, obj_fn=None):

    device = setup["device"]
    op, f, g, domain = build_pdhg_problem(setup, params)

    # Step sizes: σ τ ‖K‖² < 1  (10 % margin on the operator norm).
    op_norm = 1.1 * odl.power_method_opnorm(op, maxiter=50)
    tau = sigma = 1.0 / op_norm

    callbacks = []
    if verbose:
        callbacks.append(odl.solvers.CallbackPrintIteration())
    tracker = _ObjectiveTracker(obj_fn, device) if obj_fn is not None else None
    if tracker is not None:
        callbacks.append(tracker)
    callback = None
    for cb in callbacks:
        callback = cb if callback is None else (callback & cb)

    x = op.domain.zero()
    odl.solvers.pdhg(x, f, g, op, niter=niter, tau=tau, sigma=sigma,
                     callback=callback)

    rec = _odl_image_to_torch(x[0], device)
    if tracker is not None:
        return rec, tracker.hist
    return rec



