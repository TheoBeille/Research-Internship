"""TGV² tomography baseline solved with PDHG (Chambolle–Pock).

This is an *independent* solver for the **exact same** variational problem that
the monotone-inclusion code solves with `run_zero`:

    min_u  ½‖A u − y‖²  +  α₁‖∇u − w‖₁  +  α₂‖E w‖₁          (minimised over w)

where

  * `A`  is the parallel-beam ray transform, **normalised so ‖A‖ = 1**
         (exactly the operator `get_setup` wraps as `A_torch`);
  * `y`  is the (normalised) sinogram stored in `setup['data']`;
  * `∇`  is the forward gradient, `E` the symmetrised gradient — the same TGV
         operators built in `get_setup`;
  * the two ‖·‖₁ are the **anisotropic** (componentwise) ℓ¹ norms, which are the
         convex conjugates of the componentwise ℓ∞-ball projections used inside
         the inclusion's resolvent.  Using the anisotropic norm here is what
         makes the two solvers minimise *the same* objective.

So PDHG and `run_zero` must converge to the same minimiser; comparing their
reconstructions (PSNR vs the phantom, and visually) is a clean cross-check that
the inclusion is set up and solved correctly.

The function `run_pdhg(setup, params, ...)` takes the same `setup` dict produced
by `Algo_setuptorch.get_setup`, so both solvers see an identical problem.

Reference: K. Bredies and M. Holler, *A TGV-based framework ... Part II:
Numerics*, SIAM J. Imaging Sci. 8(4):2851-2886, 2015.
"""

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
    A = setup["ray_transform"]          # raw (un-normalised) ODL ray transform

    # Normalisation constant so the operator equals A_torch = A / ‖A‖.
    norm_A_const = odl.power_method_opnorm(A, maxiter=30)
    A_n = (1.0 / norm_A_const) * A

    # Normalised sinogram already stored by get_setup -> ODL element in A.range.
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

    # Column operator K acting on (u, w):
    #   1. A_n u            (data)
    #   2. ∇u − w           (first-order TGV term)
    #   3. E w              (second-order TGV term)
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
    """PDHG callback that records a torch-side objective at every iteration.

    `obj_fn(u, w)` receives the current primal iterate as torch tensors
    (u : [1,1,H,W], w : [1,2,H,W]) and must return a float. Storing the
    objective with the *same* function used for `run_zero` / `run_learned`
    guarantees the three curves are strictly comparable.
    """

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
    """Solve the TGV² tomography problem with PDHG.

    Args:
        setup   : dict from `get_setup` (same problem as `run_zero`).
        params  : `Params` instance (uses `alpha1`, `alpha2`).
        niter   : number of PDHG iterations.
        verbose : print the iteration counter.
        obj_fn  : optional callable obj_fn(u, w) -> float. If given, the
                  objective is recorded at every iteration and returned.

    Returns:
        rec                 if `obj_fn` is None
        (rec, obj_history)  if `obj_fn` is given.
    """
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


if __name__ == "__main__":

    from NN_tomo.Algo_setuptorch import Params, get_setup
    from NN_tomo.PSNR import psnr_history

    SIZE, N_ANGLES = 128, 180
    params = Params(size=SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup = get_setup(SIZE, n_angles=N_ANGLES, seed=0, noise_level=0.02,
                      device=device)
    rec = run_pdhg(setup, params, niter=300, verbose=True)
    psnr = psnr_history([rec], setup["phantom"])[0]
    print(f"PDHG reconstruction PSNR = {psnr:.2f} dB")
