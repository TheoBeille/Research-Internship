import numpy as np
import odl


def build_pdhg_problem(setup, params):
    """
    The same TGV2 problem written for odl.solvers.pdhg (reference solver):

        min_x  f(x) + g(op(x)),    x = (u, w)

    Returns (op, f, g, domain).
    """
    U = setup["space"]
    K = setup["ray_transform"]
    K_n = (1.0 / odl.power_method_opnorm(K, maxiter=30)) * K     # ||K|| = 1
    y = K.range.element(np.asarray(setup["data"].cpu())[0, 0])

    # TGV operators, identical to get_setup
    G = odl.Gradient(U, method="forward", pad_mode="symmetric")
    V = G.range
    Dx = odl.PartialDerivative(U, 0, method="forward", pad_mode="symmetric")
    Dy = odl.PartialDerivative(U, 1, method="forward", pad_mode="symmetric")
    E = odl.operator.ProductSpaceOperator([[Dx, 0], [0, Dy], [0.5 * Dy, 0.5 * Dx]])

    domain = odl.ProductSpace(U, V)

    # op(u, w) = (K u, grad u - w, E w)
    op = odl.BroadcastOperator(
        K_n * odl.ComponentProjection(domain, 0),
        odl.ReductionOperator(G, odl.ScalingOperator(V, -1)),
        E * odl.ComponentProjection(domain, 1))

    f = odl.solvers.ZeroFunctional(domain)
    g = odl.solvers.SeparableSum(
        0.5 * odl.solvers.L2NormSquared(K.range).translated(y),
        params.alpha1 * odl.solvers.L1Norm(V),
        params.alpha2 * odl.solvers.L1Norm(E.range))

    return op, f, g, domain


def run_pdhg(setup, params, objective, clean, T, snapshots=(10,), tau=0.5):
    """
    Run T iterations of PDHG from zero and time them.

    objective : function (u, w) -> TGV2 objective (see make_objective)
    tau       : primal step; the dual step follows from
                tau * sigma * ||op||^2 = 1 / 1.1^2. None gives the balanced
                steps tau = sigma.
    Returns the same dict as utils.experiments.run_fbs (without "kkt").
    """
    import time
    import torch
    from utils.PSNR import psnr_history

    op, f, g, _ = build_pdhg_problem(setup, params)
    balanced = 1.0 / (1.1 * odl.power_method_opnorm(op, maxiter=50))
    if tau is None:
        tau = balanced
    sigma = balanced ** 2 / tau
    device = setup["device"]
    result = {"objective": [], "psnr": [], "images": {}}

    def record(x):
        u = torch.tensor(np.asarray(x[0]), device=device)[None, None]
        w = torch.tensor(np.asarray(x[1]), device=device)[None]
        result["objective"].append(objective(u, w).item())
        result["psnr"].append(psnr_history([u], clean)[0])
        n = len(result["psnr"])
        if n in snapshots or n == T:
            result["images"][n] = u

    odl.solvers.pdhg(op.domain.zero(), f, g, op, niter=T, tau=tau, sigma=sigma,
                     callback=record)

    # timing: a second, short run without the monitoring
    n_timing = min(T, 20)
    start = time.perf_counter()
    odl.solvers.pdhg(op.domain.zero(), f, g, op, niter=n_timing, tau=tau, sigma=sigma)
    result["ms_per_iter"] = 1000 * (time.perf_counter() - start) / n_timing

    result["objective"] = np.array(result["objective"])
    result["psnr"] = np.array(result["psnr"])
    return result
