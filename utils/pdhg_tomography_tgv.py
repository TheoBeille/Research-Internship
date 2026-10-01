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
