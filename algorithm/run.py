"""
Evaluation runs (no gradients). Each returns

    kkt       : list of KKT residuals, one per iteration
    residuals : list of fixed-point residuals ||p_n - y_n||
    x_hist    : list of primal iterates [u, w], one per iteration
"""

import torch

from algorithm.fbs_step import unroll


def _floats(kkt, residuals, x_hist):
    return [t.item() for t in kkt], [t.item() for t in residuals], x_hist


def run_zero(initial_state, functions, params, shapes, T, device):
    """Baseline: no deviations."""
    with torch.no_grad():
        return _floats(*unroll(functions, params, shapes, T, device,
                               keep_history=True, progress=True))


def run_random(initial_state, functions, params, shapes, T=100, device="cuda",
               alpha=0.99, seed=0):
    """Control: random directions, rescaled by the safeguard."""
    torch.manual_seed(seed)

    def direction(x, *_):
        return [torch.randn_like(b) for b in x], [torch.randn_like(b) for b in x]

    with torch.no_grad():
        return _floats(*unroll(functions, params, shapes, T, device,
                               direction=direction, alpha=alpha,
                               keep_history=True, progress=True))


def run_learned(model, initial_state, clean, functions, T_test=500, return_all=False,
                use_safeguard=True):
    """Learned deviations. With return_all, also returns {"x": x_hist}."""
    model.eval()
    with torch.no_grad():
        kkt, residuals, x_hist = _floats(*model(
            functions, T=T_test, use_safeguard=use_safeguard,
            keep_history=return_all, progress=True))
    if return_all:
        return kkt, residuals, {"x": x_hist}
    return kkt, residuals


def run_learned_nosafe(model, initial_state, clean, functions, T_test=500,
                       return_all=False):
    """Learned deviations applied directly, without the safeguard
    (no convergence guarantee)."""
    return run_learned(model, initial_state, clean, functions, T_test, return_all,
                       use_safeguard=False)
