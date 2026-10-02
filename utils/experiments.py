"""Helpers shared by the figure notebooks."""

import os
import time

import numpy as np
import torch

from Algo_setuptorch import Params, get_setup, build_algo_functions, make_objective
from algorithm.fbs_step import unroll
from algorithm.unrolled_model import UnrolledFBS
from data.mayo_dataset import MayoSliceDataset
from utils.PSNR import psnr_history

RESULTS_DIR = "results"


def build_test_instance(size=512, n_angles=180, noise_level=0.05, gamma=2,
                        patient="L014", slice_idx=0,
                        cache_dir="./data/mayo_cache_512", device=None, **params_kwargs):
    """One test problem: a dict with everything the runs below need.
    Extra keywords (e.g. primal_step) are passed to Params."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = Params(size=size, gamma0=gamma, **params_kwargs)
    image = MayoSliceDataset([patient], cache_dir)[slice_idx].squeeze(0).numpy()
    setup = get_setup(size, n_angles=n_angles, seed=slice_idx,
                      noise_level=noise_level, device=device, phantom_array=image)
    return dict(
        setup=setup,
        params=params,
        functions=build_algo_functions(setup, params),
        objective=make_objective(setup, params),
        clean=setup["phantom"],
        back_projection=setup["initial_state"],
        device=device,
        shapes=[(1, 1, size, size), (1, 2, size, size), (1, 2, size, size), (1, 3, size, size)],
        tag=(f"{patient}_slice{slice_idx}_size{size}_angles{n_angles}_noise{noise_level}"
             f"_alpha{params.alpha1:g}_{params.alpha2:g}_tau{params.primal_step}"),
    )


def load_model(ckpt_path, inst, T=10, use_safeguard=True):
    """Trained model and its checkpoint (which holds the loss histories)."""
    model = UnrolledFBS(inst["params"], inst["shapes"], n_channels=3, T=T,
                        use_safeguard=use_safeguard).to(inst["device"])
    ckpt = torch.load(ckpt_path, map_location=inst["device"])
    model.load_state_dict(ckpt["model"])
    trained_step = ckpt.get("primal_step", "unknown (older checkpoint)")
    if trained_step != inst["params"].primal_step:
        print(f"WARNING: {ckpt_path} was trained with primal_step = {trained_step}, "
              f"but this problem uses primal_step = {inst['params'].primal_step}.")
    return model.eval(), ckpt


def random_direction(seed=0):
    """Random directions on all blocks (control experiment)."""
    generator = None

    def direction(x, *_):
        nonlocal generator
        if generator is None:
            generator = torch.Generator(device=x[0].device).manual_seed(seed)
        def draw():
            return [torch.randn(b.shape, generator=generator, device=b.device) for b in x]
        return draw(), draw()

    return direction


def run_fbs(inst, T, direction=None, use_safeguard=True, snapshots=(10,), n_timing=20):
    """
    Run T iterations of the forward-backward scheme on the test instance.

    direction : None (zero deviations), a trained UnrolledFBS model, or a
                function such as random_direction()

    Returns a dict with, per iteration, "kkt", "objective" and "psnr", the
    reconstructions "images" {iteration: image} at the requested iterations
    and at the last one, and "ms_per_iter" (measured on a separate short run
    without any monitoring).
    """
    alpha = 0.99
    if isinstance(direction, UnrolledFBS):
        direction, alpha = direction.dev_net, direction.alpha
    args = (inst["functions"], inst["params"], inst["shapes"])
    kwargs = dict(device=inst["device"], direction=direction, alpha=alpha,
                  use_safeguard=use_safeguard)

    with torch.no_grad():
        kkt, _, x_hist = unroll(*args, T, keep_history=True, progress=True, **kwargs)
        objective = [inst["functions"]["objective"](x).item() for x in x_hist]

        n_timing = min(T, n_timing)
        if inst["device"].type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        unroll(*args, n_timing, monitor=False, **kwargs)
        if inst["device"].type == "cuda":
            torch.cuda.synchronize()
        ms_per_iter = 1000 * (time.perf_counter() - start) / n_timing

    return dict(
        kkt=np.array([k.item() for k in kkt]),
        objective=np.array(objective),
        psnr=np.array(psnr_history(x_hist, inst["clean"])),
        images={n: x_hist[n - 1][0] for n in set(snapshots) | {T} if n <= T},
        ms_per_iter=ms_per_iter,
    )


def reference_solution(inst, T_ref=6000):
    """
    High-accuracy solution of the TGV2 problem (long zero-deviation run).
    Returns {"f_star", "psnr", "image"}; cached in results/ because it is slow.
    """
    path = os.path.join(RESULTS_DIR, f"reference_{inst['tag']}_T{T_ref}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location=inst["device"])

    with torch.no_grad():
        _, _, x_hist = unroll(inst["functions"], inst["params"], inst["shapes"], T_ref,
                              inst["device"], progress=True, monitor=False)
        x = x_hist[-1]
        ref = dict(f_star=inst["functions"]["objective"](x).item(),
                   psnr=psnr_history([x], inst["clean"])[0], image=x[0])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(ref, path)
    return ref


def objective_gap(result, f_star, floor=1e-6):
    """F(x_n) - F*, clipped from below so that it can be drawn on a log scale."""
    return np.clip(result["objective"] - f_star, floor, None)


def time_table(results, name="time_comparison"):
    """
    Time comparison between methods.

    results : dict {method name: result of run_fbs / run_pdhg}
    Columns: time per iteration and PSNR after 10 iterations.
    Returns a pandas DataFrame and writes it as LaTeX to plots/<name>.tex.
    """
    import pandas as pd

    rows = [{"Method": method,
             "Time / iteration (ms)": f"{r['ms_per_iter']:.0f}",
             "PSNR after 10 it. (dB)": f"{r['psnr'][9]:.2f}"}
            for method, r in results.items()]
    table = pd.DataFrame(rows).set_index("Method")

    os.makedirs("plots", exist_ok=True)
    with open(os.path.join("plots", f"{name}.tex"), "w") as file:
        file.write(table.to_latex(escape=True))
    return table
