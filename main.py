#%% Run to have everything  --  TOMOGRAPHY  (train ALL gammas, no-time net)

import torch
import numpy as np
import matplotlib.pyplot as plt

from Algo_setuptorch import Params
from data.dataset import build_train_test_data
from algorithm.unrolled_model import UnrolledFBS
from training.train import train
from utils.plots import (
    apply_paper_style,
    plot_convergence_2,
    plot_convergence_multi_gamma,
    train_plot,
)
from algorithm.run import run_zero, run_learned
from utils.PSNR import psnr_history


apply_paper_style()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- problem size / geometry -------------------------------------------------
params = Params(size=128)
N_ANGLES = 180

size = params.size
SHAPES = [
    (1, 1, size, size),   # u  (image)
    (1, 2, size, size),   # w
    (1, 2, size, size),   # p  (dual of grad u - w)
    (1, 3, size, size),   # q  (dual of E w)
]
N_CH_primal = sum(s[1] for s in SHAPES[:2])    # = 3

TRAIN_SEEDS = list(range(40))
TEST_SEEDS = list(range(1000, 1007))


T = 10
N_EPOCHS = 100
LR = 1e-3

GAMMAS = [1.8]


def gamma_tag(g):
    """Filename tag matching the existing checkpoints (0.1 -> 01, 1.0 -> 1)."""
    if abs(g - round(g)) < 1e-9:
        return str(int(round(g)))
    s = ("%g" % g)
    return s.replace("0.", "0") if s.startswith("0.") else s


train_data, test_data = build_train_test_data(
    train_seeds=TRAIN_SEEDS,
    test_seeds=TEST_SEEDS,
    params=params,
    device=device,
    n_angles=N_ANGLES,
)

initial_state, clean, functions = test_data[0]
coc_max = 2.0 / (params.lam0 * params.beta_bar)
print(f"beta_bar = {params.beta_bar:.3f}   "
      f"2/(lam0*beta_bar) = {coc_max:.4f}  (gamma must be < this)")


#%% Baseline (zero-deviation) per gamma + train one model per gamma -----------
curves_zero = {}
curves_learned = {}

for g in GAMMAS:
    tag = gamma_tag(g)
    print("\n" + "=" * 70)
    print(f"GAMMA = {g}   (checkpoint tag '{tag}')")
    print("=" * 70)

    params.gamma0 = g

    print("[run_zero] baseline ...")
    AxCx_zero, _res, x_hist = run_zero(
        initial_state, functions, params, SHAPES, T=100, device=device)
    rec0 = x_hist[-1][0]
    psnr0 = psnr_history([rec0], clean)[0]
    print(f"  KKT {AxCx_zero[0]:.3e} -> {AxCx_zero[-1]:.3e}   PSNR = {psnr0:.2f} dB")
    curves_zero[g] = np.asarray(AxCx_zero)

    model = UnrolledFBS(
        params=params,
        shapes=SHAPES,
        n_channels=N_CH_primal,
        T=T,
        alpha=0.99,
    ).to(device).float()

    model, train_hist, val_hist = train(
        model=model,
        train_data=train_data,
        val_data=test_data,
        n_epochs=N_EPOCHS,
        lr=LR,
        device=device,
        print_every=5,
    )

    ckpt_path = f"kkt_tomo_128_gamma_{tag}_alpha_04.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "train_loss_history": train_hist,
            "val_loss_history": val_hist,
            "lr": LR,
            "epochs": N_EPOCHS,
            "gamma": g,
            "T": T,
        },
        ckpt_path,
    )
    print(f"  saved {ckpt_path}")


    print("[run_learned] ...")
    AxCx_learned, _ = run_learned(
        model, initial_state, clean, functions, T_test=100)
    curves_learned[g] = np.asarray(AxCx_learned)

    plot_convergence_2(
        AxCx_zero, AxCx_learned, label3="learned",
        title=f"Convergence_tomo_gamma_{tag}")
    train_plot(
        train_hist, val_hist,
        title=f"Training_Validation_tomo_gamma_{tag}")

# restore automatic gamma
params.gamma0 = 0.85 * coc_max



plot_convergence_multi_gamma(
    curves_zero, curves_learned, title="Convergence_all_gamma")


