#%% Run to have everything  --  TOMOGRAPHY

import torch
import numpy as np
import matplotlib.pyplot as plt

from Algo_setuptorch import Params
from data.dataset import build_train_test_data
from algorithm.unrolled_model import UnrolledFBS
from training.train import train
from plots import plot_convergence_2, train_plot
from run import run_zero, run_learned
from PSNR import psnr_history


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
TEST_SEEDS = list(range(1000, 1008))

train_data, test_data = build_train_test_data(
    train_seeds=TRAIN_SEEDS,
    test_seeds=TEST_SEEDS,
    params=params,
    device=device,
    n_angles=N_ANGLES,
)

initial_state, clean, functions = test_data[0]
print(f"gamma = {params.gamma0:.4f}   beta_bar = {params.beta_bar:.3f}   "
      f"(gamma*||B|| = 0.7 < 1 by construction)")

#%% 1) ZERO-DEVIATION BASELINE  (the acceptance test: must converge alone) -----
print("\n[run_zero] baseline ...")
AxCx_zero, _res, x_hist = run_zero(
    initial_state, functions, params, SHAPES, T=100, device=device)
rec0 = x_hist[-1][0]
psnr0 = psnr_history([rec0], clean)[0]
print(f"  KKT {AxCx_zero[0]:.3e} -> {AxCx_zero[-1]:.3e}   PSNR = {psnr0:.2f} dB")


#%% 2) TRAIN THE UNROLLED MODEL (learned, safeguarded deviations) --------------
model = UnrolledFBS(
    params=params,
    shapes=SHAPES,
    n_channels=N_CH_primal,
    T=10,
    alpha=0.99,
).to(device).float()

from reference import load_reference


model, train_hist, val_hist = train(
    model=model,
    train_data=train_data,
    val_data=test_data,
    n_epochs=70,
    lr=1e-3,
    device=device,
    print_every=5,
          # loss = ||u_T - u*|| / ||u*||
)

checkpoint = {
    "model": model.state_dict(),
    "train_loss_history": train_hist,
    "val_loss_history": val_hist,
    "lr": 1e-3,
    "epochs": 100,
}
torch.save(checkpoint, "checkpoint_tomo_128_gamma_1.pt")

#%% 3) LEARNED CONVERGENCE + FIGURES ------------------------------------------
print("\n[run_learned] ...")
AxCx_learned, _ = run_learned(model, initial_state, clean, functions, T_test=100)

plot_convergence_2(AxCx_zero, AxCx_learned, label3="learned",
                   title="Convergence_tomo")
train_plot(train_hist, val_hist, title="Training_Validation_tomo")
print("\nDone. Figures in ./plots, checkpoint in ./checkpoint_tomo.pt")