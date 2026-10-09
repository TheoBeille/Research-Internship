"""Baseline run -> training -> comparison, on Mayo CT slices."""

import os
import random
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

from Algo_setuptorch import Params
from algorithm.run import run_zero, run_learned
from algorithm.unrolled_model import UnrolledFBS
from data.mayo_dataset import build_train_test_data_mayo
from training.train import train
from utils.plots import apply_paper_style, plot_curves
from utils.PSNR import psnr_history

# --- configuration -----------------------------------------------------------
SIZE = 512                  # 128 or 512
N_ANGLES = 180
NOISE_LEVEL = 0.05          # relative noise on the sinogram
GAMMA = 2
PRIMAL_STEP = 1             # primal step tau (see Params); must match the notebooks
USE_SAFEGUARD = False        # False: train the network without the safeguard

# L014 is kept untouched for the figures (notebooks): it is never used here
TRAIN_PATIENTS = ["L004", "L006", "L012"]
VAL_PATIENTS = ["L019"]
MAYO_CACHE_DIR = f"./data/mayo_cache_{SIZE}"
MAX_TRAIN_SLICES = None      # evenly spaced over the patients; None = all slices
MAX_VAL_SLICES = 10

T = 10                      # learned iterations (training: T + randint(0, T)); after
                            # them the classical scheme restarts from p_T
ALPHA = 0.99                # fraction of the safeguard budget (on the norm)
N_EPOCHS = 50
LR = 1e-3
ACCUM = 8                   # slices per optimizer step (gradient accumulation)
SEED = 0
RUN_NAME = f"{SIZE}" + ("" if USE_SAFEGUARD else "_nosafe")
CKPT_PATH = f"model_{RUN_NAME}.pt"

SHAPES = [
    (1, 1, SIZE, SIZE),     # u  (image)
    (1, 2, SIZE, SIZE),     # w
    (1, 2, SIZE, SIZE),     # p  (dual of grad u - w)
    (1, 3, SIZE, SIZE),     # q  (dual of E w)
]

# --- setup -------------------------------------------------------------------
random.seed(SEED)
torch.manual_seed(SEED)
apply_paper_style()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.9)     # the GPU is shared

params = Params(size=SIZE, gamma0=GAMMA, primal_step=PRIMAL_STEP)

train_data, val_data = build_train_test_data_mayo(
    TRAIN_PATIENTS, VAL_PATIENTS, params, device,
    n_angles=N_ANGLES, cache_dir=MAYO_CACHE_DIR, noise_level=NOISE_LEVEL,
    max_train_slices=MAX_TRAIN_SLICES, max_test_slices=MAX_VAL_SLICES,
)
initial_state, clean, functions = val_data[0]

# --- baseline: zero deviations -----------------------------------------------
kkt_zero, _, x_hist = run_zero(initial_state, functions, params, SHAPES, T=100, device=device)
psnr_zero = psnr_history([x_hist[-1]], clean)[0]
del x_hist
print(f"Zero deviation: KKT {kkt_zero[0]:.3e} -> {kkt_zero[-1]:.3e}, PSNR = {psnr_zero:.2f} dB")

# --- training ----------------------------------------------------------------
model = UnrolledFBS(params, SHAPES, n_channels=3, T=T, alpha=ALPHA,
                    use_safeguard=USE_SAFEGUARD)
model, train_hist, val_hist = train(
    model, train_data, val_data=val_data, n_epochs=N_EPOCHS, lr=LR, device=device,
    accum=ACCUM, ckpt_path=CKPT_PATH)   # best epoch, saved after every epoch
print(f"Saved {CKPT_PATH}")

# --- learned vs. baseline ----------------------------------------------------
kkt_learned, _ = run_learned(model, initial_state, clean, functions, T_test=100)
print(f"Learned:        KKT {kkt_learned[0]:.3e} -> {kkt_learned[-1]:.3e}")

plot_curves({"Zero deviation": kkt_zero, "Learned": kkt_learned}, "KKT residual",
            name=f"main_convergence_{RUN_NAME}", slopes=True, horizon=T)
plot_curves({"Training": train_hist, "Validation": val_hist},
            "objective at the last unrolled iteration", name=f"main_training_{RUN_NAME}")
