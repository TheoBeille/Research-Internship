"""Baseline run -> training -> comparison, on Mayo CT slices at 512x512."""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

from Algo_setuptorch import Params
from algorithm.run import run_zero, run_learned
from algorithm.unrolled_model import UnrolledFBS
from data.mayo_dataset import build_train_test_data_mayo
from training.train import train
from utils.plots import apply_paper_style, plot_convergence, train_plot
from utils.PSNR import psnr_history

# --- configuration -----------------------------------------------------------
SIZE = 512
N_ANGLES = 180
NOISE_LEVEL = 0.05          # relative noise on the sinogram
GAMMA = 2                   # needs gamma * beta_bar < 4 - 2 * lam0

TRAIN_PATIENTS = ["L004", "L006", "L012", "L019"]
TEST_PATIENTS = ["L014"]
MAYO_CACHE_DIR = "./data/mayo_cache_512"
MAX_TRAIN_SLICES = 1        # set to None to use all slices
MAX_TEST_SLICES = 1

T = 10                      # unrolled iterations
N_EPOCHS = 50
LR = 1e-3
CKPT_PATH = f"kkt_tomo_{SIZE}_gamma_{GAMMA}_alpha_04.pt"

SHAPES = [
    (1, 1, SIZE, SIZE),     # u  (image)
    (1, 2, SIZE, SIZE),     # w
    (1, 2, SIZE, SIZE),     # p  (dual of grad u - w)
    (1, 3, SIZE, SIZE),     # q  (dual of E w)
]

# --- setup -------------------------------------------------------------------
apply_paper_style()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.9)     # the GPU is shared

params = Params(size=SIZE, gamma0=GAMMA)
assert GAMMA * params.beta_bar < 4 - 2 * params.lam0

train_data, test_data = build_train_test_data_mayo(
    TRAIN_PATIENTS, TEST_PATIENTS, params, device,
    n_angles=N_ANGLES, cache_dir=MAYO_CACHE_DIR, noise_level=NOISE_LEVEL,
    max_train_slices=MAX_TRAIN_SLICES, max_test_slices=MAX_TEST_SLICES,
)
initial_state, clean, functions = test_data[0]

# --- baseline: zero deviations -----------------------------------------------
kkt_zero, _, x_hist = run_zero(initial_state, functions, params, SHAPES, T=100, device=device)
psnr_zero = psnr_history([x_hist[-1]], clean)[0]
del x_hist
print(f"Zero deviation: KKT {kkt_zero[0]:.3e} -> {kkt_zero[-1]:.3e}, PSNR = {psnr_zero:.2f} dB")

# --- training ----------------------------------------------------------------
model = UnrolledFBS(params, SHAPES, n_channels=3, T=T, alpha=0.99)
model, train_hist, val_hist = train(
    model, train_data, val_data=test_data, n_epochs=N_EPOCHS, lr=LR, device=device)

torch.save({"model": model.state_dict(), "train_loss_history": train_hist,
            "val_loss_history": val_hist, "lr": LR, "epochs": N_EPOCHS,
            "gamma": GAMMA, "T": T}, CKPT_PATH)
print(f"Saved {CKPT_PATH}")

# --- learned vs. baseline ----------------------------------------------------
kkt_learned, _ = run_learned(model, initial_state, clean, functions, T_test=100)
print(f"Learned:        KKT {kkt_learned[0]:.3e} -> {kkt_learned[-1]:.3e}")

plot_convergence(kkt_zero, kkt_learned, name=f"convergence_gamma_{GAMMA}")
train_plot(train_hist, val_hist, name=f"training_gamma_{GAMMA}")
