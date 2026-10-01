import os
import shutil

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


PAPER = {
    "blue":    "#3B5BA5",
    "orange":  "#C07820",
    "green":   "#3C8C5A",
    "red":     "#B5485D",
    "purple":  "#6E5BA0",
    "teal":    "#2E8B8B",
    "ochre":   "#9C7A2E",
    "gray":    "#7A7A7A",
}


def _latex_available():
    """True if a LaTeX install can render a trivial figure."""
    if shutil.which("latex") is None:
        return False
    try:
        with matplotlib.rc_context({"text.usetex": True}):
            fig = plt.figure()
            fig.text(0.5, 0.5, r"$x^2$")
            fig.canvas.draw()
        return True
    except Exception:
        return False
    finally:
        plt.close("all")


def apply_paper_style(use_tex=True):
    """Paper-like matplotlib style. Text is rendered with LaTeX when it is
    installed, otherwise with matplotlib's mathtext."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": "#BBBBBB",
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "text.usetex": use_tex and _latex_available(),
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    })


def _ensure_plots_dir(dir_name="plots"):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def _save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(_ensure_plots_dir(), f"{name}.pdf"))
    plt.close()


def plot_convergence(kkt_zero, kkt_learned, name="convergence"):
    """KKT residual of the baseline and of the learned scheme (log-log),
    with O(1/t) and O(1/t^2) reference slopes. Saved as plots/<name>.pdf."""
    plt.figure(figsize=(6, 4))
    t = np.arange(1, len(kkt_zero) + 1)
    plt.loglog(t, kkt_zero, label="Zero deviation", color=PAPER["blue"])
    plt.loglog(np.arange(1, len(kkt_learned) + 1), kkt_learned, label="Learned",
               color=PAPER["orange"], linestyle="--")
    plt.loglog(t, kkt_zero[0] / t, color=PAPER["gray"], lw=1.1, ls="--", label=r"$O(1/t)$")
    plt.loglog(t, kkt_zero[0] / t ** 2, color=PAPER["gray"], lw=1.1, ls=":", label=r"$O(1/t^2)$")
    plt.xlabel("Iteration")
    plt.ylabel("KKT residual")
    plt.legend()
    _save(name)


def train_plot(train_loss_hist, val_loss_hist, name="training"):
    """Training and validation loss per epoch. Saved as plots/<name>.pdf."""
    plt.figure(figsize=(6, 4))
    plt.semilogy(train_loss_hist, label="Training", color=PAPER["blue"])
    plt.semilogy(val_loss_hist, label="Validation", color=PAPER["orange"])
    plt.xlabel("Epoch")
    plt.ylabel("KKT residual at the last unrolled iteration")
    plt.legend()
    _save(name)
