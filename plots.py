

import os
import re
import numpy as np
import matplotlib.pyplot as plt


def _safe_filename(title: str) -> str:
    name = title.replace(' ', '_')
    name = re.sub(r'[^A-Za-z0-9_.-]', '', name)
    return name


def _ensure_plots_dir(dir_name: str = 'plots') -> str:
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def _add_rate_lines(reference):
    """Overlay O(1/t) and O(1/t^2) reference rates, anchored on the first
    residual value so the slopes are directly comparable to the curves."""
    reference = np.asarray(reference, dtype=float)
    if reference.size == 0:
        return
    t = np.arange(1, reference.size + 1)
    c = reference[0]                      # anchor both lines at the first point
    plt.loglog(t, c / t,      color='gray', lw=1.2, ls='--',  label=r'$O(1/t)$')
    plt.loglog(t, c / t ** 2, color='gray', lw=1.2, ls=':',   label=r'$O(1/t^2)$')


def plot_convergence(res_zero, res_learned,res_learned_primal,label3, title="Convergence",label1="Zero (baseline)",label2="Learned"):
    plt.figure(figsize=(6,4))

    t = range(1, len(res_zero) + 1)
    plt.loglog(t, res_zero, label=label1, linewidth=2)
    plt.loglog(range(1, len(res_learned) + 1), res_learned, label=label2, linewidth=2, linestyle='--')
    plt.loglog(range(1, len(res_learned_primal) + 1), res_learned_primal, label=label3, linewidth=2, linestyle='dotted')
    _add_rate_lines(res_zero)
    plt.xlabel("Iteration")
    plt.ylabel("Residual (log scale)")
    plt.title(title)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_dir = _ensure_plots_dir()
    fname = f"{_safe_filename(title)}.pdf"
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()
    
def plot_convergence_2(res_zero,res_learned_primal,label3="learned_primal", title="Convergence",label1="Zero (baseline)"):

    
    plt.figure(figsize=(6,4))

    plt.loglog(range(1, len(res_zero) + 1), res_zero, label=label1, linewidth=2)
    plt.loglog(range(1, len(res_learned_primal) + 1), res_learned_primal, label=label3, linewidth=2, linestyle='dotted')
    _add_rate_lines(res_zero)
    plt.xlabel("Iteration")
    plt.ylabel("Residual (log scale)")
    plt.title(title)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_dir = _ensure_plots_dir()
    fname = f"{_safe_filename(title)}.pdf"
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    
    plt.show()
    plt.close()

def train_plot(train_loss_hist,val_loss_hist,title="Training_Validation Errors"):
    plt.figure(figsize=(10, 5))
    plt.semilogy(train_loss_hist, label="Training error",   color="#3B5BA5", linewidth=1.5)
    plt.semilogy(val_loss_hist,   label="Validation error", color="#C07820", linewidth=1.2)
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_dir = _ensure_plots_dir()
    fname = f"{_safe_filename(title)}.pdf"
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()

