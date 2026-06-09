

import os
import re
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Paper-style palette & styling
# ============================================================
# Muted, colorblind-friendly colors in the spirit of academic figures
# (close to the Tableau "muted" / Wong palettes).

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


PAPER_CYCLE = [
    PAPER["blue"], PAPER["orange"], PAPER["green"], PAPER["red"],
    PAPER["purple"], PAPER["teal"], PAPER["ochre"], PAPER["gray"],
]


def _latex_available():
    """True if a real LaTeX install can render a trivial figure."""
    import shutil
    if shutil.which("latex") is None and shutil.which("pdflatex") is None:
        return False
    import matplotlib
    prev = matplotlib.rcParams["text.usetex"]
    try:
        matplotlib.rcParams["text.usetex"] = True
        fig = plt.figure()
        fig.text(0.5, 0.5, r"$x^2$")
        fig.canvas.draw()
        plt.close(fig)
        return True
    except Exception:
        plt.close("all")
        return False
    finally:
        matplotlib.rcParams["text.usetex"] = prev


def apply_paper_style(use_tex=True):
    """Apply a clean, paper-like matplotlib style with LaTeX-rendered text.

    If `use_tex` is True and a LaTeX install is found, all figure text is
    rendered through LaTeX (Computer Modern). Otherwise it falls back to
    matplotlib's built-in mathtext so figures still render everywhere.
    """
    tex = bool(use_tex) and _latex_available()

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
        # --- LaTeX rendering -------------------------------------------------
        "text.usetex": tex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    })
    if tex:
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

    print(f"[plots] LaTeX rendering: {'ON' if tex else 'OFF (mathtext fallback)'}")


def _safe_filename(title: str) -> str:
    name = title.replace(' ', '_')
    name = re.sub(r'[^A-Za-z0-9_.-]', '', name)
    return name


def _ensure_plots_dir(dir_name: str = 'plots') -> str:
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def _disp(s):
    """Make a text string safe for display when LaTeX rendering is on.

    With text.usetex, `_` is a special character and raw underscores in
    titles/labels crash LaTeX. We escape them, but leave any string that
    already contains math ($...$) untouched.
    """
    if plt.rcParams.get("text.usetex", False) and isinstance(s, str) and "$" not in s:
        return s.replace("_", r"\_")
    return s


def _add_rate_lines(reference):
    """Overlay O(1/t) and O(1/t^2) reference rates, anchored on the first
    residual value so the slopes are directly comparable to the curves."""
    reference = np.asarray(reference, dtype=float)
    if reference.size == 0:
        return
    t = np.arange(1, reference.size + 1)
    c = reference[0]                      # anchor both lines at the first point
    plt.loglog(t, c / t,      color=PAPER["gray"], lw=1.1, ls='--', label=r'$O(1/t)$')
    plt.loglog(t, c / t ** 2, color=PAPER["gray"], lw=1.1, ls=':',  label=r'$O(1/t^2)$')


def plot_convergence(res_zero, res_learned, res_learned_primal, label3,
                     title="Convergence", label1="Zero (baseline)",
                     label2="Learned"):
    plt.figure(figsize=(6, 4))

    t = range(1, len(res_zero) + 1)
    plt.loglog(t, res_zero, label=_disp(label1), color=PAPER["blue"], linewidth=2)
    plt.loglog(range(1, len(res_learned) + 1), res_learned, label=_disp(label2),
               color=PAPER["orange"], linewidth=2, linestyle='--')
    plt.loglog(range(1, len(res_learned_primal) + 1), res_learned_primal,
               label=_disp(label3), color=PAPER["green"], linewidth=2, linestyle='dotted')
    _add_rate_lines(res_zero)
    plt.xlabel("Iteration")
    plt.ylabel("Residual (log scale)")
    plt.title(_disp(title))
    plt.legend()
    plt.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    out_dir = _ensure_plots_dir()
    fname = f"{_safe_filename(title)}.pdf"
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()


def plot_convergence_2(res_zero, res_learned_primal, label3="learned_primal",
                       title="Convergence", label1="Zero (baseline)"):

    plt.figure(figsize=(6, 4))

    plt.loglog(range(1, len(res_zero) + 1), res_zero, label=_disp(label1),
               color=PAPER["blue"], linewidth=2)
    plt.loglog(range(1, len(res_learned_primal) + 1), res_learned_primal,
               label=_disp(label3), color=PAPER["orange"], linewidth=2, linestyle='--')
    _add_rate_lines(res_zero)
    plt.xlabel("Iteration")
    plt.ylabel("Residual (log scale)")
    plt.title(_disp(title))
    plt.legend()
    plt.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    out_dir = _ensure_plots_dir()
    fname = f"{_safe_filename(title)}.pdf"
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()


def plot_convergence_multi_gamma(curves_zero, curves_learned,
                                 title="Convergence_all_gamma"):
    """Overlay learned vs baseline convergence for several gamma values.

    Args:
        curves_zero    : dict {gamma: AxCx baseline array}
        curves_learned : dict {gamma: AxCx learned  array}
    Each gamma gets one color from the paper cycle; baseline is solid,
    learned is dashed.
    """
    plt.figure(figsize=(7, 5))

    gammas = sorted(curves_learned.keys())
    for i, g in enumerate(gammas):
        color = PAPER_CYCLE[i % len(PAPER_CYCLE)]
        if g in curves_zero:
            cz = np.asarray(curves_zero[g], dtype=float)
            plt.loglog(np.arange(1, len(cz) + 1), cz, color=color,
                       linewidth=1.6, linestyle='-', alpha=0.55,
                       label=f"$\\gamma={g}$ (baseline)")
        cl = np.asarray(curves_learned[g], dtype=float)
        plt.loglog(np.arange(1, len(cl) + 1), cl, color=color,
                   linewidth=2.2, linestyle='--',
                   label=f"$\\gamma={g}$ (learned)")

    plt.xlabel("Iteration")
    plt.ylabel("KKT residual (log scale)")
    plt.title(_disp(title))
    plt.legend(ncol=2)
    plt.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    out_dir = _ensure_plots_dir()
    plt.savefig(os.path.join(out_dir, f"{_safe_filename(title)}.pdf"))
    plt.close()


def train_plot(train_loss_hist, val_loss_hist, title="Training_Validation Errors"):
    plt.figure(figsize=(10, 5))
    plt.semilogy(train_loss_hist, label="Training error",
                 color=PAPER["blue"], linewidth=1.8)
    plt.semilogy(val_loss_hist, label="Validation error",
                 color=PAPER["orange"], linewidth=1.4)
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.title(_disp(title))
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_dir = _ensure_plots_dir()
    fname = f"{_safe_filename(title)}.pdf"
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()
