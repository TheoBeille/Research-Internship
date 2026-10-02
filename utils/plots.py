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


# one fixed style per method, so that all figures are consistent
STYLES = {
    "Zero deviation":         dict(color=PAPER["blue"], ls="-"),
    "Learned":                dict(color=PAPER["orange"], ls="--"),
    "Random":                 dict(color=PAPER["green"], ls=":"),
    "Learned (no safeguard)": dict(color=PAPER["red"], ls="--"),
    "PDHG":                   dict(color=PAPER["purple"], ls="-"),
}


def plot_curves(curves, ylabel, name, loglog=True, slopes=False, horizon=None,
                hline=None, show=False):
    """
    Plot one curve per method against the iteration number.

    curves  : dict {label: values, one per iteration}
    slopes  : add O(1/n) and O(1/n^2) reference lines
    horizon : draw a vertical line at the training horizon
    hline   : (value, label) horizontal reference line
    Saved as plots/<name>.pdf.
    """
    plt.figure(figsize=(6, 6))
    plot = plt.loglog if loglog else plt.semilogx
    for label, values in curves.items():
        plot(np.arange(1, len(values) + 1), values, label=label, **STYLES.get(label, {}))

    if slopes:
        first = next(iter(curves.values()))
        n = np.arange(1, len(first) + 1)
        plot(n, first[0] / n, color=PAPER["gray"], lw=1.0, ls="--", label=r"$O(1/n)$")
        plot(n, first[0] / n ** 2, color=PAPER["gray"], lw=1.0, ls=":", label=r"$O(1/n^2)$")
    if horizon is not None:
        plt.axvline(horizon, color=PAPER["gray"], lw=1.0, ls="-.", label=f"$T={horizon}$ (training)")
    if hline is not None:
        plt.axhline(hline[0], color=PAPER["gray"], lw=1.0, ls="--", label=hline[1])

    plt.xlabel("iteration $n$")
    plt.ylabel(ylabel)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(_ensure_plots_dir(), f"{name}.pdf"))
    plt.show() if show else plt.close()


def show_images(images, name, clim=(0, 1), show=False):
    """
    Row of grayscale images. images : dict {title: tensor [1, 1, H, W]}.
    clim : gray-level range, or None to scale each image separately.
    Saved as plots/<name>.pdf.
    """
    vmin, vmax = clim if clim is not None else (None, None)
    fig, axes = plt.subplots(1, len(images), figsize=(3.2 * len(images), 3.6))
    for ax, (title, image) in zip(np.atleast_1d(axes), images.items()):
        ax.imshow(image.squeeze().cpu().numpy(), cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(_ensure_plots_dir(), f"{name}.pdf"))
    plt.show() if show else plt.close()
