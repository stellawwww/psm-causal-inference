"""All figure generation lives here, so every plot in the project shares one style
and one export convention.

The rule: never call ``plt.savefig`` directly. Always go through :func:`save_fig`.
It writes to ``outputs/figures/psm_<name>.png`` with a consistent size, resolution
and white background, which is what makes the figures drop straight into a web page
later without any rework.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")           # no display needed; notebooks still show inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"

# Defaults chosen for the portfolio template: its body column is roughly 735 px
# wide, so 9 x 5 inches at 160 dpi (1440 x 800) stays sharp on retina screens.
FIGSIZE = (9, 5)
DPI = 160
PREFIX = "psm_"


def new_fig(figsize: tuple = FIGSIZE, **kwargs):
    """Create a figure and axes with the project's default size."""
    return plt.subplots(figsize=figsize, **kwargs)


def save_fig(fig, name: str, dpi: int = DPI, close: bool = True) -> pathlib.Path:
    """Save ``fig`` to ``outputs/figures/psm_<name>.png`` and return the path.

    Every figure that should appear in the README or on the portfolio page must be
    saved this way. A plot that is only displayed inline lives as base64 inside the
    notebook and cannot be reused anywhere else.

    Parameters
    ----------
    fig : matplotlib Figure
    name : short slug, no extension and no ``psm_`` prefix (it is added here)
    close : close the figure after saving; set False to keep showing it in a notebook
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = name if name.startswith(PREFIX) else PREFIX + name
    path = FIG_DIR / f"{stem}.png"
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    if close:
        plt.close(fig)
    return path


# --------------------------------------------------------------------- plots
def love_plot(bal: pd.DataFrame, name: str, title: str = "") -> pathlib.Path:
    """Standardized mean differences before vs after adjustment, one row per covariate.

    ``bal`` comes from :func:`src.psm.balance_table`: a frame indexed by covariate with
    a ``before`` column and any of ``matched`` / ``ipw``. Dashed guides mark the
    conventional 0.1 and 0.25 balance thresholds.
    """
    b = bal.reindex(bal["before"].abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(FIGSIZE[0], 0.35 * len(b) + 1.5))
    for col, mk in zip(b.columns, ["o", "s", "^"]):
        ax.scatter(b[col].abs(), b.index, marker=mk, label=col)
    for x in (0.1, 0.25):
        ax.axvline(x, ls="--", lw=0.8, color="grey")
    ax.set_xlabel("|standardized mean difference|")
    ax.set_title(title)
    ax.legend()
    return save_fig(fig, name)


def overlap_plot(ps: np.ndarray, t: np.ndarray, name: str, title: str = "") -> pathlib.Path:
    """Propensity score distributions by arm: the visual check for common support."""
    fig, ax = new_fig()
    bins = np.linspace(0, 1, 41)
    ax.hist(ps[t == 0], bins, alpha=.5, density=True, label="control")
    ax.hist(ps[t == 1], bins, alpha=.5, density=True, label="treated")
    ax.set_xlabel("propensity score")
    ax.set_title(title)
    ax.legend()
    return save_fig(fig, name)


def forest_plot(results: pd.DataFrame, truth: float, name: str, title: str = "",
                label_col: str = "method", est_col: str = "est",
                lo_col: str = "ci_lo", hi_col: str = "ci_hi") -> pathlib.Path:
    """Point estimate and 95% CI for each method, against a vertical benchmark line.

    This is the cover figure: it lets someone with no statistics background see the
    whole conclusion at once. ``results`` should already be ordered the way you want
    the rows to read from top to bottom.

    TODO(stella): implement. Suggested shape --
      - one row per method, ``ax.errorbar`` with horizontal CI bars
      - ``ax.axvline(truth)`` dashed, annotated with the benchmark value
      - rows whose CI misses the benchmark drawn in a different colour
    """
    raise NotImplementedError("forest_plot is yours to write in notebook 04")


def dag_plot(covariates: list, name: str, treatment: str = "Treatment\nNSW program",
             outcome: str = "Outcome\n1978 earnings",
             unobserved: str = "U  (unobserved)\nmotivation, health,\nconviction history,\nlocal labor market",
             title: str = "") -> pathlib.Path:
    """Draw the causal graph behind the covariate choice.

    ``covariates`` is a list of dicts with ``name``, ``causes_treatment`` and
    ``causes_outcome``. The same list drives the printed justification table in the
    notebook, so the picture and the prose cannot drift apart.

    Laid out by hand rather than by a graph library. A force-directed layout of a
    graph this small reads worse than a deliberate one, and hand placement keeps the
    thing the reader needs to see -- that every arrow points *into* treatment and
    outcome, and that the dashed ones are the reason the backdoor is not closed --
    in the same place every time.
    """
    n = len(covariates)
    fig, ax = plt.subplots(figsize=(FIGSIZE[0] * 1.15, 0.47 * n + 2.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    x_cov, x_t, x_y = 0.50, 0.11, 0.89
    y_top, y_bot = 0.80, 0.16
    ys = np.linspace(y_top, y_bot, n) if n > 1 else [0.5]
    y_arm, y_u = 0.075, 0.94

    def box(x, y, text, fc, ec, ls="-", fs=9, weight="normal"):
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, weight=weight, zorder=3,
                bbox=dict(boxstyle="round,pad=0.42", facecolor=fc, edgecolor=ec,
                          linestyle=ls, linewidth=1.3))

    def arrow(p0, p1, color, ls="-", lw=1.0, alpha=0.55, rad=0.0, z=1):
        ax.annotate("", xy=p1, xytext=p0, zorder=z,
                    arrowprops=dict(arrowstyle="-|>", color=color, linestyle=ls,
                                    linewidth=lw, alpha=alpha, shrinkA=13, shrinkB=15,
                                    connectionstyle=f"arc3,rad={rad}"))

    C_T, C_Y, C_U, C_COV = "#55a868", "#4c72b0", "#8172b2", "#f0f0f0"

    # covariate -> treatment (left) and covariate -> outcome (right)
    for y, c in zip(ys, covariates):
        if c["causes_treatment"]:
            arrow((x_cov, y), (x_t, y_arm), C_T, lw=1.0)
        if c["causes_outcome"]:
            arrow((x_cov, y), (x_y, y_arm), C_Y, lw=1.0)
        box(x_cov, y, c["name"], C_COV, "#999999")

    # the effect we are after
    arrow((x_t, y_arm), (x_y, y_arm), "black", lw=2.4, alpha=0.9, z=2)
    ax.text(0.5, y_arm - 0.055, "the effect we want to estimate",
            ha="center", va="center", fontsize=8.5, style="italic", color="#444444")

    # unobserved causes: the backdoor that stays open
    box(x_cov, y_u, unobserved, "white", C_U, ls="--", fs=8)
    arrow((x_cov, y_u), (x_t, y_arm), C_U, ls="--", lw=1.4, alpha=0.85, rad=0.30)
    arrow((x_cov, y_u), (x_y, y_arm), C_U, ls="--", lw=1.4, alpha=0.85, rad=-0.30)

    box(x_t, y_arm, treatment, "#dbeddb", C_T, fs=9, weight="bold")
    box(x_y, y_arm, outcome, "#dbe3f0", C_Y, fs=9, weight="bold")

    ax.text(0.02, 0.985, "solid = measured, and controlled for\ndashed = unmeasured, and cannot be",
            ha="left", va="top", fontsize=8, color="#444444")
    if title:
        ax.set_title(title, fontsize=11, pad=14)
    return save_fig(fig, name)
