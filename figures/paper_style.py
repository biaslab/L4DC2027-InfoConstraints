"""Shared style of the result figures of main.tex and main-extended.tex.

The panels come from the visualisation scripts of two experiments,
experiment-integrator/plot_paper.py and experiment-arm/plot_paper.py. Both
import this module, so that an agent has the same colour and marker in
every panel, and every panel has the same size and plot area, so that two
subfigures fill one column and four fill one row over the text width.
Running this module writes the shared legends of the figures:

    python figures/paper_style.py      # writes figures/legend_rmse.pdf (main.tex)
                                       # and figures/legend_results.pdf

Colours are the first seven slots of the reference categorical palette in
their validated order (adjacent pairs pass the colour-vision-deficiency
checks), with a distinct marker per agent as a second channel. The
baselines without an ARX model share neutral greys and differ by marker and
line style.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

TEXT_WIDTH = 7.0                       # \textwidth of ieeeconf (505.89 pt) [in]
COLUMN_WIDTH = 3.40                     # \columnwidth of ieeeconf (245.72 pt) [in]
PANEL_SIZE = (0.24 * TEXT_WIDTH, 1.32)  # one subfigure, four per row [in]
PANEL_MARGINS = dict(left=0.235, right=0.965, bottom=0.27, top=0.95)
# one subfigure of a figure of three over the text width (the probing-weight
# panels of main.tex) [in]
WIDE_PANEL_SIZE = (0.32 * TEXT_WIDTH, PANEL_SIZE[1])

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#898781"

# Blue ramp for ordered values of one agent (the CBFE controller's slot).
BLUE_RAMP = ["#0d366b", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef"]

AGENTS = {
    "kkt": dict(label=r"CBFE, KKT $\gamma$ (ours)", color="#2a78d6", marker="o", lw=1.8, zorder=10),
    "dual": dict(label=r"dual MPC, tuned $\rho$", color="#eb6834", marker="s"),
    "dual_100": dict(label=r"dual MPC, $\rho = 2\gamma_{\max}$", color="#eb6834", marker="s",
                     ls="--", mfc="white"),
    "mpcx": dict(label="MPC-X, tuned", color="#1baf7a", marker="D", ms=3.2),
    "gamma0": dict(label=r"CBFE, $\gamma = 0$ (cautious)", color="#eda100", marker="v"),
    "gamma1": dict(label=r"CBFE, $\gamma = 1$ (EFE)", color="#e87ba4", marker="^"),
    "ce": dict(label="certainty equivalence", color="#008300", marker="P"),
    "negentropy": dict(label=r"CBFE, floor $-H[p_*]$", color="#6250d6", marker="h"),
    "optimal": dict(label="optimal dual (DP)", color=INK, marker=None, ls="--", lw=1.0),
    "pd": dict(label="PD", color=INK_SECONDARY, marker="<", ls=":"),
    "pilco": dict(label="PILCO", color=INK_SECONDARY, marker=">", ls="-."),
    "tdmpc2": dict(label="TD-MPC2", color=MUTED, marker="p", ls=":"),
}

# Ordered values of the dual MPC (the prior means of the integrator), dark to
# light in the dual MPC's hue.
ORANGE_RAMP = ["#7c2d0c", "#b4461a", "#eb6834", "#f19566", "#f6c4a3"]

# Probing-weight panels: both share their axes, so that the windows of good
# probing weights of the two examples can be compared.
SENSITIVITY_XLIM = (0.25, 3.5e5)
SENSITIVITY_YLIM = (0.88, 4.3)


def sensitivity_axes(ax, rho_cbfe=None) -> None:
    """Log axes of a probing-weight panel, the reference line of CBFE at 1
    and, if given, a marker at twice the median multiplier of CBFE."""
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    ax.axhline(1.0, color=AGENTS["kkt"]["color"], lw=1.6, zorder=2)
    if rho_cbfe is not None:
        ax.plot([rho_cbfe], [1.0], marker="D", ms=4.2, color=AGENTS["kkt"]["color"], mec="white",
                mew=0.6, zorder=12, ls="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*SENSITIVITY_XLIM)
    ax.set_ylim(*SENSITIVITY_YLIM)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 1e2, 1e3, 1e4, 1e5]))
    ax.set_xticklabels(["1", "10", r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 4]))
    ax.set_yticklabels(["1", "1.5", "2", "3", "4"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"probing weight $\rho$ (log)")
    ax.tick_params(axis="y", pad=1.5)
    ax.set_ylabel("relative RMSE (log)", labelpad=2)


# Single-row results figure of main.tex (PMLR, \textwidth = 6.0 in): three
# panels of 0.32 \textwidth with the same plot area and a legend strip above
# the axes, so that the axes of the three panels align.
PMLR_TEXT_WIDTH = 6.0
ROW_PANEL_SIZE = (0.32 * PMLR_TEXT_WIDTH, 1.42)
ROW_MARGINS = dict(left=0.46, right=0.08, bottom=0.33, top=0.2)   # [in]


def row_panel():
    """A figure of one panel of the single-row results figure."""
    width, height = ROW_PANEL_SIZE
    fig = plt.figure(figsize=ROW_PANEL_SIZE)
    fig.subplots_adjust(left=ROW_MARGINS["left"] / width, right=1 - ROW_MARGINS["right"] / width,
                        bottom=ROW_MARGINS["bottom"] / height, top=1 - ROW_MARGINS["top"] / height)
    return fig, fig.add_subplot()


def top_legend(fig, handles, labels, **legend_kwargs) -> None:
    """A legend strip in one row above the axes of a row panel."""
    spacing = dict(loc="upper center", ncol=len(handles), frameon=False, fontsize=5.6, handlelength=1.5,
                   columnspacing=0.8, handletextpad=0.3, borderaxespad=0.05)
    spacing.update(legend_kwargs)
    fig.legend(handles, labels, **spacing)


# Lines that coincide with another agent's line are drawn first and wider,
# so that they show as a band beneath it.
UNDERLAY = {"gamma0": dict(lw=2.6)}


def apply() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.linewidth": 0.6, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK, "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "ytick.minor.size": 0, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.solid_capstyle": "round",
        "pdf.fonttype": 42, "savefig.dpi": 300,
    })


def panel(size=PANEL_SIZE, left=None):
    """A figure of one subfigure of ``size`` [in]. The side margins are
    those of PANEL_SIZE in inches, so that a wider panel keeps the room for
    the tick labels and widens only the plot area; ``left`` overrides the
    left margin [in]."""
    fig = plt.figure(figsize=size)
    margins = dict(PANEL_MARGINS)
    left_in = PANEL_MARGINS["left"] * PANEL_SIZE[0] if left is None else left
    margins["left"] = left_in / size[0]
    margins["right"] = 1 - (1 - PANEL_MARGINS["right"]) * PANEL_SIZE[0] / size[0]
    fig.subplots_adjust(**margins)
    return fig, fig.add_subplot()


def line_style(key: str, **overrides) -> dict:
    """matplotlib keyword arguments for an agent's line: series colour,
    marker with a white ring, and a thicker line for the CBFE controller."""
    style = dict(AGENTS[key])
    style.pop("label")
    kwargs = dict(color=style.pop("color"), marker=style.pop("marker", None), lw=style.pop("lw", 1.1),
                  ls=style.pop("ls", "-"), ms=style.pop("ms", 3.6), mec="white", mew=0.5,
                  zorder=style.pop("zorder", 3))
    mfc = style.pop("mfc", None)
    if mfc is not None:
        kwargs.update(mfc=mfc, mec=kwargs["color"], mew=0.8)
    kwargs.update(style)
    kwargs.update(overrides)
    return kwargs


def plot(ax, key: str, x, y, **overrides):
    return ax.plot(x, y, **line_style(key, **overrides))


def legend_strip(keys, path, ncol: int = 4, fontsize: float = 6.5, width: float = TEXT_WIDTH,
                 labels: dict | None = None, **legend_kwargs) -> None:
    """The legend of a figure, as a strip over ``width`` (the text width by
    default). ``labels`` overrides the labels of AGENTS per key, and
    ``legend_kwargs`` the spacing of the legend."""
    handles = [Line2D([], [], **line_style(key)) for key in keys]
    labels = [(labels or {}).get(key, AGENTS[key]["label"]) for key in keys]
    rows = -(-len(keys) // ncol)
    fig = plt.figure(figsize=(width, 0.12 + 0.13 * rows))
    spacing = dict(handlelength=2.0, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.0)
    spacing.update(legend_kwargs)
    fig.legend(handles, labels, loc="center", ncol=ncol, frameon=False, fontsize=fontsize, **spacing)
    fig.savefig(path)
    plt.close(fig)


def save(fig, path) -> None:
    fig.savefig(path)
    plt.close(fig)


# agents in the legend of the four-panel figure of main-extended.tex
RESULTS_LEGEND = ["kkt", "gamma0", "gamma1", "negentropy", "dual", "dual_100", "mpcx", "ce",
                  "optimal", "pd", "pilco", "tdmpc2"]

# agents in the legend of the RMSE figure of main.tex, in one row over the
# column width; the caption says that the baselines are tuned
RMSE_LEGEND = ["kkt", "dual", "mpcx", "optimal"]
RMSE_LABELS = {
    "kkt": "CBFE (ours)",
    "dual": "dual MPC",
    "mpcx": "MPC-X",
    "optimal": "optimal dual (DP)",
}

if __name__ == "__main__":
    from pathlib import Path
    apply()
    here = Path(__file__).resolve().parent
    legend_strip(RESULTS_LEGEND, here / "legend_results.pdf", ncol=6, fontsize=6.2)
    legend_strip(RMSE_LEGEND, here / "legend_rmse.pdf", ncol=len(RMSE_LEGEND), fontsize=6.2,
                 width=COLUMN_WIDTH, labels=RMSE_LABELS, handlelength=1.8, columnspacing=1.1)
