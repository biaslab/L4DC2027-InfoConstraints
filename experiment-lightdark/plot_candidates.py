"""Candidate panels for the results figure of main.tex (light-dark task with
a known gain, Section 5.2), from results/paper_known_test.pkl of
run_paper.py, written to figures/candidates/:

    fig_lightdark_episode_mechanism.pdf  one episode under the tight goal (the
                                         seed with the median goal cost of
                                         CBFE): the state with the lit region,
                                         and the belief standard deviation
                                         against the goal tolerance
    fig_lightdark_cost.pdf               goal cost per step of the controllers
                                         under both goals, with bootstrap 95%
                                         intervals of the means

The panels are sized for one third of the PMLR text width (6.0 in).

    python experiment-lightdark/plot_candidates.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "figures"), str(HERE.parent)]

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

import lightdark as L  # noqa: E402
import paper_style as S  # noqa: E402
import run_paper as R  # noqa: E402

FIGURES = HERE / "figures" / "candidates"
PMLR_TEXT_WIDTH = 6.0                          # \textwidth of l4dc2027.cls [in]
PANEL = (0.32 * PMLR_TEXT_WIDTH, 1.35)         # one of three panels in a row [in]
TALL_PANEL = (PANEL[0], 2.3)                   # two stacked axes [in]
LIGHT = "#fbf3d6"                              # shade of the lit region, as in plot.py
LIT_FROM = 1.0                                 # r(x) < 0.01 for x >= 1.0

# labels of run_paper.py -> styles; BSP takes the eighth slot of the
# reference categorical palette, which no other agent of the paper uses
STYLES = {
    "CBFE": dict(S.line_style("kkt"), label="CBFE"),
    # the dot plot needs a marker for the optimal controller, whose lines have none
    "DP": dict(S.line_style("optimal", marker="D", ms=3.2), label="optimal (DP)"),
    "dual MPC, rho = 10": dict(S.line_style("dual"), label="dual MPC"),
    "BSP": dict(S.line_style("dual", color="#e34948", marker="X"), label="BSP"),
    "cautious": dict(S.line_style("gamma0"), label="cautious"),
}


def figure(size, rows: int = 1, top_in: float = 0.07, left_in: float = 0.46):
    fig, axes = plt.subplots(rows, 1, figsize=size, sharex=True, squeeze=False)
    fig.subplots_adjust(left=left_in / size[0], right=1 - 0.12 / size[0], bottom=0.36 / size[1],
                        top=1 - top_in / size[1], hspace=0.18)
    return fig, axes[:, 0]


def style(label: str, **overrides) -> dict:
    kwargs = {k: v for k, v in STYLES[label].items() if k != "label"}
    kwargs.update(overrides)
    return kwargs


def plot_episode(results, path) -> int:
    """State and belief standard deviation in one episode under the tight
    goal: CBFE returns to the light whenever its belief spreads beyond the
    goal tolerance, whereas the cautious controller never leaves the dark."""
    tight = [r for r in results if r["goal"] == "tight"]
    cbfe = sorted((R.metrics(r, R.KNOWN)["nll"], r["seed"]) for r in tight if r["label"] == "CBFE")
    seed = cbfe[len(cbfe) // 2][1]
    episodes = {r["label"]: r["episode"] for r in tight if r["seed"] == seed}
    tolerance = math.sqrt(L.GOALS["tight"])
    fig, (top, bottom) = figure(TALL_PANEL, rows=2, top_in=0.22)
    k = np.arange(len(episodes["CBFE"].x))
    top.axhspan(LIT_FROM, 3.0, color=LIGHT, lw=0, zorder=0)
    top.text(4, 2.85, "light", ha="left", va="top", fontsize=5.6, color=S.INK_SECONDARY)
    top.axhline(0.0, color=S.MUTED, lw=0.6, ls=":", zorder=1)
    for label in ("cautious", "DP", "CBFE"):
        ep = episodes[label]
        kw = dict(markevery=40, ms=2.8, lw=1.0 if label != "CBFE" else 1.4)
        if label == "DP":
            kw["marker"] = None
        top.plot(k, ep.x, **style(label, **kw))
        bottom.plot(k, np.sqrt(ep.beliefs[:, 1]), **style(label, **kw))
    top.set_ylim(-0.8, 3.0)
    top.set_ylabel(r"state $x_k$", labelpad=2)
    bottom.axhline(tolerance, color=S.MUTED, lw=0.9, ls=":", zorder=1)
    bottom.text(28, tolerance * 1.1, r"goal tolerance $\sqrt{S_*}$", ha="left", va="bottom", fontsize=5.6,
                color=S.INK_SECONDARY)
    active = np.flatnonzero(episodes["CBFE"].multipliers[:, 1] > 0.0)
    bottom.plot(active, np.full(len(active), 2.2e-3), ls="none", marker="|", ms=4, mew=0.8,
                color=S.AGENTS["kkt"]["color"], zorder=6)
    bottom.text(k[-1], 3.4e-3, "constraint active", ha="right", va="bottom", fontsize=5.6, color=S.INK_SECONDARY)
    bottom.set_yscale("log")
    bottom.set_ylim(1.5e-3, 1.2)
    bottom.set_ylabel(r"belief std. $\sqrt{V_k}$", labelpad=2)
    bottom.set_xlabel(r"step $k$")
    bottom.set_xlim(0, k[-1])
    bottom.set_xticks([0, 50, 100, 150, 200])
    labels = ("CBFE", "DP", "cautious")
    fig.legend([plt.Line2D([], [], **style(lab, ms=2.8, **({"marker": None} if lab == "DP" else {})))
                for lab in labels], [STYLES[lab]["label"] for lab in labels],
               loc="upper center", ncol=3, frameon=False, fontsize=5.6, handlelength=1.5, columnspacing=0.8,
               handletextpad=0.3, borderaxespad=0.05)
    S.save(fig, path)
    return seed


def mean_interval(values: np.ndarray, rng, draws: int = 10_000):
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def plot_cost(path) -> dict:
    """Goal cost per step of every controller under the tight (filled) and
    the loose goal (open markers), on a logarithmic axis."""
    test = R.frame("paper_known_test", "known")
    rng = np.random.default_rng(0)
    rows = ["CBFE", "DP", "dual MPC, rho = 10", "BSP", "cautious"]
    fig, (ax,) = figure((PANEL[0], PANEL[1] + 0.15), top_in=0.2, left_in=0.72)
    summary = {}
    for i, label in enumerate(rows):
        y = len(rows) - 1 - i
        for goal, offset in (("tight", 0.12), ("loose", -0.12)):
            mean, low, high = mean_interval(test[(test.label == label) & (test.goal == goal)].nll.to_numpy(), rng)
            summary[(label, goal)] = (round(mean, 3), round(low, 3), round(high, 3))
            kw = style(label, ms=3.6, ls="none", zorder=4)
            if goal == "loose":
                kw.update(mfc="white", mec=kw["color"], mew=0.8)
            ax.plot([low, high], [y + offset] * 2, color=kw["color"], lw=0.9, solid_capstyle="butt", zorder=3)
            ax.plot([mean], [y + offset], **kw)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([STYLES[label]["label"] for label in reversed(rows)])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10]))
    ax.set_xticklabels(["0.1", "0.3", "1", "3", "10"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0.09, 11)
    ax.set_xlabel("goal cost per step (log)")
    handles = [plt.Line2D([], [], ls="none", marker="o", ms=3.6, color=S.INK_SECONDARY, mec="white", mew=0.5),
               plt.Line2D([], [], ls="none", marker="o", ms=3.6, color=S.INK_SECONDARY, mfc="white", mew=0.8)]
    fig.legend(handles, [r"tight, $S_* = 0.01$", r"loose, $S_* = 0.1$"], loc="upper center", ncol=2,
               frameon=False, fontsize=5.6, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.05)
    S.save(fig, path)
    return summary


def main() -> None:
    S.apply()
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = L.load("paper_known_test")
    seed = plot_episode(results, FIGURES / "fig_lightdark_episode_mechanism.pdf")
    print("episode: seed", seed)
    for key, value in plot_cost(FIGURES / "fig_lightdark_cost.pdf").items():
        print("cost", key, value)
    print(FIGURES)


if __name__ == "__main__":
    main()
