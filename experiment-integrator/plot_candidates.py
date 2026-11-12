"""Candidate panels for the results figure of main.tex, from
results/benchmark.pkl of run.py, written to figures/candidates/:

    fig_integrator_episode_mechanism.pdf  one episode under the symmetric
                                          prior b0 = 0: the output and the
                                          posterior variance of the gain
    fig_integrator_activity_paper.pdf     episodes with an active information
                                          constraint, per step and prior mean
    fig_integrator_rmse_paper.pdf         RMSE per prior mean for the
                                          controllers of Section 5.1

All panels use the controllers of the paper (analysis.py), i.e. CBFE with
the predictive floor on the next node and the dual MPC with the Beta input
cost, and are sized for one third of the PMLR text width (6.0 in).

    python experiment-integrator/plot_candidates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "figures")]

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

import integrator as I  # noqa: E402
import paper_style as S  # noqa: E402
import run as Run  # noqa: E402

FIGURES = HERE / "figures" / "candidates"
PMLR_TEXT_WIDTH = 6.0                          # \textwidth of l4dc2027.cls [in]
PANEL = (0.32 * PMLR_TEXT_WIDTH, 1.35)         # one of three panels in a row [in]
TALL_PANEL = (PANEL[0], 2.3)                   # two stacked axes [in]

# labels of run.py -> agents of paper_style, for the controllers of the paper
PAPER = {
    "CBFE-ARX, KKT β, predictive floor, next node": "kkt",
    "CBFE-ARX, β = 0": "gamma0",
    Run.dual_beta_label(Run.LAMBDA_BETA): "dual",
    f"MPC-X, γ = {Run.GAMMA:g}": "mpcx",
    "optimal dual (DP)": "optimal",
}
NAMES = {"kkt": "CBFE", "gamma0": "cautious", "dual": r"dual MPC, $\rho = 5$", "mpcx": "MPC-X",
         "optimal": "optimal (DP)"}


def figure(size, rows: int = 1, top_in: float = 0.07):
    """A figure of ``rows`` axes with a shared x-axis, a left margin of
    0.46 in for the tick labels and ``top_in`` [in] above the axes."""
    fig, axes = plt.subplots(rows, 1, figsize=size, sharex=True, squeeze=False)
    left = 0.46 / size[0]
    right = 1 - 0.06 / size[0]
    bottom = 0.36 / size[1]
    top = 1 - top_in / size[1]
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=0.18)
    return fig, axes[:, 0]


def example_episode(results):
    """The episode of plot_paper.py: the first seed under b0 = 0 whose gain
    has 0.6 < |b| < 1."""
    seed = next(r["seed"] for r in results if r["b0"] == 0.0 and 0.6 < abs(r["episode"].b) < 1.0)
    return seed, {PAPER[r["label"]]: r["episode"] for r in results
                  if r["b0"] == 0.0 and r["seed"] == seed and r["label"] in PAPER}


def plot_episode(results, path) -> None:
    """Output y_t / sigma and posterior variance P_t of the gain in one
    episode: CBFE probes at the first step and learns the gain, whereas the
    cautious controller applies no input, never learns and drifts."""
    seed, episodes = example_episode(results)
    sigma = I.Benchmark().sigma
    fig, (top, bottom) = figure(TALL_PANEL, rows=2, top_in=0.22)
    t = np.arange(len(episodes["kkt"].y))
    for key in ("gamma0", "optimal", "kkt"):
        extra = {"zorder": 11} if key == "optimal" else {}
        S.plot(top, key, t, episodes[key].y / sigma, markevery=10, ms=3.0, **extra)
        S.plot(bottom, key, t, episodes[key].P, markevery=10, ms=3.0, **extra)
    top.axhline(0.0, color=S.MUTED, lw=0.6, ls=":", zorder=1)
    top.set_ylabel(r"output $y_t/\sigma$", labelpad=2)
    bottom.set_yscale("log")
    bottom.set_ylabel(r"gain variance $P_t$", labelpad=2)
    bottom.set_xlabel(r"step $t$")
    bottom.set_xticks([0, 10, 20, 30, 40, 50])
    bottom.text(48, 0.55, "cautious: no learning", ha="right", va="top", fontsize=5.6,
                color=S.INK_SECONDARY)
    bottom.text(12, 0.05, "CBFE probes at $t = 1$", ha="left", va="bottom", fontsize=5.6,
                color=S.INK_SECONDARY)
    keys = ("kkt", "gamma0", "optimal")
    fig.legend([plt.Line2D([], [], **S.line_style(k, ms=3.0)) for k in keys], [NAMES[k] for k in keys],
               loc="upper center", ncol=3, frameon=False, fontsize=5.6, handlelength=1.5,
               columnspacing=0.8, handletextpad=0.3, borderaxespad=0.05)
    S.save(fig, path)
    print(f"episode: seed {seed}, b = {episodes['kkt'].b:.2f}")


def plot_activity(results, path, steps: int = 12) -> None:
    """Share of the episodes whose information constraint is active at each
    step, per prior mean: active at the first step for b0 <= 0.5 and never
    for b0 = 4."""
    cbfe = [r for r in results if PAPER.get(r["label"]) == "kkt"]
    k = np.arange(1, steps + 1)
    fig, (ax,) = figure(PANEL)
    for color, b0 in zip(S.BLUE_RAMP, Run.B0):
        active = np.mean([r["episode"].beta[:steps] > 0 for r in cbfe if r["b0"] == b0], axis=0)
        ax.plot(k, 100 * active, color=color, marker="o", ms=3.0, mec="white", mew=0.4, lw=1.1,
                label=fr"$b_0 = {b0:g}$")
    ax.set(xlabel=r"step $t$", ylim=(-4, 104), xticks=[1, 4, 8, 12])
    ax.set_ylabel("active episodes [%]", labelpad=2)
    ax.legend(loc="upper right", frameon=False, handlelength=1.4, labelspacing=0.15, borderaxespad=0.1,
              fontsize=5.6)
    S.save(fig, path)


def rmse_per_prior(results):
    rmse = {}
    for r in results:
        if r["label"] in PAPER:
            rmse.setdefault((PAPER[r["label"]], r["b0"]), []).append(np.sqrt(np.mean(r["episode"].y[1:] ** 2)))
    return {key: float(np.mean(v)) for key, v in rmse.items()}


def plot_rmse(results, path) -> None:
    """RMSE per prior mean: CBFE tracks the tuned dual MPC and MPC-X for
    every prior mean, whereas the cautious controller fails under b0 = 0."""
    rmse = rmse_per_prior(results)
    fig, (ax,) = figure((PANEL[0], PANEL[1] + 0.2), top_in=0.29)
    x = np.arange(len(Run.B0))
    for key in ("optimal", "mpcx", "dual", "kkt"):
        S.plot(ax, key, x, [rmse[(key, b0)] for b0 in Run.B0], ms=3.2)
    ax.set_ylim(0.096, 0.137)
    ax.set_yticks([0.10, 0.11, 0.12, 0.13])
    ax.set_yticklabels(["0.10", "0.11", "0.12", "0.13"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b0:g}" for b0 in Run.B0])
    ax.set_xlabel(r"prior mean $b_0$ of the gain")
    ax.set_ylabel("RMSE", labelpad=2)
    ax.annotate(f"cautious: {rmse[('gamma0', 0.0)]:.2f}, off scale", xy=(0, 0.1365), xytext=(0.2, 0.1352),
                fontsize=5.6,
                color=S.INK_SECONDARY, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=S.MUTED, lw=0.6, shrinkA=0, shrinkB=0,
                                mutation_scale=5))
    keys = ("kkt", "dual", "mpcx", "optimal")
    fig.legend([plt.Line2D([], [], **S.line_style(k, ms=3.0)) for k in keys], [NAMES[k] for k in keys],
               loc="upper center", ncol=2, frameon=False, fontsize=5.6, handlelength=1.5, columnspacing=0.8,
               handletextpad=0.3, labelspacing=0.15, borderaxespad=0.05)
    S.save(fig, path)
    print("RMSE per prior mean:", {f"{k}, b0={b:g}": round(v, 3) for (k, b), v in sorted(rmse.items())})


def main() -> None:
    S.apply()
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = I.load("benchmark")
    plot_episode(results, FIGURES / "fig_integrator_episode_mechanism.pdf")
    plot_activity(results, FIGURES / "fig_integrator_activity_paper.pdf")
    plot_rmse(results, FIGURES / "fig_integrator_rmse_paper.pdf")
    print(FIGURES)


if __name__ == "__main__":
    main()
