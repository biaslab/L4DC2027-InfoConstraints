"""Panels of the result figures, from results/benchmark.pkl of run.py,
written to figures/:

    fig_integrator_rmse.pdf      RMSE of the output per prior mean b0
                                 (main.tex, Fig. 3a)
    fig_integrator_loss.pdf      mean loss relative to the optimal dual
                                 controller, per prior mean b0 (|b| > 0.3)
    fig_integrator_activity.pdf  episodes in which the information constraint
                                 is active, per step and prior mean
    fig_integrator_episode.pdf   one episode under the symmetric prior b0 = 0
                                 (main-extended.tex only)
    fig_integrator_constraint.pdf  episodes in which the information constraint
                                 of the paper's CBFE is active, per step and
                                 prior mean (main.tex, Fig. 3 right)
    fig_integrator_sensitivity.pdf  RMSE of the dual MPC relative to that of
                                 CBFE against its probing weight, per prior
                                 mean b0 (main.tex, Fig. 3 left)

The two panels of Fig. 3 are drawn at their printed widths, 0.45 and 0.4 of
the PMLR text width, and at the height of the panels of Fig. 2, with their
legends inside the axes.
    legend_integrator.pdf       the legend of this row in main-extended.tex

The legends of the figures of main.tex come from figures/paper_style.py.

    python experiment-integrator/plot_paper.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "figures")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

import integrator as I  # noqa: E402
import paper_style as S  # noqa: E402
import run as Run  # noqa: E402

FIGURES = HERE / "figures"
# labels of run.py -> controllers of paper_style
AGENT = {
    "CBFE-ARX, KKT β": "kkt",
    "CBFE-ARX, β = 0": "gamma0",
    "CBFE-ARX, β = 1 (EFE)": "gamma1",
    "CBFE-ARX, KKT β, floor −H[p*]": "negentropy",
    f"dual MPC, λ = {Run.LAMBDA:g}": "dual",
    f"MPC-X, γ = {Run.GAMMA:g}": "mpcx",
    "certainty equivalence": "ce",
    "optimal dual (DP)": "optimal",
}
LOSS_ORDER = ["gamma0", "gamma1", "ce", "negentropy", "dual", "mpcx", "kkt"]
# controllers of the RMSE panel; the optimal dual controller is drawn first
RMSE_ORDER = ["dual", "mpcx", "kkt"]
ACTIVITY_STEPS = 15
EPISODE_AGENTS = ["gamma0", "optimal", "kkt"]
# Panels of Fig. 3 of main.tex, at their printed width and at the height and
# margins of the panels of Fig. 2 (experiment-quadcopter/plot.py, COMPACT)
PAPER_HEIGHT = 1.15                                                     # [in]
PAPER_MARGINS = dict(left=0.42, right=0.08, bottom=0.33, top=0.06)      # [in]


def paper_panel(fraction: float):
    """A panel of ``fraction`` of the PMLR text width, drawn at its printed size."""
    import matplotlib.pyplot as plt
    width, m = fraction * S.PMLR_TEXT_WIDTH, PAPER_MARGINS
    fig = plt.figure(figsize=(width, PAPER_HEIGHT))
    fig.subplots_adjust(left=m["left"] / width, right=1 - m["right"] / width,
                        bottom=m["bottom"] / PAPER_HEIGHT, top=1 - m["top"] / PAPER_HEIGHT)
    return fig, fig.add_subplot()


def loss_ratio(results) -> pd.DataFrame:
    """Mean loss per agent and prior mean over the episodes with |b| > 0.3,
    divided by the mean loss of the optimal dual controller on the same
    episodes."""
    df = pd.DataFrame([dict(agent=AGENT[r["label"]], b0=r["b0"], loss=r["loss"], b=r["episode"].b)
                       for r in results if r["label"] in AGENT])
    mean = df[df.b.abs() > 0.3].pivot_table(index="agent", columns="b0", values="loss", aggfunc="mean")
    return mean / mean.loc["optimal"]


def plot_loss(results, path) -> None:
    ratio = loss_ratio(results)
    fig, ax = S.panel()
    x = np.arange(len(Run.B0))
    ax.axhline(1.0, **S.line_style("optimal"))
    for key in LOSS_ORDER:
        # gamma = 0 and gamma = 1 coincide; the wider gamma = 0 line shows beneath
        S.plot(ax, key, x, ratio.loc[key, list(Run.B0)].to_numpy(), **S.UNDERLAY.get(key, {}))
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20])
    ax.set_yticklabels(["1", "2", "5", "10", "20"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.85, 32)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b0:g}" for b0 in Run.B0])
    ax.set(xlabel=r"prior mean $b_0$ of the gain", ylabel="loss / optimum")
    S.save(fig, path)


def rmse_per_prior(results) -> pd.DataFrame:
    """Mean over the episodes of the RMSE of the outputs y_1, ..., y_T from
    the goal y_* = 0, per agent and prior mean b0 (all gains b)."""
    df = pd.DataFrame([dict(agent=AGENT[r["label"]], b0=r["b0"],
                            rmse=np.sqrt(np.mean(r["episode"].y[1:] ** 2)))
                       for r in results if r["label"] in AGENT])
    return df.pivot_table(index="agent", columns="b0", values="rmse", aggfunc="mean")


def plot_rmse(results, path) -> None:
    rmse = rmse_per_prior(results)
    fig, ax = S.panel()
    x = np.arange(len(Run.B0))
    for key in ["optimal"] + RMSE_ORDER:
        S.plot(ax, key, x, rmse.loc[key, list(Run.B0)].to_numpy())
    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.12, 0.14, 0.16])
    ax.set_yticklabels(["0.10", "0.12", "0.14", "0.16"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.095, 0.165)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b0:g}" for b0 in Run.B0])
    ax.set_xlabel(r"prior mean $b_0$ of the gain")
    # the four-character tick labels need tighter padding to keep the label inside the panel
    ax.tick_params(axis="y", pad=1.5)
    ax.set_ylabel("RMSE", labelpad=2)
    S.save(fig, path)


CBFE = "CBFE-ARX, KKT β, predictive floor, next node"


def plot_sensitivity(results, path) -> None:
    """G (``integrator.Episode.G``) of the dual MPC with the Beta input cost
    over its probing weight, relative to G of CBFE, per prior mean b0. The
    weight 0 coincides with 0.5 and is omitted on the logarithmic axis."""
    sigma = I.Benchmark().sigma
    perf = {}
    for r in results:
        if r["label"] == CBFE or r["label"].startswith("dual MPC, Beta cost"):
            perf.setdefault((r["label"], r["b0"]), []).append(r["episode"].G(sigma))
    perf = {key: np.mean(values) for key, values in perf.items()}
    weights = [w for w in Run.LAMBDAS if w > 0]
    fig, ax = paper_panel(0.45)
    beta = np.concatenate([r["episode"].beta for r in results if r["label"] == CBFE])
    S.sensitivity_axes(ax, rho_cbfe=2 * np.median(beta[beta > 0]))
    # The panel shows the integrator only, so the x-axis spans this grid
    # of weights. The y-axis is linear with even ticks and stops at TOP, so
    # that the window of good weights stays visible; the weights that leave
    # the dual MPC idle under b0 = 0 lie far above it, and a label gives
    # their ratio.
    top = 2.2
    ax.set_xlim(0.35, 75)
    ax.xaxis.set_major_locator(FixedLocator([1, 10]))
    ax.set_xticklabels(["1", "10"])
    ax.set_yscale("linear")
    ax.set_ylim(0.85, top)
    ax.yaxis.set_major_locator(FixedLocator([1.0, 1.5, 2.0]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel(r"relative $G$", labelpad=2)
    for color, b0 in zip(S.ORANGE_RAMP, Run.B0):
        ratio = [perf[(Run.dual_beta_label(w), b0)] / perf[(CBFE, b0)] for w in weights]
        ax.plot(weights, ratio, color=color, marker="s", ms=2.8, mec="white", mew=0.4, lw=1.0,
                label=fr"$b_0 = {b0:g}$", zorder=4)
        off = [w for w, q in zip(weights, ratio) if q > top]
        if off:
            x = float(np.sqrt(off[0] * off[-1]))
            ax.annotate(f"{max(ratio):.1f}", xy=(x, top), xytext=(x, top - 0.32), ha="center", va="bottom",
                        fontsize=6, color=color, annotation_clip=False,
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.7, shrinkA=0, shrinkB=0))
    ax.legend(loc="upper right", ncol=2, frameon=False, handlelength=1.1, labelspacing=0.15, columnspacing=0.6,
              handletextpad=0.4, borderaxespad=0.15, fontsize=5.6)
    S.save(fig, path)


def plot_activity(results, path) -> None:
    kkt = [r for r in results if r["label"] == "CBFE-ARX, KKT β"]
    steps = np.arange(1, ACTIVITY_STEPS + 1)
    fig, ax = S.panel()
    for color, b0 in zip(S.BLUE_RAMP, Run.B0):
        active = np.mean([r["episode"].beta[:ACTIVITY_STEPS] > 0 for r in kkt if r["b0"] == b0], axis=0)
        ax.plot(steps, 100 * active, color=color, marker="o", ms=3.0, mec="white", mew=0.4, lw=1.1,
                label=fr"$b_0 = {b0:g}$")
    ax.set(xlabel=r"step $t$", ylabel="active [%]", ylim=(-4, 104),
           xticks=[1, 5, 10, 15])
    ax.legend(loc="upper right", frameon=False, handlelength=1.4, labelspacing=0.2,
              borderaxespad=0.2, fontsize=6)
    S.save(fig, path)


def plot_constraint(results, path, steps: int = 10) -> None:
    """Right panel of Fig. 3 of main.tex: share of the episodes in which the
    information constraint of CBFE (the controller of the paper) is active,
    per step and prior mean b0. The legend fills the empty upper right,
    because the constraint acts only in the first steps."""
    cbfe = [r for r in results if r["label"] == CBFE]
    k = np.arange(1, steps + 1)
    fig, ax = paper_panel(0.4)
    for color, b0 in zip(S.BLUE_RAMP, Run.B0):
        active = np.mean([r["episode"].beta[:steps] > 0 for r in cbfe if r["b0"] == b0], axis=0)
        ax.plot(k, 100 * active, color=color, marker="o", ms=2.8, mec="white", mew=0.4, lw=1.1,
                zorder=4 - Run.B0.index(b0) / 10, label=fr"$b_0 = {b0:g}$")
    ax.set(xlim=(0.6, steps + 0.4), ylim=(-4, 104), xticks=[1, 4, 7, 10], yticks=[0, 25, 50, 75, 100])
    ax.set_xlabel(r"step $t$")
    ax.set_ylabel("active [%]", labelpad=2)
    ax.legend(loc="upper right", frameon=False, handlelength=1.3, labelspacing=0.15, borderaxespad=0.2,
              fontsize=6)
    S.save(fig, path)


def example_episode(results):
    """The episode of the integrator notebook: the first seed under b0 = 0
    whose gain has 0.6 < |b| < 1."""
    seed = next(r["seed"] for r in results if r["b0"] == 0.0 and 0.6 < abs(r["episode"].b) < 1.0)
    return seed, {AGENT[r["label"]]: r["episode"] for r in results
                  if r["b0"] == 0.0 and r["seed"] == seed and r["label"] in AGENT}


def plot_episode(results, path) -> None:
    seed, episodes = example_episode(results)
    sigma = I.Benchmark().sigma
    fig, ax = S.panel()
    for key in EPISODE_AGENTS:
        y = episodes[key].y / sigma
        # the optimal controller's dashed line goes on top of the CBFE line
        S.plot(ax, key, np.arange(len(y)), y, markevery=5,
               **({"zorder": 11} if key == "optimal" else {}))
    ax.set(xlabel=r"step $t$", ylabel=r"output $y_t / \sigma$", xticks=[0, 10, 20, 30, 40, 50],
           yticks=[-6, -4, -2, 0, 2, 4])
    S.save(fig, path)
    print(f"example episode: seed {seed}, b = {episodes['kkt'].b:.2f}")


def main() -> None:
    S.apply()
    FIGURES.mkdir(exist_ok=True)
    results = I.load("benchmark")
    plot_rmse(results, FIGURES / "fig_integrator_rmse.pdf")
    plot_sensitivity(results, FIGURES / "fig_integrator_sensitivity.pdf")
    plot_constraint(results, FIGURES / "fig_integrator_constraint.pdf")
    plot_loss(results, FIGURES / "fig_integrator_loss.pdf")
    plot_activity(results, FIGURES / "fig_integrator_activity.pdf")
    plot_episode(results, FIGURES / "fig_integrator_episode.pdf")
    S.legend_strip(["kkt", "gamma0", "gamma1", "negentropy", "dual", "mpcx", "ce", "optimal"],
                   FIGURES / "legend_integrator.pdf")
    print(loss_ratio(results).round(2).loc[LOSS_ORDER])
    print("RMSE per prior mean:\n", rmse_per_prior(results).round(3).loc[["optimal"] + RMSE_ORDER])


if __name__ == "__main__":
    main()
