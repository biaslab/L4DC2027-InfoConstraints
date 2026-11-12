"""Figure of the light-dark experiments of main.tex (Section V-C), from the
results of ``run_paper.py``:

    figures/fig_lightdark_paths.pdf        state trajectories with direction-
                                           dependent gains under the tight goal
                                           (the seed with the median goal cost of
                                           CBFE), with the lit region shaded
    figures/fig_lightdark_sensitivity.pdf  goal cost of the dual MPC relative to
                                           CBFE against the probing weight, with
                                           a known gain, per goal
    figures/fig_lightdark_belief.pdf       belief standard deviation of one
                                           episode under the tight goal, with the
                                           steps of an active constraint
                                           (main.tex, Fig. 2b)
    figures/fig_lightdark_weights.pdf      goal cost of the dual MPC against the
                                           probing weight per goal, with CBFE as
                                           horizontal lines (main.tex, Fig. 2c)

    python experiment-lightdark/plot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "figures")]

import numpy as np  # noqa: E402

import lightdark as L  # noqa: E402
import paper_style as S  # noqa: E402
import run_paper as R  # noqa: E402

FIGURES = HERE / "figures"
STEPS = 30


def paths(path: Path) -> None:
    import matplotlib.pyplot as plt
    results = L.load("paper_directional_test")
    tight = [r for r in results if r["goal"] == "tight"]
    cbfe = sorted((R.metrics(r, R.DIRECTIONAL)["nll"], r["seed"]) for r in tight if r["label"] == "CBFE")
    seed = cbfe[len(cbfe) // 2][1]
    episodes = {r["label"]: r["episode"] for r in tight if r["seed"] == seed}
    fig, ax = S.panel()
    low = min(-0.6, min(float(ep.x[:STEPS + 1].min()) for ep in episodes.values()) - 0.25)
    ax.axhspan(1.0, 3.2, color="#fbf3d6", lw=0, zorder=0)
    ax.text(0.6, 3.05, "light", ha="left", va="top", fontsize=6, color=S.INK_SECONDARY)
    ax.axhline(0.0, color=S.MUTED, lw=0.6, ls=":", zorder=1)
    k = np.arange(STEPS + 1)
    for label, key, name in (("cautious", "gamma0", "cautious"), ("CBFE, state only", "negentropy", "state only"),
                             ("CBFE", "kkt", "CBFE")):
        S.plot(ax, key, k, episodes[label].x[:STEPS + 1], ms=2.4, label=name)
    ax.set_xlim(0, STEPS)
    ax.set_ylim(low, 3.2)
    ax.set_xlabel("step $k$")
    ax.set_ylabel("state $x_k$", labelpad=2)
    ax.legend(frameon=False, loc="upper right", fontsize=5.8, handlelength=1.6)
    S.save(fig, path)
    plt.close(fig)
    return seed


def sensitivity(path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    test = R.frame("paper_known_test", "known")
    tuning = R.frame("paper_known_tuning", "known")
    shared = float(tuning.groupby("label").rmse.mean().idxmin().split("= ")[-1])
    curve = test[test.label.str.startswith("dual MPC")].copy()
    curve["rho"] = curve.label.str.split("= ").str[-1].astype(float)
    cbfe = test[test.label == "CBFE"].groupby("goal").nll.mean()
    relative = curve.pivot_table(index="rho", columns="goal", values="nll", aggfunc="mean") / cbfe
    fig, ax = S.panel(S.WIDE_PANEL_SIZE, left=0.25 * S.PANEL_SIZE[0])
    ax.axhline(1.0, color=S.AGENTS["kkt"]["color"], lw=1.6, zorder=2, label="CBFE")
    for goal, color, marker in (("tight", S.ORANGE_RAMP[1], "s"), ("loose", S.ORANGE_RAMP[3], "o")):
        ax.plot(relative.index, relative[goal], color=color, marker=marker, ms=3.4, mec="white", mew=0.5, lw=1.1,
                label=f"dual MPC, {goal}")
    ax.axvline(shared, color=S.MUTED, lw=0.6, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100]))
    ax.set_xticklabels(["1", "10", r"$10^2$"])
    ax.xaxis.set_minor_locator(NullLocator())
    ticks = [t for t in (0.5, 1, 2, 4, 8, 16, 32) if relative.min().min() / 1.3 <= t <= relative.max().max() * 1.3]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"probing weight $\rho$ (log)")
    ax.set_ylabel("relative cost (log)", labelpad=2)
    ax.legend(frameon=False, loc="upper left", fontsize=5.6, handlelength=1.5, borderaxespad=0.1)
    S.save(fig, path)
    plt.close(fig)


def belief(path: Path) -> int:
    """Panel (b) of the results figure of main.tex: belief standard
    deviation in the episode with the median goal cost of CBFE under the
    tight goal, with a known gain. The ticks at the top mark the steps at
    which the state constraint of CBFE is active."""
    import math
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    results = [r for r in L.load("paper_known_test") if r["goal"] == "tight"]
    cbfe = sorted((R.metrics(r, R.KNOWN)["nll"], r["seed"]) for r in results if r["label"] == "CBFE")
    seed = cbfe[len(cbfe) // 2][1]
    episodes = {r["label"]: r["episode"] for r in results if r["seed"] == seed}
    fig, ax = S.row_panel()
    k = np.arange(len(episodes["CBFE"].x))
    tolerance = math.sqrt(L.GOALS["tight"])
    ax.axhline(tolerance, color=S.MUTED, lw=0.9, ls=":", zorder=1)
    ax.text(6, tolerance * 1.12, r"tolerance $\sqrt{S_*}$", ha="left", va="bottom", fontsize=5.6,
            color=S.INK_SECONDARY)
    keys = {"cautious": "gamma0", "DP": "optimal", "CBFE": "kkt"}
    for label, key in keys.items():
        S.plot(ax, key, k, np.sqrt(episodes[label].beliefs[:, 1]), markevery=40, ms=2.8,
               lw=1.4 if key == "kkt" else 1.0, zorder=5 if key == "kkt" else 3)
    active = np.flatnonzero(episodes["CBFE"].multipliers[:, 1] > 0.0)
    ax.plot(active, np.full(len(active), 1.25), ls="none", marker="|", ms=4.5, mew=0.9,
            color=S.AGENTS["kkt"]["color"], zorder=6, clip_on=False)
    ax.set_yscale("log")
    ax.set_ylim(0.02, 1.25)
    ax.yaxis.set_major_locator(FixedLocator([0.03, 0.1, 0.3, 1]))
    ax.set_yticklabels(["0.03", "0.1", "0.3", "1"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0, k[-1])
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xlabel("step $k$")
    ax.set_ylabel(r"belief std. $\sqrt{V_k}$", labelpad=2)
    names = {"kkt": "CBFE", "optimal": "optimal", "gamma0": "cautious"}
    S.top_legend(fig, [Line2D([], [], **S.line_style(key, ms=2.8)) for key in names], list(names.values()))
    S.save(fig, path)
    return seed


def weights(path: Path) -> float:
    """Panel (c) of the results figure of main.tex: goal cost per step of
    the dual MPC against its probing weight under the tight (solid) and the
    loose goal (dashed), with the costs of CBFE as horizontal lines and the
    weight tuned on held-out seeds as a dotted line."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    test = R.frame("paper_known_test", "known")
    tuning = R.frame("paper_known_tuning", "known")
    tuned = float(tuning.groupby("label").rmse.mean().idxmin().split("= ")[-1])
    curve = test[test.label.str.startswith("dual MPC")].copy()
    curve["rho"] = curve.label.str.split("= ").str[-1].astype(float)
    dual = curve.pivot_table(index="rho", columns="goal", values="nll", aggfunc="mean")
    cbfe = test[test.label == "CBFE"].groupby("goal").nll.mean()
    fig, ax = S.row_panel()
    ax.axvline(tuned, color=S.MUTED, lw=0.6, ls=":", zorder=1)
    for goal, ls in (("tight", "-"), ("loose", "--")):
        ax.axhline(cbfe[goal], color=S.AGENTS["kkt"]["color"], lw=1.4, ls=ls, zorder=2)
        overrides = dict(ls=ls) if goal == "tight" else dict(ls=ls, mfc="white", mec=S.AGENTS["dual"]["color"],
                                                             mew=0.8)
        S.plot(ax, "dual", dual.index, dual[goal], ms=3.2, **overrides)
        ax.text(0.75, cbfe[goal] * 1.12, goal, ha="left", va="bottom", fontsize=5.6, color=S.INK_SECONDARY)
    ax.text(tuned / 1.3, 0.1, "tuned", ha="right", va="bottom", fontsize=5.6, color=S.INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.65, 420)
    ax.set_ylim(0.09, 13)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100]))
    ax.set_xticklabels(["1", "10", "100"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10]))
    ax.set_yticklabels(["0.1", "0.3", "1", "3", "10"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"probing weight $\rho$ (log)")
    ax.set_ylabel("goal cost (log)", labelpad=2)
    handles = [Line2D([], [], color=S.AGENTS["kkt"]["color"], lw=1.4),
               Line2D([], [], **S.line_style("dual", ms=3.2))]
    S.top_legend(fig, handles, ["CBFE", "dual MPC"])
    S.save(fig, path)
    return tuned


def main() -> None:
    S.apply()
    FIGURES.mkdir(exist_ok=True)
    seed = paths(FIGURES / "fig_lightdark_paths.pdf")
    print("paths: seed", seed)
    sensitivity(FIGURES / "fig_lightdark_sensitivity.pdf")
    print("belief: seed", belief(FIGURES / "fig_lightdark_belief.pdf"))
    print("weights: tuned rho", weights(FIGURES / "fig_lightdark_weights.pdf"))
    print(FIGURES)


if __name__ == "__main__":
    main()
