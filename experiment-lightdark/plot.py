"""Figure of the light-dark experiments of main.tex (Section V-C), from the
results of ``run_paper.py``:

    figures/fig_lightdark_paths.pdf        state trajectories with direction-
                                           dependent gains under the tight goal
                                           (the seed with the median goal cost of
                                           CBFE), with the lit region shaded
    figures/fig_lightdark_sensitivity.pdf  goal cost of the dual MPC relative to
                                           CBFE against the probing weight, with
                                           a known gain, per goal

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


def main() -> None:
    S.apply()
    FIGURES.mkdir(exist_ok=True)
    seed = paths(FIGURES / "fig_lightdark_paths.pdf")
    print("paths: seed", seed)
    sensitivity(FIGURES / "fig_lightdark_sensitivity.pdf")
    print(FIGURES)


if __name__ == "__main__":
    main()
