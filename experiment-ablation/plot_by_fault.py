"""Candidate panels for the quadcopter figure: goal cost plus free energy per
step over the flight, split by the size of the compass fault |chi|. Pooled over
all faults, two regimes mix: caution costs the most when the fault is small,
and exploitation when it is large. The split shows each difference where it
arises:

    figures/fig_ablation_fault_legend.pdf  legend strip with the oracle
    figures/fig_ablation_fault_small.pdf   |chi| < 1 rad
    figures/fig_ablation_fault_medium.pdf  1 <= |chi| < 2 rad
    figures/fig_ablation_fault_large.pdf   |chi| >= 2 rad

The goal cost is that of ``plot.py`` on the true position, the free energy that
of ``free_energy.py``; the lines are means over the flights in the bin at the
nominal IMU noise with a ribbon of one standard error, on logarithmic axes
that share their limits. The oracle knows the wind and the fault.

    python experiment-ablation/plot_by_fault.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "figures")]

import numpy as np  # noqa: E402

import ablation as A  # noqa: E402
import paper_style as S  # noqa: E402
import plot as P  # noqa: E402
import plot_total as T  # noqa: E402
import run as R  # noqa: E402

ORACLE = A.REFERENCE
ORACLE_STYLE = dict(color=S.INK, ls="--", lw=1.0, zorder=4)
ORACLE_LABEL = "oracle (known wind and fault)"
BINS = {"small": (0.0, 1.0, r"$|\chi| < 1$ rad"), "medium": (1.0, 2.0, r"$1 \leq |\chi| < 2$ rad"),
        "large": (2.0, np.inf, r"$|\chi| \geq 2$ rad")}
YLIM = (200, 3e4)


def faults() -> np.ndarray:
    episodes = {r["seed"]: r["psi"] for r in A.load(A.DATA / "episodes.pkl")
                if r["label"] == "CBFE" and r["imu_noise"] == R.NOMINAL}
    return np.abs([episodes[seed] for seed in R.SEEDS])


def legend(path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], **P.line(key)) for key in P.ARMS] + [Line2D([], [], **ORACLE_STYLE)]
    names = [P.LABELS[key] for key in P.ARMS] + [ORACLE_LABEL]
    fig = plt.figure(figsize=(S.PMLR_TEXT_WIDTH, 0.17))
    fig.legend(handles, names, loc="center", ncol=len(names), frameon=False, fontsize=6.5, handlelength=2.4,
               columnspacing=1.6, handletextpad=0.4)
    S.save(fig, path)


def panel(total: dict, chi: np.ndarray, name: str, path: Path) -> None:
    low, high, text = BINS[name]
    select = (chi >= low) & (chi < high)
    bench = A.Benchmark()
    time = bench.dt * np.arange(1, bench.T + 1)
    fig, ax = P.panel(P.WIDTH)
    oracle = T.stack(total, ORACLE, R.NOMINAL)[select]
    mean, se = oracle.mean(axis=0), oracle.std(axis=0, ddof=1) / np.sqrt(len(oracle))
    ax.fill_between(time, mean - se, mean + se, color=S.INK, alpha=0.10, lw=0, zorder=1)
    ax.plot(time, mean, **ORACLE_STYLE)
    for label in P.ARMS:
        T.ribbon(ax, time, T.stack(total, label, R.NOMINAL)[select], label, markevery=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(*YLIM)
    P.ticks(ax.xaxis, [0.5, 1, 2, 5, 10])
    P.ticks(ax.yaxis, [1000, 10000], scale=1000)
    ax.text(0.97, 0.95, f"{text}\n{select.sum()} flights", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color=S.INK_SECONDARY)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(T.YLABEL, labelpad=2)
    S.save(fig, path)


def report(total: dict, chi: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    for name, (low, high, _) in BINS.items():
        select = (chi >= low) & (chi < high)
        flights = {label: T.stack(total, label, R.NOMINAL)[select].mean(axis=1) for label in P.ARMS + [ORACLE]}
        print(f"{name:6s} ({select.sum()} flights): " + ", ".join(f"{P.LABELS.get(l, 'oracle')} {v.mean():.0f}"
                                                              for l, v in flights.items()))
        for label in ("no constraint", "certainty equivalence"):
            d = flights[label] - flights["CBFE"]
            m = d[rng.integers(0, len(d), (10000, len(d)))].mean(axis=1)
            print(f"    {P.LABELS[label]} - CBFE {d.mean():+.0f} [{np.quantile(m, .025):+.0f}, "
                  f"{np.quantile(m, .975):+.0f}], CBFE lower in {100 * np.mean(d > 0):.0f}%")


def main():
    S.apply()
    total, chi = T.totals(), faults()
    legend(P.FIGURES / "fig_ablation_fault_legend.pdf")
    for name in BINS:
        panel(total, chi, name, P.FIGURES / f"fig_ablation_fault_{name}.pdf")
    report(total, chi)


if __name__ == "__main__":
    main()
