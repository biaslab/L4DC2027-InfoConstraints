"""Candidate panels for the quadcopter figure with the goal cost plus the free
energy of the filter on the y-axis, in absolute values, from the outputs of
``run.py`` and ``free_energy.py``:

    figures/fig_ablation_total_time.pdf       goal cost + free energy per step over the
                                              flight at the nominal IMU noise (log-log)
    figures/fig_ablation_total_noise.pdf      goal cost + free energy per step against
                                              the IMU noise
    figures/fig_ablation_total_wallclock.pdf  goal cost + free energy per step of every
                                              flight against its wall-clock time
    figures/fig_ablation_total_box.pdf        goal cost + free energy per step of every
                                              flight at the nominal IMU noise in the
                                              first, middle and last third of the flight

The goal cost of step k is |p_k - p_*|^2 / (2 S_*) on the true position, as in
``plot.py``, and the free energy F_k = -ln N(y_k; y_hat_k, S_k) is that of the
filter's update. Ribbons show one standard error over the flights. The boxes
span the quartiles, the whiskers reach 1.5 times the interquartile range, and
the white dots mark the means.

    python experiment-ablation/plot_total.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "figures")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ablation as A  # noqa: E402
import paper_style as S  # noqa: E402
import plot as P  # noqa: E402
import run as R  # noqa: E402

THIRDS = {"start": slice(0, 7), "middle": slice(7, 14), "end": slice(14, 20)}     # steps 1-7, 8-14, 15-20
YLABEL = r"goal cost + $F$ ($10^3$)"


def totals() -> dict:
    """Goal cost on the true position plus free energy, per step, by (label, noise, seed)."""
    with open(A.DATA / "free_energy.pkl", "rb") as f:
        energy = {(t["label"], t["imu_noise"], t["seed"]): t["F"] for t in pickle.load(f)}
    return {(r["label"], r["imu_noise"], r["seed"]): r["true_cost"] + energy[(r["label"], r["imu_noise"], r["seed"])]
            for r in A.load(A.DATA / "episodes.pkl")}


def stack(total: dict, label: str, noise: float) -> np.ndarray:
    return np.array([total[(label, noise, seed)] for seed in R.SEEDS])


def ribbon(ax, x, values: np.ndarray, label: str, **overrides) -> None:
    mean, se = values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))
    style = P.line(label, **overrides)
    ax.fill_between(x, mean - se, mean + se, color=style["color"], alpha=0.18, lw=0, zorder=2)
    ax.plot(x, mean, **style)


def time_course(total: dict, path: Path) -> None:
    bench = A.Benchmark()
    fig, ax = P.panel(P.WIDTH)
    time = bench.dt * np.arange(1, bench.T + 1)
    for label in P.ARMS:
        ribbon(ax, time, stack(total, label, R.NOMINAL), label, markevery=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(500, 2e4)
    P.ticks(ax.xaxis, [0.5, 1, 2, 5, 10])
    P.ticks(ax.yaxis, [1000, 10000], scale=1000)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(YLABEL, labelpad=2)
    S.save(fig, path)


def noise_sweep(total: dict, path: Path) -> None:
    fig, ax = P.panel(P.WIDTH)
    for label in P.ARMS:
        ribbon(ax, R.NOISES, np.array([stack(total, label, noise).mean(axis=1) for noise in R.NOISES]).T, label)
    ax.set_xscale("log")
    ax.set_xlim(min(R.NOISES) / 1.15, max(R.NOISES) * 1.15)
    ax.set_ylim(2400, 7200)
    P.ticks(ax.xaxis, [0.2, 0.5, 1, 2, 3])
    P.ticks(ax.yaxis, [3000, 4000, 5000, 6000, 7000], scale=1000)
    ax.set_xlabel(r"IMU noise std $\sigma_f$ [m/s$^2$]")
    ax.set_ylabel(YLABEL, labelpad=2)
    S.save(fig, path)


def wallclock(total: dict, path: Path) -> None:
    runs = pd.read_csv(A.RESULTS / "runs.csv")
    timed = runs[(runs.imu_noise == R.NOMINAL) & runs.timed]
    fig, ax = P.panel(P.WIDTH)
    for label in ["no constraint", "certainty equivalence", "CBFE"]:
        sel = timed[timed.label == label]
        P.scatter(ax, sel.wall, [total[(label, R.NOMINAL, seed)].mean() for seed in sel.seed], label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    P.ticks(ax.xaxis, [0.3, 1, 3, 10])
    P.ticks(ax.yaxis, [1000, 10000], scale=1000)
    ax.set_xlabel("wall clock [s]")
    ax.set_ylabel(YLABEL, labelpad=1)
    S.save(fig, path)


def boxes(total: dict, path: Path) -> None:
    """Grouped boxplot of the mean per step of every flight in each third."""
    fig, ax = P.panel(P.WIDTH)
    width, gap = 0.24, 0.05
    for i, window in enumerate(THIRDS.values()):
        for j, label in enumerate(P.ARMS):
            values = stack(total, label, R.NOMINAL)[:, window].mean(axis=1)
            colour = P.STYLES[label]["color"]
            position = i + (j - 1) * (width + gap)
            ax.boxplot(values, positions=[position], widths=width, whis=1.5, patch_artist=True,
                       boxprops=dict(facecolor=colour, edgecolor=colour, alpha=0.35, lw=0.8),
                       medianprops=dict(color=colour, lw=1.4), whiskerprops=dict(color=colour, lw=0.8),
                       capprops=dict(color=colour, lw=0.8),
                       flierprops=dict(marker=P.STYLES[label]["marker"], ms=2.2, mfc=colour, mec=colour, alpha=0.6,
                                       ls="none"))
            ax.plot(position, values.mean(), marker="o", ms=2.6, mfc="white", mec=S.INK, mew=0.6, zorder=8)
    ax.set_yscale("log")
    ax.set_xlim(-0.55, len(THIRDS) - 0.45)
    ax.set_xticks(range(len(THIRDS)), list(THIRDS))
    ax.tick_params(axis="x", length=0)
    P.ticks(ax.yaxis, [100, 1000, 10000, 100000], scale=1000)
    ax.set_xlabel("third of the flight")
    ax.set_ylabel(YLABEL, labelpad=1)
    S.save(fig, path)


def report(total: dict) -> None:
    rng = np.random.default_rng(0)
    for name, window in THIRDS.items():
        flights = {label: stack(total, label, R.NOMINAL)[:, window].mean(axis=1) for label in P.ARMS}
        line = f"{name:7s}" + "".join(f" {P.LABELS[l]}: mean {v.mean():6.0f} median {np.median(v):6.0f} |"
                                      for l, v in flights.items())
        print(line)
        for label in ("no constraint", "certainty equivalence"):
            d = flights[label] - flights["CBFE"]
            m = d[rng.integers(0, len(d), (10000, len(d)))].mean(axis=1)
            print(f"         {P.LABELS[label]} - CBFE {d.mean():+6.0f} [{np.quantile(m, .025):+6.0f}, "
                  f"{np.quantile(m, .975):+6.0f}], CBFE lower in {100 * np.mean(d > 0):.0f}% of flights")


def main():
    S.apply()
    total = totals()
    time_course(total, P.FIGURES / "fig_ablation_total_time.pdf")
    noise_sweep(total, P.FIGURES / "fig_ablation_total_noise.pdf")
    wallclock(total, P.FIGURES / "fig_ablation_total_wallclock.pdf")
    boxes(total, P.FIGURES / "fig_ablation_total_box.pdf")
    report(total)


if __name__ == "__main__":
    main()
