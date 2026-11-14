"""Candidate panels for the quadcopter figure with the free energy of the
filter on the y-axis, from the outputs of ``free_energy.py``:

    figures/fig_ablation_fe_time.pdf       free energy per step over the flight at the
                                           nominal IMU noise, minus that of CBFE on the
                                           same seed
    figures/fig_ablation_fe_noise.pdf      free energy per step against the IMU noise,
                                           minus that of CBFE on the same seed
    figures/fig_ablation_fe_wallclock.pdf  free energy per step of every flight against
                                           its wall-clock time at the nominal IMU noise

The free energy of step k is -ln N(y_k; y_hat_k, S_k) of the filter's one-step
prediction. The panels use the styles of ``plot.py``; ribbons show one
standard error of the paired difference over the flights. The oracle is
left out, because its filter fixes the parameters at the truth while the
simulator also rotates the rotor plane by the heading fault, which the model
omits, so its predictions are overconfident.

    python experiment-quadcopter/plot_free_energy.py
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

REFERENCE = "CBFE"


def traces(all_traces: list, label: str, noise: float) -> dict:
    return {t["seed"]: t["F"] for t in all_traces if t["label"] == label and t["imu_noise"] == noise}


def paired(all_traces: list, label: str, noise: float) -> np.ndarray:
    """Free energy per step minus that of CBFE on the same seed, (flights, T)."""
    ref, own = traces(all_traces, REFERENCE, noise), traces(all_traces, label, noise)
    return np.array([own[seed] - ref[seed] for seed in sorted(ref)])


def time_course(all_traces: list, path: Path) -> None:
    bench = A.Benchmark()
    fig, ax = P.panel(P.WIDTH)
    P.zero(ax)
    time = bench.dt * np.arange(1, bench.T + 1)
    for label in P.ARMS:
        d = paired(all_traces, label, R.NOMINAL)
        mean, se = d.mean(axis=0), d.std(axis=0, ddof=1) / np.sqrt(len(d))
        style = P.line(label, markevery=2)
        ax.fill_between(time, mean - se, mean + se, color=style["color"], alpha=0.18, lw=0, zorder=2)
        ax.plot(time, mean, **style)
    ax.set_xscale("log")
    ax.set_xlim(time[0], time[-1])
    P.ticks(ax.xaxis, [0.5, 1, 2, 5, 10])
    P.ticks(ax.yaxis, [-2, -1, 0, 1])
    ax.set_ylim(-2.2, 1.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$F - F_{\mathrm{CBFE}}$ [nat]", labelpad=2)
    S.save(fig, path)


def noise_sweep(all_traces: list, path: Path) -> None:
    fig, ax = P.panel(P.WIDTH)
    P.zero(ax)
    for label in P.ARMS:
        rows = [paired(all_traces, label, noise).mean(axis=1) for noise in R.NOISES]
        mean = np.array([r.mean() for r in rows])
        se = np.array([r.std(ddof=1) / np.sqrt(len(r)) for r in rows])
        style = P.line(label)
        ax.fill_between(R.NOISES, mean - se, mean + se, color=style["color"], alpha=0.18, lw=0, zorder=2)
        ax.plot(R.NOISES, mean, **style)
    ax.set_xscale("log")
    ax.set_xlim(min(R.NOISES) / 1.15, max(R.NOISES) * 1.15)
    P.ticks(ax.xaxis, [0.2, 0.5, 1, 2, 3])
    P.ticks(ax.yaxis, [-0.5, 0, 0.5, 1])
    ax.set_xlabel(r"IMU noise std $\sigma_f$ [m/s$^2$]")
    ax.set_ylabel(r"$F - F_{\mathrm{CBFE}}$ [nat]", labelpad=2)
    S.save(fig, path)


def wallclock(table: pd.DataFrame, path: Path) -> None:
    fig, ax = P.panel(P.WIDTH)
    for label in ["no constraint", "certainty equivalence", "CBFE"]:
        sel = table[table.label == label]
        P.scatter(ax, sel.wall, sel.F, label)
    ax.set_xscale("log")
    P.ticks(ax.xaxis, [0.3, 1, 3, 10])
    P.ticks(ax.yaxis, [6, 8, 10, 12, 14])
    ax.set_xlabel("wall clock [s]")
    ax.set_ylabel("free energy [nat]", labelpad=1)
    S.save(fig, path)


def main():
    S.apply()
    with open(A.DATA / "free_energy.pkl", "rb") as f:
        all_traces = pickle.load(f)
    runs = pd.read_csv(A.RESULTS / "runs.csv")
    energy = pd.read_csv(A.RESULTS / "free_energy.csv")
    table = energy.merge(runs[["label", "imu_noise", "seed", "wall", "timed"]], on=["label", "imu_noise", "seed"])
    table = table[(table.imu_noise == R.NOMINAL) & table.timed]
    time_course(all_traces, P.FIGURES / "fig_ablation_fe_time.pdf")
    noise_sweep(all_traces, P.FIGURES / "fig_ablation_fe_noise.pdf")
    wallclock(table, P.FIGURES / "fig_ablation_fe_wallclock.pdf")
    print(table.groupby("label").F.describe()[["min", "50%", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
