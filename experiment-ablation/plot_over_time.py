"""Panels (a)-(c) of the quadcopter figure of main.tex and candidate views of
the performance over the flight, which accumulate the differences between the
controllers instead of averaging them per step, in absolute values. The
performance is G of eq. (metric) of main.tex: the average over the T steps of
half the precision-weighted prediction error of the filter's one-step
prediction of y_k = (position fix, IMU sample) plus the goal cost on the true
position. Panel (a) accumulates the terms of G up to time t, so it ends at G,
and the wall-clock times are averages per step.

    figures/fig_ablation_cumulative.pdf  cumulative goal cost + free energy up to time t,
                                         mean over the flights at the nominal IMU noise
                                         with a ribbon of one standard error
    figures/fig_ablation_cumulative_noise.pdf      cumulative cost of the flight against the
                                                   IMU noise, on the y-axis of the panel above
    figures/fig_ablation_cumulative_wallclock.pdf  cumulative cost of every flight against
                                                   its wall-clock time at the nominal IMU noise
    figures/fig_ablation_arrival.pdf     share of flights that have come within 1 m of the
                                         target by time t
    figures/fig_ablation_legend_oracle.pdf   legend strip with the oracle, for the figure
                                         of main.tex whose panel (a) is the cumulative cost

The goal cost is that of ``plot.py`` on the true position, and the prediction
error e_k' S_k^-1 e_k comes from ``free_energy.py``. The noise panel marks the
nominal IMU noise of the other panels with a dashed vertical line. The oracle
knows the wind and the fault.

    python experiment-ablation/plot_over_time.py
    python experiment-ablation/plot_over_time.py --fresh   # the panels of main.tex

With ``--fresh``, the panels show the fresh seeds 4000-4099 without the seeds
on which the oracle fails (``paper_data.py``), and their file names end in
"_fresh".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "figures")]

import pickle  # noqa: E402

import numpy as np  # noqa: E402

import ablation as A  # noqa: E402
import paper_data as D  # noqa: E402
import paper_style as S  # noqa: E402
import plot as P  # noqa: E402
import plot_by_fault as B  # noqa: E402
import plot_total as T  # noqa: E402
import run as R  # noqa: E402
import run_baselines as RB  # noqa: E402

PAPER_ARMS = ["CBFE", "no constraint", "dual MPC", "MPC-X"]     # info-CBFE, no-info-CBFE and the tuned baselines

RADIUS = 1.0      # [m]
YLIM = (0, 6750)                                          # shared by G over time and over the noise
YTICKS = [0, 2000, 4000, 6000]
STEPS = A.Benchmark().T
YLABEL = r"$G$ ($10^3$)"
FRESH = False     # set by --fresh: the flights of paper_data.flights()


def metric() -> dict:
    """Per-step terms of G, eq. (metric) of main.tex, by (label, noise, seed),
    for the flights of ``run.py`` and the tuned baselines of ``run_baselines.py``,
    which record the prediction error of their filter."""
    if FRESH:
        return D.terms(flights())
    with open(A.DATA / "free_energy.pkl", "rb") as f:
        error = {(t["label"], t["imu_noise"], t["seed"]): 0.5 * t["error"] for t in pickle.load(f)}
    out = {(r["label"], r["imu_noise"], r["seed"]): r["true_cost"] + error[(r["label"], r["imu_noise"], r["seed"])]
           for r in A.load(A.DATA / "episodes.pkl")}
    out.update({(r["label"], r["imu_noise"], r["seed"]): r["true_cost"] + 0.5 * r["error"]
                for r in A.load(A.DATA / "baselines_test.pkl")})
    return out


def flights() -> list:
    """Every flight of ``run.py`` and of the tuned baselines."""
    if FRESH:
        return D.flights(fresh=True)
    return A.load(A.DATA / "episodes.pkl") + A.load(A.DATA / "baselines_test.pkl")


def arms():
    return PAPER_ARMS + [B.ORACLE]


def power(value: float) -> str:
    """LaTeX of value as a multiple of a power of ten, such as 3 \\cdot 10^{3}."""
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10.0**exponent
    return rf"10^{{{exponent}}}" if np.isclose(mantissa, 1.0) else rf"{mantissa:g} \cdot 10^{{{exponent}}}"


def labels() -> dict:
    t = RB.tuned()
    return {**P.LABELS, "dual MPC": rf"dual MPC, tuned $\rho = {power(t['rho'])}$",
            "MPC-X": rf"MPC-X, tuned $\kappa = {t['kappa']:g}$", B.ORACLE: "oracle"}


def legend(path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    names = labels()
    handles = [Line2D([], [], **P.line(key)) for key in PAPER_ARMS] + [Line2D([], [], **B.ORACLE_STYLE)]
    fig = plt.figure(figsize=(S.PMLR_TEXT_WIDTH, 0.17))
    fig.legend(handles, [names[key] for key in arms()], loc="center", ncol=len(handles), frameon=False, fontsize=6.5,
               handlelength=2.2, columnspacing=1.3, handletextpad=0.4)
    S.save(fig, path)


def draw(ax, x, values: np.ndarray, label: str, markevery=2) -> None:
    mean, se = values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))
    style = dict(B.ORACLE_STYLE) if label == B.ORACLE else P.line(label, markevery=markevery)
    ax.fill_between(x, mean - se, mean + se, color=style["color"], alpha=0.10 if label == B.ORACLE else 0.18, lw=0,
                    zorder=1)
    ax.plot(x, mean, **style)


def cumulative(total: dict, path: Path) -> None:
    bench = A.Benchmark()
    time = bench.dt * np.arange(0, bench.T + 1)
    fig, ax = P.panel(P.WIDTH)
    for label in arms():
        values = np.cumsum(T.stack(total, label, R.NOMINAL), axis=1) / STEPS
        draw(ax, time, np.hstack([np.zeros((len(values), 1)), values]), label)
    ax.set_xlim(0, time[-1])
    ax.set_ylim(*YLIM)
    P.ticks(ax.xaxis, [0, 2, 4, 6, 8, 10])
    P.ticks(ax.yaxis, YTICKS, scale=1000)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$G$ up to $t$ ($10^3$)", labelpad=2)
    S.save(fig, path)


def noise_sweep(total: dict, path: Path) -> None:
    """Cumulative cost of the whole flight against the IMU noise."""
    fig, ax = P.panel(P.WIDTH)
    ax.axvline(R.NOMINAL, color=S.INK, ls=(0, (4, 2)), lw=0.8, zorder=1)
    for label in arms():
        ends = np.array([T.stack(total, label, noise).mean(axis=1) for noise in R.NOISES]).T    # (flights, noises)
        draw(ax, R.NOISES, ends, label, markevery=1)
    ax.set_xscale("log")
    ax.set_xlim(min(R.NOISES) / 1.15, max(R.NOISES) * 1.15)
    ax.set_ylim(*YLIM)
    P.ticks(ax.xaxis, [0.2, 0.5, 1, 2, 3])
    P.ticks(ax.yaxis, YTICKS, scale=1000)
    ax.set_xlabel(r"noise std $\sigma_f$ [m/s$^2$]")
    ax.set_ylabel(YLABEL, labelpad=2)
    S.save(fig, path)


def wallclock(total: dict, path: Path) -> None:
    """G of every flight against its wall-clock time per step. The flights at
    the nominal noise ran on the four fast cores only, so their times compare."""
    nominal = [r for r in flights() if r["imu_noise"] == R.NOMINAL]
    fig, ax = P.panel(P.WIDTH)
    for label in ["no constraint", "MPC-X", "dual MPC", "CBFE"]:
        sel = [r for r in nominal if r["label"] == label]
        P.scatter(ax, [1000.0 * r["wall"] / STEPS for r in sel], [total[(label, R.NOMINAL, r["seed"])].mean() for r in sel],
                  label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 700)
    ax.set_ylim(6e2, 6e4)
    P.ticks(ax.xaxis, [10, 100])
    P.ticks(ax.yaxis, [1e3, 1e4])
    ax.yaxis.set_major_formatter(lambda value, _: f"$10^{{{round(np.log10(value))}}}$")
    ax.set_xlabel("wall clock [ms/step]")
    ax.set_ylabel(r"$G$", labelpad=1)
    S.save(fig, path)


def arrival(path: Path) -> None:
    bench = A.Benchmark()
    time = bench.dt * np.arange(0, bench.T + 1)
    episodes = {(r["label"], r["seed"]): r for r in flights() if r["imu_noise"] == R.NOMINAL}
    fig, ax = P.panel(P.WIDTH)
    for label in arms():
        distance = np.array([np.linalg.norm(episodes[(label, seed)]["p"] - episodes[(label, seed)]["target"], axis=1)
                             for seed in R.SEEDS])
        arrived = np.maximum.accumulate(distance < RADIUS, axis=1).astype(float)
        draw(ax, time, 100.0 * arrived, label)
        print(f"{P.LABELS.get(label, 'oracle'):24s} within {RADIUS} m by t: " +
              " ".join(f"{t:g}s {100 * arrived[:, int(t / bench.dt)].mean():.0f}%" for t in (2, 3, 4, 5, 7, 10)))
    ax.set_xlim(0, time[-1])
    ax.set_ylim(0, 100)
    P.ticks(ax.xaxis, [0, 2, 4, 6, 8, 10])
    P.ticks(ax.yaxis, [0, 25, 50, 75, 100])
    ax.set_xlabel("time [s]")
    ax.set_ylabel(f"within {RADIUS:g} m (%)", labelpad=2)
    S.save(fig, path)


def main():
    global FRESH
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fresh", action="store_true", help="the fresh seeds without the seeds on which the oracle fails")
    FRESH = parser.parse_args().fresh
    tag = "_fresh" if FRESH else ""
    if FRESH:
        R.SEEDS = D.seeds(fresh=True)        # plot_total.stack and arrival() read the seeds from run.py
    S.apply()
    total = metric()
    legend(P.FIGURES / "fig_ablation_legend_oracle.pdf")
    cumulative(total, P.FIGURES / f"fig_ablation_cumulative{tag}.pdf")
    noise_sweep(total, P.FIGURES / f"fig_ablation_cumulative_noise{tag}.pdf")
    wallclock(total, P.FIGURES / f"fig_ablation_cumulative_wallclock{tag}.pdf")
    arrival(P.FIGURES / f"fig_ablation_arrival{tag}.pdf")
    for label in arms():
        final = T.stack(total, label, R.NOMINAL).mean(axis=1)
        print(f"{P.LABELS.get(label, 'oracle'):24s} G: {final.mean():.0f} +- {final.std(ddof=1) / np.sqrt(len(final)):.0f}")


if __name__ == "__main__":
    main()
