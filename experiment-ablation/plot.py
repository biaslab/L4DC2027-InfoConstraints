"""Panels of the quadcopter figure of main.tex, from the outputs of ``run.py``.
Every panel shows the learning cost, i.e. the goal cost on the true position
minus that of the oracle that knows the wind, the drag and the faults on the
same seed:

    figures/fig_ablation_legend.pdf     one legend strip over the text width
    figures/fig_ablation_time.pdf       learning cost per step over the flight at
                                        the nominal IMU noise (log-log)
    figures/fig_ablation_noise.pdf      learning cost per step against the IMU
                                        noise (log-log)
    figures/fig_ablation_wallclock.pdf  learning cost against the wall-clock time
                                        of every flight at the nominal IMU noise
    figures/fig_ablation_activity.pdf   wall-clock time per step of every CBFE flight
                                        against its share of steps with an active
                                        constraint, with a least-squares line

Lines carry a ribbon of one standard error of the paired difference to the
oracle over the flights. The colour encodes the information constraint (blue
with, amber without), and the line style and the marker fill encode the
trace term (solid and filled with, dashed and open without); MPC is the
certainty-equivalent controller without both, in the green of paper_style. The panels are
drawn at their printed size, 0.24 of the PMLR text width each. Panel (a) has
logarithmic axes, so its curves leave the plot where the learning cost
approaches zero at the end of the flight.

    python experiment-ablation/plot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "figures")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ablation as A  # noqa: E402
import paper_style as S  # noqa: E402
import run as R  # noqa: E402

FIGURES = HERE / "figures"
BLUE, AMBER = S.AGENTS["kkt"]["color"], S.AGENTS["gamma0"]["color"]
DASH = (0, (3, 1.5))
GREEN = S.AGENTS["ce"]["color"]
# no-info-CBFE is magenta, because the amber of the cautious controller lies
# within 14 OKLab units of the orange of the dual MPC under normal vision
MAGENTA, ORANGE, TEAL = "#b5179e", S.AGENTS["dual"]["color"], S.AGENTS["mpcx"]["color"]
STYLES = {
    "CBFE": dict(color=BLUE, ls="-", marker="o", mfc=BLUE, lw=1.8, zorder=6),
    "no constraint": dict(color=MAGENTA, ls="-", marker="v", mfc=MAGENTA, lw=1.2, zorder=5),
    "certainty equivalence": dict(color=GREEN, ls="-", marker="P", mfc=GREEN, lw=1.2, zorder=5),
    "dual MPC": dict(color=ORANGE, ls="-", marker="s", mfc=ORANGE, lw=1.2, zorder=5),
    "MPC-X": dict(color=TEAL, ls="-", marker="D", mfc=TEAL, lw=1.2, zorder=5),
}
LABELS = {"CBFE": "info-CBFE", "no constraint": "no-info-CBFE", "certainty equivalence": "MPC",
          "dual MPC": "dual MPC", "MPC-X": "MPC-X"}
ARMS = ["CBFE", "no constraint", "certainty equivalence"]
ORACLE = A.REFERENCE
WIDTH, HEIGHT = 0.24 * S.PMLR_TEXT_WIDTH, 1.3                                    # [in]
MARGINS = dict(left=0.42, right=0.06, bottom=0.33, top=0.06)                     # [in]


def panel(width: float):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(width, HEIGHT))
    fig.subplots_adjust(left=MARGINS["left"] / width, right=1 - MARGINS["right"] / width,
                        bottom=MARGINS["bottom"] / HEIGHT, top=1 - MARGINS["top"] / HEIGHT)
    return fig, fig.add_subplot()


def line(label: str, markers: bool = True, **overrides) -> dict:
    style = STYLES[label]
    kwargs = dict(color=style["color"], ls=style["ls"], lw=style["lw"], zorder=style["zorder"])
    if markers:
        kwargs.update(marker=style["marker"], ms=3.4, mfc=style["mfc"], mec=style["color"], mew=0.7)
    kwargs.update(overrides)
    return kwargs


def ticks(axis, values, scale: float = 1.0) -> None:
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    axis.set_major_locator(FixedLocator(values))
    axis.set_major_formatter(lambda value, _: f"{value / scale:g}")
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


def zero(ax) -> None:
    ax.axhline(0.0, color=S.MUTED, lw=0.6, zorder=1)


def legend(path: Path) -> None:
    """One legend strip for the whole figure."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(S.PMLR_TEXT_WIDTH, 0.17))
    fig.legend([Line2D([], [], **line(key)) for key in ARMS], [LABELS[key] for key in ARMS], loc="center",
               ncol=len(ARMS), frameon=False, fontsize=6.5, handlelength=2.4, columnspacing=1.6, handletextpad=0.4)
    S.save(fig, path)


def learning(episodes: list, label: str, noise: float) -> np.ndarray:
    """Learning cost per step of every flight, paired with the oracle by seed."""
    oracle = {r["seed"]: r["true_cost"] for r in episodes if r["label"] == ORACLE and r["imu_noise"] == noise}
    return np.array([r["true_cost"] - oracle[r["seed"]] for r in episodes
                     if r["label"] == label and r["imu_noise"] == noise])


def time_course(episodes: list, path: Path) -> None:
    """(a) Learning cost per step over the flight."""
    bench = A.Benchmark()
    fig, ax = panel(WIDTH)
    time = bench.dt * np.arange(1, bench.T + 1)
    for label in ARMS:
        costs = learning(episodes, label, R.NOMINAL)
        mean, se = costs.mean(axis=0), costs.std(axis=0, ddof=1) / np.sqrt(len(costs))
        style = line(label, markevery=2)
        ax.fill_between(time, np.maximum(mean - se, 1.0), mean + se, color=style["color"], alpha=0.18, lw=0,
                        zorder=2)
        ax.plot(time, np.where(mean > 0.0, mean, np.nan), **style)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(100, 3e4)
    ticks(ax.xaxis, [0.5, 1, 2, 5, 10])
    ticks(ax.yaxis, [100, 1000, 10000], scale=1000)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"learning cost ($10^3$)", labelpad=2)
    S.save(fig, path)


def noise_sweep(runs: pd.DataFrame, path: Path) -> None:
    """(b) Learning cost per step against the IMU noise, with ribbons."""
    fig, ax = panel(WIDTH)
    for label in ARMS:
        sel = runs[runs.label == label].groupby("imu_noise").regret
        mean, se = sel.mean(), sel.std(ddof=1) / np.sqrt(sel.size())
        style = line(label)
        ax.fill_between(mean.index, mean - se, mean + se, color=style["color"], alpha=0.18, lw=0, zorder=2)
        ax.plot(mean.index, mean.values, **style)
    ax.set_xscale("log")
    ax.set_xlim(min(R.NOISES) / 1.15, max(R.NOISES) * 1.15)
    ticks(ax.xaxis, [0.2, 0.5, 1, 2, 3])
    ax.set_yscale("log")
    ticks(ax.yaxis, [1000, 2000, 3000, 5000], scale=1000)
    ax.set_xlabel(r"IMU noise std $\sigma_f$ [m/s$^2$]")
    ax.set_ylabel(r"learning cost ($10^3$)", labelpad=2)
    S.save(fig, path)


def scatter(ax, x, y, label: str) -> None:
    style = STYLES[label]
    ax.scatter(x, y, s=6, marker=style["marker"], facecolors=style["mfc"], edgecolors=style["color"], linewidths=0.5,
               alpha=0.8, zorder=style["zorder"])


def wallclock(runs: pd.DataFrame, path: Path) -> None:
    """(c) Learning cost against the wall-clock time of every flight."""
    fig, ax = panel(WIDTH)
    zero(ax)
    nominal = runs[(runs.imu_noise == R.NOMINAL) & runs.timed]
    for label in ["no constraint", "certainty equivalence", "CBFE"]:
        sel = nominal[nominal.label == label]
        scatter(ax, sel.wall, sel.regret, label)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1000, linscale=0.6)
    ticks(ax.xaxis, [0.3, 1, 3, 10])
    ticks(ax.yaxis, [-10000, 0, 1000, 10000], scale=1000)
    ax.set_xlabel("wall clock [s]")
    ax.set_ylabel(r"learning cost ($10^3$)", labelpad=1)
    S.save(fig, path)


def activity(runs: pd.DataFrame, path: Path) -> None:
    """(d) Wall-clock time per step of every CBFE flight against its share of
    active steps, with the least-squares line of the time on the share."""
    fig, ax = panel(WIDTH)
    steps = A.Benchmark().T
    nominal = runs[(runs.imu_noise == R.NOMINAL) & runs.timed & (runs.label == "CBFE")]
    share = 100.0 * nominal.active_steps.to_numpy() / steps
    wall = 1000.0 * nominal.wall.to_numpy() / steps
    scatter(ax, share, wall, "CBFE")
    slope, intercept = np.polyfit(share, wall, 1)
    grid = np.linspace(0.0, share.max() * 1.05, 100)
    ax.plot(grid, intercept + slope * grid, color=S.INK, lw=0.9, zorder=8)
    r2 = 1.0 - np.sum((wall - intercept - slope * share)**2) / np.sum((wall - wall.mean())**2)
    print(f"activity fit: wall per step = {intercept:.0f} ms + {slope:.2f} ms per % active (R^2 = {r2:.2f})")
    ax.set_ylim(100, 500)
    ticks(ax.yaxis, [100, 200, 300, 400, 500])
    ticks(ax.xaxis, [0, 20, 40, 60])
    ax.set_xlim(0, 66)
    ax.set_xlabel("active steps (%)")
    ax.set_ylabel("wall clock [ms/step]", labelpad=1)
    S.save(fig, path)


def main():
    S.apply()
    FIGURES.mkdir(parents=True, exist_ok=True)
    episodes = A.load(A.DATA / "episodes.pkl")
    runs = pd.read_csv(A.RESULTS / "runs.csv")
    legend(FIGURES / "fig_ablation_legend.pdf")
    time_course(episodes, FIGURES / "fig_ablation_time.pdf")
    noise_sweep(runs, FIGURES / "fig_ablation_noise.pdf")
    wallclock(runs, FIGURES / "fig_ablation_wallclock.pdf")
    activity(runs, FIGURES / "fig_ablation_activity.pdf")
    print("\n".join(str(p) for p in sorted(FIGURES.glob("fig_ablation_*.pdf"))))


if __name__ == "__main__":
    main()
