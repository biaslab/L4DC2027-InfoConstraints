"""Panels of the light-dark figure of main.tex, from the outputs of
``run_baselines.py`` and ``performance.py``, in the layout of the quadcopter
figure. Every panel shows the performance G of eq. (metric) of main.tex, the
mean over the T steps of half the precision-weighted prediction error of the
filter plus the goal cost on the true state:

    figures/fig_lightdark_legend.pdf     one legend strip over the text width
    figures/fig_lightdark_time.pdf       G accumulated up to step t at the nominal
                                         light noise, so that it ends at G
    figures/fig_lightdark_noise.pdf      G against the noise in the light, with a
                                         dashed line at the nominal light noise
    figures/fig_lightdark_wallclock.pdf  G of every episode against its wall-clock
                                         time per step at the nominal light noise

Lines carry a ribbon of one standard error over the episodes. The panels are
drawn at their printed size, 0.32 of the PMLR text width each, with the
colours of paper_style: CBFE blue, the dual MPC with the tuned weight orange
and MPC-X green. Linear axes carry evenly spaced ticks and logarithmic axes
carry decades.

    python experiment-lightdark/plot_baselines.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "figures")]

import numpy as np  # noqa: E402

import lightdark as L  # noqa: E402
import paper_style as S  # noqa: E402
import run_baselines as R  # noqa: E402

FIGURES = HERE / "figures"
STYLES = {
    "CBFE": dict(color=S.AGENTS["kkt"]["color"], ls="-", marker="o", mfc=S.AGENTS["kkt"]["color"], lw=1.8, zorder=6),
    "dual MPC, tuned": dict(color=S.AGENTS["dual"]["color"], ls="-", marker="s", mfc=S.AGENTS["dual"]["color"],
                            lw=1.2, zorder=5),
    "MPC-X": dict(color=S.AGENTS["mpcx"]["color"], ls="-", marker="D", mfc=S.AGENTS["mpcx"]["color"], lw=1.2,
                  zorder=5),
}
ARMS = list(STYLES)
WIDTH, HEIGHT = 0.32 * S.PMLR_TEXT_WIDTH, 1.3
MARGINS = dict(left=0.42, right=0.12, bottom=0.36, top=0.06)


def labels() -> dict:
    t = R.tuned()
    return {"CBFE": "info-CBFE", "dual MPC, tuned": rf"dual MPC, tuned $\rho = {t['rho']:g}$",
            "MPC-X": rf"MPC-X, tuned $\kappa = {t['gamma']:g}$"}


def panel():
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    fig.subplots_adjust(left=MARGINS["left"] / WIDTH, right=1 - MARGINS["right"] / WIDTH,
                        bottom=MARGINS["bottom"] / HEIGHT, top=1 - MARGINS["top"] / HEIGHT)
    return fig, fig.add_subplot()


def line(label: str, **overrides) -> dict:
    style = STYLES[label]
    kwargs = dict(color=style["color"], ls=style["ls"], lw=style["lw"], zorder=style["zorder"], marker=style["marker"],
                  ms=3.2, mfc=style["mfc"], mec=style["color"], mew=0.7)
    kwargs.update(overrides)
    return kwargs


def ticks(axis, values) -> None:
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    axis.set_major_locator(FixedLocator(values))
    axis.set_major_formatter(lambda value, _: f"{value:g}")
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


def scatter(ax, x, y, label: str) -> None:
    style = STYLES[label]
    ax.scatter(x, y, s=6, marker=style["marker"], facecolors=style["mfc"], edgecolors=style["color"], linewidths=0.5,
               alpha=0.8, zorder=style["zorder"])


def legend(path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    names = labels()
    fig = plt.figure(figsize=(S.PMLR_TEXT_WIDTH, 0.17))
    fig.legend([Line2D([], [], **line(key)) for key in ARMS], [names[key] for key in ARMS], loc="center",
               ncol=len(ARMS), frameon=False, fontsize=6.5, handlelength=2.4, columnspacing=1.6, handletextpad=0.4)
    S.save(fig, path)


def performance() -> list:
    """Per-step terms of G for every episode, from performance.py."""
    with open(L.RESULTS / "baselines_performance.pkl", "rb") as f:
        return pickle.load(f)


def stack(rows: list, file: str, label: str, light: float) -> np.ndarray:
    """g_k of every episode of one controller and light noise, (episodes, T)."""
    chosen = sorted((r for r in rows if r["file"] == file and r["label"] == label and r["light"] == light),
                    key=lambda r: r["seed"])
    return np.array([r["goal"] + r["prediction"] for r in chosen])


def episodes(rows: list, label: str, light: float) -> np.ndarray:
    """g_k at a light noise, from the test runs at the nominal noise and the sweep otherwise."""
    return stack(rows, "baselines_test" if light == R.NOMINAL_LIGHT else "baselines_sweep", label, light)


def ribbon(ax, x, mean: np.ndarray, se: np.ndarray, label: str, **overrides) -> None:
    style = line(label, **overrides)
    ax.fill_between(x, mean - se, mean + se, color=style["color"], alpha=0.18, lw=0, zorder=2)
    ax.plot(x, mean, **style)


def time_course(rows: list, path: Path) -> None:
    """(a) G accumulated up to step t."""
    fig, ax = panel()
    T = R.bench().T
    steps = np.arange(0, T + 1)
    for label in ARMS:
        g = episodes(rows, label, R.NOMINAL_LIGHT)
        cumulative = np.hstack([np.zeros((len(g), 1)), np.cumsum(g, axis=1) / T])
        ribbon(ax, steps, cumulative.mean(axis=0), cumulative.std(axis=0, ddof=1) / np.sqrt(len(g)), label,
               markevery=25)
    ax.set_xlim(0, T)
    ax.set_ylim(0, 3)
    ticks(ax.xaxis, [0, 50, 100, 150, 200])
    ticks(ax.yaxis, [0, 1, 2, 3])
    ax.set_xlabel("step $t$")
    ax.set_ylabel(r"$G$ up to $t$", labelpad=2)
    S.save(fig, path)


def noise_sweep(rows: list, path: Path) -> None:
    """(b) G against the measurement noise in the light."""
    fig, ax = panel()
    ax.axvline(R.NOMINAL_LIGHT, color=S.INK, ls=(0, (4, 2)), lw=0.8, zorder=1)
    for label in ARMS:
        per_level = [episodes(rows, label, light).mean(axis=1) for light in R.LIGHTS]
        mean = np.array([v.mean() for v in per_level])
        se = np.array([v.std(ddof=1) / np.sqrt(len(v)) for v in per_level])
        ribbon(ax, np.array(R.LIGHTS), mean, se, label)
    ax.set_xscale("log")
    ax.set_xlim(min(R.LIGHTS) / 1.2, max(R.LIGHTS) * 1.2)
    ax.set_ylim(0, 12.5)
    ticks(ax.xaxis, [0.01, 0.03, 0.1, 0.3])
    ticks(ax.yaxis, [0, 3, 6, 9, 12])
    ax.set_xlabel(r"light noise std $\sqrt{r_L}$")
    ax.set_ylabel(r"$G$", labelpad=2)
    S.save(fig, path)


def wallclock(rows: list, path: Path) -> None:
    """(c) G of every episode against its wall-clock time per step."""
    fig, ax = panel()
    for label in ["dual MPC, tuned", "MPC-X", "CBFE"]:
        chosen = [r for r in rows if r["file"] == "baselines_test" and r["label"] == label]
        scatter(ax, [1000.0 * r["wall"] / r["steps"] for r in chosen],
                [float(np.mean(r["goal"] + r["prediction"])) for r in chosen], label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 50)
    ax.set_ylim(0.6, 30)
    ticks(ax.xaxis, [1, 10])
    ticks(ax.yaxis, [1, 10])
    ax.set_xlabel("wall clock [ms/step]")
    ax.set_ylabel(r"$G$", labelpad=1)
    S.save(fig, path)


def main():
    S.apply()
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = performance()
    legend(FIGURES / "fig_lightdark_legend.pdf")
    time_course(rows, FIGURES / "fig_lightdark_time.pdf")
    noise_sweep(rows, FIGURES / "fig_lightdark_noise.pdf")
    wallclock(rows, FIGURES / "fig_lightdark_wallclock.pdf")
    print("\n".join(str(p) for p in sorted(FIGURES.glob("fig_lightdark_[ltnw]*.pdf"))))


if __name__ == "__main__":
    main()
