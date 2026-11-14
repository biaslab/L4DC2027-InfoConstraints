"""Tuned baselines on the quadcopter, from the outputs of ``run_baselines.py``:

    figures/fig_ablation_transfer_legend.pdf  legend strip of the IMU noise levels
    figures/fig_ablation_transfer_dual.pdf    G of the dual MPC relative to CBFE
                                              against the probing weight rho
    figures/fig_ablation_transfer_mpcx.pdf    G of MPC-X relative to CBFE against
                                              the accuracy kappa

One line per IMU noise, from light grey (sigma_f = 0.2) to black (3), the
nominal noise drawn thicker. G is relative to that of CBFE at the same noise
on the same 50 held-out seeds, so values below the blue line at 1 beat CBFE.
The dashed vertical line marks the value tuned at the nominal noise. The
script also prints, per noise, the best value of each grid and the paired
differences to CBFE of the tuned values on the held-out and the test seeds.

    python experiment-quadcopter/plot_baselines.py
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
import run as R  # noqa: E402
import run_baselines as B  # noqa: E402

FIGURES = HERE / "figures"
SHADES = ["#c9c8c3", "#a5a39d", "#7d7b75", "#52514e", S.INK]       # IMU noise, low to high


def per_flight(results: list) -> dict:
    """G of every flight, by (label, noise) as an array over the sorted seeds."""
    out = {}
    for r in sorted(results, key=lambda r: r["seed"]):
        out.setdefault((r["label"], r["imu_noise"]), []).append(float(np.mean(r["true_cost"] + 0.5 * r["error"])))
    return {key: np.array(values) for key, values in out.items()}


def interval(d: np.ndarray) -> tuple:
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def legend(path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=shade, lw=2.2 if noise == R.NOMINAL else 1.2, marker="o", ms=3)
               for noise, shade in zip(R.NOISES, SHADES)]
    handles.append(Line2D([], [], color=P.BLUE, lw=1.4))
    names = [rf"$\sigma_f = {noise:g}$" for noise in R.NOISES] + ["CBFE"]
    fig = plt.figure(figsize=(S.PMLR_TEXT_WIDTH, 0.17))
    fig.legend(handles, names, loc="center", ncol=len(names), frameon=False, fontsize=6.5, handlelength=2.0,
               columnspacing=1.4, handletextpad=0.4)
    S.save(fig, path)


def decades(axis, values) -> None:
    """Ticks at the decades in ``values``, labelled as powers of ten."""
    import math
    P.ticks(axis, values)
    axis.set_major_formatter(lambda value, _: rf"$10^{{{round(math.log10(value))}}}$")


def panel(g: dict, prefix: str, values, tuned: float, xlabel: str, xticks, xlim, path: Path) -> None:
    fig, ax = P.panel(P.WIDTH)
    fig.subplots_adjust(bottom=0.40 / P.HEIGHT)          # room for the powers of ten
    ax.axhline(1.0, color=P.BLUE, lw=1.4, zorder=3)
    ax.axvline(tuned, color=S.INK, ls=(0, (4, 2)), lw=0.8, zorder=1)
    for noise, shade in zip(R.NOISES, SHADES):
        ratio = [g[(f"{prefix}{v:g}", noise)].mean() / g[("CBFE", noise)].mean() for v in values]
        nominal = noise == R.NOMINAL
        ax.plot(values, ratio, color=shade, lw=2.2 if nominal else 1.2, marker="o", ms=3, zorder=6 if nominal else 5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(0.9, 12)
    decades(ax.xaxis, xticks)
    P.ticks(ax.yaxis, [1, 10])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$G$ relative to CBFE", labelpad=2)
    S.save(fig, path)


def report(g: dict, tuned: dict) -> None:
    for noise in R.NOISES:
        cbfe = g[("CBFE", noise)]
        best_rho = min(B.RHOS, key=lambda r: g[(f"dual MPC, rho = {r:g}", noise)].mean())
        best_kappa = min(B.KAPPAS, key=lambda k: g[(f"MPC-X, kappa = {k:g}", noise)].mean())
        line = f"sigma_f {noise:g}: CBFE {cbfe.mean():.0f}"
        for name, prefix, best, value in (("rho", "dual MPC, rho = ", best_rho, tuned["rho"]),
                                          ("kappa", "MPC-X, kappa = ", best_kappa, tuned["kappa"])):
            mean, low, high = interval(g[(f"{prefix}{value:g}", noise)] - cbfe)
            line += (f" | best {name} {best:g} ({g[(f'{prefix}{best:g}', noise)].mean():.0f}),"
                     f" tuned {value:g}: {g[(f'{prefix}{value:g}', noise)].mean():.0f}, - CBFE {mean:+.0f} [{low:+.0f}, {high:+.0f}]")
        print(line)
    test_path = A.DATA / "baselines_test.pkl"
    if not test_path.exists():
        return
    test = per_flight(A.load(test_path))
    import pickle
    with open(A.DATA / "free_energy.pkl", "rb") as f:
        error = {(t["label"], t["imu_noise"], t["seed"]): t["error"] for t in pickle.load(f)}
    stored = {}
    for r in sorted(A.load(A.DATA / "episodes.pkl"), key=lambda r: r["seed"]):
        g_flight = float(np.mean(r["true_cost"] + 0.5 * error[(r["label"], r["imu_noise"], r["seed"])]))
        stored.setdefault((r["label"], r["imu_noise"]), []).append(g_flight)
    stored = {key: np.array(values) for key, values in stored.items()}
    print("test seeds 1000-1099 (tuned values):")
    for noise in R.NOISES:
        cbfe = stored[("CBFE", noise)]
        line = f"sigma_f {noise:g}: CBFE {cbfe.mean():.0f}, MPC {stored[('certainty equivalence', noise)].mean():.0f}"
        for label in ("dual MPC", "MPC-X"):
            mean, low, high = interval(test[(label, noise)] - cbfe)
            line += f" | {label} {test[(label, noise)].mean():.0f}, - CBFE {mean:+.0f} [{low:+.0f}, {high:+.0f}]"
        print(line)


def main():
    S.apply()
    g = per_flight(A.load(A.DATA / "baselines_transfer.pkl"))
    tuned = B.tuned()
    report(g, tuned)
    legend(FIGURES / "fig_ablation_transfer_legend.pdf")
    panel(g, "dual MPC, rho = ", B.RHOS, tuned["rho"], r"probing weight $\rho$", [1e2, 1e3, 1e4, 1e5], (60, 1.7e5),
          FIGURES / "fig_ablation_transfer_dual.pdf")
    panel(g, "MPC-X, kappa = ", B.KAPPAS, tuned["kappa"], r"accuracy $\kappa$", [1e-2, 1e-1, 1], (2e-3, 1.6),
          FIGURES / "fig_ablation_transfer_mpcx.pdf")


if __name__ == "__main__":
    main()
