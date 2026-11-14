"""Candidate panels on tuning transfer in the light-dark experiment, from the
outputs of ``run_transfer.py`` and ``performance.py``:

    figures/fig_lightdark_transfer_legend.pdf  legend strip of the light noise levels
    figures/fig_lightdark_transfer_dual.pdf    G of the dual MPC relative to CBFE
                                               against the probing weight rho
    figures/fig_lightdark_transfer_mpcx.pdf    G of MPC-X relative to CBFE against
                                               the accuracy kappa

One line per light noise, from light grey (sqrt r_L = 0.01) to black (0.3),
the nominal noise drawn thicker. G is relative to that of CBFE at the same
light noise on the same 50 held-out seeds, so values below the blue line at 1
beat CBFE. The dashed vertical line marks the value tuned at the nominal noise.

    python experiment-lightdark/plot_transfer.py
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
import plot_baselines as B  # noqa: E402
import run_baselines as R  # noqa: E402
import run_transfer as X  # noqa: E402

SHADES = ["#c9c8c3", "#a5a39d", "#7d7b75", "#52514e", S.INK]       # light noise, low to high


def table() -> dict:
    """Mean G over the held-out seeds, by (label, light)."""
    with open(L.RESULTS / "baselines_performance.pkl", "rb") as f:
        rows = [r for r in pickle.load(f) if r["file"] == "transfer"]
    out = {}
    for r in rows:
        out.setdefault((r["label"], r["light"]), []).append(float(np.mean(r["goal"] + r["prediction"])))
    return {key: float(np.mean(values)) for key, values in out.items()}


def legend(path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=shade, lw=2.2 if light == R.NOMINAL_LIGHT else 1.2, marker="o", ms=3)
               for light, shade in zip(R.LIGHTS, SHADES)]
    handles.append(Line2D([], [], color=B.STYLES["CBFE"]["color"], lw=1.4))
    names = [rf"$\sqrt{{r_L}} = {light:.1g}$" for light in R.LIGHTS] + ["CBFE"]
    fig = plt.figure(figsize=(S.PMLR_TEXT_WIDTH, 0.17))
    fig.legend(handles, names, loc="center", ncol=len(names), frameon=False, fontsize=6.5, handlelength=2.0,
               columnspacing=1.4, handletextpad=0.4)
    S.save(fig, path)


def panel(g: dict, prefix: str, values, tuned: float, xlabel: str, xticks, path: Path) -> None:
    fig, ax = B.panel()
    ax.axhline(1.0, color=B.STYLES["CBFE"]["color"], lw=1.4, zorder=3)
    ax.axvline(tuned, color=S.INK, ls=(0, (4, 2)), lw=0.8, zorder=1)
    for light, shade in zip(R.LIGHTS, SHADES):
        ratio = [g[(f"{prefix}{v:g}", light)] / g[("CBFE", light)] for v in values]
        nominal = light == R.NOMINAL_LIGHT
        ax.plot(values, ratio, color=shade, lw=2.2 if nominal else 1.2, marker="o", ms=3, zorder=6 if nominal else 5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = B.ticks
    ticks(ax.xaxis, xticks)
    ticks(ax.yaxis, [1, 10])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$G$ relative to CBFE", labelpad=2)
    S.save(fig, path)


def main():
    S.apply()
    g = table()
    for light in R.LIGHTS:
        best_rho = min(X.RHOS, key=lambda r: g[(f"dual MPC, rho = {r:g}", light)])
        best_kappa = min(X.KAPPAS, key=lambda k: g[(f"MPC-X, kappa = {k:g}", light)])
        print(f"sqrt r_L {light:.3g}: CBFE {g[('CBFE', light)]:.2f} | best rho {best_rho:g} "
              f"({g[(f'dual MPC, rho = {best_rho:g}', light)]:.2f}), rho=100 {g[('dual MPC, rho = 100', light)]:.2f} | "
              f"best kappa {best_kappa:g} ({g[(f'MPC-X, kappa = {best_kappa:g}', light)]:.2f}), "
              f"kappa=0.3 {g[('MPC-X, kappa = 0.3', light)]:.2f}")
    legend(B.FIGURES / "fig_lightdark_transfer_legend.pdf")
    panel(g, "dual MPC, rho = ", X.RHOS, 100.0, r"probing weight $\rho$", [10, 100, 1000],
          B.FIGURES / "fig_lightdark_transfer_dual.pdf")
    panel(g, "MPC-X, kappa = ", X.KAPPAS, 0.3, r"accuracy $\kappa$", [0.1, 1],
          B.FIGURES / "fig_lightdark_transfer_mpcx.pdf")


if __name__ == "__main__":
    main()
