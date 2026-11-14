"""Numbers of Section 5.1 of main.tex and Section S2.6 of the supplement, from
results/benchmark.pkl of run.py.

    python experiment-integrator/analysis.py

The performance of an episode is G of eq. (metric) of main.tex
(``integrator.Episode.G``): the mean over the steps of half the
precision-weighted error of the Kalman filter's one-step prediction of y_t
plus the goal cost y_t^2 / (2 sigma^2). The script prints, for the
controllers of the paper,
  - G per prior mean b0 and over all prior means, averaged over episodes;
  - paired bootstrap 95% intervals of the differences in G between CBFE and
    each baseline (the seeds are shared);
  - the activity of the information constraint and the KKT multipliers;
  - G of the dual MPC with the Beta input cost over its probing weights, per
    prior mean, relative to CBFE (the probing-weight curve of Fig. 3).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import integrator as I  # noqa: E402
import run as Run  # noqa: E402

CBFE = "CBFE-ARX, KKT β, predictive floor, next node"
PAPER = {
    CBFE: "CBFE",
    "CBFE-ARX, KKT β": "CBFE, task floor, all nodes",
    "CBFE-ARX, KKT β, next node": "CBFE, task floor, next node",
    "CBFE-ARX, KKT β, predictive floor": "CBFE, predictive floor, all nodes",
    "CBFE-ARX, β = 0": "cautious (γ = 0)",
    "CBFE-ARX, β = 1 (EFE)": "EFE (γ = 1)",
    Run.dual_beta_label(Run.LAMBDA_BETA): f"dual MPC, ρ = {Run.LAMBDA_BETA:g}",
    f"dual MPC, λ = {Run.LAMBDA:g}": f"dual MPC, quadratic, ρ = {Run.LAMBDA:g}",
    f"MPC-X, γ = {Run.GAMMA:g}": "MPC-X",
    "optimal dual (DP)": "optimal dual (DP)",
}
BOOTSTRAP = 10_000
WINDOW = 1.06            # the good probing weights keep the dual MPC within 6% of CBFE at every prior mean


def episodes(results) -> pd.DataFrame:
    sigma = I.Benchmark().sigma
    rows = [dict(label=r["label"], b0=r["b0"], seed=r["seed"], G=r["episode"].G(sigma),
                 rmse=float(np.sqrt(np.mean(r["episode"].y[1:] ** 2))), beta=r["episode"].beta) for r in results]
    return pd.DataFrame(rows)


def paired_interval(a: np.ndarray, b: np.ndarray, rng) -> tuple:
    """Mean of a - b and its percentile bootstrap 95% interval."""
    d = a - b
    means = d[rng.integers(0, len(d), (BOOTSTRAP, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def curve(df: pd.DataFrame) -> pd.DataFrame:
    """Mean G of the dual MPC with the Beta input cost per probing weight and prior mean."""
    dual = df[df.label.str.startswith("dual MPC, Beta cost")].copy()
    dual["rho"] = dual.label.str.split("= ").str[-1].astype(float)
    table = dual.pivot_table(index="rho", columns="b0", values="G", aggfunc="mean")
    table["all"] = table.mean(axis=1)
    return table


def main() -> None:
    pd.set_option("display.width", 200)
    df = episodes(I.load("benchmark"))
    paper = df[df.label.isin(PAPER)].assign(name=lambda d: d.label.map(PAPER))
    table = paper.pivot_table(index="name", columns="b0", values="G", aggfunc="mean", sort=False)
    table["all"] = paper.groupby("name", sort=False).G.mean()
    table["RMSE"] = paper.groupby("name", sort=False).rmse.mean()
    print("G per prior mean and over all (and the mean RMSE for reference)\n", table.round(4).to_string(), "\n")

    rng = np.random.default_rng(0)
    g = df.pivot_table(index=["b0", "seed"], columns="label", values="G")
    print("paired difference in G, CBFE - baseline, mean [95% interval]")
    for label in PAPER:
        if label == CBFE:
            continue
        cells = [paired_interval(g[CBFE].to_numpy(), g[label].to_numpy(), rng)]
        cells += [paired_interval(g.loc[b0, CBFE].to_numpy(), g.loc[b0, label].to_numpy(), rng) for b0 in Run.B0]
        print(f"  {PAPER[label]:36s}" + "  ".join(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]" for m, lo, hi in cells))
    print("  columns: all, then b0 =", Run.B0, "\n")

    kkt = df[df.label == CBFE]
    beta = np.concatenate(kkt.beta.to_numpy())
    active = beta > 0
    print(f"CBFE: constraint active on {100 * active.mean():.1f}% of the steps; multiplier median "
          f"{np.median(beta[active]):.2f}, max {beta.max():.2f}")
    for b0 in Run.B0:
        b = np.stack(kkt[kkt.b0 == b0].beta.to_numpy())
        print(f"  b0 = {b0:g}: active on {100 * (b > 0).mean():.1f}% of the steps, at the first step in "
              f"{100 * (b[:, 0] > 0).mean():.0f}% and at the second in {100 * (b[:, 1] > 0).mean():.0f}% of the episodes")
    times = [r["episode"].plan_time for r in I.load("benchmark") if r["label"] == CBFE]
    print(f"  planning time per step: median {1e3 * np.median(np.concatenate(times)):.2f} ms\n")

    dual = curve(df)
    cbfe = table.loc["CBFE"]
    ratio = dual / cbfe[list(Run.B0) + ["all"]]
    print("dual MPC with the Beta input cost: G per probing weight and prior mean\n", dual.round(4).to_string())
    print("... relative to CBFE\n", ratio.round(3).to_string())
    worst = ratio[list(Run.B0)].max(axis=1)
    print("largest ratio over the prior means per probing weight:", worst.round(3).to_dict())
    print(f"probing weights within {100 * (WINDOW - 1):.0f}% of CBFE at every prior mean:",
          [rho for rho, w in worst.items() if w <= WINDOW])
    print("CBFE relative to the best probing weight per prior mean:",
          (cbfe[list(Run.B0)] / dual[list(Run.B0)].min()).round(3).to_dict())


if __name__ == "__main__":
    main()
