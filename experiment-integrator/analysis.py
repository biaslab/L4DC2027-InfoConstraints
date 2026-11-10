"""Numbers of Sec. V-A of main.tex, from results/benchmark.pkl of run.py.

    python experiment-integrator/analysis.py

Prints, for the controllers of the paper,
  - the RMSE of y_1, ..., y_T per prior mean b0 and over all prior means,
    and the NLL under the goal prior N(0, sigma^2), averaged over episodes;
  - paired bootstrap 95% intervals of the RMSE differences between CBFE and
    each baseline (the seeds are shared);
  - the activity of the information constraint and the KKT multipliers;
  - the RMSE of the dual MPC with the Beta input cost over its probing
    weights, per prior mean (the probing-weight curve).
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


def episodes(results) -> pd.DataFrame:
    sigma = I.Benchmark().sigma
    rows = []
    for r in results:
        y = r["episode"].y[1:]
        rows.append(dict(label=r["label"], b0=r["b0"], seed=r["seed"],
                         rmse=float(np.sqrt(np.mean(y ** 2))),
                         nll=float(np.mean(0.5 * y ** 2 / sigma ** 2 + 0.5 * np.log(2 * np.pi * sigma ** 2))),
                         beta=r["episode"].beta))
    return pd.DataFrame(rows)


def paired_interval(a: np.ndarray, b: np.ndarray, rng) -> tuple:
    """Mean of a - b and its percentile bootstrap 95% interval."""
    d = a - b
    means = d[rng.integers(0, len(d), (BOOTSTRAP, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    pd.set_option("display.width", 200)
    df = episodes(I.load("benchmark"))
    paper = df[df.label.isin(PAPER)].assign(name=lambda d: d.label.map(PAPER))
    table = paper.pivot_table(index="name", columns="b0", values="rmse", aggfunc="mean", sort=False)
    table["all"] = paper.groupby("name", sort=False).rmse.mean()
    table["NLL"] = paper.groupby("name", sort=False).nll.mean()
    print("RMSE per prior mean, over all, and NLL\n", table.round(4).to_string(), "\n")

    rng = np.random.default_rng(0)
    rmse = df.pivot_table(index=["b0", "seed"], columns="label", values="rmse")
    print("paired RMSE difference CBFE - baseline, mean [95% interval]")
    for label in PAPER:
        if label == CBFE:
            continue
        cells = [paired_interval(rmse[CBFE].to_numpy(), rmse[label].to_numpy(), rng)]
        cells += [paired_interval(rmse.loc[b0, CBFE].to_numpy(), rmse.loc[b0, label].to_numpy(), rng)
                  for b0 in Run.B0]
        print(f"  {PAPER[label]:36s}" + "  ".join(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]" for m, lo, hi in cells))
    print("  columns: all, then b0 =", Run.B0, "\n")

    kkt = df[df.label == CBFE]
    beta = np.concatenate(kkt.beta.to_numpy())
    active = beta > 0
    print(f"CBFE: constraint active on {100 * active.mean():.1f}% of the steps; multiplier median "
          f"{np.median(beta[active]):.2f}, max {beta.max():.2f}")
    for b0 in Run.B0:
        b = np.stack(kkt[kkt.b0 == b0].beta.to_numpy())
        print(f"  b0 = {b0:g}: active on {100 * (b > 0).mean():.1f}% of the steps, "
              f"at the first step in {100 * (b[:, 0] > 0).mean():.0f}% of the episodes")
    times = [r["episode"].plan_time for r in I.load("benchmark") if r["label"] == CBFE]
    print(f"  planning time per step: median {1e3 * np.median(np.concatenate(times)):.2f} ms\n")

    curve = df[df.label.str.startswith("dual MPC, Beta cost")].copy()
    curve["rho"] = curve.label.str.split("= ").str[-1].astype(float)
    curve = curve.pivot_table(index="rho", columns="b0", values="rmse", aggfunc="mean")
    curve["all"] = curve.mean(axis=1)
    print("dual MPC with the Beta input cost: RMSE per probing weight and prior mean\n",
          curve.round(4).to_string())
    cbfe = table.loc["CBFE"]
    print("CBFE relative to the best probing weight per prior mean:",
          (cbfe[list(Run.B0)] / curve[list(Run.B0)].min()).round(3).to_dict())


if __name__ == "__main__":
    main()
