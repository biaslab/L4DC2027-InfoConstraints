"""Numbers of the planar positioning benchmark, from results/tuning.pkl and
results/test.pkl of run.py.

    python experiment-planar/analysis.py

Prints the weights selected on the held-out seeds (one for all disturbance
levels, and one per level), the goal cost per step on the test seeds, paired
bootstrap 95% intervals of the differences to CBFE, the worst-case cost of
every fixed weight relative to CBFE, and the activity and the multipliers of
the information constraint.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import planar as PL  # noqa: E402
import run as Run  # noqa: E402

BOOTSTRAP = 10_000
NORMALISED = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


def frame(results) -> pd.DataFrame:
    return pd.DataFrame([dict(label=r["label"], sigma=r["sigma"], seed=r["seed"], G=r["G"],
                              rmse=r["rmse"]) for r in results])


def paired_interval(a: np.ndarray, b: np.ndarray, rng) -> tuple:
    """Mean of a - b and its percentile bootstrap 95% interval."""
    d = a - b
    means = d[rng.integers(0, len(d), (BOOTSTRAP, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    pd.set_option("display.width", 200)
    tuning, results = frame(PL.load("tuning")), PL.load("test")
    test = frame(results)
    duals = [Run.dual_label(r) for r in Run.RHOS]
    mpcx = [Run.mpcx_label(g) for g in Run.GAMMAS]

    rmse = tuning.pivot_table(index="label", columns="sigma", values="rmse", aggfunc="mean")
    single, single_x = rmse.loc[duals].mean(axis=1).idxmin(), rmse.loc[mpcx].mean(axis=1).idxmin()
    per = {s: rmse.loc[duals, s].idxmin() for s in Run.SIGMAS}
    print(f"held-out seeds, mean RMSE over all levels selects: {single}; {single_x}")
    print("per level:", {s: per[s] for s in Run.SIGMAS})

    cost = test.pivot_table(index="label", columns="sigma", values="G", aggfunc="mean", sort=False)
    print("\ntest seeds: goal cost per step\n" + cost.round(3).to_string())
    print("\nrelative to CBFE\n" + (cost / cost.loc["CBFE"]).round(2).to_string())
    worst = (cost.loc[duals] / cost.loc["CBFE"]).max(axis=1)
    print("\nworst case over the levels, cost relative to CBFE, per fixed weight\n" + worst.round(2).to_string())

    rng = np.random.default_rng(0)
    print("\npaired differences of the goal cost to CBFE, mean [95% interval]")
    for s in Run.SIGMAS:
        cbfe = test[(test.label == "CBFE") & (test.sigma == s)].sort_values("seed").G.values
        for label in dict.fromkeys(["cautious", single, per[s], single_x, "oracle"]):
            other = test[(test.label == label) & (test.sigma == s)].sort_values("seed").G.values
            m, lo, hi = paired_interval(other, cbfe, rng)
            print(f"  sigma = {s:<5g} {label:24s} {m:+9.3f} [{lo:+.3f}, {hi:+.3f}]")

    # A weight normalised by the noise-to-tolerance ratio tr(S_*^{-1} R)/n_y = sigma^2/S_*, the
    # scale of the leverage analysis in the supplement, with one constant c for all levels.
    grid, goal = np.array(Run.RHOS), PL.Benchmark().goal_variance

    def schedule(c):
        return {s: Run.dual_label(grid[np.argmin(abs(np.log(grid / (c * s**2 / goal))))]) for s in Run.SIGMAS}

    scores = {c: np.mean([rmse.loc[label, s] for s, label in schedule(c).items()]) for c in NORMALISED}
    c = min(scores, key=scores.get)
    print(f"\nnormalised weight rho = c sigma^2/S_* on the grid; held-out seeds select c = {c:g}")
    for s, label in schedule(c).items():
        other = test[(test.label == label) & (test.sigma == s)].sort_values("seed").G.values
        cbfe = test[(test.label == "CBFE") & (test.sigma == s)].sort_values("seed").G.values
        m, lo, hi = paired_interval(other, cbfe, rng)
        print(f"  sigma = {s:<5g} {label:24s} {other.mean() / cbfe.mean():5.2f} x CBFE"
              f"   difference {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\nCBFE: share of steps with an active constraint, last active step, median active multiplier")
    for s in Run.SIGMAS:
        eps = [r for r in results if r["label"] == "CBFE" and r["sigma"] == s]
        active = np.array([r["multiplier"] > 0 for r in eps])
        last = [np.flatnonzero(a).max() + 1 if a.any() else 0 for a in active]
        values = np.concatenate([r["multiplier"][r["multiplier"] > 0] for r in eps])
        print(f"  sigma = {s:<5g} {100 * active.mean():5.1f}%   median last step {np.median(last):.0f}"
              f"   median multiplier {np.median(values):.2f}")


if __name__ == "__main__":
    main()
