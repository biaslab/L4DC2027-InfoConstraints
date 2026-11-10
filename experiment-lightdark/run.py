"""Run the light-dark benchmark of ``lightdark.py`` with the tuning protocol of
the paper. The probing weight rho of the dual MPC (weight rho / 2 on the
state information of every step) is chosen on the held-out seeds 3000-3049
by the mean RMSE over both goals (one value for both goals); the best value
per goal is reported as well. CBFE, the cautious controller, the belief-space
planner and the DP have nothing to tune. The test seeds are 2000-2199.

    python experiment-lightdark/run.py tune     # results/tuning.pkl
    python experiment-lightdark/run.py test     # results/test.pkl
    python experiment-lightdark/run.py report
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lightdark as L  # noqa: E402

BENCH = L.Benchmark()
RHOS = (0.3, 1.0, 3.0, 10.0, 20.0, 30.0, 50.0, 100.0, 200.0, 300.0, 1000.0)
TUNING_SEEDS = range(3000, 3050)
TEST_SEEDS = range(2000, 2200)
FIXED = {"CBFE": {"kind": "agent", "planner": "cbfe"},
         "cautious": {"kind": "agent", "planner": "none"},
         "BSP": {"kind": "agent", "planner": "bsp"},
         "DP": {"kind": "dp"},
         "oracle": {"kind": "oracle"}}
BOOTSTRAP = 10_000


def dual_label(rho: float) -> str:
    return f"dual MPC, rho = {rho:g}"


def dual_specs() -> dict:
    return {dual_label(r): {"kind": "agent", "planner": "dual", "weight": r / 2.0} for r in RHOS}


def run(name: str, specs: dict, seeds, processes=None) -> None:
    jobs = [dict(label=label, spec=spec, bench=BENCH, goal=goal, seed=seed)
            for label, spec in specs.items() for goal in L.GOALS for seed in seeds]
    start = time.perf_counter()
    results = L.run_many(jobs, processes=processes)
    print(f"{len(jobs)} episodes in {time.perf_counter() - start:.0f} s", flush=True)
    print(L.save(results, name), flush=True)


def frame(name: str) -> pd.DataFrame:
    return pd.DataFrame([L.metrics(r, BENCH) for r in L.load(name)])


def interval(d: np.ndarray, rng) -> str:
    means = d[rng.integers(0, len(d), (BOOTSTRAP, len(d)))].mean(axis=1)
    return f"{d.mean():+.4f} [{np.quantile(means, 0.025):+.4f}, {np.quantile(means, 0.975):+.4f}]"


def report() -> None:
    pd.set_option("display.width", 220)
    tuning = frame("tuning")
    print("held-out seeds: mean RMSE and goal NLL per goal")
    print(tuning.pivot_table(index="label", columns="goal", values=["rmse", "nll"], aggfunc="mean")
          .round(4).to_string(), "\n")
    shared = tuning.groupby("label").rmse.mean().idxmin()
    per_goal = {goal: tuning[tuning.goal == goal].groupby("label").rmse.mean().idxmin() for goal in L.GOALS}
    print(f"tuned: shared {shared}; per goal {per_goal}")
    if not (L.RESULTS / "test.pkl").exists():
        return
    test = frame("test")
    rows = list(FIXED) + [shared] + [label for label in dict.fromkeys(per_goal.values()) if label != shared]
    print("\ntest seeds: means per goal")
    print(test.pivot_table(index="label", columns="goal", values=["rmse", "nll", "final", "inside", "light", "active"],
                           aggfunc="mean").reindex(rows).round(4).to_string(), "\n")
    rng = np.random.default_rng(0)
    for goal in L.GOALS:
        sub = test[test.goal == goal].pivot_table(index="seed", columns="label", values=["rmse", "nll"])
        print(f"goal {goal}: paired differences to CBFE, mean [95% interval]")
        for label in rows[1:]:
            for metric in ("rmse", "nll"):
                d = (sub[(metric, label)] - sub[(metric, "CBFE")]).to_numpy()
                print(f"   {label:24s} {metric:5s} {interval(d, rng)}")
    print("\ntest seeds: mean RMSE of the dual MPC per rho, relative to CBFE")
    curve = test[test.label.str.startswith("dual MPC")].copy()
    curve["rho"] = curve.label.str.split("= ").str[-1].astype(float)
    cbfe = test[test.label == "CBFE"].groupby("goal").rmse.mean()
    rel = curve.pivot_table(index="rho", columns="goal", values="rmse", aggfunc="mean") / cbfe
    rel["both"] = curve.groupby("rho").rmse.mean() / test[test.label == "CBFE"].rmse.mean()
    print(rel.round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("tune", "test", "report"))
    parser.add_argument("--processes", type=int, default=None)
    args = parser.parse_args()
    if args.mode == "tune":
        run("tuning", dual_specs(), TUNING_SEEDS, args.processes)
    elif args.mode == "test":
        run("test", {**FIXED, **dual_specs()}, TEST_SEEDS, args.processes)
    report()


if __name__ == "__main__":
    main()
