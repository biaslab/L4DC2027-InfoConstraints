"""Light-dark experiments of the paper (main.tex, Section V-C), with the tuning
protocol of Sections V-A and V-B.

One benchmark (``lightdark.py``): a one-sided light (x >= 1 lit, the target
x_* = 0 in the dark), T = 200 steps from x_0 ~ N(0, 0.25), a tight (S_* = 0.01)
and a loose (S_* = 0.1) goal. Two conditions:

    known        input gain 1; CBFE constrains the state information
    directional  unknown gains b_+ (u >= 0) and b_- (u < 0); CBFE constrains
                 the gain and the state information (two layers), with both
                 single-constraint ablations

The probing weight rho of the dual MPC (weight rho / 2 on every information
term of every step) is chosen on held-out seeds by the mean RMSE over both
goals; the best value per goal is reported as well. The goal cost is split
into the transient (steps 1-20) and the steady state (steps 21-200).

    python experiment-lightdark/run_paper.py tune known        # results/paper_known_tuning.pkl
    python experiment-lightdark/run_paper.py test known        # results/paper_known_test.pkl
    python experiment-lightdark/run_paper.py tune directional
    python experiment-lightdark/run_paper.py test directional
    python experiment-lightdark/run_paper.py report known|directional
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lightdark as L  # noqa: E402
from controllers.cbfe_lightdark import LightDarkModel  # noqa: E402

KNOWN = replace(L.Benchmark(model=LightDarkModel(one_sided=True)), T=200)
DIRECTIONAL = replace(KNOWN, gain_variance=0.09, directional=True)
BENCH = {"known": KNOWN, "directional": DIRECTIONAL}
RHOS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
TUNING_SEEDS = {"known": range(3000, 3050), "directional": range(3000, 3030)}
TEST_SEEDS = {"known": range(2000, 2200), "directional": range(2000, 2100)}
TRANSIENT = 20
FIXED = {"known": {"CBFE": {"kind": "agent", "planner": "cbfe"},
                   "cautious": {"kind": "agent", "planner": "none"},
                   "BSP": {"kind": "agent", "planner": "bsp"},
                   "DP": {"kind": "dp"},
                   "oracle": {"kind": "oracle"}},
         "directional": {"CBFE": {"kind": "agent", "planner": "cbfe"},
                         "CBFE, state only": {"kind": "agent", "planner": "cbfe", "constraints": ("state",)},
                         "CBFE, gain only": {"kind": "agent", "planner": "cbfe", "constraints": ("parameter",)},
                         "cautious": {"kind": "agent", "planner": "none"},
                         "BSP": {"kind": "agent", "planner": "bsp"},
                         "oracle": {"kind": "oracle"}}}
BOOTSTRAP = 10_000


def dual_label(rho: float) -> str:
    return f"dual MPC, rho = {rho:g}"


def dual_specs(condition: str) -> dict:
    if condition == "known":
        return {dual_label(r): {"kind": "agent", "planner": "dual", "weight": r / 2.0} for r in RHOS}
    return {dual_label(r): {"kind": "agent", "planner": "dual", "weights": (r / 2.0, r / 2.0)} for r in RHOS}


def run(name: str, condition: str, specs: dict, seeds, processes=None) -> None:
    bench = BENCH[condition]
    jobs = [dict(label=label, spec=spec, bench=bench, goal=goal, seed=seed)
            for label, spec in specs.items() for goal in L.GOALS for seed in seeds]
    jobs.sort(key=lambda job: job["spec"].get("planner") not in ("cbfe", "bsp", "dual"))
    start = time.perf_counter()
    results = L.run_many(jobs, processes=processes)
    print(f"{name}: {len(jobs)} episodes in {time.perf_counter() - start:.0f} s", flush=True)
    print(L.save(results, name), flush=True)


def trips(ep) -> int:
    lit = (ep.x[1:] >= 0.9).astype(int)
    return int(np.sum(np.diff(lit) == 1) + lit[0])


def metrics(result: dict, bench) -> dict:
    ep = result["episode"]
    S = L.GOALS[result["goal"]]
    error = ep.x[1:] - bench.target
    cost = error**2 / (2.0 * S)
    return dict(label=result["label"], goal=result["goal"], seed=result["seed"],
                nll=float(np.mean(cost)), transient=float(np.mean(cost[:TRANSIENT])),
                steady=float(np.mean(cost[TRANSIENT:])), rmse=float(np.sqrt(np.mean(error**2))),
                inside=100.0 * float(np.mean(np.abs(error[TRANSIENT:]) <= 2.0 * math.sqrt(S))),
                trips=trips(ep), gain_variance=float(ep.beliefs[-1, 3]),
                state_active=100.0 * float(np.mean(ep.multipliers[:, 1] > 0.0)),
                gain_active=100.0 * float(np.mean(ep.multipliers[:, 0] > 0.0)))


def frame(name: str, condition: str) -> pd.DataFrame:
    return pd.DataFrame([metrics(r, BENCH[condition]) for r in L.load(name)])


def interval(d: np.ndarray, rng) -> str:
    means = d[rng.integers(0, len(d), (BOOTSTRAP, len(d)))].mean(axis=1)
    return f"{d.mean():+.3f} [{np.quantile(means, 0.025):+.3f}, {np.quantile(means, 0.975):+.3f}]"


def report(condition: str) -> None:
    pd.set_option("display.width", 250)
    tuning = frame(f"paper_{condition}_tuning", condition)
    print(f"{condition}: held-out seeds, mean RMSE per goal")
    print(tuning.pivot_table(index="label", columns="goal", values="rmse", aggfunc="mean").round(4).to_string())
    shared = tuning.groupby("label").rmse.mean().idxmin()
    per_goal = {goal: tuning[tuning.goal == goal].groupby("label").rmse.mean().idxmin() for goal in L.GOALS}
    print(f"tuned: shared {shared}; per goal {per_goal}\n")
    if not (L.RESULTS / f"paper_{condition}_test.pkl").exists():
        return
    test = frame(f"paper_{condition}_test", condition)
    rows = list(FIXED[condition]) + [shared] + [label for label in dict.fromkeys(per_goal.values()) if label != shared]
    print(f"{condition}: test seeds, means per goal")
    print(test.pivot_table(index="label", columns="goal",
                           values=["nll", "transient", "steady", "rmse", "inside", "trips", "gain_variance"],
                           aggfunc="mean").reindex(rows).round(3).to_string(), "\n")
    rng = np.random.default_rng(0)
    for goal in L.GOALS:
        sub = test[test.goal == goal].pivot_table(index="seed", columns="label", values=["nll", "transient", "steady"])
        print(f"goal {goal}: paired differences to CBFE, mean [95% interval]")
        for label in rows[1:]:
            print("   " + f"{label:22s}" + "  ".join(
                f"{metric} {interval((sub[(metric, label)] - sub[(metric, 'CBFE')]).to_numpy(), rng)}"
                for metric in ("nll", "transient", "steady")))
    curve = test[test.label.str.startswith("dual MPC")].copy()
    curve["rho"] = curve.label.str.split("= ").str[-1].astype(float)
    cbfe = test[test.label == "CBFE"].groupby("goal").nll.mean()
    print("\ntest seeds: goal cost of the dual MPC per rho, relative to CBFE")
    print((curve.pivot_table(index="rho", columns="goal", values="nll", aggfunc="mean") / cbfe).round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("tune", "test", "report"))
    parser.add_argument("condition", choices=("known", "directional"))
    parser.add_argument("--processes", type=int, default=None)
    args = parser.parse_args()
    c = args.condition
    if args.mode == "tune":
        run(f"paper_{c}_tuning", c, dual_specs(c), TUNING_SEEDS[c], args.processes)
    elif args.mode == "test":
        run(f"paper_{c}_test", c, {**FIXED[c], **dual_specs(c)}, TEST_SEEDS[c], args.processes)
    report(c)


if __name__ == "__main__":
    main()
