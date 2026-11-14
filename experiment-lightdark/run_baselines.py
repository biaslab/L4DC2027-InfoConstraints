"""Light-dark experiment of main.tex with the baselines of the parameter
experiments: CBFE against the dual MPC with a tuned and an untuned weight
(rho = 1) and against MPC-X, under the tight goal (S_* = 0.01).

    x_k = x_{k-1} + u_k + w_k ,   w_k ~ N(0, 10^-4) ,
    y_k = x_k + v_k ,             v_k ~ N(0, r(x_k)) ,

with the noise r(x) of ``agents.cbfe_lightdark.LightDarkModel`` (one-sided:
r_L in the light x >= 1.5, rising to 100 in the dark x <= 0), x_0 ~ N(0, 0.25)
and T = 200 steps. The information is gathered for the hidden state x.

    python experiment-lightdark/run_baselines.py tune    # results/baselines_tuning.pkl
    python experiment-lightdark/run_baselines.py test    # results/baselines_test.pkl, baselines_sweep.pkl
    python experiment-lightdark/run_baselines.py report

Tuning selects rho and the accuracy gamma of MPC-X on the held-out seeds
3000-3049 by the mean goal cost at the nominal light noise. The test runs
the seeds 2000-2199 at the nominal light noise in four processes on the four
fast cores, so that the wall-clock times are comparable, and the seeds
2000-2099 at the other light noises on all cores, with the weights tuned at
the nominal noise.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lightdark as L  # noqa: E402
from agents.cbfe_lightdark import LightDarkModel  # noqa: E402

GOAL = "tight"
NOMINAL_LIGHT = math.sqrt(LightDarkModel().r0)      # std of the measurement noise in the light
LIGHTS = (0.01, NOMINAL_LIGHT, 0.1, 0.2, 0.3)
RHOS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
GAMMAS = (0.1, 0.3, 1.0, 3.0, 10.0)
TUNING_SEEDS, TEST_SEEDS, SWEEP_SEEDS = range(3000, 3050), range(2000, 2200), range(2000, 2100)
FAST_CORES = {0, 1, 2, 3}
UNTUNED_RHO = 1.0


def bench(light: float = NOMINAL_LIGHT) -> L.Benchmark:
    return replace(L.Benchmark(model=LightDarkModel(one_sided=True, r0=light**2)), T=200)


def dual(rho: float) -> dict:
    return {"kind": "agent", "planner": "dual", "weight": rho / 2.0}


def mpcx(gamma: float) -> dict:
    return {"kind": "agent", "planner": "mpcx", "accuracy": gamma}


def run(specs: dict, seeds, light: float, processes: int) -> list:
    jobs = [dict(label=label, spec=spec, bench=bench(light), goal=GOAL, seed=seed)
            for label, spec in specs.items() for seed in seeds]
    out = L.run_many(jobs, processes=processes)
    for result in out:
        result["light"] = light
    return out


def frame(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        ep = r["episode"]
        cost = (ep.x[1:] - 0.0)**2 / (2.0 * L.GOALS[GOAL])
        rows.append(dict(label=r["label"], light=r["light"], seed=r["seed"], G=float(cost.mean()),
                         wall=float(ep.plan_time.sum()), active=float(np.mean(ep.multipliers[:, 1] > 0.0)),
                         lit=float(np.mean(ep.x[1:] >= 1.0))))
    return pd.DataFrame(rows)


def tuned() -> dict:
    """The rho and gamma that the held-out seeds select by the mean goal cost."""
    tuning = frame(L.load("baselines_tuning")).groupby("label").G.mean()
    rho = min(RHOS, key=lambda r: tuning[f"dual MPC, rho = {r:g}"])
    gamma = min(GAMMAS, key=lambda g: tuning[f"MPC-X, gamma = {g:g}"])
    return dict(rho=rho, gamma=gamma)


def controllers() -> dict:
    t = tuned()
    return {"CBFE": {"kind": "agent", "planner": "cbfe"},
            "dual MPC, tuned": dual(t["rho"]),
            "dual MPC, rho = 1": dual(UNTUNED_RHO),
            "MPC-X": mpcx(t["gamma"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("tune", "test", "report"))
    args = parser.parse_args()
    start = time.perf_counter()
    if args.mode == "tune":
        specs = {**{f"dual MPC, rho = {r:g}": dual(r) for r in RHOS}, **{f"MPC-X, gamma = {g:g}": mpcx(g) for g in GAMMAS}}
        print(L.save(run(specs, TUNING_SEEDS, NOMINAL_LIGHT, processes=12), "baselines_tuning"))
    elif args.mode == "test":
        affinity = os.sched_getaffinity(0)
        os.sched_setaffinity(0, FAST_CORES)
        timed = run(controllers(), TEST_SEEDS, NOMINAL_LIGHT, processes=len(FAST_CORES))
        os.sched_setaffinity(0, affinity)
        print(L.save(timed, "baselines_test"))
        sweep = [r for light in LIGHTS if light != NOMINAL_LIGHT for r in run(controllers(), SWEEP_SEEDS, light, 12)]
        print(L.save(sweep, "baselines_sweep"))
    print(f"{args.mode}: {time.perf_counter() - start:.0f} s")
    if args.mode != "tune" and (L.RESULTS / "baselines_test.pkl").exists():
        test = frame(L.load("baselines_test"))
        print("tuned:", tuned())
        print(test.groupby("label")[["G", "wall", "active", "lit"]].mean().round(4).to_string())
        rng = np.random.default_rng(0)
        for label in ("dual MPC, tuned", "dual MPC, rho = 1", "MPC-X"):
            d = (test[test.label == label].sort_values("seed").G.to_numpy()
                 - test[test.label == "CBFE"].sort_values("seed").G.to_numpy())
            m = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            print(f"  {label:18s} - CBFE {d.mean():+.3f} [{np.quantile(m, .025):+.3f}, {np.quantile(m, .975):+.3f}]"
                  f" CBFE better on {100 * np.mean(d > 0):.0f}%")
    elif args.mode == "tune":
        tuning = frame(L.load("baselines_tuning"))
        print(tuning.groupby("label")[["G", "active", "lit"]].mean().round(4).to_string())
        print("tuned:", tuned())


if __name__ == "__main__":
    main()
