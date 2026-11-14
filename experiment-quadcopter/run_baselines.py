"""Tuned baselines on the quadcopter of ``ablation.py``: the dual MPC with a
fixed probing weight rho on the information of every planned step, and
MPC-X with the experiment-design requirement of accuracy kappa
(``controllers.cbfe_quad``, module docstring).

    python experiment-quadcopter/run_baselines.py pilot      # a few flights per value, to place the grids
    python experiment-quadcopter/run_baselines.py transfer   # data/baselines_transfer.pkl
    python experiment-quadcopter/run_baselines.py test       # data/baselines_test.pkl
    python experiment-quadcopter/run_baselines.py test --fresh   # data/baselines_test_fresh.pkl
    python experiment-quadcopter/run_baselines.py report

``transfer`` flies CBFE and every grid value at every IMU noise of
``run.NOISES`` on the held-out seeds 3000-3049. The values with the lowest
mean G at the nominal noise sigma_f = 2 are the tuned ones, and the other
noise levels show whether the tuning transfers. ``test`` flies the tuned
values on the test seeds 1000-1099 of ``run.py`` at every noise, the nominal
level on the four fast cores so that the wall-clock times compare with
data/episodes.pkl. With ``--fresh``, ``test`` flies the fresh seeds 4000-4099
of ``run.py`` instead, with the same tuned values. G is the metric of main.tex: the goal cost on the true
position plus half the precision-weighted prediction error of the filter.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ablation as A  # noqa: E402
import run as R  # noqa: E402

RHOS = (1e2, 1e3, 3e3, 1e4, 3e4, 1e5)
KAPPAS = (3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
TUNING_SEEDS, PILOT_SEEDS = range(3000, 3050), range(3000, 3024)


def dual(rho: float) -> dict:
    return dict(caution=True, constrained=False, planner="dual", probing_weight=rho / 2.0)


def mpcx(kappa: float) -> dict:
    return dict(caution=False, constrained=False, planner="mpcx", accuracy=kappa, fallback="nominal")


def grid() -> dict:
    return {"CBFE": None, **{f"dual MPC, rho = {r:g}": dual(r) for r in RHOS},
            **{f"MPC-X, kappa = {k:g}": mpcx(k) for k in KAPPAS}}


def jobs(specs: dict, noises, seeds) -> list:
    return [dict(label=label, spec=spec, bench=A.Benchmark(imu_noise=noise), seed=seed)
            for label, spec in specs.items() for noise in noises for seed in seeds]


def frame(results: list) -> pd.DataFrame:
    return pd.DataFrame([dict(label=r["label"], imu_noise=r["imu_noise"], seed=r["seed"],
                              G=float(np.mean(r["true_cost"] + 0.5 * r["error"])),
                              ms_per_step=1e3 * r["wall"] / len(r["u"]),
                              active=float(np.mean(r["multiplier"] > 0.0))) for r in results])


def tuned() -> dict:
    """rho and kappa with the lowest mean G at the nominal noise on the tuning seeds."""
    runs = frame(A.load(A.DATA / "baselines_transfer.pkl"))
    nominal = runs[runs.imu_noise == R.NOMINAL].groupby("label").G.mean()
    return dict(rho=min(RHOS, key=lambda r: nominal[f"dual MPC, rho = {r:g}"]),
                kappa=min(KAPPAS, key=lambda k: nominal[f"MPC-X, kappa = {k:g}"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("pilot", "transfer", "test", "report"))
    parser.add_argument("--fresh", action="store_true", help="test on the seeds 4000-4099 instead of 1000-1099")
    args = parser.parse_args()
    seeds, tag = (R.FRESH_SEEDS, "_fresh") if args.fresh else (R.SEEDS, "")
    start = time.perf_counter()
    pd.set_option("display.width", 200)
    if args.mode == "pilot":
        specs = {**grid(), "MPC": dict(caution=False, constrained=False)}
        out = A.run_many(jobs(specs, [R.NOMINAL], PILOT_SEEDS), processes=12)
        print(frame(out).groupby("label")[["G", "ms_per_step", "active"]].mean().round(2).to_string())
    elif args.mode == "transfer":
        out = A.run_many(jobs(grid(), R.NOISES, TUNING_SEEDS), processes=12)
        print(A.save(out, A.DATA / "baselines_transfer.pkl"))
    elif args.mode == "test":
        t = tuned()
        specs = {"dual MPC": dual(t["rho"]), "MPC-X": mpcx(t["kappa"])}
        affinity = os.sched_getaffinity(0)
        os.sched_setaffinity(0, R.FAST_CORES)
        timed = A.run_many(jobs(specs, [R.NOMINAL], seeds), processes=len(R.FAST_CORES))
        os.sched_setaffinity(0, affinity)
        rest = A.run_many(jobs(specs, [n for n in R.NOISES if n != R.NOMINAL], seeds), processes=12)
        print(A.save(timed + rest, A.DATA / f"baselines_test{tag}.pkl"), "tuned:", t)
    print(f"{args.mode}: {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
