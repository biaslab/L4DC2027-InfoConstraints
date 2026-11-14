"""Run the quadcopter ablation of ``ablation.py``: CBFE with and without the
information constraint and the trace term, and the controller that knows the
wind, over a sweep of the IMU noise.

    python experiment-ablation/run.py          # data/episodes.pkl, results/runs.csv,
                                               # results/levels.csv
    python experiment-ablation/run.py --fresh  # the same on the fresh seeds, with
                                               # "_fresh" added to the file names

Every controller flies the seeds 1000-1099 at every noise level. These test
seeds also served to select the setting (supplementary S2.5), so ``--fresh``
flies the seeds 4000-4099 instead, which played no role in the design. The nominal
level, whose wall-clock times Fig. 2 shows, runs in four processes on the
four fast cores only: the Ryzen AI 9 HX 370 mixes four fast (Zen 5) and eight
slower (Zen 5c) cores, and runs on both kinds give two clusters of times. The
other levels run on all cores, and their times are not comparable. The run
takes about half an hour.
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

NOISES = (0.2, 0.5, 1.0, 2.0, 3.0)   # sigma_f [m/s^2]
NOMINAL = 2.0                                        # level of Figs. 1 and 2
SEEDS = range(1000, 1100)
FRESH_SEEDS = range(4000, 4100)   # untouched by the design search (seeds 2000-2029, 3000-3099)
FAST_CORES = {0, 1, 2, 3}     # the four Zen 5 cores; the other eight are slower Zen 5c


def summary(results: list) -> pd.DataFrame:
    """One row per episode, with the regret against the controller that knows
    the wind on the same seed. G is the goal cost on the true position, and
    G_fix the goal cost on the position fix."""
    runs = pd.DataFrame([dict(label=r["label"], imu_noise=r["imu_noise"], seed=r["seed"], G=r["G_true"],
                              G_fix=r["G"], miss=r["miss"],
                              speed=r["speed"], wall=r["wall"], ms_per_step=1e3 * r["wall"] / len(r["u"]),
                              active_steps=int((r["multiplier"] > 0.0).sum()), plans=int(r["plans"].sum()),
                              timed=bool(r["imu_noise"] == NOMINAL)) for r in results])
    known = runs[runs.label == A.REFERENCE].set_index(["imu_noise", "seed"]).G
    runs["regret"] = runs.G.values - known.loc[list(zip(runs.imu_noise, runs.seed))].values
    return runs


def levels(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard error of the goal cost, the regret and the final miss
    per controller and noise level."""
    se = lambda g: g.std(ddof=1) / np.sqrt(len(g))  # noqa: E731
    out = runs.groupby(["label", "imu_noise"]).agg(
        G=("G", "mean"), G_se=("G", se), regret=("regret", "mean"), regret_se=("regret", se),
        miss=("miss", "mean"), active_share=("active_steps", "mean"), runs=("G", "size")).reset_index()
    out["active_share"] /= A.Benchmark().T
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fresh", action="store_true", help="fly the seeds 4000-4099 instead of 1000-1099")
    args = parser.parse_args()
    seeds, tag = (FRESH_SEEDS, "_fresh") if args.fresh else (SEEDS, "")
    jobs = lambda noises: [dict(label=label, bench=A.Benchmark(imu_noise=noise), seed=seed)  # noqa: E731
                           for label in A.CONTROLLERS for noise in noises for seed in seeds]
    start = time.perf_counter()
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, FAST_CORES)
    timed = A.run_many(jobs([NOMINAL]), processes=len(FAST_CORES))
    os.sched_setaffinity(0, affinity)
    rest = A.run_many(jobs([n for n in NOISES if n != NOMINAL]), processes=12)
    results = timed + rest
    print(f"{len(results)} episodes in {time.perf_counter() - start:.0f} s")
    print(A.save(results, A.DATA / f"episodes{tag}.pkl"))
    runs = summary(results)
    A.RESULTS.mkdir(parents=True, exist_ok=True)
    runs.to_csv(A.RESULTS / f"runs{tag}.csv", index=False)
    table = levels(runs)
    table.to_csv(A.RESULTS / f"levels{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
