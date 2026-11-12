"""Run the planar positioning benchmark of ``planar.py``.

Every controller reaches a target at distance 0.5 within S_* = 0.01 I over
T = 40 steps, for the disturbance levels SIGMAS. The probing weight of the
dual MPC and the accuracy of MPC-X are chosen on held-out seeds (2000-2029)
by the mean RMSE over all disturbance levels, as in Section 5 of main.tex;
the best value per level is reported as well. The test seeds are 1000-1099.

    python experiment-planar/run.py tune    # results/tuning.pkl
    python experiment-planar/run.py test    # results/test.pkl

Both runs take a few minutes on 24 cores.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import planar as PL  # noqa: E402

SIGMAS = (0.01, 0.03, 0.1, 0.3)
RHOS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
GAMMAS = (0.01, 0.1, 1.0)
TUNING_SEEDS = range(2000, 2030)
TEST_SEEDS = range(1000, 1100)


def dual_label(rho: float) -> str:
    return f"dual MPC, rho = {rho:g}"


def mpcx_label(gamma: float) -> str:
    return f"MPC-X, gamma = {gamma:g}"


def controllers() -> dict:
    out = {"CBFE": {"kind": "cbfe", "fixed": None},
           "cautious": {"kind": "cbfe", "fixed": 0.0},
           "oracle": {"kind": "oracle"}}
    out.update({mpcx_label(g): {"kind": "mpcx", "gamma": g} for g in GAMMAS})
    out.update({dual_label(r): {"kind": "dual", "rho": r} for r in RHOS})
    return out


def run(seeds, processes=None) -> list:
    jobs = [dict(label=label, spec=spec, bench=PL.Benchmark(sigma=sigma), seed=seed)
            for label, spec in controllers().items() for sigma in SIGMAS for seed in seeds]
    # slow controllers first, so that the pool finishes evenly
    order = sorted(range(len(jobs)), key=lambda i: jobs[i]["spec"]["kind"] != "mpcx")
    start = time.perf_counter()
    out = PL.run_many([jobs[i] for i in order], processes=processes)
    print(f"{len(jobs)} episodes in {time.perf_counter() - start:.0f} s", flush=True)
    results = [None] * len(jobs)
    for i, r in zip(order, out):
        results[i] = r
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("tune", "test"))
    parser.add_argument("--processes", type=int, default=None)
    args = parser.parse_args()
    seeds, name = (TUNING_SEEDS, "tuning") if args.mode == "tune" else (TEST_SEEDS, "test")
    print(PL.save(run(seeds, processes=args.processes), name))


if __name__ == "__main__":
    main()
