"""Tuning transfer in the light-dark experiment: the dual MPC over a grid of
probing weights rho and MPC-X over a grid of accuracies kappa, at every light
noise of ``run_baselines.LIGHTS``, together with the untuned CBFE, on the
held-out tuning seeds. It shows whether the parameter that is best at one
light noise stays best at the others.

    python experiment-lightdark/run_transfer.py    # results/transfer.pkl
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lightdark as L  # noqa: E402
import run_baselines as R  # noqa: E402

RHOS = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
KAPPAS = (0.03, 0.1, 0.3, 1.0, 3.0)


def main():
    start = time.perf_counter()
    specs = {"CBFE": {"kind": "agent", "planner": "cbfe"},
             **{f"dual MPC, rho = {r:g}": R.dual(r) for r in RHOS},
             **{f"MPC-X, kappa = {k:g}": R.mpcx(k) for k in KAPPAS}}
    results = [r for light in R.LIGHTS for r in R.run(specs, R.TUNING_SEEDS, light, processes=12)]
    print(L.save(results, "transfer"), f"{time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
