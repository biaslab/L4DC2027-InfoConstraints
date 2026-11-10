"""Runs of the revision of 2026-11-07 (reviews/plan_paper_2026-11-07_1750.md):

1. the CBFE-ARX agent with the predictive floor and the constraint on the
   node of the next input, and each change on its own, on the test seeds;
2. the dual MPC with the control prior as its input cost over the probing
   weights ``compare.DUAL_WEIGHTS``, on the test seeds (probing-weight
   curve) and on the held-out seeds 100-105 (tuning), under a known and an
   unknown polarity;
3. with ``--mpcx``, MPC-X over the accuracies ``compare.MPCX_ACCURACIES``,
   on the held-out seeds (tuning) and the test seeds.

    python experiment-arm/run_revision.py [--processes 10] [--mpcx]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

import compare as C  # noqa: E402

CONDITIONS = ["no change", "unknown polarity"]
CBFE = ["CBFE-ARX, KKT β, predictive floor, next node", "CBFE-ARX, KKT β, next node",
        "CBFE-ARX, KKT β, predictive floor"]
DUAL = [f"dual MPC, Beta cost, λ = {weight:g}" for weight in C.DUAL_WEIGHTS]
MPCX = [f"MPC-X, γ = {accuracy:g}" for accuracy in C.MPCX_ACCURACIES]
TUNING = C.HERE / "results" / "compare_tuning"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processes", type=int, default=10)
    parser.add_argument("--mpcx", action="store_true",
                        help="only the accuracies of MPC-X, on the held-out and the test seeds")
    args = parser.parse_args()
    if args.mpcx:
        C.run(MPCX, seeds=range(100, 106), processes=args.processes, out=TUNING, conds=CONDITIONS)
        new = [label for label in MPCX if label not in ("MPC-X, γ = 0.0004", "MPC-X, γ = 0.003")]
        C.run(new, seeds=range(10), processes=args.processes, conds=CONDITIONS)
        return
    C.run(CBFE, seeds=range(10), processes=args.processes, conds=CONDITIONS)
    C.run(DUAL, seeds=range(100, 106), processes=args.processes, out=TUNING, conds=CONDITIONS)
    C.run(DUAL, seeds=range(10), processes=args.processes, conds=CONDITIONS)


if __name__ == "__main__":
    main()
