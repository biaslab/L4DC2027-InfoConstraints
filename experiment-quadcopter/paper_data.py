"""Flights behind the quadcopter numbers and Fig. 2 of main.tex.

The paper reports the fresh seeds 4000-4099 of ``run.py --fresh`` and
``run_baselines.py test --fresh``, which played no role in the design. It
drops every seed on which the oracle, which knows the wind and the fault, ends
farther from the target than it started, at any IMU noise. On such a seed the
gusts exceed what any controller can hold, so the target is out of reach, and
the cost of the one flight dominates every mean. The rule drops seed 4076 of
the fresh seeds (oracle miss 13 m) and seed 1023 of the test seeds 1000-1099
(10 m); no other oracle flight ends more than 4.8 m from the target.

    python experiment-quadcopter/paper_data.py    # print the dropped seeds of both seed sets
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import ablation as A  # noqa: E402
import run as R  # noqa: E402

START = float(np.linalg.norm([5.0, 0.0, 1.0]))    # distance of every target from the start [m]


def load(fresh: bool) -> list:
    """All flights of ``run.py`` and of the tuned baselines on one seed set."""
    tag = "_fresh" if fresh else ""
    return A.load(A.DATA / f"episodes{tag}.pkl") + A.load(A.DATA / f"baselines_test{tag}.pkl")


def failed(flights: list) -> set:
    """Seeds on which the oracle ends farther from the target than it started."""
    return {r["seed"] for r in flights if r["label"] == A.REFERENCE and r["miss"] > START}


def seeds(fresh: bool, flights: list | None = None) -> list:
    """The seeds of the seed set without the seeds on which the oracle fails."""
    drop = failed(load(fresh) if flights is None else flights)
    return [seed for seed in (R.FRESH_SEEDS if fresh else R.SEEDS) if seed not in drop]


def flights(fresh: bool = True) -> list:
    """The flights of the paper: the seed set without the failed seeds."""
    every = load(fresh)
    keep = set(seeds(fresh, every))
    return [r for r in every if r["seed"] in keep]


def terms(flights: list) -> dict:
    """Per-step terms of G, eq. (metric) of main.tex, by (label, noise, seed):
    the goal cost on the true position plus half the precision-weighted
    prediction error of the filter. The flights of ``run.py`` on the test seeds
    predate the stored error, so ``free_energy.py`` replays it for them."""
    replay = {}
    if any("error" not in r for r in flights):
        with open(A.DATA / "free_energy.pkl", "rb") as f:
            replay = {(t["label"], t["imu_noise"], t["seed"]): t["error"] for t in pickle.load(f)}
    out = {}
    for r in flights:
        key = (r["label"], r["imu_noise"], r["seed"])
        out[key] = r["true_cost"] + 0.5 * (r["error"] if "error" in r else replay[key])
    return out


if __name__ == "__main__":
    for fresh in (False, True):
        every = load(fresh)
        miss = sorted(((r["miss"], r["seed"]) for r in every if r["label"] == A.REFERENCE), reverse=True)
        print(f"{'fresh 4000-4099' if fresh else 'test 1000-1099'}: drop {sorted(failed(every))}, "
              f"largest oracle misses {[(s, round(m, 1)) for m, s in miss[:6]]}")
