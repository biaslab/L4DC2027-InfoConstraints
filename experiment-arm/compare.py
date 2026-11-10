"""Comparison of all agents on the arm.

Every agent runs the reaching protocol of ``reaching.py`` for four rounds
of the four targets (48 s), under three conditions: the payload stays at
0.5 kg; it rises to 1.5 kg after the first round; or the motor polarities
are unknown (``Settings.polarity_known = False``), i.e. each motor may be
reversed, with the polarity pair of the seed (``reaching.POLARITIES``). Ten
plant noise seeds per agent and condition. All agents see the joint
encoders, act through the same torque limit every 0.1 s, and take their
prior knowledge from the nominal arm (the uncertain parameters at their
prior means). Under unknown polarity, the ARX agents forget the
coefficients that change sign with a motor (``reaching.forget_polarity``);
PILCO, TD-MPC2 and the PD law keep the nominal polarity, because their
priors cannot express an unknown sign:

``CBFE-ARX``  ARX prior from 600 nominal steps at weight 0.15; β = 0 or KKT
              with the task floor of ``Settings``; the planner of ``Settings``
              (backward messages)
``dual MPC``  the same model and prior; λ = 0 was the best probing weight on
              held-out seeds (100-105), λ = 100 matches the KKT β = 50 of the
              earlier floor -H[p_*]
``MPC-X``     MPC with integrated experiment design (Larsson et al., ECC
              2013) on the same model and prior; γ = 4e-4 was the best
              accuracy on held-out seeds (100-105), γ = 3e-3 switches the
              requirement off at the model accuracy where the task floor does
``PILCO``     90 nominal transitions (the effective size of the ARX prior);
              0.5 s rollouts, refits every second, anchor weight 3, chosen
              on held-out seeds
``TD-MPC2``   pretrained on the nominal arm (``train_tdmpc2.py``), keeps
              learning online; one entry per checkpoint, plus the largest
              checkpoint with frozen weights
``PD``        the fixed PD law of the nominal excitation, no learning

Results go to one pickle per agent in ``results/compare``, so that agents
and conditions can be added without rerunning the others. ``results/compare/sweeps_planner``
keeps the CBFE-ARX episodes of the earlier one-step planner, which ``load``
does not read; ``results/compare/negentropy_floor`` keeps the KKT episodes
of the earlier floor -H[p_*]. TD-MPC2 needs torch:

    python experiment-arm/compare.py                      # all agents
    python experiment-arm/compare.py --agents PD "dual MPC, λ = 0"
    python experiment-arm/compare.py --conditions "unknown polarity"
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

import numpy as np  # noqa: E402

import reaching as R  # noqa: E402

RESULTS = HERE / "results" / "compare"
CHECKPOINTS = HERE / "results" / "tdmpc2"
NEW_PAYLOAD = 1.5
ROUNDS = 4
# probing weights of the dual MPC with the control prior as its input cost,
# whose cost is then twice the negative log-posterior of CBFE-ARX with the
# multiplier λ / 2 on every node
# multiplier λ / 2 on every node. By eq. (26) of the paper it probes only for
# λ > 2 tr(S_*⁻¹ R) / D_y, about 115 on the arm, hence the wide grid.
DUAL_WEIGHTS = (0.0, 10.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e5)
# accuracies of MPC-X; 4e-4 was the best on held-out seeds of the payload rise
MPCX_ACCURACIES = (1e-5, 3e-5, 1e-4, 4e-4, 1e-3, 3e-3, 1e-2)


def agent_specs() -> dict:
    specs = {
        "CBFE-ARX, β = 0": {"kind": "cbfe", "fixed_beta": 0.0},
        "CBFE-ARX, KKT β": {"kind": "cbfe", "fixed_beta": None},
        "CBFE-ARX, β = 1 (EFE)": {"kind": "cbfe", "fixed_beta": 1.0},
        # the information constraint on the node of the next input (S2) with
        # the predictive floor (S1), and each change on its own
        "CBFE-ARX, KKT β, predictive floor, next node": {
            "kind": "cbfe", "fixed_beta": None, "info_floor": "predictive",
            "constraint_nodes": "next", "beta_max": 1e6},
        "CBFE-ARX, KKT β, next node": {"kind": "cbfe", "fixed_beta": None,
                                       "constraint_nodes": "next"},
        "CBFE-ARX, KKT β, predictive floor": {"kind": "cbfe", "fixed_beta": None,
                                              "info_floor": "predictive", "beta_max": 1e6},
        "dual MPC, λ = 0": {"kind": "dual", "lambda_dual": 0.0},
        "dual MPC, λ = 100": {"kind": "dual", "lambda_dual": 100.0},
        **{f"dual MPC, Beta cost, λ = {weight:g}": {
            "kind": "dual", "lambda_dual": weight, "input_cost": "beta", "rest_tolerance": 1e-7}
           for weight in DUAL_WEIGHTS},
        **{f"MPC-X, γ = {accuracy:g}": {"kind": "mpcx", "gamma": accuracy}
           for accuracy in MPCX_ACCURACIES},
        "PILCO": {"kind": "pilco", "horizon": 5, "refit_every": 10, "theta_anchor": 3.0},
        "PD": {"kind": "pd"},
    }
    paths = sorted(CHECKPOINTS.glob("tdmpc2_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for path in paths:
        steps = int(path.stem.split("_")[1])
        specs[f"TD-MPC2, {steps // 1000}k steps"] = {"kind": "tdmpc2", "checkpoint": str(path)}
    if paths:
        steps = int(paths[-1].stem.split("_")[1])
        specs[f"TD-MPC2, {steps // 1000}k steps, frozen"] = {
            "kind": "tdmpc2", "checkpoint": str(paths[-1]), "learn_online": False}
    return specs


def conditions(s: R.Settings) -> dict:
    """Condition -> (targets, payload changes, settings of the episode)."""
    targets, rise = R.payload_change(NEW_PAYLOAD, s, rounds=ROUNDS)
    return {"no change": (targets, None, s),
            f"rise to {NEW_PAYLOAD} kg": (targets, rise, s),
            "unknown polarity": (targets, None, replace(s, polarity_known=False))}


def slug(label: str) -> str:
    label = label.lower().replace("β", "beta").replace("λ", "lambda").replace("γ", "gamma")
    return re.sub(r"[^a-z0-9]+", "_", label).strip("_")


def run(labels, seeds=range(10), processes=None, out=RESULTS, s: R.Settings = R.Settings(),
        conds=None):
    """Run the agents with the given labels under the conditions ``conds``
    (default: all) and save one pickle per agent, which keeps the records of
    the other conditions."""
    specs, all_conds = agent_specs(), conditions(s)
    conds = {c: all_conds[c] for c in (conds or all_conds)}
    jobs, keys = [], []
    for label in labels:
        for cond, (targets, changes, s_cond) in conds.items():
            for seed in seeds:
                jobs.append(dict(agent=specs[label], seed=seed, s=s_cond, targets=targets, changes=changes))
                keys.append((label, cond, seed))
    # expensive agents first, so that the pool finishes evenly
    cost = {"pilco": 0, "tdmpc2": 1, "mpcx": 2, "dual": 2, "cbfe": 3, "pd": 4}
    order = sorted(range(len(jobs)), key=lambda i: cost[jobs[i]["agent"]["kind"]])
    t0 = time.perf_counter()
    episodes = R.run_many([jobs[i] for i in order], processes=processes)
    print(f"{len(jobs)} episodes in {time.perf_counter() - t0:.0f} s", flush=True)
    by_key = {keys[i]: ep for i, ep in zip(order, episodes)}
    out.mkdir(parents=True, exist_ok=True)
    for label in labels:
        path = out / f"{slug(label)}.pkl"
        records = []
        if path.exists():
            with open(path, "rb") as f:
                records = [rec for rec in pickle.load(f)["records"] if rec["condition"] not in conds]
        records += [dict(agent=label, condition=c, seed=sd, episode=by_key[(label, c, sd)])
                    for (lb, c, sd) in keys if lb == label]
        with open(path, "wb") as f:
            pickle.dump({"label": label, "spec": specs[label], "settings": s, "records": records}, f)


def load(out=RESULTS) -> list:
    """All saved records: dicts with agent, condition, seed, episode and the
    settings of the run."""
    records = []
    for path in sorted(out.glob("*.pkl")):
        with open(path, "rb") as f:
            data = pickle.load(f)
        records.extend(dict(rec, settings=data["settings"]) for rec in data["records"])
    return records


def per_round(records, s: R.Settings = R.Settings()):
    """One row per agent, condition, seed and round, with the scores of
    ``reaching.summarise`` and the mean planning time per step. Records
    that carry their settings are scored with those, others with ``s``."""
    import pandas as pd
    rows = []
    for rec in records:
        ep = rec["episode"]
        s = rec.get("settings", s)
        n = len(R.TARGETS) * s.steps_per_target
        for r in range(len(ep.actions) // n):
            seg = slice(len(R.TARGETS) * r, len(R.TARGETS) * (r + 1))
            rows.append({"agent": rec["agent"], "condition": rec["condition"], "seed": rec["seed"],
                         "round": r + 1, "plan_time": float(ep.plan_time[r * n:(r + 1) * n].mean()),
                         **R.summarise(ep, s, seg)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agents", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--conditions", nargs="*", default=None)
    args = parser.parse_args()
    labels = args.agents or list(agent_specs())
    unknown = [lb for lb in labels if lb not in agent_specs()]
    if unknown:
        raise SystemExit(f"unknown agents {unknown}; choose from {list(agent_specs())}")
    names = list(conditions(R.Settings()))
    if args.conditions and set(args.conditions) - set(names):
        raise SystemExit(f"unknown conditions {args.conditions}; choose from {names}")
    run(labels, seeds=range(args.seeds), processes=args.processes, conds=args.conditions)


if __name__ == "__main__":
    main()
