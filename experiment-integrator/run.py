"""Run the integrator benchmark of ``integrator.py``.

Every controller regulates the integrator for T = 50 steps from y_0 = 0,
with the gain drawn from its prior N(b0, 1) per seed, for the prior means
b0 in B0. Small b0 = beta0 is the regime where the cautious controller
turns off. The probing weight of the dual MPC and the accuracy of MPC-X are
chosen on held-out seeds (1000-1049) by the mean loss over all priors:

    python experiment-integrator/run.py tune    # writes results/tuning.pkl
    python experiment-integrator/run.py         # writes results/benchmark.pkl

``--only LABEL ...`` reruns the named controllers and merges them into the
saved results. ``--dither STD`` adds a known Gaussian perturbation of that
standard deviation to every applied input (the same for every controller)
and writes results/<name>_dither.pkl; it checks that no result depends on
whether a planner returns exact zeros at a symmetric point. The dual MPC
keeps the plan at rest whenever the solver's plan improves on it by less
than REST_TOLERANCE (1 + |J|), as the CBFE-ARX message planner keeps its
warm start on ties. The first run solves the dynamic program of the optimal dual
controller (about three minutes on 24 cores) and caches it in
results/dual_dp_T50.npz.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import integrator as I  # noqa: E402

B0 = (0.0, 0.5, 1.0, 2.0, 4.0)
SEEDS = range(200)
TUNING_SEEDS = range(1000, 1050)
LAMBDAS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
GAMMAS = (0.01, 0.03, 0.1, 0.3, 1.0)
# chosen with ``run.py tune``; LAMBDA was 2 before the dual MPC kept the plan at rest
# within REST_TOLERANCE (2026-11-07), because round-off moved it off rest at b0 = 0
LAMBDA = 5.0
GAMMA = 0.1
REST_TOLERANCE = 1e-7     # the stopping tolerance of the message planner


# The dual MPC with the control prior as its input cost minimises twice the
# negative log-posterior of CBFE-ARX with the multiplier λ / 2 on every node.
LAMBDA_BETA = 5.0         # chosen with ``run.py tune`` (2026-11-07)
# The CBFE-ARX agent of the paper: the information constraint on the node of
# the next input, solved on the plan, with the predictive floor; the cap of
# the multiplier is far above every root.
CBFE = {"kind": "cbfe", "fixed_beta": None, "info_floor": "predictive",
        "constraint_nodes": "next", "max_multiplier": 1e6}


def dual(weight: float, input_cost: str = "quadratic") -> dict:
    return {"kind": "dual", "lambda_dual": weight, "rest_tolerance": REST_TOLERANCE,
            "input_cost": input_cost}


def dual_beta_label(weight: float) -> str:
    return f"dual MPC, Beta cost, λ = {weight:g}"


def controllers() -> dict:
    return {
        "certainty equivalence": {"kind": "ce"},
        "cautious": {"kind": "cautious"},
        "optimal dual (DP)": {"kind": "dual_dp"},
        "approximate dual (5.1)": {"kind": "dual_approx"},
        "CBFE-ARX, β = 0": {"kind": "cbfe", "fixed_beta": 0.0},
        "CBFE-ARX, β = 1 (EFE)": {"kind": "cbfe", "fixed_beta": 1.0},
        "CBFE-ARX, KKT β": {"kind": "cbfe", "fixed_beta": None, "info_floor": "task",
                            "constraint_nodes": "all"},
        "CBFE-ARX, KKT β, floor −H[p*]": {"kind": "cbfe", "fixed_beta": None, "info_floor": "negentropy"},
        "CBFE-ARX, KKT β, predictive floor, next node": CBFE,
        "CBFE-ARX, KKT β, next node": {"kind": "cbfe", "fixed_beta": None, "info_floor": "task",
                                       "constraint_nodes": "next"},
        "CBFE-ARX, KKT β, predictive floor": {"kind": "cbfe", "fixed_beta": None,
                                              "info_floor": "predictive", "constraint_nodes": "all",
                                              "max_multiplier": 1e6},
        f"dual MPC, λ = {LAMBDA:g}": dual(LAMBDA),
        # λ = 2 has the cost of the CBFE-ARX agent with β = 1 (EFE), up to the input cost
        "dual MPC, λ = 2": dual(2.0),
        f"MPC-X, γ = {GAMMA:g}": {"kind": "mpcx", "gamma": GAMMA},
        # the probing-weight curve of the dual MPC with the Beta input cost
        **{dual_beta_label(v): dual(v, "beta") for v in LAMBDAS},
    }


def tuning_candidates() -> dict:
    out = {f"dual MPC, λ = {v:g}": dual(v) for v in LAMBDAS}
    out.update({dual_beta_label(v): dual(v, "beta") for v in LAMBDAS})
    out.update({f"MPC-X, γ = {v:g}": {"kind": "mpcx", "gamma": v} for v in GAMMAS})
    return out


def run(specs: dict, seeds, bench: I.Benchmark = I.Benchmark(), processes=None) -> list:
    jobs = [dict(label=label, spec=spec, bench=replace(bench, b0=b0), seed=seed)
            for label, spec in specs.items() for b0 in B0 for seed in seeds]
    # expensive controllers first, so that the pool finishes evenly
    order = sorted(range(len(jobs)), key=lambda i: jobs[i]["spec"]["kind"] not in ("mpcx", "cbfe", "dual_dp"))
    t0 = time.perf_counter()
    out = I.run_many([jobs[i] for i in order], processes=processes)
    print(f"{len(jobs)} episodes in {time.perf_counter() - t0:.0f} s", flush=True)
    results = [None] * len(jobs)
    for i, r in zip(order, out):
        results[i] = r
    return results


def summary(results) -> str:
    import pandas as pd
    df = pd.DataFrame([dict(label=r["label"], b0=r["b0"], loss=r["loss"]) for r in results])
    table = df.pivot_table(index="label", columns="b0", values="loss", aggfunc="mean", sort=False)
    table["all"] = df.groupby("label", sort=False).loss.mean()
    return table.round(1).to_string()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", nargs="?", default="benchmark", choices=("benchmark", "tune"))
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--dither", type=float, default=0.0, metavar="STD",
                        help="known input perturbation of every controller")
    parser.add_argument("--only", nargs="+", default=None, metavar="LABEL",
                        help="rerun these controllers and merge them into the saved results")
    args = parser.parse_args()
    I.dual_dp(I.Benchmark().T, verbose=True)
    name, specs, seeds = (("tuning", tuning_candidates(), TUNING_SEEDS) if args.mode == "tune"
                          else ("benchmark", controllers(), SEEDS))
    if args.only:
        unknown = set(args.only) - set(specs)
        if unknown:
            parser.error(f"unknown controllers {sorted(unknown)}; choose from {list(specs)}")
        specs = {label: spec for label, spec in specs.items() if label in args.only}
    bench = I.Benchmark()
    if args.dither > 0.0:
        bench, name = replace(bench, dither=args.dither), f"{name}_dither"
    results = run(specs, seeds, bench=bench, processes=args.processes)
    if args.only:
        results = [r for r in I.load(name) if r["label"] not in specs] + results
    print(I.save(results, name))
    print(summary(results))


if __name__ == "__main__":
    main()
