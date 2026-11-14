"""Pilot of the floor fix: shared budget and end-of-task spread (one-sided light, T = 60)."""
import sys, time
from dataclasses import replace
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import lightdark as L
from controllers.cbfe_lightdark import LightDarkModel

OLD = {"task_floor": False, "shared_budget": False}
KNOWN = {"CBFE": {"kind": "agent", "planner": "cbfe"},
         "CBFE, old floor": {"kind": "agent", "planner": "cbfe", **OLD},
         "cautious": {"kind": "agent", "planner": "none"},
         "DP": {"kind": "dp"}, "oracle": {"kind": "oracle"}}
UNKNOWN = {"CBFE": {"kind": "agent", "planner": "cbfe"},
           "CBFE, old floor": {"kind": "agent", "planner": "cbfe", **OLD},
           "CBFE, mean-field belief": {"kind": "agent", "planner": "cbfe", "joint_belief": False},
           "CBFE, mean-field, old floor": {"kind": "agent", "planner": "cbfe", "joint_belief": False, **OLD},
           "CBFE, end-of-task only": {"kind": "agent", "planner": "cbfe", "shared_budget": False},
           "CBFE, shared budget only": {"kind": "agent", "planner": "cbfe", "task_floor": False},
           "CBFE, state only": {"kind": "agent", "planner": "cbfe", "constraints": ("state",)},
           "CBFE, parameter only": {"kind": "agent", "planner": "cbfe", "constraints": ("parameter",)},
           "cautious": {"kind": "agent", "planner": "none"},
           "dual 10": {"kind": "agent", "planner": "dual", "weights": (5.0, 5.0)},
           "oracle": {"kind": "oracle"}}


def trips(ep):
    inside = (ep.x[1:] >= 0.9).astype(int)
    return int(np.sum(np.diff(inside) == 1) + inside[0])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    known = replace(L.Benchmark(model=LightDarkModel(one_sided=True)), T=60)
    unknown = replace(known, gain_variance=0.09)
    jobs = [dict(label=l, spec=s, bench=b, goal=g, seed=5000 + i, condition=c)
            for c, b, specs in (("known", known, KNOWN), ("unknown", unknown, UNKNOWN))
            for l, s in specs.items() for g in L.GOALS for i in range(n)]
    jobs.sort(key=lambda j: j["condition"] != "unknown")
    t = time.perf_counter(); res = L.run_many(jobs); print(f"{len(jobs)} episodes in {time.perf_counter()-t:.0f} s")
    for r, j in zip(res, jobs): r["condition"] = j["condition"]
    L.save(res, "pilot_fix")
    df = pd.DataFrame([dict(L.metrics(r, known), condition=r["condition"], trips=trips(r["episode"])) for r in res])
    pd.set_option("display.width", 250)
    for c, specs in (("known", KNOWN), ("unknown", UNKNOWN)):
        print(f"\n{c} gain")
        sub = df[df.condition == c]
        table = sub.pivot_table(index="label", columns="goal", values=["nll", "rmse", "final", "inside", "trips"], aggfunc="mean")
        table[("nll_max", "tight")] = sub[sub.goal == "tight"].groupby("label").nll.max()
        print(table.reindex(list(specs)).round(3).to_string())
