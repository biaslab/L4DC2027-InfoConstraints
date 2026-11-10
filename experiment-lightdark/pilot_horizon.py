"""Pilot: episode length T = 60 with known and unknown gain."""
import sys, time
from dataclasses import replace
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import lightdark as L
from pilot_gain import SPECS as GAIN_SPECS

KNOWN = {"CBFE": {"kind": "agent", "planner": "cbfe"}, "cautious": {"kind": "agent", "planner": "none"},
         "BSP": {"kind": "agent", "planner": "bsp"},
         **{f"dual {r:g}": {"kind": "agent", "planner": "dual", "weight": r / 2} for r in (10, 30, 100)},
         "DP": {"kind": "dp"}, "oracle": {"kind": "oracle"}}
UNKNOWN = {k: v for k, v in GAIN_SPECS.items() if k not in ("dual 300",)}

if __name__ == "__main__":
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    known, unknown = replace(L.Benchmark(), T=T), replace(L.Benchmark(), T=T, gain_variance=0.09)
    jobs = [dict(label=l, spec=s, bench=b, goal=g, seed=5000 + i, condition=c)
            for c, b, specs in (("known", known, KNOWN), ("unknown", unknown, UNKNOWN))
            for l, s in specs.items() for g in L.GOALS for i in range(n)]
    jobs.sort(key=lambda j: j["condition"] != "unknown")
    t = time.perf_counter(); res = L.run_many(jobs); print(f"{len(jobs)} episodes in {time.perf_counter()-t:.0f} s")
    for r, j in zip(res, jobs): r["condition"] = j["condition"]
    L.save(res, f"pilot_T{T}")
    rows = [dict(L.metrics(r, known), condition=r["condition"]) for r in res]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    for c, specs in (("known", KNOWN), ("unknown", UNKNOWN)):
        print(f"\n{c} gain, T = {T}")
        print(df[df.condition == c].pivot_table(index="label", columns="goal", values=["nll", "rmse", "final", "inside", "light"], aggfunc="mean").reindex(list(specs)).round(3).to_string())
