"""Pilot of the light-dark experiment with an unknown actuator gain."""
import sys, time
from dataclasses import replace
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import lightdark as L

SPECS = {"CBFE": {"kind": "agent", "planner": "cbfe"},
         "CBFE, state only": {"kind": "agent", "planner": "cbfe", "constraints": ("state",)},
         "CBFE, parameter only": {"kind": "agent", "planner": "cbfe", "constraints": ("parameter",)},
         "cautious": {"kind": "agent", "planner": "none"},
         "BSP": {"kind": "agent", "planner": "bsp"},
         **{f"dual {r:g}": {"kind": "agent", "planner": "dual", "weights": (r / 2, r / 2)} for r in (3, 10, 30, 100, 300)},
         "oracle": {"kind": "oracle"}}

if __name__ == "__main__":
    bench = replace(L.Benchmark(), gain_variance=0.09)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    jobs = [dict(label=l, spec=s, bench=bench, goal=g, seed=5000 + i) for l, s in SPECS.items() for g in L.GOALS for i in range(n)]
    jobs.sort(key=lambda j: j["spec"]["planner"] not in ("cbfe", "bsp") if j["spec"]["kind"] == "agent" else True)
    t = time.perf_counter(); res = L.run_many(jobs); print(f"{len(jobs)} episodes in {time.perf_counter()-t:.0f} s")
    L.save(res, "pilot_gain")
    df = pd.DataFrame([L.metrics(r, bench) for r in res])
    pd.set_option("display.width", 250)
    print(df.pivot_table(index="label", columns="goal", values=["nll", "rmse", "final", "inside", "light", "active", "parameter_active", "final_P"], aggfunc="mean").reindex(list(SPECS)).round(3).to_string())
