"""Pilot of the light-dark experiment: 100 seeds per goal, the main controllers."""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import lightdark as L

SPECS = {"CBFE": {"kind": "agent", "planner": "cbfe"},
         "cautious": {"kind": "agent", "planner": "none"},
         "BSP": {"kind": "agent", "planner": "bsp"},
         **{f"dual {r:g}": {"kind": "agent", "planner": "dual", "weight": r / 2} for r in (1, 3, 10, 30, 100)},
         "DP": {"kind": "dp"}, "oracle": {"kind": "oracle"}}

if __name__ == "__main__":
    bench = L.Benchmark()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    jobs = [dict(label=l, spec=s, bench=bench, goal=g, seed=5000 + i) for l, s in SPECS.items() for g in L.GOALS for i in range(n)]
    t = time.perf_counter(); res = L.run_many(jobs); print(f"{len(jobs)} episodes in {time.perf_counter()-t:.0f} s")
    L.save(res, "pilot")
    df = pd.DataFrame([L.metrics(r, bench) for r in res])
    pd.set_option("display.width", 220)
    print(df.pivot_table(index="label", columns="goal", values=["nll", "rmse", "final", "inside", "light", "active"], aggfunc="mean").reindex(list(SPECS)).round(3).to_string())
