"""Pilot of option 1: start at the light with a known state and an unknown gain,
the target in the dark (one-sided light, T = 60)."""
import sys, time
from dataclasses import replace
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import lightdark as L
from agents.cbfe_lightdark import LightDarkModel
from pilot_fix import trips

SPECS = {"CBFE": {"kind": "agent", "planner": "cbfe"},
         "CBFE, state only": {"kind": "agent", "planner": "cbfe", "constraints": ("state",)},
         "CBFE, parameter only": {"kind": "agent", "planner": "cbfe", "constraints": ("parameter",)},
         "cautious": {"kind": "agent", "planner": "none"},
         "BSP": {"kind": "agent", "planner": "bsp"},
         **{f"dual {r:g}": {"kind": "agent", "planner": "dual", "weights": (r / 2, r / 2)} for r in (3, 10, 30, 100)},
         "oracle": {"kind": "oracle"}}

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    bench = replace(L.Benchmark(model=LightDarkModel(one_sided=True)), T=60, prior_mean=1.5,
                    prior_variance=0.0025, gain_variance=0.09)
    jobs = [dict(label=l, spec=s, bench=bench, goal=g, seed=5000 + i) for l, s in SPECS.items() for g in L.GOALS for i in range(n)]
    t = time.perf_counter(); res = L.run_many(jobs); print(f"{len(jobs)} episodes in {time.perf_counter()-t:.0f} s")
    L.save(res, "pilot_beacon")
    df = pd.DataFrame([dict(L.metrics(r, bench), trips=trips(r["episode"])) for r in res])
    pd.set_option("display.width", 250)
    print(df.pivot_table(index="label", columns="goal", values=["nll", "rmse", "final", "inside", "trips", "parameter_active", "active", "final_P"],
                         aggfunc="mean").reindex(list(SPECS)).round(3).to_string())
