"""Pilot of option 2: direction-dependent gains b_+ (u >= 0) and b_- (u < 0),
start in the dark, one-sided light, T = 60."""
import sys, time
from dataclasses import replace
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import lightdark as L
from agents.cbfe_lightdark import LightDarkModel
from pilot_fix import trips
from pilot_beacon import SPECS

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    bench = replace(L.Benchmark(model=LightDarkModel(one_sided=True)), T=60, gain_variance=0.09, directional=True)
    jobs = [dict(label=l, spec=s, bench=bench, goal=g, seed=5000 + i) for l, s in SPECS.items() for g in L.GOALS for i in range(n)]
    jobs.sort(key=lambda j: j["spec"].get("planner") not in ("cbfe", "bsp"))
    t = time.perf_counter(); res = L.run_many(jobs); print(f"{len(jobs)} episodes in {time.perf_counter()-t:.0f} s")
    L.save(res, "pilot_directional")
    df = pd.DataFrame([dict(L.metrics(r, bench), trips=trips(r["episode"])) for r in res])
    pd.set_option("display.width", 250)
    print(df.pivot_table(index="label", columns="goal", values=["nll", "rmse", "final", "inside", "trips", "parameter_active", "active", "final_P"],
                         aggfunc="mean").reindex(list(SPECS)).round(3).to_string())
