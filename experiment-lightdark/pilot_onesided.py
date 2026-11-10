"""Pilot: one-sided light, T = 60, unknown gain (and known gain for reference)."""
import sys, time
from dataclasses import replace
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import lightdark as L
from agents.cbfe_lightdark import LightDarkModel
from pilot_horizon import KNOWN, UNKNOWN

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    model = LightDarkModel(one_sided=True)
    known = replace(L.Benchmark(model=model), T=60)
    unknown = replace(known, gain_variance=0.09)
    jobs = [dict(label=l, spec=s, bench=b, goal=g, seed=5000 + i, condition=c)
            for c, b, specs in (("known", known, KNOWN), ("unknown", unknown, UNKNOWN))
            for l, s in specs.items() for g in L.GOALS for i in range(n)]
    jobs.sort(key=lambda j: j["condition"] != "unknown")
    t = time.perf_counter(); res = L.run_many(jobs); print(f"{len(jobs)} episodes in {time.perf_counter()-t:.0f} s")
    for r, j in zip(res, jobs): r["condition"] = j["condition"]
    L.save(res, "pilot_onesided")
    df = pd.DataFrame([dict(L.metrics(r, known), condition=r["condition"]) for r in res])
    pd.set_option("display.width", 250)
    for c, specs in (("known", KNOWN), ("unknown", UNKNOWN)):
        print(f"\n{c} gain, one-sided light, T = 60")
        print(df[df.condition == c].pivot_table(index="label", columns="goal", values=["nll", "rmse", "final", "inside", "light"], aggfunc="mean").reindex(list(specs)).round(3).to_string())
        sub = df[(df.condition == c) & (df.label == "CBFE")]
        print("CBFE median nll", sub.groupby("goal").nll.median().round(3).to_dict(), "max", sub.groupby("goal").nll.max().round(1).to_dict())
