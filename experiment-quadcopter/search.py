"""Logged search for a quadcopter configuration in which the information
constraint binds and helps:

    python experiment-quadcopter/search.py '[["name", {"goal_std": 0.03, ...}], ...]'

Every configuration overrides fields of ``ablation.Benchmark`` and runs CBFE
and the controller without the constraint on the search seeds 2000-2029.
results/search_log.jsonl keeps every trial. The selected configurations were
then validated on the fresh seeds 3000-3099, and run.py uses the seeds
1000-1099, which the search never saw."""
import dataclasses, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import ablation as A

LOG = Path(__file__).resolve().parent / "results" / "search_log.jsonl"

def evaluate(name, overrides, seeds, labels=("CBFE", "no constraint", "certainty equivalence")):
    bench = dataclasses.replace(A.Benchmark(), **overrides)
    jobs = [dict(label=l, bench=bench, seed=s) for l in labels for s in seeds]
    t0 = time.perf_counter()
    res = A.run_many(jobs, processes=12)
    by = {l: sorted([r for r in res if r["label"] == l], key=lambda r: r["seed"]) for l in labels}
    out = dict(name=name, overrides=overrides, seeds=[seeds[0], seeds[-1]], seconds=round(time.perf_counter() - t0))
    rng = np.random.default_rng(0)
    for l in labels:
        out[l] = dict(G=float(np.mean([r["G"] for r in by[l]])), G_true=float(np.mean([r["G_true"] for r in by[l]])),
                      miss=float(np.mean([r["miss"] for r in by[l]])),
                      active=float(np.mean([(r["multiplier"] > 0).mean() for r in by[l]])))
    for metric in ("G", "G_true"):
        d = np.array([a[metric] - b[metric] for a, b in zip(by["no constraint"], by["CBFE"])])
        m = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        out[f"gain_{metric}"] = [float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]
    with open(LOG, "a") as f:
        f.write(json.dumps(out) + "\n")
    d = np.array([a["G_true"] - b["G_true"] for a, b in zip(by["no constraint"], by["CBFE"])])
    out["median_gain_true"], out["win_rate"] = float(np.median(d)), float(np.mean(d > 0))
    out["relative_gain_true"] = float(d.mean() / np.mean([r["G_true"] for r in by["no constraint"]]))
    with open(LOG, "a") as f:
        f.write(json.dumps(dict(name=name, extra=dict(median=out["median_gain_true"], win=out["win_rate"]))) + "\n")
    if "certainty equivalence" in by:
        d = np.array([a["G_true"] - b["G_true"] for a, b in zip(by["certainty equivalence"], by["CBFE"])])
        m = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        out["gain_vs_ce"] = [float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]
        out["win_vs_ce"] = float(np.mean(d > 0))
        out["relative_vs_ce"] = float(d.mean() / np.mean([r["G_true"] for r in by["certainty equivalence"]]))
        with open(LOG, "a") as f:
            f.write(json.dumps(dict(name=name, vs_ce=out["gain_vs_ce"], win_vs_ce=out["win_vs_ce"])) + "\n")
    g, c = out["gain_G_true"], out.get("gain_vs_ce", [np.nan] * 3)
    print(f"{name:40s} active {100*out['CBFE']['active']:4.1f}% | CBFE {out['CBFE']['G_true']:8.0f} | "
          f"vs no-constraint {g[0]:+7.0f} [{g[1]:+.0f}, {g[2]:+.0f}] ({100*out['relative_gain_true']:+.0f}%, wins {100*out['win_rate']:.0f}%) | "
          f"vs CE {c[0]:+7.0f} [{c[1]:+.0f}, {c[2]:+.0f}] ({100*out.get('relative_vs_ce', np.nan):+.0f}%, wins {100*out.get('win_vs_ce', np.nan):.0f}%) "
          f"| {out['seconds']} s", flush=True)
    return out

if __name__ == "__main__":
    seeds = list(range(2000, 2030))
    configs = json.loads(sys.argv[1])
    for name, overrides in configs:
        evaluate(name, overrides, seeds)
