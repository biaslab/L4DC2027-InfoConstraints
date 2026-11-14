"""Numbers of the quadcopter Results of main.tex and of supplement S2, from
the flights that ``paper_data.py`` selects: the seed set without the seeds on
which the oracle fails.

    python experiment-quadcopter/summary.py            # test seeds 1000-1099 -> results/summary.csv
    python experiment-quadcopter/summary.py --fresh    # fresh seeds 4000-4099 -> results/summary_fresh.csv

G of a flight is the mean over its steps of the goal cost on the true position
plus half the precision-weighted prediction error. Intervals are paired
bootstrap 95% intervals over the flights with 1e4 resamples and
default_rng(0), as in plot_baselines.interval. The learning cost is the cost
above that of the oracle on the same seed. The script also prints the
headline numbers with every seed, which supplement S2.5 reports.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ablation as A  # noqa: E402
import paper_data as D  # noqa: E402
import run as R  # noqa: E402

ARMS = {"info-CBFE": "CBFE", "no-info-CBFE": "no constraint", "dual MPC": "dual MPC", "MPC-X": "MPC-X",
        "oracle": A.REFERENCE, "MPC": "certainty equivalence"}
BASELINES = ("dual MPC", "MPC-X", "no-info-CBFE")
EARLY = 8                                    # steps in the first 4 s


def interval(d: np.ndarray) -> str:
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    return f"{d.mean():.0f} [{np.quantile(m, 0.025):.0f}, {np.quantile(m, 0.975):.0f}]"


class Flights:
    """Flights by (arm, noise), sorted by seed, with the per-step terms of G."""

    def __init__(self, flights: list):
        terms = D.terms(flights)
        self.by = {}
        for r in sorted(flights, key=lambda r: r["seed"]):
            self.by.setdefault((r["label"], r["imu_noise"]), []).append(r)
        self.steps = {key: np.array([terms[(r["label"], r["imu_noise"], r["seed"])] for r in rs])
                      for key, rs in self.by.items()}

    def get(self, arm: str, noise: float = R.NOMINAL) -> list:
        return self.by[(ARMS[arm], noise)]

    def G(self, arm: str, noise: float = R.NOMINAL, steps: int | None = None) -> np.ndarray:
        """G per flight, or its part accrued in the first ``steps`` steps."""
        terms = self.steps[(ARMS[arm], noise)]
        return terms[:, :steps].sum(axis=1) / terms.shape[1]


def headline(f: Flights) -> dict:
    g = {arm: f.G(arm) for arm in ARMS}
    out = {f"G {arm}": f"{g[arm].mean():.0f}" for arm in ARMS}
    for arm in BASELINES:
        out[f"{arm} - info-CBFE"] = interval(g[arm] - g["info-CBFE"])
    for arm in ("MPC-X", "no-info-CBFE"):
        excess = lambda a: g[a].mean() - g["oracle"].mean()  # noqa: E731
        out[f"learning-cost cut vs {arm}"] = f"{100 * (1 - excess('info-CBFE') / excess(arm)):.0f}%"
    return out


def numbers(f: Flights, seeds: list, dropped: list) -> dict:
    T = A.Benchmark().T
    out = {"seeds": f"{len(seeds)} ({seeds[0]}-{seeds[-1]})", "dropped": str(dropped)}
    out.update(headline(f))
    for noise in R.NOISES:
        means = {arm: f.G(arm, noise).mean() for arm in ("info-CBFE", "no-info-CBFE", "dual MPC", "MPC-X", "oracle")}
        out[f"sigma_f {noise:g}: means"] = ", ".join(f"{a} {m:.0f}" for a, m in means.items())
        for arm in BASELINES:
            out[f"sigma_f {noise:g}: {arm} - info-CBFE"] = interval(f.G(arm, noise) - f.G("info-CBFE", noise))
    cum = {arm: np.cumsum(f.steps[(ARMS[arm], R.NOMINAL)], axis=1).mean(0) for arm in BASELINES + ("info-CBFE",)}
    lowest = all(cum["info-CBFE"][t] <= min(cum[a][t] for a in BASELINES) for t in range(T))
    out["info-CBFE lowest cumulative mean at every step"] = str(lowest)
    share = max(np.mean([0.5 * np.mean(r["error"] if "error" in r else 0.0) for r in f.get(arm)]) / f.G(arm).mean()
                for arm in ("info-CBFE", "no-info-CBFE", "dual MPC", "MPC-X"))
    out["prediction term, largest share of G"] = f"{100 * share:.3f}%"
    # mechanism at the nominal noise
    for arm in ("info-CBFE", "no-info-CBFE", "MPC-X"):
        tilt = [np.degrees(np.arccos(np.cos(0.4 * r["u"][0, 1]) * np.cos(0.4 * r["u"][0, 2]))) for r in f.get(arm)]
        out[f"first tilt {arm} (mean)"] = f"{np.mean(tilt):.0f} deg"
    active = np.array([r["multiplier"] > 0.0 for r in f.get("info-CBFE")])
    out["info-CBFE active at step 1 / step 2 / later"] = (f"{100 * active[:, 0].mean():.0f}% / {100 * active[:, 1].mean():.0f}% / "
                                                          f"{100 * active[:, 2:].mean():.0f}%")
    for arm in ("no-info-CBFE", "MPC-X"):
        out[f"{arm} - info-CBFE, first 4 s"] = interval(f.G(arm, steps=EARLY) - f.G("info-CBFE", steps=EARLY))
    binds = np.concatenate([r["multiplier"] for noise in R.NOISES for r in f.get("MPC-X", noise)]) > 0.0
    out["MPC-X requirement binds (all noises)"] = f"{100 * binds.mean():.2f}% of {binds.size} steps"
    g = {arm: f.G(arm) for arm in ("info-CBFE", "MPC-X")}
    out["MPC-X median above info-CBFE"] = f"{100 * (np.median(g['MPC-X']) / np.median(g['info-CBFE']) - 1):.0f}%"
    worst = lambda x: np.sort(x)[-max(1, round(len(x) / 10)):].mean()  # noqa: E731
    out["MPC-X worst 10% above info-CBFE worst 10%"] = f"{100 * (worst(g['MPC-X']) / worst(g['info-CBFE']) - 1):.0f}%"
    chi = np.abs([r["psi"] for r in f.get("info-CBFE")])
    better = g["info-CBFE"] < g["MPC-X"]
    out["info-CBFE beats MPC-X, |chi| < 1 / > 2 rad"] = (f"{100 * better[chi < 1].mean():.0f}% of {np.sum(chi < 1)} / "
                                                         f"{100 * better[chi > 2].mean():.0f}% of {np.sum(chi > 2)}")
    for arm in BASELINES:
        out[f"info-CBFE beats {arm}"] = f"{100 * np.mean(f.G('info-CBFE') < f.G(arm)):.0f}% of flights"
    # wall clock at the nominal noise, which ran on the four fast cores
    for arm in ("info-CBFE", "no-info-CBFE", "dual MPC", "MPC-X"):
        out[f"ms/step {arm} (median over flights)"] = f"{np.median([1e3 * r['wall'] / T for r in f.get(arm)]):.0f}"
    flights = f.get("info-CBFE")
    seconds = np.concatenate([r["seconds"][r["multiplier"] > 0.0] for r in flights])
    out["info-CBFE active steps: median / max / share > 0.5 s"] = (f"{np.median(seconds):.2f} s / {seconds.max():.2f} s / "
                                                                  f"{100 * np.mean(seconds > 0.5):.0f}%")
    share = 100 * np.array([np.mean(r["multiplier"] > 0.0) for r in flights])
    ms = np.array([1e3 * r["wall"] / T for r in flights])
    slope, intercept = np.polyfit(share, ms, 1)
    out["info-CBFE ms/step per % active steps"] = f"{slope:.1f} (intercept {intercept:.0f} ms)"
    others = np.concatenate([r["seconds"] for arm in ("no-info-CBFE", "dual MPC", "MPC-X") for r in f.get(arm)])
    out["longest step of the other controllers"] = f"{others.max():.2f} s"
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fresh", action="store_true", help="the fresh seeds 4000-4099 instead of 1000-1099")
    fresh = parser.parse_args().fresh
    every = D.load(fresh)
    seeds = D.seeds(fresh, every)
    out = numbers(Flights([r for r in every if r["seed"] in set(seeds)]), seeds, sorted(D.failed(every)))
    out.update({f"every seed: {key}": value for key, value in headline(Flights(every)).items()})
    path = A.RESULTS / f"summary{'_fresh' if fresh else ''}.csv"
    pd.Series(out, name="value").rename_axis("quantity").to_csv(path)
    width = max(map(len, out))
    print("\n".join(f"{key:{width}s}  {value}" for key, value in out.items()))
    print(path)


if __name__ == "__main__":
    main()
