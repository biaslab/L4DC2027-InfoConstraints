"""Performance G of eq. (metric) of main.tex for every light-dark episode of
``run_baselines.py``, by replaying the grid filter on the stored inputs and
outputs. Per step k,

    g_k = 1/2 (y_k - mu_k)^2 / Sigma_k  +  1/2 x_k^2 / S_* ,

with mu_k and Sigma_k the mean and variance of the filter's one-step
predictive distribution of y_k, which mixes the noise r(x) over the predicted
state, and the goal term on the true state, as in the quadcopter experiment;
G is the mean of g_k over the T steps. The replay checks that the filter
reproduces the stored beliefs.

    python experiment-lightdark/performance.py    # results/baselines_performance.pkl
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import lightdark as L  # noqa: E402
import run_baselines as R  # noqa: E402
from controllers.cbfe_lightdark.agent import GridFilter  # noqa: E402

FILES = ("baselines_tuning", "baselines_test", "baselines_sweep", "transfer")


def replay(item: tuple) -> dict:
    name, result = item
    bench, ep = R.bench(result["light"]), result["episode"]
    grid = GridFilter(bench.model, bench.prior_mean, bench.prior_variance)
    T = len(ep.u)
    prediction, drift = np.zeros(T), 0.0
    for k in range(T):
        grid.update(ep.u[k], ep.y[k])
        prediction[k] = 0.5 * (ep.y[k] - grid.last_output_mean)**2 / grid.last_output_variance
        drift = max(drift, abs(grid.moments()[0] - ep.beliefs[k + 1][0]))
    goal = ep.x[1:]**2 / (2.0 * L.GOALS[R.GOAL])
    return dict(file=name, label=result["label"], light=result["light"], seed=result["seed"], goal=goal,
                prediction=prediction, wall=float(ep.plan_time.sum()), steps=T, drift=drift)


def main():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    items = [(name, result) for name in FILES if (L.RESULTS / f"{name}.pkl").exists() for result in L.load(name)]
    with ProcessPoolExecutor(12, mp_context=get_context("forkserver")) as pool:
        rows = list(pool.map(replay, items, chunksize=8))
    print(f"{len(rows)} episodes; largest deviation from the stored posterior means: {max(r['drift'] for r in rows):.1e}")
    with open(L.RESULTS / "baselines_performance.pkl", "wb") as f:
        pickle.dump(rows, f)


if __name__ == "__main__":
    main()
