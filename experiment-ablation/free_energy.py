"""Free energy of the filter of every controller in the quadcopter runs of
``run.py``, at every time step, by replaying the stored flights.

The filter of every controller conditions on y_k = (position fix, IMU sample)
after each input. The output map is linear in the state and the parameters,
so the Gaussian measurement update is exact given the predicted belief, and
its free energy is

    F_k = -ln N(y_k ; y_hat_k, S_k)
        = 1/2 e_k' S_k^-1 e_k  +  1/2 ln |2 pi S_k| ,   e_k = y_k - y_hat_k ,

the precision-weighted prediction error plus the log-determinant of the
predictive covariance. Their sum over the flight is the negative log marginal
likelihood of the measurements under the filter. The replay rebuilds every
agent, draws the same noise from the seed, feeds it the stored inputs and
checks that its parameter estimates match the stored ones.

    python experiment-ablation/free_energy.py    # data/free_energy.pkl, results/free_energy.csv
"""
from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ablation as A  # noqa: E402

BLOCKS = {"position": slice(0, 3), "imu": slice(3, 6)}


def marginal(residual: np.ndarray, innovation: np.ndarray) -> float:
    root = np.linalg.cholesky(innovation)
    white = np.linalg.solve(root, residual)
    return 0.5 * (len(residual) * math.log(2.0 * math.pi) + 2.0 * np.sum(np.log(np.diag(root))) + white @ white)


def replay(r: dict) -> dict:
    """Free energy, prediction error and log-determinant of every update."""
    bench = A.Benchmark(imu_noise=r["imu_noise"])
    d = A.draw(bench, r["seed"])
    agent = A.make(r["label"], bench, d["target"], d["lean"])
    agent.observe(position=r["fixes"][0])
    T = bench.T
    out = {key: np.zeros(T) for key in ("F", "error", "logdet", "F_position", "F_imu")}
    drift = 0.0
    for t in range(T):
        if r["label"] == A.REFERENCE:
            e, psi = d["efficiency"][t + 1], d["psi"]
            agent.set_parameters(np.concatenate([e * (d["k"] * d["wind"][t + 1] + d["bias"][t + 1]),
                                                 [e * d["k"], e * math.cos(psi), e * math.sin(psi), e]]))
        f = (A.specific_force(r["u"][t], r["v"][t], d["wind"][t + 1], d["k"], agent.config, d["psi"],
                              d["efficiency"][t + 1], d["bias"][t + 1]) + bench.force_noise * d["force"][t])
        agent.update(r["u"][t], r["fixes"][t + 1], f + bench.imu_noise * d["imu"][t])
        out["F"][t], out["error"][t], out["logdet"][t] = agent.last_free_energy, agent.last_error, agent.last_logdet
        for name, block in BLOCKS.items():
            out[f"F_{name}"][t] = marginal(agent.last_residual[block], agent.last_innovation[block, block])
        drift = max(drift, float(np.max(np.abs(agent.mean[A.NX_PARAMS] - r["estimate"][t]))))
    return dict(label=r["label"], imu_noise=r["imu_noise"], seed=r["seed"], drift=drift, **out)


def main():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    episodes = A.load(A.DATA / "episodes.pkl")
    with ProcessPoolExecutor(12, mp_context=get_context("forkserver")) as pool:
        traces = list(pool.map(replay, episodes, chunksize=8))
    print(f"largest deviation from the stored parameter estimates: {max(t['drift'] for t in traces):.2e}")
    with open(A.DATA / "free_energy.pkl", "wb") as f:
        pickle.dump(traces, f)
    rows = [dict(label=t["label"], imu_noise=t["imu_noise"], seed=t["seed"],
                 **{key: float(t[key].mean()) for key in ("F", "error", "logdet", "F_position", "F_imu")})
            for t in traces]
    table = pd.DataFrame(rows)
    table.to_csv(A.RESULTS / "free_energy.csv", index=False)
    print(table.groupby(["imu_noise", "label"])[["F", "error", "logdet", "F_position", "F_imu"]].mean()
          .round(2).to_string())


if __name__ == "__main__":
    main()
