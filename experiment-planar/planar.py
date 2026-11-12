"""Planar positioning with an unknown actuator orientation (candidate for
main.tex, Section 5.1; see reviews/setting_2026-11-12.md).

The problem
-----------
A planar stage (or a robot servoed on image features) moves its measured
position by an input whose frame is rotated by an unknown angle,

    y_k = y_{k-1} + B u_k + w_k ,   w_k ~ N(0, sigma^2 I) ,   u_k in [-1, 1]^2 ,
    B = R(phi) ,                    phi ~ U[0, 2 pi) unknown and constant,

so theta = vec(B) and the state is measured (g = identity, v = 0). From
y_0 = 0, the controller reaches a target y_* at distance d in a random
direction, within the tolerance S_* I, over T steps. Every seed fixes the
orientation, the target direction and the noise, so all controllers of a
comparison see the same draws.

With B unknown, every input only adds predictive spread, so the cautious
controller never moves: the dual effect of an unknown polarity, in two
dimensions. The disturbance level sigma is the condition: it sets how large
a probe must be to stand out from the noise, and hence the probing weight
that a dual MPC needs.

The model
---------
All controllers share the ARX model of ``agents.common`` with features
(1, u_1, u_2, y_1, y_2): the intercept and the output coefficients known
(prior precision ``known`` in noise units), the entries of B independent
N(0, P0) a priori with P0 = 1/2, which matches the first two moments of a
uniform orientation, and the noise level known (nu0 = 1e4). The posterior is
then the exact Gaussian posterior of B.

Controllers (``make``)
----------------------
    {"kind": "cbfe", "fixed": None}    the CBFE-ARX agent of the paper (KKT
                                       multiplier, predictive floor, next node)
    {"kind": "cbfe", "fixed": 0.0}     the cautious controller
    {"kind": "dual", "rho": rho}       dual MPC with the Beta input cost
    {"kind": "mpcx", "gamma": gamma}   MPC-X with accuracy gamma
    {"kind": "oracle"}                 the known B, deadbeat within the bound
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

from agents.common import MatrixNormalWishart, MultivariateNormal  # noqa: E402

RESULTS = HERE / "results"


@dataclass(frozen=True)
class Benchmark:
    sigma: float = 0.1            # disturbance standard deviation
    goal_variance: float = 0.01   # S_*
    distance: float = 0.5         # d, distance of the target from y_0 = 0
    T: int = 40                   # steps
    K: int = 5                    # planning horizon
    P0: float = 0.5               # prior variance of every entry of B
    known: float = 1e6            # prior precision of the known coefficients (noise units)
    nu0: float = 1e4              # Wishart degrees of freedom: the noise level is known


def prior(bench: Benchmark) -> MatrixNormalWishart:
    """The shared MNW prior over the coefficients of (1, u_1, u_2, y_1, y_2)."""
    s2 = bench.sigma**2
    mean = np.zeros((5, 2))
    mean[3, 0] = mean[4, 1] = 1.0
    precision = np.diag([bench.known, s2 / bench.P0, s2 / bench.P0, bench.known, bench.known])
    return MatrixNormalWishart(coefficient_mean=mean, coefficient_precision=precision,
                               noise_scale=s2 * (bench.nu0 - 3.0) * np.eye(2),
                               degrees_of_freedom=bench.nu0)


def make(spec: dict, bench: Benchmark, target: np.ndarray):
    """The agent of ``spec`` on the shared model (None for the oracle)."""
    goal = MultivariateNormal(mean=target, covariance=bench.goal_variance * np.eye(2))
    common = dict(input_dim=2, output_dim=2, input_memory=0, output_memory=1,
                  goal_prior=goal, horizon=bench.K, num_harmonics=0)
    kind = spec["kind"]
    if kind == "oracle":
        return None
    if kind == "cbfe":
        from agents.cbfe_arx import Agent
        agent = Agent.build(**common, fixed_multiplier=spec.get("fixed"),
                            information_floor_type="predictive", constraint_nodes="next",
                            max_multiplier=1e6)
    elif kind == "dual":
        from agents.dual_mpc import DualMPC
        agent = DualMPC.build(**common, probing_weight=spec["rho"], effort_weight=2.0)
        agent.config.rest_tolerance = 1e-7
        agent.config.input_cost = "beta"
    elif kind == "mpcx":
        from agents.mpcx import MPCX
        agent = MPCX.build(**common, accuracy=spec["gamma"], experiment_length=bench.T,
                           effort_weight=2.0)
    else:
        raise ValueError(f"unknown controller kind '{kind}'")
    agent.estimator.posterior = prior(bench)
    agent.estimator.output_buffer, agent.estimator.input_buffer = [np.zeros(2)], []
    return agent


def draw(bench: Benchmark, seed: int):
    """The input matrix, the target and the disturbances of ``seed``."""
    rng = np.random.default_rng(seed)
    phi, psi = rng.uniform(0.0, 2.0 * np.pi, 2)
    B = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    target = bench.distance * np.array([np.cos(psi), np.sin(psi)])
    return B, target, bench.sigma * rng.standard_normal((bench.T, 2))


def run_job(job: dict) -> dict:
    """One episode; G is the goal cost per step, sum_k ||y_k - y_*||^2 / (2 S_* T)."""
    bench, spec, seed = job["bench"], job["spec"], job["seed"]
    B, target, noise = draw(bench, seed)
    agent = make(spec, bench, target)
    y = np.zeros((bench.T + 1, 2))
    u = np.zeros((bench.T, 2))
    multiplier = np.full(bench.T, np.nan)
    start = time.perf_counter()
    for t in range(bench.T):
        if agent is None:
            u[t] = np.clip(np.linalg.solve(B, target - y[t]), -1.0, 1.0)
        else:
            u[t] = np.clip(agent.plan(y[t]), -1.0, 1.0)
            multiplier[t] = float(getattr(agent.planner, "last_multiplier", np.nan))
        y[t + 1] = y[t] + B @ u[t] + noise[t]
        if agent is not None:
            agent.update(y[t + 1], u[t])
    squared = np.sum((y[1:] - target) ** 2, axis=1)
    return dict(label=job["label"], sigma=bench.sigma, seed=seed, y=y, u=u, multiplier=multiplier,
                G=float(squared.mean() / (2.0 * bench.goal_variance)),
                rmse=float(np.sqrt(squared.mean())), seconds=time.perf_counter() - start)


def run_many(jobs, processes: Optional[int] = None) -> list:
    """``run_job`` for every job in parallel processes, in the order of the jobs."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    jobs = list(jobs)
    processes = processes or min(len(jobs), os.cpu_count() or 1)
    if processes <= 1:
        return [run_job(job) for job in jobs]
    with ProcessPoolExecutor(processes, mp_context=get_context("forkserver")) as pool:
        return list(pool.map(run_job, jobs, chunksize=1))


def save(results: list, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(results, f)
    return path


def load(name: str) -> list:
    with open(RESULTS / f"{name}.pkl", "rb") as f:
        return pickle.load(f)
