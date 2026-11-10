"""Light-dark positioning (main.tex, Section V-C).

    x_k = x_{k-1} + u_k + w_k ,   w_k ~ N(0, q) ,
    y_k = x_k + v_k ,             v_k ~ N(0, r(x_k)) ,   r(x) = r0 + c (x - x_L)^2 ,

with |u| <= 1 and the input gain 1. With ``gain_variance > 0``, the input
enters as b u_k with an unknown gain b ~ N(gain_mean, gain_variance), and the
agents estimate it jointly with the state. The state starts at x_0 ~ N(0, V_0),
the controllers know this prior, and every episode regulates the state to the target x_* = 0 over T
steps under a tight (S_* = 0.01) or a loose (S_* = 0.1) goal. The target lies
in the dark (r(0) = 1); the light at x_L = 1 measures the state precisely. A
seed fixes the initial state and the noise, so all controllers of a
comparison see the same draws.

Controllers (``make_controller``), all on the exact grid filter of
``agents.cbfe_lightdark``:

    {"kind": "agent", "planner": "cbfe" | "none" | "bsp"}
    {"kind": "agent", "planner": "dual", "weight": rho / 2}
    {"kind": "dp"}       one-step lookahead on the optimal value function of the
                         Gaussian belief MDP (the filter moments as the belief)
    {"kind": "oracle"}   deadbeat control with the true state
"""
from __future__ import annotations

import math
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

from agents.cbfe_lightdark import (DirectionalAgent, GainConfig, GainLightDarkAgent, GridFilter,  # noqa: E402
                                   JointGridFilter, LightDarkAgent, LightDarkConfig, LightDarkModel)

RESULTS = HERE / "results"
GOALS = {"tight": 0.01, "loose": 0.1}
BOUND = 0.98


@dataclass(frozen=True)
class Benchmark:
    model: LightDarkModel = field(default_factory=LightDarkModel)
    T: int = 30
    horizon: int = 5
    prior_mean: float = 0.0
    prior_variance: float = 0.25
    target: float = 0.0
    gain_mean: float = 1.0
    gain_variance: float = 0.0      # 0: the gain is known
    directional: bool = False       # True: unknown gains b_+ (u >= 0) and b_- (u < 0)


@dataclass
class Episode:
    x: np.ndarray            # (T+1,) true states
    u: np.ndarray            # (T,)
    y: np.ndarray            # (T,)
    gain: object             # true input gain, or (b_+, b_-)
    beliefs: np.ndarray      # (T+1, 4) filter m, V, b_hat, P
    multipliers: np.ndarray  # (T, 2) parameter and state multipliers
    floors: np.ndarray       # (T, 2) parameter and state floors
    plan_time: np.ndarray    # (T,) [s]


def draw(bench: Benchmark, seed: int):
    rng = np.random.default_rng(seed)
    x0 = bench.prior_mean + math.sqrt(bench.prior_variance) * rng.standard_normal()
    w = math.sqrt(bench.model.q) * rng.standard_normal(bench.T)
    e = rng.standard_normal(bench.T)
    gain = bench.gain_mean
    if bench.gain_variance > 0.0:
        draws = np.random.default_rng(seed + 1_000_000).standard_normal(2)
        gain += math.sqrt(bench.gain_variance) * draws[0]
        if bench.directional:
            gain = (gain, bench.gain_mean + math.sqrt(bench.gain_variance) * draws[1])
    return x0, w, e, gain


def effective_gain(gain, u: float) -> float:
    """The gain that acts on the input u: b_+ for u >= 0 and b_- for u < 0."""
    if isinstance(gain, tuple):
        return gain[0] if u >= 0.0 else gain[1]
    return gain


# ----- optimal controller of the Gaussian belief MDP -----

@dataclass
class BeliefDP:
    """Dynamic program on the Gaussian belief (m, V) with the measurement noise
    at the predicted mean (an extended Kalman filter) and the stage cost
    ((m^- - x_*)^2 + V^-) / (2 S_*) + c(u) of the planners."""
    bench: Benchmark
    goal_variance: float
    m_grid: np.ndarray = field(default_factory=lambda: np.linspace(-2.0, 3.0, 101))
    log_v_grid: np.ndarray = field(default_factory=lambda: np.linspace(math.log(1e-5), math.log(2.0), 50))
    u_grid: np.ndarray = field(default_factory=lambda: np.linspace(-BOUND, BOUND, 99))

    def solve(self) -> "BeliefDP":
        b, model = self.bench, self.bench.model
        gh, w = np.polynomial.hermite.hermgauss(9)
        gh, w = math.sqrt(2.0) * gh, w / math.sqrt(math.pi)
        M, LV = np.meshgrid(self.m_grid, self.log_v_grid, indexing="ij")
        V = np.exp(LV)
        control_cost = -np.log1p(-self.u_grid**2)
        dm, dlv = self.m_grid[1] - self.m_grid[0], self.log_v_grid[1] - self.log_v_grid[0]
        value = np.zeros_like(M)
        self.values = [value]
        for _ in range(b.T):
            mean = M[..., None] + self.u_grid
            prior = (V + model.q)[..., None] * np.ones_like(self.u_grid)
            r = model.noise(mean)
            posterior = prior * r / (prior + r)
            spread = prior / np.sqrt(prior + r)
            stage = ((mean - b.target)**2 + prior) / (2.0 * self.goal_variance) + control_cost
            nxt = mean[..., None] + spread[..., None] * gh
            im = np.clip((nxt - self.m_grid[0]) / dm, 0, len(self.m_grid) - 1)
            iv = np.clip((np.log(posterior)[..., None] - self.log_v_grid[0]) / dlv, 0, len(self.log_v_grid) - 1)
            iv = np.broadcast_to(iv, im.shape)
            future = map_coordinates(value, [im.ravel(), iv.ravel()], order=1, mode="nearest").reshape(im.shape)
            value = np.min(stage + np.sum(w * future, axis=-1), axis=-1)
            self.values.append(value)
        self.values = self.values[::-1]     # values[k]: cost-to-go with T - k steps left
        return self

    def act(self, k: int, m: float, V: float) -> float:
        b, model = self.bench, self.bench.model
        gh, w = np.polynomial.hermite.hermgauss(9)
        gh, w = math.sqrt(2.0) * gh, w / math.sqrt(math.pi)
        u = self.u_grid
        mean, prior = m + u, V + model.q
        r = model.noise(mean)
        posterior = prior * r / (prior + r)
        spread = prior / np.sqrt(prior + r)
        stage = ((mean - b.target)**2 + prior) / (2.0 * self.goal_variance) - np.log1p(-u**2)
        nxt = mean[:, None] + spread[:, None] * gh
        dm, dlv = self.m_grid[1] - self.m_grid[0], self.log_v_grid[1] - self.log_v_grid[0]
        im = np.clip((nxt - self.m_grid[0]) / dm, 0, len(self.m_grid) - 1)
        iv = np.clip((np.log(posterior)[:, None] - self.log_v_grid[0]) / dlv, 0, len(self.log_v_grid) - 1)
        iv = np.broadcast_to(iv, im.shape)
        future = map_coordinates(self.values[k + 1], [im.ravel(), iv.ravel()], order=1,
                                 mode="nearest").reshape(im.shape)
        return float(u[int(np.argmin(stage + np.sum(w * future, axis=-1)))])


_DP_CACHE = {}


def belief_dp(bench: Benchmark, goal_variance: float) -> BeliefDP:
    key = (bench, goal_variance)
    if key not in _DP_CACHE:
        _DP_CACHE[key] = BeliefDP(bench, goal_variance).solve()
    return _DP_CACHE[key]


class FilterController:
    def __init__(self, bench: Benchmark):
        self.filter = GridFilter(bench.model, bench.prior_mean, bench.prior_variance)
        self.last_multipliers = {"parameter": 0.0, "state": 0.0}
        self.last_floors = {"parameter": 0.0, "state": 0.0}

    def update(self, u, y):
        self.filter.update(u, y)


class DPController(FilterController):
    def __init__(self, bench, goal_variance):
        super().__init__(bench)
        self.dp = belief_dp(bench, goal_variance)
        self.k = 0

    def plan(self):
        m, V = self.filter.moments()
        u = self.dp.act(self.k, m, V)
        self.k += 1
        return u


class Oracle(FilterController):
    def __init__(self, bench, states, gain):
        super().__init__(bench)
        self.states, self.gain, self.k = states, gain, 0

    def plan(self):
        x = self.states[self.k]
        u = float(np.clip(-x / effective_gain(self.gain, -x), -BOUND, BOUND))
        self.k += 1
        return u


class KnownGainAgent(LightDarkAgent):
    """The light-dark agent with the multiplier and floor records of the
    unknown-gain agent."""

    @property
    def last_multipliers(self):
        return {"parameter": 0.0, "state": self.last_multiplier}

    @property
    def last_floors(self):
        return {"parameter": 0.0, "state": self.last_floor}


def make_controller(spec: dict, bench: Benchmark, goal_variance: float, states=None, gain: float = 1.0):
    if spec["kind"] == "agent" and bench.gain_variance > 0.0:
        agent_class = DirectionalAgent if bench.directional else GainLightDarkAgent
        config = GainConfig(goal_mean=bench.target, goal_variance=goal_variance, horizon=bench.horizon,
                            planner=spec["planner"],
                            constraints=tuple(spec.get("constraints", ("parameter", "state"))),
                            weights=tuple(spec.get("weights", (0.0, 0.0))),
                            task_length=bench.T if spec.get("task_floor", False) else None,
                            shared_budget=spec.get("shared_budget", False),
                            joint_belief=spec.get("joint_belief", True))
        return agent_class(config, bench.model, bench.prior_mean, bench.prior_variance,
                           bench.gain_mean, bench.gain_variance)
    if spec["kind"] == "agent":
        config = LightDarkConfig(goal_mean=bench.target, goal_variance=goal_variance, horizon=bench.horizon,
                                 planner=spec["planner"], weight=spec.get("weight", 0.0),
                                 task_length=bench.T if spec.get("task_floor", False) else None)
        return KnownGainAgent(config, bench.model, bench.prior_mean, bench.prior_variance)
    if spec["kind"] == "dp":
        return DPController(bench, goal_variance)
    if spec["kind"] == "oracle":
        return Oracle(bench, states, gain)
    raise ValueError(spec["kind"])


def _moments(controller) -> np.ndarray:
    """m, V and the mean and variance of the gain (of b_- with directional gains)."""
    moments = controller.filter.moments()
    if isinstance(moments[0], np.ndarray):
        mean, cov = moments
        return np.array([mean[0], cov[0, 0], mean[2], cov[2, 2]])
    if len(moments) == 2:
        return np.array([moments[0], moments[1], 1.0, 0.0])
    return np.array(moments[:4])


def run_episode(spec: dict, bench: Benchmark, goal: str, seed: int) -> Episode:
    x0, w, e, gain = draw(bench, seed)
    T = bench.T
    x, u, y = np.zeros(T + 1), np.zeros(T), np.zeros(T)
    x[0] = x0
    controller = make_controller(spec, bench, GOALS[goal], states=x, gain=gain)
    beliefs = np.zeros((T + 1, 4))
    beliefs[0] = _moments(controller)
    multipliers, floors, plan_time = np.zeros((T, 2)), np.zeros((T, 2)), np.zeros(T)
    for k in range(T):
        start = time.perf_counter()
        u[k] = float(np.clip(controller.plan(), -1.0, 1.0))
        plan_time[k] = time.perf_counter() - start
        multipliers[k] = [controller.last_multipliers[n] for n in ("parameter", "state")]
        floors[k] = [controller.last_floors[n] for n in ("parameter", "state")]
        x[k + 1] = x[k] + effective_gain(gain, u[k]) * u[k] + w[k]
        y[k] = x[k + 1] + math.sqrt(float(bench.model.noise(x[k + 1]))) * e[k]
        controller.update(u[k], y[k])
        beliefs[k + 1] = _moments(controller)
    return Episode(x=x, u=u, y=y, gain=gain, beliefs=beliefs, multipliers=multipliers, floors=floors,
                   plan_time=plan_time)


def run_job(job: dict) -> dict:
    episode = run_episode(job["spec"], job["bench"], job["goal"], job["seed"])
    return dict(label=job["label"], goal=job["goal"], seed=job["seed"], episode=episode)


def run_many(jobs, processes: Optional[int] = None) -> list:
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    jobs = list(jobs)
    processes = processes or min(len(jobs), os.cpu_count() or 1)
    if processes <= 1:
        return [run_job(job) for job in jobs]
    with ProcessPoolExecutor(processes, mp_context=get_context("forkserver")) as pool:
        return list(pool.map(run_job, jobs, chunksize=2))


def save(results: list, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(results, f)
    return path


def load(name: str) -> list:
    with open(RESULTS / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def metrics(result: dict, bench: Benchmark) -> dict:
    ep = result["episode"]
    S = GOALS[result["goal"]]
    error = ep.x[1:] - bench.target
    return dict(label=result["label"], goal=result["goal"], seed=result["seed"],
                nll=float(np.mean(error**2) / (2.0 * S)), rmse=float(np.sqrt(np.mean(error**2))),
                final=float(abs(error[-1])), inside=100.0 * float(np.mean(np.abs(error) <= 2.0 * math.sqrt(S))),
                light=100.0 * float(np.mean(np.abs(ep.x[1:] - bench.model.light) < 0.6)),
                active=100.0 * float(np.mean(ep.multipliers[:, 1] > 0.0)),
                parameter_active=100.0 * float(np.mean(ep.multipliers[:, 0] > 0.0)),
                final_P=float(ep.beliefs[-1, 3]),
                plan_time=float(np.median(ep.plan_time)))
