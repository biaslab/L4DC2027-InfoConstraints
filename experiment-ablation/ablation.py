"""Ablation of the information constraint and the trace term on a quadcopter
that navigates in wind.

The problem
-----------
The quadcopter flies from p_0 = (0, 0, 2) m to a target 5 m away in a random
horizontal direction and 1 m higher, within S_* = 0.03^2 m^2, over T = 20
steps of dt = 0.5 s. Its controller commands the collective thrust and the
attitude (``agents.cbfe_quad``). The wind w_k acts through rotor drag, which
grows with the thrust and the relative airspeed in the rotor plane:

    f_k = tau n - k tau P (v_{k-1} - w_k) + eta_k ,

with the drag coefficient k, the unmodelled force eta_k ~ N(0, 0.2^2 I) m/s^2,
a GNSS-grade position fix with noise sigma_p = 0.3 m, and an IMU that
measures f_k with noise sigma_f = 2 m/s^2 (a vibrating airframe, averaged over
a step). The controller knows the model and the noise levels, but neither the
wind nor the drag coefficient. The goal cost is scored on the true position,
because the position fix is ten times noisier than the tolerance.

Faults. Magnetic interference (steel structures, power lines) biases the
compass, so the heading that the flight controller uses is off by an unknown
angle psi ~ U[-150, 150] deg, and every commanded tilt pushes the
quadcopter in a direction rotated by psi; GPS-hold drones with this fault
spiral around their target ("toilet bowling"). The controller models the
horizontal thrust as [[a, -b], [b, a]] tau n_h with unknown (a, b) =
(cos psi, sin psi), and its prior on (a, b) matches the first two moments of
this fault distribution: mean (0.20, 0), standard deviations (0.61, 0.76).
A compass-trusting prior, intermittent power loss and a buckled rotor arm
are available as options (prior, power_rate, buckle_prob); the latter two
are off by default, because their variance swamped the comparison.

The defaults come from a logged search (``search.py``, results/search_log.jsonl)
for a case in which the information constraint binds and helps: it binds
only when the goal-reaching inputs spread the predicted position by more
than about 3 S_* beyond the planned input, which needs a tight tolerance,
uncertain wind and a noisy IMU. The wind is capped at 8 m/s, below the
speed at which rotor drag overpowers the tilt authority.

Every seed draws the vehicle's drag coefficient k ~ U[0.025, 0.055] s/m and
the wind:
  - the mean speed from a Weibull distribution (shape 2, scale 5 m/s, clipped
    to [0.5, 8] m/s), the usual model of wind speeds;
  - the mean direction uniformly;
  - Ornstein-Uhlenbeck fluctuations of the speed (20% of the mean speed), the
    direction (0.35 rad) and the vertical wind (0.3 m/s), whose correlation
    times are log-uniform on [1, 10] s, so the wind varies smoothly with a
    smoothness that differs between seeds.
The noise draws are standard normal and the noise levels scale them, so all
controllers and all noise levels see the same draws.

The controller's prior on the wind force c = k w is N(0, diag(s_h^2, s_h^2,
s_v^2)) with s_h = 0.04 * 5 / sqrt(2) and s_v = 0.04 * 0.4, so that its
horizontal variance matches k_bar^2 E[w_x^2] under the Weibull prior. The
prior on k is N(0.04, 0.01^2). The wind force follows a random walk with
steps of 0.025 per step.

Controllers (``CONTROLLERS``: trace term, information constraint)
-----------------------------------------------------------------
    "CBFE"                    both
    "no constraint"           the trace term with gamma = 0 (cautious)
    "no trace"                the KKT multiplier without the trace term
    "certainty equivalence"   neither
    "known model"             knows the wind, the drag and the faults at every step (reference)
"""
from __future__ import annotations

import math
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

from agents.cbfe_quad import Agent, Config  # noqa: E402
from agents.cbfe_quad.agent import GRAVITY  # noqa: E402

DATA = HERE / "data"
RESULTS = HERE / "results"
CONTROLLERS = {"CBFE": (True, True), "no constraint": (True, False), "no trace": (False, True),
               "certainty equivalence": (False, False), "known model": (True, False)}
REFERENCE = "known model"
NX_PARAMS = slice(9, 13)      # wind force c and drag coefficient k in the filter mean


@dataclass(frozen=True)
class Benchmark:
    imu_noise: float = 2.0        # sigma_f [m/s^2]
    position_noise: float = 0.3   # sigma_p [m], GNSS grade
    force_noise: float = 0.2      # sigma_eta [m/s^2]
    dt: float = 0.5
    T: int = 20
    K: int = 6
    start: tuple = (0.0, 0.0, 2.0)
    distance: float = 5.0
    climb: float = 1.0
    goal_std: float = 0.03
    wind_scale: float = 5.0       # Weibull scale of the mean wind speed [m/s]
    wind_max: float = 8.0
    turbulence: float = 0.2       # speed fluctuation / mean speed
    direction_std: float = 0.35   # [rad]
    vertical_std: float = 0.3     # [m/s]
    correlation: tuple = (1.0, 10.0)   # range of the correlation times [s]
    drag: tuple = (0.025, 0.055)  # range of the drag coefficient k [s/m]
    drag_prior: tuple = (0.04, 0.01)
    wind_step: float = 0.025
    # faults (all off by default)
    heading_max: float = 2.6      # heading misalignment psi ~ U[-heading_max, heading_max] [rad]
    power_rate: float = 0.0       # onset rate of intermittent power losses [1/s]
    power_dwell: float = 1.0      # mean duration of a power loss [s]
    power_low: tuple = (0.5, 0.8)  # thrust efficiency during a loss
    buckle_prob: float = 0.0      # probability that an arm buckles during the flight
    buckle_time: tuple = (2.0, 8.0)  # [s]
    buckle_bias: float = 0.15     # horizontal force of the buckled rotor per unit thrust
    buckle_loss: float = 0.1      # thrust efficiency lost by the buckled rotor
    # prior on the thrust gains (a, b, eta) and their random walk in the controller
    prior: str = "calibrated"     # "calibrated" or "compass"
    prior_gain_std: float = 0.5   # std of (a, b) under the compass prior
    gain_step: float = 0.02
    gain_lean: float = 0.0        # the prior mean of (a, b) moves by up to this much in a random direction


def ou_path(rng, n: int, dt: float, std: float, tau: float) -> np.ndarray:
    """Stationary Ornstein-Uhlenbeck path at the steps dt."""
    rho = math.exp(-dt / tau)
    out = np.empty(n)
    out[0] = std * rng.standard_normal()
    for i in range(1, n):
        out[i] = rho * out[i - 1] + std * math.sqrt(1.0 - rho**2) * rng.standard_normal()
    return out


def draw(bench: Benchmark, seed: int) -> dict:
    """Target, drag coefficient, wind and noise draws of ``seed``."""
    rng = np.random.default_rng(seed)
    heading = rng.uniform(0.0, 2.0 * math.pi)
    start = np.asarray(bench.start, float)
    target = start + np.array([bench.distance * math.cos(heading), bench.distance * math.sin(heading), bench.climb])
    k = rng.uniform(*bench.drag)
    speed = float(np.clip(bench.wind_scale * rng.weibull(2.0), 0.5, bench.wind_max))
    direction = rng.uniform(0.0, 2.0 * math.pi)
    taus = np.exp(rng.uniform(*np.log(bench.correlation), size=3))
    n = bench.T + 1
    speeds = np.maximum(speed + ou_path(rng, n, bench.dt, bench.turbulence * speed, taus[0]), 0.0)
    directions = direction + ou_path(rng, n, bench.dt, bench.direction_std, taus[1])
    vertical = ou_path(rng, n, bench.dt, bench.vertical_std, taus[2])
    wind = np.column_stack([speeds * np.cos(directions), speeds * np.sin(directions), vertical])
    out = dict(target=target, start=start, k=k, speed=speed, direction=direction, taus=taus, wind=wind,
               force=rng.standard_normal((bench.T, 3)), position=rng.standard_normal((n, 3)),
               imu=rng.standard_normal((bench.T, 3)))
    # faults, drawn after everything else so that the draws above do not change
    psi = rng.uniform(-bench.heading_max, bench.heading_max)
    low = rng.uniform(*bench.power_low)
    onset, recovery = 1.0 - math.exp(-bench.power_rate * bench.dt), 1.0 - math.exp(-bench.dt / bench.power_dwell)
    lost, efficiency = False, np.ones(n)
    flips = rng.uniform(size=n)
    for i in range(1, n):
        lost = (flips[i] >= recovery) if lost else (flips[i] < onset)
        efficiency[i] = low if lost else 1.0
    buckles = rng.uniform() < bench.buckle_prob
    t_buckle = rng.uniform(*bench.buckle_time)
    chi = rng.uniform(0.0, 2.0 * math.pi)
    after = (np.arange(n) * bench.dt >= t_buckle) & buckles
    efficiency = efficiency * np.where(after, 1.0 - bench.buckle_loss, 1.0)
    bias = np.outer(after, bench.buckle_bias * np.array([math.cos(chi), math.sin(chi), 0.0]))
    lean = rng.uniform(0.0, bench.gain_lean) * np.array([math.cos(c := rng.uniform(0.0, 2 * math.pi)), math.sin(c)])
    out.update(psi=psi, efficiency=efficiency, bias=bias, buckles=bool(buckles), t_buckle=t_buckle, lean=lean)
    return out


def gain_prior(bench: Benchmark):
    """Mean and variance of the controller's prior on (a, b, eta). The
    calibrated prior matches the first two moments of (cos psi, sin psi) under
    the heading misalignment, and the compass prior trusts the heading with
    the standard deviation prior_gain_std."""
    if bench.prior == "compass":
        return np.array([1.0, 0.0, 1.0]), np.array([bench.prior_gain_std**2] * 2 + [0.05**2])
    psi = np.random.default_rng(0).uniform(-bench.heading_max, bench.heading_max, 100_000)
    return (np.array([np.cos(psi).mean(), 0.0, 1.0]),
            np.array([max(np.cos(psi).var(), 1e-6), max(np.sin(psi).var(), 1e-6), 0.05**2]))


def make(label: str, bench: Benchmark, target: np.ndarray, lean: Optional[np.ndarray] = None) -> Agent:
    caution, constrained = CONTROLLERS[label]
    config = Config(dt=bench.dt, horizon=bench.K, position_noise=bench.position_noise, imu_noise=bench.imu_noise,
                    force_noise=bench.force_noise, wind_step=bench.wind_step, gain_step=bench.gain_step,
                    goal=tuple(target), goal_std=bench.goal_std, constrained=constrained, caution=caution)
    faults = bench.heading_max > 0.0 or bench.power_rate > 0.0 or bench.buckle_prob > 0.0
    gain_mean, gain_var = gain_prior(bench) if faults else (np.array([1.0, 0.0, 1.0]), np.zeros(3))
    if lean is not None:
        gain_mean = gain_mean + np.concatenate([lean, [0.0]])
    s_h = bench.drag_prior[0] * bench.wind_scale / math.sqrt(2.0)
    s_v = bench.drag_prior[0] * 0.4
    return Agent(config,
                 state_mean=np.concatenate([bench.start, np.zeros(3), [0.0, 0.0, GRAVITY]]),
                 state_variance=np.array([bench.position_noise**2] * 3 + [0.05**2] * 3 + [1.0] * 3),
                 parameter_mean=np.concatenate([[0.0, 0.0, 0.0, bench.drag_prior[0]], gain_mean]),
                 parameter_variance=np.concatenate([[s_h**2, s_h**2, s_v**2, bench.drag_prior[1]**2], gain_var]))


def specific_force(u: np.ndarray, v: np.ndarray, wind: np.ndarray, k: float, config: Config, psi: float = 0.0,
                   efficiency: float = 1.0, bias: Optional[np.ndarray] = None) -> np.ndarray:
    """Specific force of the true quadcopter without the unmodelled force: the
    thrust axis is rotated by the heading misalignment psi, the thrust scaled
    by the efficiency, and a buckled rotor adds a horizontal force."""
    tau = efficiency * GRAVITY * (1.0 + config.thrust_range * u[0])
    phi, pitch = config.tilt_max * u[1], config.tilt_max * u[2]
    n = np.array([math.cos(phi) * math.sin(pitch), -math.sin(phi), math.cos(phi) * math.cos(pitch)])
    rotation = np.array([[math.cos(psi), -math.sin(psi), 0.0], [math.sin(psi), math.cos(psi), 0.0], [0.0, 0.0, 1.0]])
    n = rotation @ n
    rotor_plane = np.eye(3) - np.outer(n, n)
    out = tau * n - k * tau * rotor_plane @ (v - wind)
    return out if bias is None else out + tau * bias


def run_job(job: dict) -> dict:
    """One episode. The goal cost of step k is |y_p,k - p_*|^2 / (2 S_*) on the
    position fix, and the wall-clock time counts planning and filtering."""
    bench, seed, label = job["bench"], job["seed"], job["label"]
    d = draw(bench, seed)
    agent = make(label, bench, d["target"], d["lean"])
    T, dt = bench.T, bench.dt
    p, v = np.zeros((T + 1, 3)), np.zeros((T + 1, 3))
    p[0] = d["start"]
    u = np.zeros((T, 3))
    multiplier, plans, seconds = np.zeros(T), np.zeros(T, int), np.zeros(T)
    estimate = np.zeros((T, 4))
    gains = np.zeros((T, 3))
    fixes = np.zeros((T + 1, 3))
    fixes[0] = p[0] + bench.position_noise * d["position"][0]
    agent.observe(position=fixes[0])
    gravity = np.array([0.0, 0.0, GRAVITY])
    for t in range(T):
        if label == REFERENCE:
            e, psi = d["efficiency"][t + 1], d["psi"]
            agent.set_parameters(np.concatenate([e * (d["k"] * d["wind"][t + 1] + d["bias"][t + 1]),
                                                 [e * d["k"], e * math.cos(psi), e * math.sin(psi), e]]))
        start = time.perf_counter()
        u[t] = agent.plan()
        seconds[t] = time.perf_counter() - start
        multiplier[t], plans[t] = agent.last_multiplier, agent.last_plans
        f = (specific_force(u[t], v[t], d["wind"][t + 1], d["k"], agent.config, d["psi"], d["efficiency"][t + 1],
                            d["bias"][t + 1]) + bench.force_noise * d["force"][t])
        a = f - gravity
        v[t + 1] = v[t] + dt * a
        p[t + 1] = p[t] + dt * v[t] + 0.5 * dt**2 * a
        fixes[t + 1] = p[t + 1] + bench.position_noise * d["position"][t + 1]
        start = time.perf_counter()
        agent.update(u[t], fixes[t + 1], f + bench.imu_noise * d["imu"][t])
        seconds[t] += time.perf_counter() - start
        estimate[t] = agent.mean[NX_PARAMS]
        gains[t] = agent.mean[-3:]
    cost = np.sum((fixes[1:] - d["target"])**2, axis=1) / (2.0 * bench.goal_std**2)
    true_cost = np.sum((p[1:] - d["target"])**2, axis=1) / (2.0 * bench.goal_std**2)
    return dict(label=label, imu_noise=bench.imu_noise, seed=seed, target=d["target"], k=d["k"], speed=d["speed"],
                wind=d["wind"], p=p, v=v, u=u, fixes=fixes, cost=cost, G=float(cost.mean()),
                true_cost=true_cost, G_true=float(true_cost.mean()),
                miss=float(np.linalg.norm(p[-1] - d["target"])), multiplier=multiplier, plans=plans,
                seconds=seconds, wall=float(seconds.sum()), estimate=estimate, gains=gains, psi=d["psi"],
                efficiency=d["efficiency"], buckles=d["buckles"])


def run_many(jobs, processes: Optional[int] = None) -> list:
    """``run_job`` for every job in parallel processes, in the order of the jobs."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    jobs = list(jobs)
    processes = processes or min(len(jobs), os.cpu_count() or 1)
    if processes <= 1:
        return [run_job(job) for job in jobs]
    with ProcessPoolExecutor(processes, mp_context=get_context("forkserver")) as pool:
        return list(pool.map(run_job, jobs, chunksize=2))


def save(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)
