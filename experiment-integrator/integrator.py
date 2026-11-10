"""Dual control of an integrator with unknown gain (Åström and Helmersson,
Computers & Mathematics with Applications 12(6A), 653-662, 1986).

The problem (the paper's Sec. 2)
--------------------------------
    y_{t+1} = y_t + b u_t + sigma e_{t+1},    e_t ~ N(0, 1),           (2.1)
    b ~ N(b0, P0) unknown and constant,
    loss    L = sum_{t=1}^{T} y_t^2 / sigma^2.                          (2.2)

The controller knows sigma and the prior of b. Given the data, the gain has
the Gaussian posterior N(bhat_t, P_t) of a scalar Kalman filter (2.3)-(2.4),
and (y_t, bhat_t, P_t) is the hyperstate of the dual control problem. With
the normalised output eta = y / sigma, the normalised gain estimate
beta = bhat / sqrt(P) and the normalised input nu = u sqrt(P) / sigma, one
step of the filter reads, with w ~ N(0, 1),

    eta'  = eta + beta nu + sqrt(1 + nu^2) w,                          (2.5)
    beta' = beta sqrt(1 + nu^2) + nu w,                                (2.6)

so sigma and P drop out and the Bellman equation of the unconstrained
problem lives on (eta, beta):

    V_k(eta, beta) = min_nu  (eta + beta nu)^2 + 1 + nu^2 + E_w V_{k-1}(eta', beta'),   (2.8)

with V_0 = 0 and V_k(eta, beta) = V_k(|eta|, |beta|). For k = 1 the minimiser
is the cautious law nu = -eta beta / (1 + beta^2), i.e.
u = -bhat y / (bhat^2 + P), and V_1 = 1 + eta^2 / (1 + beta^2). With a known
gain V_k = k, and V_k - k is the paper's learning loss. The cautious
controller turns off when |beta| is small, because the input only adds
variance; the dual controller probes instead. Small |beta| = |bhat| / sqrt(P)
is therefore the regime where information gathering pays. ``DualDP`` solves
the Bellman equation on a grid and acts by one-step lookahead on V; its
expected loss is the optimum of the unconstrained problem and a lower bound
for every controller here. It reproduces the paper's Figs. 1 and 3 (see
integrator.ipynb).

The protocol
------------
The paper defines the problem but no benchmark protocol; its simulations
concern a slowly varying gain. The protocol here is ours: regulation from
y_0 = 0 over T = 50 steps with sigma = 0.1, the gain drawn from its prior
N(b0, P0 = 1) in every episode, and the prior means b0 of ``run.py``.

Controllers
-----------
Classical laws (on the exact Kalman filter, inputs clipped to |u| <= u_max):
certainty equivalence u = -y / bhat and cautious control; unclipped, the
optimal dual controller and the paper's approximation (5.1) of it. The ARX
agents of ``agents/`` run on the same Bayesian model: features (1, u, y), the
intercept and the pole known (prior precision 1e6 in noise units), the gain
with prior N(b0, P0), and the noise known (nu0 = 1e4), so that their
parameter posterior is the Kalman filter. Their goal prior is N(0, sigma^2),
i.e. as tight as the noise allows, and they plan over K = 5 steps with inputs
in [-1, 1].
"""

from __future__ import annotations

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
from numpy.polynomial.legendre import leggauss  # noqa: E402
from scipy.interpolate import RectBivariateSpline  # noqa: E402

from agents.common import MatrixNormalWishart, MultivariateNormal  # noqa: E402

RESULTS = HERE / "results"


@dataclass(frozen=True)
class Benchmark:
    sigma: float = 0.1        # noise std
    b0: float = 0.0           # prior mean of the gain
    P0: float = 1.0           # prior variance of the gain
    T: int = 50               # steps; the loss sums y_1 ... y_T
    u_max: float = 1.0        # input bound of the clipped laws and the agents
    K: int = 5                # planning horizon of the MPC agents
    known: float = 1e6        # prior precision of the intercept and the pole (noise units)
    nu0: float = 1e4          # Wishart degrees of freedom: the noise level is known
    dither: float = 0.0       # std of a known perturbation added to every applied input

    @property
    def beta0(self) -> float:
        """Normalised prior gain estimate b0 / sqrt(P0)."""
        return self.b0 / np.sqrt(self.P0)


@dataclass
class Episode:
    b: float
    y: np.ndarray             # (T+1,) y_0 ... y_T
    u: np.ndarray             # (T,)
    bhat: np.ndarray          # (T+1,) gain estimate before each input
    P: np.ndarray             # (T+1,)
    beta: np.ndarray          # (T,) multiplier or probing weight of the agent (nan if none)
    plan_time: np.ndarray     # (T,) [s]

    def loss(self, sigma: float) -> float:
        return float(np.sum(self.y[1:] ** 2) / sigma**2)


def kalman(bhat, P, y, u, y_next, sigma):
    """Posterior N(bhat', P') of the gain after y_next = y + b u + e."""
    s = sigma**2 + u * u * P
    return bhat + P * u * (y_next - y - bhat * u) / s, P * sigma**2 / s


# ---------------------------------------------------------------------------
# Optimal dual control by dynamic programming
# ---------------------------------------------------------------------------

@dataclass
class DualDP:
    """Value functions V_0 ... V_T of the normalised Bellman equation (2.8)
    of the paper on a grid over (|eta|, |beta|), quadratically spaced
    towards zero.

    The numerics follow Sec. 3 of the paper. Cubic splines interpolate V_k
    on the grid mirrored to negative eta and beta, so that the spline is
    even. Beyond the grid, V grows in eta^2 with the slope of the last grid
    interval, and it is held constant in beta (V_k tends to k, the loss with
    a known gain, as beta grows). The expectation over w uses Gauss-Legendre
    panels on [-8, 8], split where eta' = 0 and around beta' = 0 (the
    paper's discontinuity points), because V has kinks there and, for large
    |nu|, beta' crosses the peak of V in a w-interval of width about 1/|nu|,
    which a fixed Gauss-Hermite rule misses. The minimisation over nu
    searches nu = nu_c + z / sqrt(1 + beta^2) on a grid of z, around the
    cautious input nu_c = -eta beta / (1 + beta^2): the immediate cost is
    then z^2 + const for every beta, so one grid resolves the minimiser both
    for well-known gains and for probing. The vertex of the parabola through
    the best grid point and its neighbours refines the minimum, and ``nu``
    refines the minimiser further by golden-section search."""
    T: int
    x_max: float = 60.0
    n_x: int = 121
    z_max: float = 12.0
    n_z: int = 161
    n_gl: int = 8
    V: Optional[np.ndarray] = field(default=None, repr=False)

    W_MAX = 8.0                 # integration range of w
    W_CUTS = (-3.0, 3.0)        # fixed panel boundaries

    def __post_init__(self):
        self.x = self.x_max * np.linspace(0.0, 1.0, self.n_x) ** 2
        self.x_gl, self.g_gl = leggauss(self.n_gl)
        self._splines = {}

    # interpolation of V_k at (|eta|, |beta|) ------------------------------
    def _spline(self, k: int) -> RectBivariateSpline:
        if k not in self._splines:
            xs = np.concatenate([-self.x[:0:-1], self.x])
            V = np.concatenate([self.V[k][:0:-1], self.V[k]], axis=0)
            V = np.concatenate([V[:, :0:-1], V], axis=1)
            self._splines[k] = RectBivariateSpline(xs, xs, V, kx=3, ky=3, s=0)
        return self._splines[k]

    def value(self, k: int, eta, beta):
        eta, beta = np.abs(eta), np.minimum(np.abs(beta), self.x[-1])
        e = np.minimum(eta, self.x[-1])
        v = self._spline(k).ev(e, beta).reshape(np.shape(e))
        slope = (self.V[k][-1] - self.V[k][-2]) / (self.x[-1] ** 2 - self.x[-2] ** 2)
        return v + np.interp(beta, self.x, slope) * np.maximum(eta**2 - self.x[-1] ** 2, 0.0)

    def expectation(self, k: int, mean, r, beta_r, nu):
        """E_w V_k(mean + r w, beta_r + nu w), w ~ N(0, 1), broadcast."""
        L = self.W_MAX
        moving = np.abs(nu) > 1e-9
        w_beta = np.where(moving, -beta_r / np.where(moving, nu, 1.0), L)
        d = np.minimum(3.0 / np.maximum(np.abs(nu), 1e-9), 2.0)
        cuts = np.broadcast_arrays(-L, -mean / r, w_beta - d, w_beta, w_beta + d, *self.W_CUTS, L)
        cuts = np.sort(np.clip(np.stack(cuts, axis=-1), -L, L), axis=-1)
        ev = 0.0
        for p in range(cuts.shape[-1] - 1):
            half = 0.5 * (cuts[..., p + 1] - cuts[..., p])
            mid = 0.5 * (cuts[..., p + 1] + cuts[..., p])
            for x, g in zip(self.x_gl, self.g_gl):
                w = mid + half * x
                ev = ev + (half * g * np.exp(-0.5 * w * w) / np.sqrt(2.0 * np.pi)
                           * self.value(k, mean + r * w, beta_r + nu * w))
        return ev

    def q_value(self, k: int, eta, beta, nu):
        """(eta + beta nu)^2 + 1 + nu^2 + E V_{k-1}(eta', beta'), broadcast."""
        r = np.sqrt(1.0 + nu * nu)
        mean = eta + beta * nu
        cost = mean**2 + 1.0 + nu * nu
        if k > 1:
            cost = cost + self.expectation(k - 1, mean, r, beta * r, nu)
        return cost

    def minimise(self, k: int, eta, beta):
        """min over nu of ``q_value`` at the points (eta, beta), and the minimiser."""
        nu_c = -eta * beta / (1.0 + beta * beta)
        scale = 1.0 / np.sqrt(1.0 + beta * beta)
        zs = np.linspace(-self.z_max, self.z_max, self.n_z)
        h = zs[1] - zs[0]
        C = np.empty(np.shape(eta) + (self.n_z,))
        for chunk in np.array_split(np.arange(self.n_z), 8):
            C[..., chunk] = self.q_value(k, eta[..., None], beta[..., None],
                                         nu_c[..., None] + scale[..., None] * zs[chunk])
        i = np.clip(C.argmin(axis=-1), 1, self.n_z - 2)
        c0, c1, c2 = (np.take_along_axis(C, (i + j)[..., None], -1)[..., 0] for j in (-1, 0, 1))
        curv = c0 - 2.0 * c1 + c2
        z = zs[i] + np.clip(0.5 * h * (c0 - c2) / np.where(curv > 0, curv, 1.0) * (curv > 0), -h, h)
        cz = self.q_value(k, eta, beta, nu_c + scale * z)
        better = cz < c1
        return np.where(better, cz, c1), nu_c + scale * np.where(better, z, zs[i])

    def solve(self, verbose: bool = False, processes: Optional[int] = None) -> "DualDP":
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        self.V = np.zeros((self.T + 1, self.n_x, self.n_x))
        E, B = np.meshgrid(self.x, self.x, indexing="ij")
        rows = np.array_split(np.arange(self.n_x), 4 * (processes or os.cpu_count() or 1))
        params = dict(T=self.T, x_max=self.x_max, n_x=self.n_x, z_max=self.z_max, n_z=self.n_z, n_gl=self.n_gl)
        t0 = time.perf_counter()
        pool = (ProcessPoolExecutor(processes, mp_context=get_context("forkserver"))
                if processes != 1 else None)
        try:
            for k in range(1, self.T + 1):
                if pool is None:
                    self._splines = {}
                    self.V[k] = self.minimise(k, E, B)[0]
                else:
                    jobs = [(params, k, self.V[k - 1], E[r], B[r]) for r in rows]
                    self.V[k] = np.concatenate(list(pool.map(_bellman_rows, jobs)))
                if verbose:
                    print(f"V_{k}: {time.perf_counter() - t0:.0f} s", flush=True)
        finally:
            if pool is not None:
                pool.shutdown()
        self._splines = {}
        return self

    def nu(self, k: int, eta: float, beta: float) -> float:
        """Optimal normalised input with k terms of the loss to go."""
        if k == 1:
            return -eta * beta / (1.0 + beta * beta)
        a, b = abs(eta), abs(beta)
        grid = (-a * b / (1.0 + b * b)
                + np.linspace(-self.z_max, self.z_max, 2 * self.n_z + 1) / np.sqrt(1.0 + b * b))
        i = int(np.argmin(self.q_value(k, a, b, grid)))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        g = (np.sqrt(5.0) - 1.0) / 2.0
        x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
        f1, f2 = self.q_value(k, a, b, x1), self.q_value(k, a, b, x2)
        for _ in range(30):
            if f1 < f2:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - g * (hi - lo)
                f1 = self.q_value(k, a, b, x1)
            else:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + g * (hi - lo)
                f2 = self.q_value(k, a, b, x2)
        nu = 0.5 * (lo + hi)
        return float(np.sign(eta if eta != 0 else 1.0) * np.sign(beta if beta != 0 else 1.0) * nu)

    def save(self, path: Path) -> None:
        np.savez(path, V=self.V, T=self.T, x_max=self.x_max, n_x=self.n_x, z_max=self.z_max,
                 n_z=self.n_z, n_gl=self.n_gl)

    @classmethod
    def load(cls, path: Path) -> "DualDP":
        d = np.load(path)
        dp = cls(T=int(d["T"]), x_max=float(d["x_max"]), n_x=int(d["n_x"]), z_max=float(d["z_max"]),
                 n_z=int(d["n_z"]), n_gl=int(d["n_gl"]))
        dp.V = d["V"]
        return dp


def _bellman_rows(job):
    """V_k on some rows of the grid, given V_{k-1} (a worker of ``DualDP.solve``)."""
    params, k, V_prev, E, B = job
    dp = DualDP(**params)
    dp.V = np.zeros((k, dp.n_x, dp.n_x))
    dp.V[k - 1] = V_prev
    return dp.minimise(k, E, B)[0]


def dual_dp(T: int, verbose: bool = False) -> DualDP:
    """The solved DP for horizon T, cached in results/."""
    path = RESULTS / f"dual_dp_T{T}.npz"
    if path.exists():
        return DualDP.load(path)
    RESULTS.mkdir(parents=True, exist_ok=True)
    dp = DualDP(T=T).solve(verbose=verbose)
    dp.save(path)
    return dp


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

class KalmanController:
    """Base class of the classical laws: acts on the exact Kalman filter."""

    def __init__(self, bench: Benchmark):
        self.bench = bench
        self.bhat, self.P = bench.b0, bench.P0
        self.last_beta = np.nan

    def learn(self, y, u, y_next):
        self.bhat, self.P = kalman(self.bhat, self.P, y, u, y_next, self.bench.sigma)


class CertaintyEquivalence(KalmanController):
    """u = -y / bhat, clipped. At bhat = 0 the law takes its limit for
    bhat -> 0+, the full input against the output, so that it does not turn
    off; the paper's certainty equivalence fails by too large inputs when
    the gain estimate is close to zero, not by turn-off."""

    def act(self, y, t):
        u = -y / self.bhat if self.bhat != 0.0 else -np.sign(y) * self.bench.u_max
        return float(np.clip(u, -self.bench.u_max, self.bench.u_max))


class Cautious(KalmanController):
    def act(self, y, t):
        u = -self.bhat * y / (self.bhat**2 + self.P)
        return float(np.clip(u, -self.bench.u_max, self.bench.u_max))


class OptimalDual(KalmanController):
    """Bayes-optimal unconstrained dual control by one-step lookahead on the
    value functions of ``DualDP``."""

    def __init__(self, bench: Benchmark, dp: DualDP):
        super().__init__(bench)
        self.dp = dp

    def act(self, y, t):
        s = self.bench.sigma
        sqrtP = np.sqrt(self.P)
        nu = self.dp.nu(self.bench.T - t, y / s, self.bhat / sqrtP)
        return float(nu * s / sqrtP)


def approximate_dual_law(eta, beta):
    """The paper's approximation (5.1) of the optimal dual law for T = 31,

        |nu| = (0.56 + |beta|) / (2.2 + 0.08 |beta| + beta^2) |eta| + 1.9 / (1.7 + beta^4),

    with the sign of the cautious input -eta beta, extended from the octant
    eta, beta > 0 by the antisymmetry of nu in eta and in beta."""
    a, b = np.abs(eta), np.abs(beta)
    nu = (0.56 + b) / (2.2 + 0.08 * b + b * b) * a + 1.9 / (1.7 + b**4)
    return -np.where(eta < 0, -1.0, 1.0) * np.where(beta < 0, -1.0, 1.0) * nu


class ApproximateDual(KalmanController):
    """The stationary approximation (5.1) of the optimal dual law, unclipped."""

    def act(self, y, t):
        s = self.bench.sigma
        sqrtP = np.sqrt(self.P)
        return float(approximate_dual_law(y / s, self.bhat / sqrtP) * s / sqrtP)


def policy_values(dp: DualDP, law, T: int) -> DualDP:
    """Expected losses V_0 ... V_T of the stationary normalised law
    nu = law(eta, beta), on the grid and with the quadrature of ``dp``; the
    law must be odd in eta and in beta, like the optimal one."""
    ev = DualDP(T=T, x_max=dp.x_max, n_x=dp.n_x, z_max=dp.z_max, n_z=dp.n_z, n_gl=dp.n_gl)
    ev.V = np.zeros((T + 1, ev.n_x, ev.n_x))
    E, B = np.meshgrid(ev.x, ev.x, indexing="ij")
    nu = law(E, B)
    for k in range(1, T + 1):
        ev.V[k] = ev.q_value(k, E, B, nu)
    return ev


def bayes_prior(bench: Benchmark) -> MatrixNormalWishart:
    """The MNW prior of the ARX agents over features (1, u, y): the gain
    with prior N(b0, P0), the intercept 0 and the pole 1 known, and the noise
    level sigma known."""
    s2 = bench.sigma**2
    return MatrixNormalWishart(coefficient_mean=np.array([[0.0], [bench.b0], [1.0]]),
                               coefficient_precision=np.diag([bench.known, s2 / bench.P0,
                                                              bench.known]),
                               noise_scale=np.array([[s2 * (bench.nu0 - 2.0)]]),
                               degrees_of_freedom=bench.nu0)


class ARXController:
    """An agent of ``agents/`` on the benchmark model (see the module
    docstring)."""

    def __init__(self, agent, bench: Benchmark):
        self.agent, self.bench = agent, bench
        agent.estimator.posterior = bayes_prior(bench)
        agent.estimator.output_buffer, agent.estimator.input_buffer = [np.array([0.0])], []
        self.last_beta = np.nan

    @property
    def bhat(self):
        return float(self.agent.estimator.posterior.coefficient_mean[1, 0])

    @property
    def P(self):
        post = self.agent.estimator.posterior
        return float(post.coefficient_covariance[1, 1] * self.bench.sigma**2)

    def act(self, y, t):
        u = float(np.clip(self.agent.plan(np.array([y]))[0], -self.bench.u_max, self.bench.u_max))
        self.last_beta = float(getattr(self.agent.planner, "last_multiplier", np.nan))
        return u

    def learn(self, y, u, y_next):
        self.agent.update(np.array([y_next]), np.array([u]))


def goal(bench: Benchmark) -> MultivariateNormal:
    return MultivariateNormal(mean=np.zeros(1), covariance=np.array([[bench.sigma**2]]))


def make_controller(spec: dict, bench: Benchmark):
    """Controller for a specification:

    {"kind": "ce"} | {"kind": "cautious"} | {"kind": "dual_dp"} | {"kind": "dual_approx"}
    {"kind": "cbfe", "fixed_beta": None or β, "info_floor": "task", "predictive" or
     "negentropy", "constraint_nodes": "all" or "next", "max_multiplier": 50.0}
    {"kind": "dual", "lambda_dual": λ, "rest_tolerance": None or tol,
     "input_cost": "quadratic" or "beta"}
    {"kind": "mpcx", "gamma": γ}
    """
    spec = dict(spec)
    kind = spec.pop("kind")
    if kind == "ce":
        return CertaintyEquivalence(bench)
    if kind == "cautious":
        return Cautious(bench)
    if kind == "dual_dp":
        return OptimalDual(bench, dual_dp(bench.T))
    if kind == "dual_approx":
        return ApproximateDual(bench)
    if kind == "cbfe":
        from agents.cbfe_arx import Agent
        agent = Agent.build(input_dim=1, output_dim=1, input_memory=0, output_memory=1,
                            goal_prior=goal(bench), horizon=bench.K, num_harmonics=0,
                            fixed_multiplier=spec.get("fixed_beta"),
                            information_floor_type=spec.get("info_floor", "task"),
                            constraint_nodes=spec.get("constraint_nodes", "all"),
                            max_multiplier=spec.get("max_multiplier", 50.0))
        return ARXController(agent, bench)
    if kind == "dual":
        from agents.dual_mpc import DualMPC
        agent = DualMPC.build(input_dim=1, output_dim=1, input_memory=0, output_memory=1,
                              goal_prior=goal(bench), horizon=bench.K,
                              probing_weight=spec["lambda_dual"], effort_weight=2.0,
                              num_harmonics=0)
        agent.config.rest_tolerance = spec.get("rest_tolerance")
        agent.config.input_cost = spec.get("input_cost", "quadratic")
        return ARXController(agent, bench)
    if kind == "mpcx":
        from agents.mpcx import MPCX
        agent = MPCX.build(input_dim=1, output_dim=1, input_memory=0, output_memory=1,
                           goal_prior=goal(bench), horizon=bench.K, accuracy=spec["gamma"],
                           experiment_length=bench.T, effort_weight=2.0, num_harmonics=0)
        return ARXController(agent, bench)
    raise ValueError(f"unknown controller kind '{kind}'")


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

def draw(bench: Benchmark, seed: int):
    """The gain b ~ N(b0, P0) and the noise sigma e_1 ... sigma e_T of ``seed``."""
    rng = np.random.default_rng(seed)
    b = bench.b0 + np.sqrt(bench.P0) * rng.standard_normal()
    return b, bench.sigma * rng.standard_normal(bench.T)


def known_gain_loss(bench: Benchmark, seed: int) -> float:
    """Loss of the minimum-variance law u = -y / b on the noise of ``seed``,
    i.e. sum e_t^2; its expectation is T."""
    return float(np.sum(draw(bench, seed)[1] ** 2) / bench.sigma**2)


def run_episode(spec: dict, bench: Benchmark, seed: int) -> Episode:
    """One episode with the gain and the noise of ``seed``. With
    ``bench.dither`` > 0, every applied input carries a known Gaussian
    perturbation of that standard deviation, drawn from its own stream of
    ``seed`` (the same for every controller), so that no result depends on
    whether a planner returns exact zeros at a symmetric point."""
    b, noise = draw(bench, seed)
    dither = bench.dither * np.random.default_rng([seed, 1]).standard_normal(bench.T)
    ctrl = make_controller(spec, bench)
    y = np.zeros(bench.T + 1)
    u, beta, plan_time = np.zeros(bench.T), np.full(bench.T, np.nan), np.zeros(bench.T)
    bhat, P = np.zeros(bench.T + 1), np.zeros(bench.T + 1)
    for t in range(bench.T):
        bhat[t], P[t] = ctrl.bhat, ctrl.P
        t0 = time.perf_counter()
        u[t] = ctrl.act(y[t], t)
        plan_time[t] = time.perf_counter() - t0
        u[t] += dither[t]          # unclipped: the optimal dual laws have no input bound
        beta[t] = ctrl.last_beta
        y[t + 1] = y[t] + b * u[t] + noise[t]
        ctrl.learn(y[t], u[t], y[t + 1])
    bhat[-1], P[-1] = ctrl.bhat, ctrl.P
    return Episode(b=float(b), y=y, u=u, bhat=bhat, P=P, beta=beta, plan_time=plan_time)


def run_job(job: dict) -> dict:
    ep = run_episode(job["spec"], job["bench"], job["seed"])
    return dict(label=job["label"], b0=job["bench"].b0, seed=job["seed"], episode=ep,
                loss=ep.loss(job["bench"].sigma))


def run_many(jobs, processes: Optional[int] = None) -> list:
    """``run_job`` for every job in parallel processes, in the order of the jobs."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    jobs = list(jobs)
    processes = processes or min(len(jobs), os.cpu_count() or 1)
    if processes <= 1:
        return [run_job(job) for job in jobs]
    if any(job["spec"]["kind"] == "dual_dp" for job in jobs):
        dual_dp(jobs[0]["bench"].T)                    # solve once before forking
    with ProcessPoolExecutor(processes, mp_context=get_context("forkserver")) as pool:
        return list(pool.map(run_job, jobs, chunksize=4))


def save(results: list, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(results, f)
    return path


def load(name: str) -> list:
    with open(RESULTS / f"{name}.pkl", "rb") as f:
        return pickle.load(f)
