"""CBFE agent for a light-dark positioning task: a hidden state whose
measurement precision depends on where it is (main.tex, Section V-C).

Model
-----
    x_k = x_{k-1} + u_k + w_k ,        w_k ~ N(0, q) ,
    y_k = x_k + v_k ,                  v_k ~ N(0, r(x_k)) ,
    r(x) = r0 (r_dark / r0)^s(x) ,     s(x) = 1 - exp(-((x - x_L) / w)^4) ,

so the state is measured precisely in a flat-bottomed light around x_L and not
at all in the dark, where the noise is r_dark everywhere.
``GridFilter`` computes the exact posterior on a grid. The planning model
takes its moments N(m, V) as the belief node of the state.

Planning model
--------------
The beliefs stay fixed within the plan (explicit dual control), so the
predicted state beliefs are N(m_k, V_k) with m_k = m_{k-1} + u_k and
V_k = V + k q, and the plan u = (u_1..u_K) minimises

    J(u) = sum_k [c(u_k) + ((m_k - x_*)^2 + V_k) / (2 S_*) - g_k I_k(m_k)] ,
    I_k(m) = 1/2 ln(1 + V_k / r(m)) ,

the state information of the observation node of step k, with the noise
at the predicted mean, and the control prior c(u) = -(a-1) ln(1 - u^2).
Without information terms, the predicted variance does not depend on the
plan, so the planner has no reason to move towards the light: it lacks the
dual effect of Bar-Shalom and Tse.

CBFE weights the first step with the KKT multiplier of

    I_1(m + u_1) >= beta ,   beta = 1/2 ln((V + q) / S_*) ,

the rate-distortion function of the predicted state N(m + u, V + q) at the
goal tolerance S_*: the least information with which the next state can be
known to within the tolerance. The floor is capped at the information of the
most informative input minus ``floor_margin``, so the constraint stays
feasible. The dual MPC weights every step with a fixed weight. The
belief-space planner ("bsp") instead propagates the variance through the
measurement updates of the most likely observations (Platt et al., 2010),

    V_k^- = V_{k-1}^+ + q ,   V_k^+ = V_k^- r(m_k) / (V_k^- + r(m_k)) ,

and minimises sum_k [c(u_k) + ((m_k - x_*)^2 + V_k^-) / (2 S_*)].

``_sweeps`` is a Gauss-Newton iterative LQR on m_k = m_{k-1} + u_k that
drops the negative curvature of the information terms, with a backtracking
line search; the KKT search is the upward bracket from one and regula falsi
of the other agents.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from agents.common import input_nodes

CHI2_95 = 3.841458820694124     # chi-square quantile of one degree of freedom at 0.95


@dataclass(frozen=True)
class LightDarkModel:
    """q: process noise variance; the noise r(x) of the module docstring with
    light position ``light``, half-width ``width``, and noise r0 in the light
    and r_dark in the dark."""
    q: float = 1e-4
    r0: float = 1e-3
    r_dark: float = 100.0
    width: float = 0.8
    light: float = 1.5
    one_sided: bool = False     # True: everything beyond the light is lit as well

    def _shape(self, x):
        z = (np.asarray(x, float) - self.light) / self.width
        if self.one_sided:
            z = np.minimum(z, 0.0)
        return 1.0 - np.exp(-z**4), z

    def noise(self, x):
        s, _ = self._shape(x)
        return self.r0 * np.exp(math.log(self.r_dark / self.r0) * s)

    def noise_derivative(self, x):
        s, z = self._shape(x)
        r = self.r0 * np.exp(math.log(self.r_dark / self.r0) * s)
        return r * math.log(self.r_dark / self.r0) * np.exp(-z**4) * 4.0 * z**3 / self.width


@dataclass
class GridFilter:
    """Exact posterior of the state on a uniform grid."""
    model: LightDarkModel
    mean: float = 0.0
    variance: float = 0.25
    low: float = -4.0
    high: float = 5.0
    points: int = 1801

    def __post_init__(self):
        self.x = np.linspace(self.low, self.high, self.points)
        self.dx = self.x[1] - self.x[0]
        density = np.exp(-0.5 * (self.x - self.mean)**2 / self.variance)
        self.p = density / density.sum()
        half = int(math.ceil(6.0 * math.sqrt(self.model.q) / self.dx))
        offsets = self.dx * np.arange(-half, half + 1)
        kernel = np.exp(-0.5 * offsets**2 / self.model.q)
        self.kernel = kernel / kernel.sum()

    def update(self, control_input: float, output: float) -> None:
        shifted = np.interp(self.x - control_input, self.x, self.p, left=0.0, right=0.0)
        predicted = np.convolve(shifted, self.kernel, mode="same")
        r = self.model.noise(self.x)
        # moments of the one-step predictive distribution of the output
        weights = predicted / predicted.sum()
        self.last_output_mean = float(weights @ self.x)
        self.last_output_variance = float(weights @ (self.x - self.last_output_mean)**2 + weights @ r)
        log_likelihood = -0.5 * ((output - self.x)**2 / r + np.log(r))
        posterior = predicted * np.exp(log_likelihood - log_likelihood.max())
        total = posterior.sum()
        if not np.isfinite(total) or total <= 0.0:       # the prediction missed the output
            posterior, total = np.exp(log_likelihood - log_likelihood.max()), None
            total = posterior.sum()
        self.p = posterior / total

    def moments(self) -> Tuple[float, float]:
        m = float(self.p @ self.x)
        return m, float(self.p @ (self.x - m)**2)


@dataclass
class LightDarkConfig:
    """Goal, horizon and the kind of planner.

    planner   "cbfe" (KKT multiplier on the state information of the next
              step), "dual" (fixed ``weight`` on every step), "bsp"
              (belief-space planning), "mpcx" (nominal MPC with an
              experiment-design constraint, see ``_solve_mpcx``), or "none"
              (no information term)
    accuracy  gamma of MPC-X: the state precision predicted at the end of the
              horizon must reach gamma chi2_0.95(1) / (2 S_*)
    """
    goal_mean: float = 0.0
    goal_variance: float = 0.01
    horizon: int = 5
    control_prior_shape: float = 2.0
    planner: str = "cbfe"
    weight: float = 0.0
    floor_margin: float = 0.01
    max_multiplier: float = 1e6
    root_tolerance: float = 1e-3
    bound_margin: float = 1e-6
    task_length: Optional[int] = None
    max_sweeps: int = 60
    sweep_tolerance: float = 1e-10
    accuracy: float = 1.0


@dataclass
class LightDarkAgent:
    """Grid filter and the planner of the module docstring. ``plan()`` returns
    the next input, ``update(u, y)`` adds the output that followed it."""
    config: LightDarkConfig
    model: LightDarkModel = field(default_factory=LightDarkModel)
    prior_mean: float = 0.0
    prior_variance: float = 0.25

    def __post_init__(self):
        self.filter = GridFilter(self.model, self.prior_mean, self.prior_variance)
        self._nodes = input_nodes(1)[:, 0]
        self._previous_plan: Optional[np.ndarray] = None
        self.last_multiplier = 0.0
        self.last_floor = 0.0
        self.t = 0

    def update(self, control_input: float, output: float) -> None:
        self.filter.update(control_input, output)
        self.t += 1

    # ----- terms -----

    def information(self, m, V):
        return 0.5 * np.log1p(V / self.model.noise(m))

    def _information_derivatives(self, m: float, V: float) -> Tuple[float, float]:
        r = float(self.model.noise(m))
        dr = float(self.model.noise_derivative(m))
        first = -0.5 * V * dr / (r * (r + V))
        h = 1e-5
        plus = -0.5 * V * float(self.model.noise_derivative(m + h)) / (
            float(self.model.noise(m + h)) * (float(self.model.noise(m + h)) + V))
        minus = -0.5 * V * float(self.model.noise_derivative(m - h)) / (
            float(self.model.noise(m - h)) * (float(self.model.noise(m - h)) + V))
        return first, (plus - minus) / (2 * h)

    def objective(self, plan: np.ndarray, weights: np.ndarray, m: float, V: float) -> float:
        c = self.config
        K = len(plan)
        means = m + np.cumsum(plan)
        variances = V + self.model.q * np.arange(1, K + 1)
        value = float(np.sum(-(c.control_prior_shape - 1.0) * np.log1p(-np.square(plan))
                             + (np.square(means - c.goal_mean) + variances) / (2.0 * c.goal_variance)))
        active = weights > 0.0
        if active.any():
            value -= float(np.sum(weights[active] * self.information(means[active], variances[active])))
        return value

    def bsp_objective(self, plan: np.ndarray, m: float, V: float) -> float:
        c = self.config
        value, variance = 0.0, V
        for k, u in enumerate(plan):
            mean = m + float(np.sum(plan[:k + 1]))
            prior = variance + self.model.q
            value += (-(c.control_prior_shape - 1.0) * math.log1p(-u * u)
                      + ((mean - c.goal_mean)**2 + prior) / (2.0 * c.goal_variance))
            r = float(self.model.noise(mean))
            variance = prior * r / (prior + r)
        return value

    # ----- planning -----

    def _sweeps(self, plan: np.ndarray, weights: np.ndarray, m: float, V: float) -> Tuple[np.ndarray, float]:
        c = self.config
        K = len(plan)
        a1 = c.control_prior_shape - 1.0
        S = c.goal_variance
        bound = 1.0 - c.bound_margin
        plan = np.clip(plan, -bound, bound)
        variances = V + self.model.q * np.arange(1, K + 1)
        value = self.objective(plan, weights, m, V)
        for _ in range(c.max_sweeps):
            means = m + np.cumsum(plan)
            previous = np.concatenate([[m], means[:-1]])
            lx = (means - c.goal_mean) / S
            lxx = np.full(K, 1.0 / S)
            for k in np.nonzero(weights > 0.0)[0]:
                d1, d2 = self._information_derivatives(float(means[k]), float(variances[k]))
                lx[k] -= weights[k] * d1
                lxx[k] += max(-weights[k] * d2, 0.0)
            gu = 2.0 * a1 * plan / (1.0 - plan**2)
            guu = 2.0 * a1 * (1.0 + plan**2) / (1.0 - plan**2)**2
            vx, vxx = 0.0, 0.0
            feedforward, feedback = np.zeros(K), np.zeros(K)
            for k in range(K - 1, -1, -1):
                wx, wxx = lx[k] + vx, lxx[k] + vxx
                qu, quu = gu[k] + wx, guu[k] + wxx
                feedforward[k], feedback[k] = -qu / quu, -wxx / quu
                vx, vxx = wx - wxx * qu / quu, wxx - wxx**2 / quu
            step, improved = 1.0, False
            for _ in range(25):
                new, state = np.zeros(K), m
                for k in range(K):
                    new[k] = plan[k] + step * feedforward[k] + feedback[k] * (state - previous[k])
                    state += new[k]
                if np.all(np.abs(new) < bound):
                    new_value = self.objective(new, weights, m, V)
                    if new_value < value:
                        improved = True
                        break
                step *= 0.5
            if not improved:
                break
            decrease = value - new_value
            plan, value = new, new_value
            if decrease < c.sweep_tolerance * (1.0 + abs(value)):
                break
        return plan, value

    def _starts(self, m: float, start: Optional[np.ndarray]) -> list:
        K = self.config.horizon
        starts = [np.zeros(K) if start is None else np.array(start, float)]
        towards = np.zeros(K)
        towards[0] = float(np.clip(self.model.light - m, -0.9, 0.9))
        starts += [np.zeros(K), towards]
        return starts

    def _solve(self, weights: np.ndarray, m: float, V: float, start: Optional[np.ndarray]):
        best, best_value = None, np.inf
        starts = self._starts(m, start) if np.any(weights > 0.0) else self._starts(m, start)[:1]
        for initial in starts:
            plan, value = self._sweeps(initial, weights, m, V)
            if value < best_value:
                best, best_value = plan, value
        return best, best_value

    def _solve_bsp(self, m: float, V: float, start: Optional[np.ndarray]):
        bound = 1.0 - self.config.bound_margin
        best = None
        for initial in self._starts(m, start):
            result = minimize(self.bsp_objective, np.clip(initial, -bound, bound), args=(m, V),
                              method="L-BFGS-B", bounds=[(-bound, bound)] * self.config.horizon,
                              options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 300})
            if best is None or result.fun < best.fun:
                best = result
        return best.x, float(best.fun)

    def nominal_cost(self, plan: np.ndarray, m: float) -> float:
        """Tracking cost of the mean predictions and the control prior."""
        c = self.config
        means = m + np.cumsum(plan)
        return float(np.sum(-(c.control_prior_shape - 1.0) * np.log1p(-np.square(plan))
                            + np.square(means - c.goal_mean) / (2.0 * c.goal_variance)))

    def end_variance(self, plan: np.ndarray, m: float, V: float) -> float:
        """State variance at the end of the horizon after the planned
        measurements, with the noise at the predicted means."""
        variance, mean = V, m
        for u in plan:
            mean += float(u)
            prior = variance + self.model.q
            r = float(self.model.noise(mean))
            variance = prior * r / (prior + r)
        return variance

    def _solve_mpcx(self, m: float, V: float, start: Optional[np.ndarray]) -> Tuple[np.ndarray, bool]:
        """MPC with integrated experiment design (Larsson et al., 2013) for the
        hidden state: the nominal MPC subject to a state precision of at least
        gamma chi2_0.95(1) / (2 S_*) at the end of the horizon. Without a
        ramp over the experiment, because the state drifts and its information
        does not accumulate. If no plan meets the requirement, the plan with
        the largest precision stands."""
        c = self.config
        K = c.horizon
        bound = 1.0 - c.bound_margin
        required = math.log(c.goal_variance / (c.accuracy * CHI2_95 / 2.0))
        plan, _ = self._solve(np.zeros(K), m, V, start)
        if math.log(self.end_variance(plan, m, V)) <= required:
            return plan, False
        towards, mean = np.zeros(K), m
        for k in range(K):
            towards[k] = float(np.clip(self.model.light + 0.2 - mean, -0.9, 0.9))
            mean += towards[k]
        margin = {"type": "ineq", "fun": lambda u: required - math.log(self.end_variance(u, m, V))}
        best = None
        for initial in (towards, plan):
            result = minimize(self.nominal_cost, np.clip(initial, -bound, bound), args=(m,), method="SLSQP",
                              bounds=[(-bound, bound)] * K, constraints=[margin],
                              options={"maxiter": 200, "ftol": 1e-10})
            if margin["fun"](result.x) >= -1e-6 and (best is None or result.fun < best.fun):
                best = result
        if best is not None:
            return best.x, True
        result = minimize(lambda u: math.log(self.end_variance(u, m, V)), np.clip(towards, -bound, bound),
                          method="L-BFGS-B", bounds=[(-bound, bound)] * K)
        return result.x, True

    def floor(self, m: float, V: float) -> float:
        """1/2 ln(A / S_*), with A = V + (T - t) q the spread by the end of the
        task (A = V + q without ``task_length``), capped at the information of
        the most informative input."""
        c = self.config
        prior = V + self.model.q
        steps = 1 if c.task_length is None else max(c.task_length - self.t, 1)
        beta = 0.5 * math.log((V + steps * self.model.q) / c.goal_variance)
        best = float(np.max(self.information(m + self._nodes, prior)))
        return min(beta, best - c.floor_margin)

    def _root(self, m: float, V: float, beta: float, plan: np.ndarray) -> Tuple[float, np.ndarray]:
        c = self.config
        K = c.horizon
        prior = V + self.model.q

        def evaluate(value, start):
            weights = np.zeros(K)
            weights[0] = value
            new_plan, _ = self._solve(weights, m, V, start)
            return float(self.information(m + new_plan[0], prior)) - beta + 1e-12, new_plan

        residual, plan0 = evaluate(0.0, plan)
        if residual >= 0.0:
            return 0.0, plan0
        low, value, start = (0.0, residual), 1.0, plan0
        while True:
            residual, new_plan = evaluate(value, start)
            if residual >= 0.0:
                high = (value, residual, new_plan)
                break
            low, start = (value, residual), new_plan
            if value >= c.max_multiplier:
                return value, new_plan
            value = min(4.0 * value, c.max_multiplier)
        if high[1] < c.root_tolerance:
            return high[0], high[2]
        side = 0
        for _ in range(20):
            (value_low, residual_low), (value_high, residual_high) = low, high[:2]
            value = value_high - residual_high * (value_high - value_low) / (residual_high - residual_low)
            residual, new_plan = evaluate(value, high[2])
            if abs(residual) < c.root_tolerance:
                return value, new_plan
            if residual > 0.0:
                high = (value, residual, new_plan)
                if side == 1:
                    low = (low[0], 0.5 * low[1])
                side = 1
            else:
                low = (value, residual)
                if side == -1:
                    high = (high[0], 0.5 * high[1], high[2])
                side = -1
        return high[0], high[2]

    def plan(self) -> float:
        c = self.config
        m, V = self.filter.moments()
        K = c.horizon
        warm = (None if self._previous_plan is None
                else np.concatenate([self._previous_plan[1:], self._previous_plan[-1:]]))
        self.last_multiplier, self.last_floor = 0.0, 0.0
        if c.planner == "bsp":
            plan, _ = self._solve_bsp(m, V, warm)
        elif c.planner == "mpcx":
            plan, active = self._solve_mpcx(m, V, warm)
            self.last_multiplier = 1.0 if active else 0.0
        elif c.planner == "dual":
            plan, _ = self._solve(np.full(K, c.weight), m, V, warm)
        else:
            plan, _ = self._solve(np.zeros(K), m, V, warm)
            if c.planner == "cbfe":
                beta = self.floor(m, V)
                self.last_floor = beta
                if beta > 0.0:
                    self.last_multiplier, plan = self._root(m, V, beta, plan)
        self._previous_plan = plan
        return float(plan[0])
