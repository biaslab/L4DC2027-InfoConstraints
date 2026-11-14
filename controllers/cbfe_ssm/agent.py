"""CBFE agent for a scalar state-space model with a hidden state and an unknown
input gain (main.tex, Section V-C):

    x_k = x_{k-1} + b u_k + w_k ,   w_k ~ N(0, q) ,
    y_k = x_k + v_k ,               v_k ~ N(0, r) .

The model is linear in (x, b) given the inputs, so the Kalman filter over the
joint state (x, b) is exact. The planning model takes the marginals
q(x_t) = N(m_t, V_t) and q(b) = N(b_hat, P) as its belief nodes (Definition 4.1
of the paper), so the predicted states of the plan are Gaussian with

    m_k = m_{k-1} + b_hat u_k ,   V_k = V_{k-1} + P u_k^2 + q .

The plan u = (u_1, ..., u_K) minimises

    J(u) = sum_k [c(u_k) + ((m_k - x_*)^2 + V_k) / (2 S_*)]
           - gamma_par I_par(u_1) - gamma_state I_state(u_1) + eta rho(u_1) ,

with the control prior c(u) = -(a - 1) ln(1 - u^2), the goal prior N(x_*, S_*)
on the state, and

    I_par(u)   = 1/2 ln(1 + P u^2 / q)             parameter information of the
                                                   transition node of u_1
    I_state(u) = 1/2 ln(1 + (V_t + q + P u^2) / r)  state information of the
                                                   observation node of y_{t+1}
    rho(u)     = q(x_{t+1} > x_max)                 risk of the next state

The KKT multipliers gamma_par, gamma_state and eta of the constraints

    I_par >= beta_par ,   I_state >= beta_state ,   rho <= epsilon

follow cyclically: each one is the smallest multiplier that meets its
constraint with the others fixed, found by an upward bracket from one and
regula falsi. For the mode of the plan, every residual is monotone in its
own multiplier. The floors are the predictive-entropy excesses of eq. (10) at
the goal-reaching inputs, i.e. the inputs that the control prior admits and
whose mean prediction reaches the goal:

    beta_par   = 1/2 ln((q + P E[u^2]) / (q + S_*)) ,
    beta_state = 1/2 ln((r + V_t + q + P E[u^2]) / (r + S_*)) .

With the chance constraint, every floor is capped at the information of the
most informative input that meets it (minus ``floor_margin``), or of the
input with the least risk if none does. The constraints are then jointly
feasible, and the agent gathers as much information as the chance
constraint permits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from controllers.common import input_nodes

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


@dataclass
class HiddenStateConfig:
    """Model, goal and constraints of the agent.

    q, r                  process and measurement noise variances
    goal_mean, goal_variance   goal prior N(x_*, S_*) on the state
    horizon               planning horizon K
    control_prior_shape   a of the symmetric Beta(a, a) control prior on [-1, 1]
    parameter_information KKT multiplier for the parameter-information constraint
                          (False: the constraint is removed)
    state_information     KKT multiplier for the state-information constraint
    chance_bound          x_max of the region {x > x_max} to avoid (None: no
                          chance constraint)
    chance_level          epsilon, the admissible risk of the next state
    max_multiplier        cap of every bracket (never reached in the paper)
    root_tolerance        stopping tolerance of regula falsi on the residual
    """
    q: float = 0.01
    r: float = 0.01
    goal_mean: float = 0.0
    goal_variance: float = 0.01
    horizon: int = 5
    control_prior_shape: float = 2.0
    parameter_information: bool = True
    state_information: bool = True
    chance_bound: Optional[float] = None
    chance_level: float = 0.05
    max_multiplier: float = 1e6
    root_tolerance: float = 1e-3
    floor_margin: float = 0.01
    bound_margin: float = 1e-6


@dataclass
class HiddenStateAgent:
    """Joint Kalman filter over (x, b) and the constrained planner of the
    module docstring. ``plan()`` returns the next input, ``update(u, y)``
    adds the output that followed it."""
    config: HiddenStateConfig
    gain_mean: float = 0.0
    gain_variance: float = 1.0
    state_mean: float = 0.0
    state_variance: float = 0.01

    mean: np.ndarray = field(init=False)
    covariance: np.ndarray = field(init=False)
    _previous_plan: Optional[np.ndarray] = field(default=None, init=False)

    def __post_init__(self):
        self.mean = np.array([self.state_mean, self.gain_mean], dtype=float)
        self.covariance = np.diag([self.state_variance, self.gain_variance]).astype(float)
        self.last_multipliers = {"parameter": 0.0, "state": 0.0, "chance": 0.0}
        self.last_floors = {"parameter": float("nan"), "state": float("nan")}
        self.last_plans = 0
        self._nodes = input_nodes(1)[:, 0]

    # ----- filtering -----

    def update(self, control_input: float, output: float) -> None:
        """Predict the joint state through the input, then condition on the
        output (exact, since the model is linear in (x, b) given u)."""
        transition = np.array([[1.0, control_input], [0.0, 1.0]])
        mean = transition @ self.mean
        covariance = transition @ self.covariance @ transition.T + np.diag([self.config.q, 0.0])
        innovation_variance = covariance[0, 0] + self.config.r
        gain = covariance[:, 0] / innovation_variance
        self.mean = mean + gain * (output - mean[0])
        self.covariance = covariance - np.outer(gain, gain) * innovation_variance
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def marginals(self) -> Tuple[float, float, float, float]:
        """m_t, V_t, b_hat, P of the belief nodes."""
        return (float(self.mean[0]), float(self.covariance[0, 0]),
                float(self.mean[1]), float(self.covariance[1, 1]))

    # ----- terms of the objective -----

    def parameter_information(self, u, P):
        return 0.5 * np.log1p(P * np.square(u) / self.config.q)

    def state_information(self, u, V, P):
        return 0.5 * np.log1p((V + self.config.q + P * np.square(u)) / self.config.r)

    def risk(self, u, m, V, b_hat, P):
        """q(x_{t+1} > x_max) under the predictive N(m + b_hat u, V + q + P u^2)."""
        spread = np.sqrt(V + self.config.q + P * np.square(u))
        return ndtr((m + b_hat * np.asarray(u) - self.config.chance_bound) / spread)

    def floors(self, m, V, b_hat, P) -> Tuple[float, float]:
        """beta_par and beta_state over the goal-reaching inputs, capped with a
        chance constraint at the information of the most informative safe
        input (see the module docstring)."""
        c = self.config
        u = self._nodes
        log_weights = ((c.control_prior_shape - 1.0) * np.log1p(-np.square(u))
                       - 0.5 * np.square(m + b_hat * u - c.goal_mean) / c.goal_variance)
        weights = np.exp(log_weights - log_weights.max())
        mean_square = float(np.sum(weights * np.square(u)) / weights.sum())
        beta_parameter = 0.5 * math.log((c.q + P * mean_square) / (c.q + c.goal_variance))
        beta_state = 0.5 * math.log((c.r + V + c.q + P * mean_square) / (c.r + c.goal_variance))
        if c.chance_bound is not None:
            risk = self.risk(u, m, V, b_hat, P)
            safe = risk <= c.chance_level
            candidates = u[safe] if safe.any() else u[[int(np.argmin(risk))]]
            beta_parameter = min(beta_parameter, float(np.max(self.parameter_information(candidates, P)))
                                 - c.floor_margin)
            beta_state = min(beta_state, float(np.max(self.state_information(candidates, V, P)))
                             - c.floor_margin)
        return beta_parameter, beta_state

    def objective(self, plan, multipliers, m, V, b_hat, P) -> Tuple[float, np.ndarray]:
        """J of the module docstring and its gradient."""
        c = self.config
        K = len(plan)
        steps = np.arange(1, K + 1)
        predicted_mean = m + b_hat * np.cumsum(plan)
        predicted_variance = V + c.q * steps + P * np.cumsum(np.square(plan))
        error = predicted_mean - c.goal_mean
        prior_cost = -(c.control_prior_shape - 1.0) * np.log1p(-np.square(plan))
        value = float(np.sum(prior_cost) + np.sum(np.square(error) + predicted_variance)
                      / (2.0 * c.goal_variance))
        gradient = (2.0 * (c.control_prior_shape - 1.0) * plan / (1.0 - np.square(plan))
                    + (b_hat * np.cumsum(error[::-1])[::-1] + P * plan * (K - steps + 1))
                    / c.goal_variance)
        u = float(plan[0])
        if multipliers["parameter"] > 0.0:
            value -= multipliers["parameter"] * float(self.parameter_information(u, P))
            gradient[0] -= multipliers["parameter"] * P * u / (c.q + P * u * u)
        if multipliers["state"] > 0.0:
            value -= multipliers["state"] * float(self.state_information(u, V, P))
            gradient[0] -= multipliers["state"] * P * u / (c.r + V + c.q + P * u * u)
        if multipliers["chance"] > 0.0:
            spread = math.sqrt(V + c.q + P * u * u)
            gap = c.chance_bound - (m + b_hat * u)
            z = gap / spread
            value += multipliers["chance"] * 0.5 * math.erfc(z / SQRT2)
            dz = -b_hat / spread - gap * P * u / spread**3
            gradient[0] -= multipliers["chance"] * math.exp(-0.5 * z * z) / SQRT2PI * dz
        return value, gradient

    # ----- planning -----

    def _solve(self, multipliers, beliefs, start: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """Mode of the plan for fixed multipliers: L-BFGS-B from the warm start
        and, with a positive multiplier, from the constant plans at +-0.5,
        because a plan at rest is a stationary point under a symmetric
        posterior."""
        self.last_plans += 1
        c = self.config
        K = c.horizon
        bound = 1.0 - c.bound_margin
        starts = [np.zeros(K) if start is None else np.clip(start, -bound, bound)]
        if any(value > 0.0 for value in multipliers.values()):
            starts += [np.full(K, 0.5), np.full(K, -0.5)]
        best, best_value = None, np.inf
        for initial in starts:
            result = minimize(self.objective, initial, args=(multipliers, *beliefs), jac=True,
                              method="L-BFGS-B", bounds=[(-bound, bound)] * K,
                              options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 200})
            if result.fun < best_value:
                best, best_value = result.x, float(result.fun)
        return best, best_value

    def _residual(self, name, u, beliefs, floors) -> float:
        m, V, b_hat, P = beliefs
        if name == "parameter":
            return float(self.parameter_information(u, P)) - floors[0]
        if name == "state":
            return float(self.state_information(u, V, P)) - floors[1]
        return self.config.chance_level - float(self.risk(u, m, V, b_hat, P))

    def _root(self, name, multipliers, beliefs, floors, plan) -> Tuple[float, np.ndarray]:
        """Smallest multiplier >= 0 for which constraint ``name`` holds at the
        first input of the mode, with the other multipliers fixed."""
        c = self.config
        trial = dict(multipliers)

        def evaluate(value, start):
            trial[name] = value
            new_plan, _ = self._solve(trial, beliefs, start)
            return self._residual(name, new_plan[0], beliefs, floors), new_plan

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
        for _ in range(15):
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
        """The next input: the first input of the mode of the plan with the
        KKT multipliers of the enabled constraints."""
        c = self.config
        beliefs = self.marginals()
        floors = self.floors(*beliefs)
        self.last_floors = {"parameter": floors[0], "state": floors[1]}
        self.last_plans = 0
        names = [name for name, enabled in (("parameter", c.parameter_information),
                                            ("state", c.state_information),
                                            ("chance", c.chance_bound is not None)) if enabled]
        multipliers = {"parameter": 0.0, "state": 0.0, "chance": 0.0}
        warm = (None if self._previous_plan is None
                else np.concatenate([self._previous_plan[1:], self._previous_plan[-1:]]))
        plan, _ = self._solve(multipliers, beliefs, warm)
        for _ in range(6):
            changed = False
            for name in names:
                if self._residual(name, plan[0], beliefs, floors) >= 0.0 and multipliers[name] == 0.0:
                    continue
                value, plan = self._root(name, multipliers, beliefs, floors, plan)
                if abs(value - multipliers[name]) > 1e-3 * (1.0 + multipliers[name]):
                    changed = True
                multipliers[name] = value
            if not changed:
                break
        self.last_multipliers = dict(multipliers)
        self._previous_plan = plan
        return float(plan[0])
