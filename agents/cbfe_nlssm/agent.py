"""CBFE agent for a nonlinear state-space model with unknown parameters and a
noisy measurement (main.tex, Sections 2-4):

    x_k = x_{k-1} - a sin(x_{k-1}) + b u_k + w_k ,   w_k ~ N(0, q) ,
    y_k = x_k + v_k ,                                v_k ~ N(0, r) ,

with u_k in [-1, 1] and the parameters theta = (a, b) unknown and constant.
The model is a geared joint under gravity whose friction dominates its
inertia: a is the gravity torque and b the motor gain per step, both
relative to the friction, and the sign of b, the motor polarity, is unknown.
The model is nonlinear in the state and linear in the parameters, so the
information of the next output has a closed form.

Filter
------
An extended Kalman filter over z = (x, a, b) approximates the posterior
p(x_t, theta | D_t) outside the planning model. The planner takes its
marginals q(x_t) = N(m, V) and q(theta) = N(theta_hat, P) as belief nodes.

Planning model
--------------
Under the factorization constraints (Definition 2), every transition node
draws x_{k-1} and theta independently, and the planner passes point masses at
the predicted means forward. With s_k = sin m_{k-1}, g_k = 1 - a_hat cos m_{k-1}
and h_k = (-s_k, u_k),

    m_k = m_{k-1} - a_hat s_k + b_hat u_k ,
    V_k = g_k^2 V_{k-1} + h_k^T P h_k + q .

The plan u = (u_1, ..., u_K) minimises eq. (14),

    J(u) = sum_k [ -(c - 1) ln(1 - u_k^2) + ((m_k - y_*)^2 + V_k + r) / (2 S_*) ]
           - gamma I(u_1) ,

with the Beta(c, c) control prior, the goal prior N(y_*, S_*), and the
information that the next output carries about the next state and the
parameters,

    I(u) = I[y_{t+1} : x_{t+1}, theta | u] = 1/2 ln(1 + V_1(u) / r) .

L-BFGS-B minimises J with its exact gradient, from the shifted previous plan
and, for gamma > 0, from the constant plans at +-0.5, because a plan at rest
is stationary under a sign-symmetric posterior.

Floor and multiplier
--------------------
The floor is eq. (10), beta = 1/2 ln(Sigma_bar / (r + S_*)), with the mean
predictive variance Sigma_bar = E_p~[V_1(u) + r] of the goal-reaching inputs
p~(u) ~ p(u) N(m_1(u); y_*, S_*) on the 399 input nodes. It is capped at the
information of the most informative input minus ``floor_margin``. The KKT
multiplier is zero if the plan at gamma = 0 meets the floor. Otherwise it is
the smallest multiplier at which the first input of the plan meets the
floor, found by a bracket that grows fourfold from 1 and regula falsi
(Illinois). With ``constrained=False``, gamma = 0 throughout, and the agent
is the cautious controller.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from agents.common import input_nodes


@dataclass
class Config:
    """Model, goal and constraint of the agent.

    q, r                 process and measurement noise variances
    goal_mean            target y_*
    goal_variance        tolerance S_*
    horizon              planning horizon K
    control_prior_shape  c of the symmetric Beta(c, c) control prior on [-1, 1]
    constrained          False: no information constraint (gamma = 0)
    max_multiplier       cap of the bracket, far above every root
    root_tolerance       stopping tolerance of regula falsi on the residual
    floor_margin         gap between the capped floor and the largest information
    bound_margin         distance of the plan from the input bounds
    caution              False: J drops the trace term V_k / (2 S_*) of the expected
                         goal log-density, so that it becomes certainty equivalence
    drift                variances of the steps of (a, b) per time step: the filter
                         and the planner add them to the parameter covariance
    gain_reversion       rho of the gain model b_{k+1} = rho b_k + e_k, a mean-reverting
                         drift towards zero for rho < 1 and a random walk for rho = 1
    """
    q: float = 1e-4
    r: float = 0.01
    goal_mean: float = 1.0
    goal_variance: float = 0.01
    horizon: int = 5
    control_prior_shape: float = 2.0
    constrained: bool = True
    max_multiplier: float = 1e6
    root_tolerance: float = 1e-3
    floor_margin: float = 0.01
    bound_margin: float = 1e-6
    caution: bool = True
    drift: Tuple[float, float] = (0.0, 0.0)
    gain_reversion: float = 1.0


Beliefs = Tuple[float, float, np.ndarray, np.ndarray]   # m, V, theta_hat, P


@dataclass
class Agent:
    """Extended Kalman filter over (x, a, b) and the planner of the module
    docstring. ``observe(y)`` conditions on an output of the current state,
    ``plan()`` returns the next input and ``update(u, y)`` adds the output
    that followed it."""
    config: Config
    prior_mean: Tuple[float, float, float] = (0.0, 0.2, 0.0)
    prior_variance: Tuple[float, float, float] = (0.0025, 0.0025, 0.25)

    def __post_init__(self):
        self.mean = np.array(self.prior_mean, dtype=float)
        self.covariance = np.diag(self.prior_variance).astype(float)
        self.last_multiplier = 0.0
        self.last_floor = float("nan")
        self.last_information = float("nan")
        self.last_plans = 0
        self._previous_plan: Optional[np.ndarray] = None
        self._nodes = input_nodes(1)[:, 0]
        self._log_prior_nodes = (self.config.control_prior_shape - 1.0) * np.log1p(-np.square(self._nodes))

    # ----- filter -----

    def observe(self, output: float) -> None:
        """Condition the belief on an output of the current state."""
        innovation_variance = self.covariance[0, 0] + self.config.r
        gain = self.covariance[:, 0] / innovation_variance
        self.mean = self.mean + gain * (output - self.mean[0])
        self.covariance = self.covariance - np.outer(gain, gain) * innovation_variance
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def update(self, control_input: float, output: float) -> None:
        """Predict the belief through the input, linearised at the mean, and
        condition it on the output that followed."""
        c = self.config
        self.mean[2] *= c.gain_reversion
        self.covariance[2, :] *= c.gain_reversion
        self.covariance[:, 2] *= c.gain_reversion
        self.covariance[1, 1] += c.drift[0]
        self.covariance[2, 2] += c.drift[1]
        x, a, b = self.mean
        jacobian = np.array([[1.0 - a * math.cos(x), -math.sin(x), control_input],
                             [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.mean = np.array([x - a * math.sin(x) + b * control_input, a, b])
        self.covariance = jacobian @ self.covariance @ jacobian.T
        self.covariance[0, 0] += c.q
        self.observe(output)

    def beliefs(self) -> Beliefs:
        """Belief nodes of the planning model: the marginals of the filter."""
        return (float(self.mean[0]), float(self.covariance[0, 0]), self.mean[1:].copy(),
                self.covariance[1:, 1:].copy())

    # ----- next node: moments, information and floor -----

    def parameters(self, beliefs: Beliefs, k: int) -> Tuple[float, float, float, float, float]:
        """a_hat, b_hat, P_aa, P_ab and P_bb of the parameters k steps ahead
        under the drift model."""
        c = self.config
        _, _, (a, b), P = beliefs
        rho = c.gain_reversion
        growth = k if rho == 1.0 else (1.0 - rho**(2 * k)) / (1.0 - rho**2)
        return (a, rho**k * b, P[0, 0] + k * c.drift[0], rho**k * P[0, 1],
                rho**(2 * k) * P[1, 1] + growth * c.drift[1])

    def next_moments(self, control_input, beliefs: Beliefs):
        """Mean and variance of the next state at the input(s)."""
        m, V = beliefs[0], beliefs[1]
        a, b, Paa, Pab, Pbb = self.parameters(beliefs, 1)
        s, g = math.sin(m), 1.0 - a * math.cos(m)
        u = np.asarray(control_input, dtype=float)
        mean = m - a * s + b * u
        variance = g * g * V + Paa * s * s - 2.0 * Pab * s * u + Pbb * u * u + self.config.q
        return mean, variance

    def information(self, control_input, beliefs: Beliefs):
        """I[y_{t+1} : x_{t+1}, theta | u] at the input(s)."""
        return 0.5 * np.log1p(self.next_moments(control_input, beliefs)[1] / self.config.r)

    def floor(self, beliefs: Beliefs) -> float:
        """Eq. (10) over the goal-reaching inputs, capped below the largest
        information on the input nodes."""
        c = self.config
        mean, variance = self.next_moments(self._nodes, beliefs)
        log_weights = self._log_prior_nodes - 0.5 * np.square(mean - c.goal_mean) / c.goal_variance
        weights = np.exp(log_weights - log_weights.max())
        spread = float(np.sum(weights * (variance + c.r)) / weights.sum())
        floor = 0.5 * math.log(spread / (c.r + c.goal_variance))
        largest = float(0.5 * np.log1p(variance.max() / c.r))
        return min(floor, largest - c.floor_margin)

    # ----- planning -----

    def objective(self, plan: np.ndarray, multiplier: float, beliefs: Beliefs) -> Tuple[float, np.ndarray]:
        """J of the module docstring and its gradient, by forward sensitivities."""
        c = self.config
        m, V = beliefs[0], beliefs[1]
        K = len(plan)
        shape = c.control_prior_shape - 1.0
        trace = 1.0 if c.caution else 0.0
        dm, dV = np.zeros(K), np.zeros(K)
        value, gradient = 0.0, np.zeros(K)
        for k in range(K):
            u = float(plan[k])
            a, b, Paa, Pab, Pbb = self.parameters(beliefs, k + 1)
            s, co = math.sin(m), math.cos(m)
            g = 1.0 - a * co
            ds, dg = co * dm, a * s * dm
            V_next = g * g * V + Paa * s * s - 2.0 * Pab * s * u + Pbb * u * u + c.q
            dV_next = 2.0 * g * V * dg + g * g * dV + 2.0 * Paa * s * ds - 2.0 * Pab * u * ds
            dV_next[k] += 2.0 * (Pbb * u - Pab * s)
            dm_next = g * dm
            dm_next[k] += b
            m, V, dm, dV = m - a * s + b * u, V_next, dm_next, dV_next
            error = m - c.goal_mean
            value += -shape * math.log1p(-u * u) + (error * error + trace * (V + c.r)) / (2.0 * c.goal_variance)
            gradient += (error * dm + trace * 0.5 * dV) / c.goal_variance
            gradient[k] += 2.0 * shape * u / (1.0 - u * u)
            if k == 0 and multiplier > 0.0:
                value -= multiplier * 0.5 * math.log1p(V / c.r)
                gradient -= multiplier * 0.5 * dV / (c.r + V)
        return value, gradient

    def _solve(self, multiplier: float, beliefs: Beliefs, start: np.ndarray) -> np.ndarray:
        """Mode of the plan for a fixed multiplier."""
        self.last_plans += 1
        c = self.config
        bound = 1.0 - c.bound_margin
        starts = [np.clip(start, -bound, bound)]
        if multiplier > 0.0:
            starts += [np.full(c.horizon, 0.5), np.full(c.horizon, -0.5)]
        best, best_value = None, np.inf
        for initial in starts:
            result = minimize(self.objective, initial, args=(multiplier, beliefs), jac=True,
                              method="L-BFGS-B", bounds=[(-bound, bound)] * c.horizon,
                              options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 200})
            if result.fun < best_value:
                best, best_value = result.x, float(result.fun)
        return best

    def _multiplier(self, beliefs: Beliefs, floor: float, plan: np.ndarray) -> Tuple[float, np.ndarray]:
        """Smallest multiplier >= 0 at which the first input of the plan
        meets the floor, and that plan."""
        c = self.config

        def residual(trial: float, start: np.ndarray):
            new_plan = self._solve(trial, beliefs, start)
            return float(self.information(new_plan[0], beliefs)) - floor, new_plan

        value = residual(0.0, plan)
        if value[0] >= 0.0:
            return 0.0, value[1]
        low, trial, start = (0.0, value[0]), 1.0, value[1]
        while True:
            value = residual(trial, start)
            if value[0] >= 0.0:
                high = (trial, value[0], value[1])
                break
            low, start = (trial, value[0]), value[1]
            if trial >= c.max_multiplier:
                return trial, value[1]
            trial = min(4.0 * trial, c.max_multiplier)
        if high[1] < c.root_tolerance:
            return high[0], high[2]
        side = 0
        for _ in range(15):
            (trial_low, residual_low), (trial_high, residual_high) = low, high[:2]
            trial = trial_high - residual_high * (trial_high - trial_low) / (residual_high - residual_low)
            value = residual(trial, high[2])
            if abs(value[0]) < c.root_tolerance:
                return trial, value[1]
            if value[0] > 0.0:
                high = (trial, value[0], value[1])
                if side == 1:
                    low = (low[0], 0.5 * low[1])
                side = 1
            else:
                low = (trial, value[0])
                if side == -1:
                    high = (high[0], 0.5 * high[1], high[2])
                side = -1
        return high[0], high[2]

    def plan(self) -> float:
        """The next input: the first input of the mode of the plan, with the
        KKT multiplier of the information constraint (or zero)."""
        c = self.config
        beliefs = self.beliefs()
        self.last_plans = 0
        start = (np.zeros(c.horizon) if self._previous_plan is None
                 else np.concatenate([self._previous_plan[1:], self._previous_plan[-1:]]))
        if c.constrained:
            floor = self.floor(beliefs)
            multiplier, plan = self._multiplier(beliefs, floor, start)
        else:
            floor, multiplier, plan = float("nan"), 0.0, self._solve(0.0, beliefs, start)
        self.last_floor, self.last_multiplier = floor, float(multiplier)
        self.last_information = float(self.information(plan[0], beliefs))
        self._previous_plan = plan
        return float(plan[0])
