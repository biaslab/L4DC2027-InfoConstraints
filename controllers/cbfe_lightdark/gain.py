"""Light-dark positioning with an unknown actuator gain: information constraints
on two nodes of the planning graph (main.tex, Section V-C).

    x_k = x_{k-1} + b u_k + w_k ,   b ~ N(b0, P0) unknown and constant,
    y_k = x_k + v_k ,               v_k ~ N(0, r(x_k)) ,

with the light-dark noise of ``agent.py``. ``JointGridFilter`` computes the
exact posterior over (x, b) on a grid. The planning model takes its moments
as one belief node of the state and the gain (Definition 4.1): means m and
b_hat, variances V and P, and covariance C. The first transition node draws
both inputs from that node, and later nodes treat the predicted state and the
gain as independent (eq. (5)), so with the beliefs fixed within the plan

    m_k = m_{k-1} + b_hat u_k ,   V_k = V + k q + P sum_{j<=k} u_j^2 + 2 C u_1 .

and the plan minimises

    J(u) = sum_k [c(u_k) + ((m_k - x_*)^2 + V_k) / (2 S_*)
                  - g_k^par I_par,k - g_k^state I_state,k] .

The chain rule splits the information of the next output about the gain
and the state into two layers, the gain above the state,

    I(b, x; y | u) = I(b; y | u) + I(x; y | b, u) ,
    I_par,k   = 1/2 ln(1 + P e_k^2 / (A_k + r(m_k)))   the output about the gain
    I_state,k = 1/2 ln(1 + A_k / r(m_k))               the output about the state,
                                                       given the gain

with A_k = V - C^2/P + k q the spread of the predicted state given the gain,
e_1 = u_1 + C/P and e_k = u_k for k > 1, so that P e_1^2 is the spread that
the gain adds to the next state, and the measurement noise at the predicted
mean. In the dark, the output reveals
neither. CBFE weights the first input with the KKT multipliers of
I_par >= beta_par and I_state >= beta_state, solved cyclically. With the
goal-reaching inputs p~ of eq. (10) and u2 = E_p~[u^2], the total floor is
the rate-distortion function of the predicted state at the goal tolerance,
1/2 ln((A + P u2) / S_*), and the two layers share it:

    beta_state = 1/2 ln(A / S_*)
    beta_par   = 1/2 ln((A + P u2) / max(A, S_*))

so the floors add up to the total whenever both are positive, and the gain
layer stays masked while the spread A of the state layer is large. A is the
spread that the state would reach by the end of the task without further
information, A = V + (T - t) q, because the goal applies at every remaining
step. In the dark, V grows by q per step while (T - t) q shrinks by q, so
drift does not raise the floors again. These two options (``shared_budget``,
``task_length``) are off by default: in the pilot, the shared budget made the
agent calibrate longer at a higher goal cost, and the end-of-task spread
changed little. By default, A = V - C^2/P + q and the gain layer uses the state
layer's spread as noise, beta_par = 1/2 ln((A + P u2) / (A + S_*)), as eq. (10).
The joint belief node (``joint_belief``) is what prevents repeated trips to
the light: with mean-field marginals the planner misses the term 2 u C of the
predicted variance and leaves the light too early.

each capped at the information of the most informative input minus
``floor_margin``. The dual MPC weights every input with fixed weights. The
belief-space planner propagates the joint Gaussian belief of (x, b) through
the measurement updates of the most likely outputs and minimises
sum_k [c(u_k) + ((m_k - x_*)^2 + V_k^-) / (2 S_*)]. All planners minimise J by
L-BFGS-B from the warm start, rest and a step towards the light.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from controllers.common import input_nodes

from .agent import LightDarkModel


@dataclass
class JointGridFilter:
    """Exact posterior over (x, b) on a grid."""
    model: LightDarkModel
    x_mean: float = 0.0
    x_variance: float = 0.25
    b_mean: float = 1.0
    b_variance: float = 0.09
    x_range: Tuple[float, float, int] = (-4.0, 8.0, 1201)
    b_range: Tuple[float, float, int] = (-0.2, 2.2, 241)

    def __post_init__(self):
        self.x = np.linspace(*self.x_range)
        self.b = np.linspace(*self.b_range)
        self.dx = self.x[1] - self.x[0]
        px = np.exp(-0.5 * (self.x - self.x_mean)**2 / self.x_variance)
        pb = np.exp(-0.5 * (self.b - self.b_mean)**2 / self.b_variance)
        self.p = np.outer(pb, px)                      # (b, x)
        self.p /= self.p.sum()
        half = max(1, int(math.ceil(6.0 * math.sqrt(self.model.q) / self.dx)))
        offsets = self.dx * np.arange(-half, half + 1)
        kernel = np.exp(-0.5 * offsets**2 / self.model.q)
        self.kernel = kernel / kernel.sum()

    def update(self, control_input: float, output: float) -> None:
        predicted = np.empty_like(self.p)
        for j, gain in enumerate(self.b):
            shifted = np.interp(self.x - gain * control_input, self.x, self.p[j], left=0.0, right=0.0)
            predicted[j] = np.convolve(shifted, self.kernel, mode="same")
        r = self.model.noise(self.x)
        log_likelihood = -0.5 * ((output - self.x)**2 / r + np.log(r))
        posterior = predicted * np.exp(log_likelihood - log_likelihood.max())[None, :]
        self.p = posterior / posterior.sum()

    def moments(self) -> Tuple[float, float, float, float, float]:
        """m, V, b_hat, P and the covariance of x and b."""
        px, pb = self.p.sum(axis=0), self.p.sum(axis=1)
        m, b_hat = float(px @ self.x), float(pb @ self.b)
        V = float(px @ (self.x - m)**2)
        P = float(pb @ (self.b - b_hat)**2)
        C = float(np.sum(self.p * np.outer(self.b - b_hat, self.x - m)))
        return m, V, b_hat, P, C


@dataclass
class GainConfig:
    """Goal, horizon, planner and constraints.

    planner      "cbfe", "dual", "bsp" or "none"
    constraints  the information constraints of CBFE ("parameter", "state")
    weights      fixed weights (parameter, state) of the dual MPC on every input
    """
    goal_mean: float = 0.0
    goal_variance: float = 0.01
    horizon: int = 5
    control_prior_shape: float = 2.0
    planner: str = "cbfe"
    constraints: Tuple[str, ...] = ("parameter", "state")
    weights: Tuple[float, float] = (0.0, 0.0)
    floor_margin: float = 0.01
    max_multiplier: float = 1e6
    root_tolerance: float = 1e-3
    bound_margin: float = 1e-6
    task_length: Optional[int] = None
    shared_budget: bool = False
    joint_belief: bool = True


@dataclass
class GainLightDarkAgent:
    config: GainConfig
    model: LightDarkModel = field(default_factory=LightDarkModel)
    x_mean: float = 0.0
    x_variance: float = 0.25
    b_mean: float = 1.0
    b_variance: float = 0.09

    def __post_init__(self):
        self.filter = JointGridFilter(self.model, self.x_mean, self.x_variance, self.b_mean, self.b_variance)
        self._nodes = input_nodes(1)[:, 0]
        self._previous_plan: Optional[np.ndarray] = None
        self.last_multipliers = {"parameter": 0.0, "state": 0.0}
        self.last_floors = {"parameter": 0.0, "state": 0.0}
        self.t = 0

    def update(self, control_input: float, output: float) -> None:
        self.filter.update(control_input, output)
        self.t += 1

    def _beliefs(self, moments):
        return moments if self.config.joint_belief else (*moments[:4], 0.0)

    # ----- terms -----

    def parameter_information(self, e, m, A, P):
        """I(b; y | u) at the predicted mean m, for the effective input e = u + C/P
        and the spread A of the predicted state given the gain."""
        return 0.5 * np.log1p(P * np.square(e) / (A + self.model.noise(m)))

    def state_information(self, m, A):
        """I(x; y | b, u)."""
        return 0.5 * np.log1p(A / self.model.noise(m))

    @staticmethod
    def conditional(beliefs) -> Tuple[float, float]:
        """Spread of the state given the gain, V - C^2/P, and the offset C/P of
        the effective input."""
        m, V, b_hat, P, C = beliefs
        if P <= 1e-12:
            return V, 0.0
        return max(V - C * C / P, 1e-12), C / P

    def objective(self, plan: np.ndarray, weights: Dict[str, np.ndarray], beliefs) -> Tuple[float, np.ndarray]:
        """J and its gradient."""
        c, q = self.config, self.model.q
        m, V, b_hat, P, C = beliefs
        Vc, offset = self.conditional(beliefs)
        K = len(plan)
        a1 = c.control_prior_shape - 1.0
        S = c.goal_variance
        means = m + b_hat * np.cumsum(plan)
        variances = V + q * np.arange(1, K + 1) + P * np.cumsum(plan**2) + 2.0 * C * plan[0]
        error = means - c.goal_mean
        value = float(np.sum(-a1 * np.log1p(-plan**2) + (error**2 + variances) / (2.0 * S)))
        # d/du_j: tracking through m_k (k >= j), caution through V_k (k >= j)
        dm = error / S                                  # dJ/dm_k
        dV = np.full(K, 0.5 / S)                        # dJ/dV_k
        wp, ws = weights["parameter"], weights["state"]
        extra_u = np.zeros(K)
        if np.any(wp > 0.0) or np.any(ws > 0.0):
            A = Vc + q * np.arange(1, K + 1)
            r = self.model.noise(means)
            dr = self.model.noise_derivative(means)
            effective = plan.copy()
            effective[0] += offset
            e2 = effective**2
            value -= float(np.sum(wp * 0.5 * np.log1p(P * e2 / (A + r))))
            value -= float(np.sum(ws * 0.5 * np.log1p(A / r)))
            extra_u = -wp * P * effective / (A + r + P * e2)
            dm = dm + wp * 0.5 * P * e2 * dr / ((A + r) * (A + r + P * e2))
            dm = dm + ws * 0.5 * A * dr / (r * (r + A))
        tail_m = np.cumsum(dm[::-1])[::-1]
        tail_V = np.cumsum(dV[::-1])[::-1]
        gradient = 2.0 * a1 * plan / (1.0 - plan**2) + b_hat * tail_m + 2.0 * P * plan * tail_V + extra_u
        gradient[0] += 2.0 * C * tail_V[0]
        return value, gradient

    def bsp_objective(self, plan: np.ndarray, moments) -> float:
        """Belief-space planning with the most likely outputs: the joint
        Gaussian of (x, b) through the extended Kalman filter."""
        c, q = self.config, self.model.q
        m, V, b_hat, P, C = moments
        mean = np.array([m, b_hat])
        cov = np.array([[V, C], [C, P]])
        a1 = c.control_prior_shape - 1.0
        value = 0.0
        for u in plan:
            F = np.array([[1.0, u], [0.0, 1.0]])
            mean = F @ mean
            cov = F @ cov @ F.T + np.diag([q, 0.0])
            value += -a1 * math.log1p(-u * u) + ((mean[0] - c.goal_mean)**2 + cov[0, 0]) / (2.0 * c.goal_variance)
            r = float(self.model.noise(mean[0]))
            gain = cov[:, 0] / (cov[0, 0] + r)
            cov = cov - np.outer(gain, cov[0, :])
        return value

    # ----- planning -----

    def _starts(self, m: float, b_hat: float, start: Optional[np.ndarray]) -> list:
        K = self.config.horizon
        towards = np.zeros(K)
        towards[0] = float(np.clip((self.model.light - m) / max(b_hat, 0.1), -0.9, 0.9))
        return [np.zeros(K) if start is None else np.array(start, float), np.zeros(K), towards]

    def _solve(self, weights, beliefs, start) -> Tuple[np.ndarray, float]:
        bound = 1.0 - self.config.bound_margin
        best = None
        for initial in self._starts(beliefs[0], beliefs[2], start):
            result = minimize(self.objective, np.clip(initial, -bound, bound), args=(weights, beliefs), jac=True,
                              method="L-BFGS-B", bounds=[(-bound, bound)] * self.config.horizon,
                              options={"ftol": 1e-13, "gtol": 1e-10, "maxiter": 400})
            if best is None or result.fun < best.fun:
                best = result
        return best.x, float(best.fun)

    def _solve_bsp(self, moments, start) -> np.ndarray:
        bound = 1.0 - self.config.bound_margin
        best = None
        for initial in self._starts(moments[0], moments[2], start):
            result = minimize(self.bsp_objective, np.clip(initial, -bound, bound), args=(moments,),
                              method="L-BFGS-B", bounds=[(-bound, bound)] * self.config.horizon,
                              options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 300})
            if best is None or result.fun < best.fun:
                best = result
        return best.x

    def floors(self, beliefs) -> Dict[str, float]:
        c, q = self.config, self.model.q
        m, V, b_hat, P, C = beliefs
        Vc, offset = self.conditional(beliefs)
        u = self._nodes
        log_weights = ((c.control_prior_shape - 1.0) * np.log1p(-u**2)
                       - 0.5 * (m + b_hat * u - c.goal_mean)**2 / c.goal_variance)
        weights = np.exp(log_weights - log_weights.max())
        gain_spread = P * float(np.sum(weights * (u + offset)**2) / weights.sum())
        steps = 1 if c.task_length is None else max(c.task_length - self.t, 1)
        A = Vc + steps * q
        budget = max(A, c.goal_variance) if c.shared_budget else A + c.goal_variance
        floors = {"parameter": 0.5 * math.log((A + gain_spread) / budget),
                  "state": 0.5 * math.log(A / c.goal_variance)}
        best_parameter = float(np.max(self.parameter_information(u + offset, m + b_hat * u, Vc + q, P)))
        best_state = float(np.max(self.state_information(m + b_hat * u, Vc + q)))
        floors["parameter"] = min(floors["parameter"], best_parameter - c.floor_margin)
        floors["state"] = min(floors["state"], best_state - c.floor_margin)
        return floors

    def _residual(self, name: str, u: float, beliefs, floors) -> float:
        m, V, b_hat, P, C = beliefs
        Vc, offset = self.conditional(beliefs)
        A = Vc + self.model.q
        if name == "parameter":
            value = float(self.parameter_information(u + offset, m + b_hat * u, A, P))
        else:
            value = float(self.state_information(m + b_hat * u, A))
        return value - floors[name] + 1e-12

    def _weights(self, multipliers: Dict[str, float]) -> Dict[str, np.ndarray]:
        K = self.config.horizon
        out = {}
        for name in ("parameter", "state"):
            w = np.zeros(K)
            w[0] = multipliers.get(name, 0.0)
            out[name] = w
        return out

    def _root(self, name, multipliers, beliefs, floors, plan) -> Tuple[float, np.ndarray]:
        c = self.config
        trial = dict(multipliers)

        def evaluate(value, start):
            trial[name] = value
            new_plan, _ = self._solve(self._weights(trial), beliefs, start)
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
        moments = self.filter.moments()
        beliefs = self._beliefs(moments)
        K = c.horizon
        warm = (None if self._previous_plan is None
                else np.concatenate([self._previous_plan[1:], self._previous_plan[-1:]]))
        self.last_multipliers = {"parameter": 0.0, "state": 0.0}
        self.last_floors = {"parameter": 0.0, "state": 0.0}
        if c.planner == "bsp":
            plan = self._solve_bsp(moments, warm)
        elif c.planner == "dual":
            weights = {"parameter": np.full(K, c.weights[0]), "state": np.full(K, c.weights[1])}
            plan, _ = self._solve(weights, beliefs, warm)
        else:
            plan, _ = self._solve(self._weights({}), beliefs, warm)
            if c.planner == "cbfe":
                floors = self.floors(beliefs)
                self.last_floors = dict(floors)
                names = [name for name in c.constraints if floors[name] > 0.0]
                multipliers = {name: 0.0 for name in names}
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
                self.last_multipliers.update(multipliers)
        self._previous_plan = plan
        return float(plan[0])
