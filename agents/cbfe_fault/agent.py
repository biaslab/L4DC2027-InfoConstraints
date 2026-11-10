"""CBFE agent for an integrator whose actuator can fail (main.tex, Section V-C).

Model
-----
A mode m selects the prior of the input gain, and the gain drives the
measured state,

    m in {nominal, degraded, reversed},   b | m ~ N(mu_m, s_m^2),
    x_k = x_{k-1} + b_k u_k + w_k ,       w_k ~ N(0, sigma^2) .

The actuator starts nominal. At every step it fails with probability h, into
the degraded or the reversed mode with equal probability, and a failed
actuator stays failed. A fault draws a new gain from the prior of its mode.

Filter
------
``FaultFilter`` is exact. Its hypotheses are "no fault yet" and "fault of type
F at step tau" for every tau <= t. Given a hypothesis, the gain before and
after the fault has a Gaussian posterior from a scalar Kalman filter, and the
hypothesis weight follows from the predictive likelihoods. The planning
model takes the next mode and the gain of every mode as belief nodes: the
weights q(m) and the moment-matched Gaussians N(b_m, P_m).

Information of the two layers
-----------------------------
The chain rule splits the information of the next state about the mode and
the gain into the two layers of the model,

    I(m, b; x' | u) = I(m; x' | u) + I(b; x' | m, u) ,

    upper layer  I_mode(u) = H[sum_m q_m N(b_m u, R_m(u))] - sum_m q_m H[N(0, R_m(u))] ,
    lower layer  I_gain(u) = sum_m q_m 1/2 ln(1 + P_m u^2 / sigma^2) ,

with R_m(u) = sigma^2 + P_m u^2. Gauss-Hermite quadrature evaluates the
mixture entropy.

Floors
------
The goal-reaching inputs weight every input by the goal density of the mean
prediction of every mode, p~(u) ~ p(u) sum_m q_m N(x_*; x + b_m u, S_*), which
is eq. (10) of the paper for a single mode. With u2 = E_p~[u^2]:

    lower layer  beta_gain = 1/2 ln((sigma^2 + Pbar u2) / (sigma^2 + S_*)),  Pbar = sum_m q_m P_m ,

which is eq. (10). Eq. (10) is the rate-distortion function of the Gaussian
prediction N(0, sigma^2 + Pbar u2) at the distortion sigma^2 + S_*. Entropy
does not measure the spread of a mixture, so the floor of the mode layer is
the rate-distortion function of its prediction,

    upper layer  beta_mode = R_Y(D),  Y ~ sum_m q_m N(b_m sqrt(u2), sigma^2 + P_m u2),
                 D = sigma^2 + Pbar u2 + S_* ,

the least information with which the goal-reaching predictions can be
described to within the noise, the spread of the lower layer and the goal
tolerance. ``rate_distortion`` computes it with the Blahut-Arimoto
algorithm. With one constraint on the total information, the floor is
R_Y(sigma^2 + S_*) with the spread of the lower layer inside Y.

Planner
-------
With point masses at the predicted means forward, the plan u = (u_1..u_K)
minimises

    J(u) = sum_k [c(u_k) + ((x_k - x_*)^2 + V_b u_k^2) / (2 S_*)
                  - g_k^mode I_mode(u_k) - g_k^gain I_gain(u_k)] + eta risk(u_1) ,

    x_k = x_{k-1} + bbar u_k ,   bbar = E[b] ,   V_b = Var[b]  (over modes and gains),

with the control prior c(u) = -(a-1) ln(1 - u^2). The expected goal
log-density of a mixture depends only on the first two moments of the gain,
so the q(m)-weighted backward messages are Gaussian. ``_sweeps`` passes them
backward in information form (a Gauss-Newton iterative LQR that drops the
negative curvature of the information and risk terms) and updates the inputs
forward with a backtracking line search. With a positive weight, it also
sweeps from the constant plans at +-0.5. CBFE weights the first input only,
with the KKT multipliers of

    I_mode >= beta_mode ,   I_gain >= beta_gain ,   risk <= epsilon ,

found cyclically by an upward bracket from one and regula falsi, as in
``agents.cbfe_ssm``. The dual MPC weights every input with fixed weights.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp, ndtr

from agents.common import input_nodes

MODES = ("nominal", "degraded", "reversed")
LOG_2PI_E = math.log(2.0 * math.pi * math.e)

_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(24)
_GH_NODES = math.sqrt(2.0) * _GH_NODES
_GH_WEIGHTS = _GH_WEIGHTS / math.sqrt(math.pi)


# ----- information and rate-distortion -----

def mixture_entropy(means: np.ndarray, variances: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Differential entropy of sum_m w_m N(mu_m, v_m), by Gauss-Hermite
    quadrature per component. ``means`` and ``variances`` have shape (..., M)."""
    means, variances = np.broadcast_arrays(np.asarray(means, float), np.asarray(variances, float))
    sd = np.sqrt(variances)
    points = means[..., :, None] + sd[..., :, None] * _GH_NODES             # (..., M, n)
    diff = points[..., :, :, None] - means[..., None, None, :]              # (..., M, n, M)
    log_components = (np.log(weights)[None, None, :] - 0.5 * diff**2 / variances[..., None, None, :]
                      - 0.5 * np.log(2.0 * np.pi * variances[..., None, None, :]))
    log_density = logsumexp(log_components, axis=-1)                       # (..., M, n)
    return -np.sum(weights * np.sum(_GH_WEIGHTS * log_density, axis=-1), axis=-1)


def rate_distortion(means: Sequence[float], variances: Sequence[float], weights: Sequence[float],
                    distortion: float, grid: int = 161, iterations: int = 40,
                    bisections: int = 16) -> float:
    """Rate-distortion function R_Y(D) [nats] of the mixture Y = sum_m w_m
    N(mu_m, v_m) under squared error, by the Blahut-Arimoto algorithm with Y
    and the reproduction values on one uniform grid, where the Gaussian
    kernel exp(-s (y - y')^2) acts by convolution. At slope s, the algorithm
    gives D(s) and R(s) = -s D(s) - E[ln c(y)], with c the normaliser of the
    test channel. Bisection on ln s, around the Gaussian slope 1/(2D), finds
    D(s) = D. For one Gaussian, R = 1/2 ln(v / D)."""
    means = np.asarray(means, float)
    variances = np.asarray(variances, float)
    weights = np.asarray(weights, float)
    keep = weights > 1e-12
    means, variances, weights = means[keep], variances[keep], weights[keep] / weights[keep].sum()
    mean = float(weights @ means)
    variance = float(weights @ (variances + means**2) - mean**2)
    if variance <= distortion:
        return 0.0
    sd = np.sqrt(variances)
    y = np.linspace(float(np.min(means - 6.0 * sd)), float(np.max(means + 6.0 * sd)), grid)
    spacing = y[1] - y[0]
    density = np.sum(weights * np.exp(-0.5 * (y[:, None] - means)**2 / variances) / sd, axis=1)
    probability = density / density.sum()
    offsets = spacing * np.arange(-(grid - 1), grid)
    squared = offsets**2

    def blahut_arimoto(slope: float, marginal: np.ndarray):
        kernel = np.exp(-slope * squared)
        for _ in range(iterations):
            normaliser = np.convolve(marginal, kernel)[grid - 1:2 * grid - 1]
            marginal = marginal * np.convolve(probability / normaliser, kernel)[grid - 1:2 * grid - 1]
            marginal = np.maximum(marginal / marginal.sum(), 1e-300)
        normaliser = np.convolve(marginal, kernel)[grid - 1:2 * grid - 1]
        achieved = float(probability @ (np.convolve(marginal, kernel * squared)[grid - 1:2 * grid - 1]
                                        / normaliser))
        rate = -slope * achieved - float(probability @ np.log(normaliser))
        return achieved, max(rate, 0.0), marginal

    centre = math.log(0.5 / distortion)
    low, high = centre - math.log(64.0), centre + math.log(64.0)
    marginal = probability.copy()
    rate = 0.0
    for _ in range(bisections):
        middle = 0.5 * (low + high)
        achieved, rate, marginal = blahut_arimoto(math.exp(middle), marginal)
        if abs(achieved - distortion) < 2e-3 * distortion:
            break
        if achieved > distortion:
            low = middle
        else:
            high = middle
    return rate


# ----- filter -----

@dataclass
class FaultModel:
    """Fault model shared by the plant and the agents.

    sigma2     process noise variance
    means      prior means of the gain per mode (nominal, degraded, reversed)
    variances  prior variances of the gain per mode
    hazard     probability per step that a nominal actuator fails
    initial    prior probabilities of the modes at the start
    """
    sigma2: float = 0.01
    means: Tuple[float, float, float] = (1.0, 0.5, -1.0)
    variances: Tuple[float, float, float] = (0.01, 0.04, 0.09)
    hazard: float = 1.0 / 60.0
    initial: Tuple[float, float, float] = (1.0, 0.0, 0.0)


@dataclass
class FaultFilter:
    """Exact filter over the fault hypotheses (see the module docstring)."""
    model: FaultModel

    def __post_init__(self):
        m = self.model
        self.t = 0
        self.nominal = np.array([m.means[0], m.variances[0]])          # gain mean, variance
        self.nominal_log_weight = 0.0                                    # log prior + log likelihood
        self.nominal_history: List[Tuple[float, float, float]] = []     # (mean, var, log weight) per t
        start = [index for index in (1, 2) if m.initial[index] > 0.0]
        self.fault_mode = np.array(start, dtype=int)                    # 1 or 2
        self.fault_mean = np.array([m.means[i] for i in start], dtype=float)
        self.fault_var = np.array([m.variances[i] for i in start], dtype=float)
        self.fault_log_weight = np.array([math.log(m.initial[i]) for i in start], dtype=float)
        if m.initial[0] > 0.0:
            self.nominal_log_weight = math.log(m.initial[0])
        else:
            self.nominal_log_weight = -np.inf

    def update(self, control_input: float, previous_state: float, state: float) -> None:
        """Add the transition x_{t-1} -> x_t under the input u_t."""
        m = self.model
        log_fault = math.log(m.hazard / 2.0)
        log_stay = math.log1p(-m.hazard)
        # a fault at this step: new hypotheses from the nominal one before the step
        self.fault_mode = np.concatenate([self.fault_mode, [1, 2]])
        self.fault_mean = np.concatenate([self.fault_mean, [m.means[1], m.means[2]]])
        self.fault_var = np.concatenate([self.fault_var, [m.variances[1], m.variances[2]]])
        self.fault_log_weight = np.concatenate(
            [self.fault_log_weight, [self.nominal_log_weight + log_fault] * 2])
        self.nominal_log_weight += log_stay
        # Kalman update of every gain and the predictive log-likelihood
        u, dx = float(control_input), float(state - previous_state)

        def kalman(mean, var):
            predictive_var = var * u * u + m.sigma2
            residual = dx - mean * u
            log_likelihood = -0.5 * (residual**2 / predictive_var + np.log(2.0 * np.pi * predictive_var))
            gain = var * u / predictive_var
            return mean + gain * residual, var - gain * u * var, log_likelihood

        mean, var, log_likelihood = kalman(self.nominal[0], self.nominal[1])
        self.nominal = np.array([mean, var])
        self.nominal_log_weight += float(log_likelihood)
        self.fault_mean, self.fault_var, log_likelihood = kalman(self.fault_mean, self.fault_var)
        self.fault_log_weight = self.fault_log_weight + log_likelihood
        self.t += 1

    def _normalised(self):
        log_weights = np.concatenate([[self.nominal_log_weight], self.fault_log_weight])
        weights = np.exp(log_weights - logsumexp(log_weights))
        return weights[0], weights[1:]

    def current_modes(self) -> np.ndarray:
        """Posterior of the mode at the latest step t."""
        nominal, faults = self._normalised()
        return np.array([nominal, faults[self.fault_mode == 1].sum(), faults[self.fault_mode == 2].sum()])

    def beliefs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Belief nodes of the planning model for the next step: the mode
        weights q(m) and the moment-matched gain mean and variance per mode."""
        m = self.model
        nominal, faults = self._normalised()
        q = np.zeros(3)
        means, variances = np.zeros(3), np.zeros(3)
        q[0] = nominal * (1.0 - m.hazard)
        means[0], variances[0] = self.nominal
        for index in (1, 2):
            select = self.fault_mode == index
            w = np.concatenate([faults[select], [nominal * m.hazard / 2.0]])
            mu = np.concatenate([self.fault_mean[select], [m.means[index]]])
            var = np.concatenate([self.fault_var[select], [m.variances[index]]])
            q[index] = w.sum()
            w = w / w.sum()
            means[index] = w @ mu
            variances[index] = w @ (var + mu**2) - means[index]**2
        return q / q.sum(), means, np.maximum(variances, 1e-12)


# ----- agent -----

@dataclass
class FaultConfig:
    """Goal, horizon and constraints of the agent.

    goal_mean, goal_variance   goal prior N(x_*, S_*) on the state
    constraints   "layers" (mode and gain information), "total" (one
                  constraint on their sum), or "none" (cautious)
    weights       fixed weights (mode, gain) on every input instead of KKT
                  multipliers (dual MPC); None for CBFE
    chance_bound  x_max of the region |x| > x_max to avoid (None: no chance
                  constraint)
    """
    goal_mean: float = 0.0
    goal_variance: float = 0.01
    horizon: int = 5
    control_prior_shape: float = 2.0
    constraints: str = "layers"
    weights: Optional[Tuple[float, float]] = None
    chance_bound: Optional[float] = None
    chance_level: float = 0.05
    max_multiplier: float = 1e6
    root_tolerance: float = 1e-3
    floor_margin: float = 0.01
    bound_margin: float = 1e-6
    max_sweeps: int = 50
    sweep_tolerance: float = 1e-9


@dataclass
class FaultAgent:
    """Exact fault filter and the constrained planner of the module docstring.
    ``plan(x)`` returns the next input, ``update(u, x, x_next)`` adds the
    transition that followed it."""
    config: FaultConfig
    model: FaultModel = field(default_factory=FaultModel)

    def __post_init__(self):
        self.filter = FaultFilter(self.model)
        self._nodes = input_nodes(1)[:, 0]
        self._previous_plan: Optional[np.ndarray] = None
        self.last_multipliers = {"mode": 0.0, "gain": 0.0, "total": 0.0, "chance": 0.0}
        self.last_floors = {"mode": 0.0, "gain": 0.0, "total": 0.0}
        self.last_plans = 0

    def update(self, control_input: float, previous_state: float, state: float) -> None:
        self.filter.update(control_input, previous_state, state)

    # ----- terms -----

    def gain_information(self, u, beliefs):
        q, means, variances = beliefs
        u2 = np.square(np.asarray(u, float))[..., None]
        return np.sum(q * 0.5 * np.log1p(variances * u2 / self.model.sigma2), axis=-1)

    def mode_information(self, u, beliefs):
        q, means, variances = beliefs
        u = np.asarray(u, float)[..., None]
        spread = self.model.sigma2 + variances * u**2
        entropy = mixture_entropy(means * u, spread, q)
        return entropy - np.sum(q * 0.5 * (LOG_2PI_E + np.log(spread)), axis=-1)

    def risk(self, u, x, beliefs):
        """q(|x'| > x_max) under the mixture prediction."""
        q, means, variances = beliefs
        u = np.asarray(u, float)[..., None]
        sd = np.sqrt(self.model.sigma2 + variances * u**2)
        bound = self.config.chance_bound
        mean = x + means * u
        return np.sum(q * (ndtr((mean - bound) / sd) + ndtr((-bound - mean) / sd)), axis=-1)

    def information(self, name: str, u, beliefs):
        if name == "mode":
            return self.mode_information(u, beliefs)
        if name == "gain":
            return self.gain_information(u, beliefs)
        return self.mode_information(u, beliefs) + self.gain_information(u, beliefs)

    def floors(self, x: float, beliefs, names: Sequence[str] = ("mode", "gain", "total")) -> Dict[str, float]:
        """Floors of the module docstring for the constraints ``names``, capped
        with a chance constraint at the information of the most informative
        safe input."""
        c, sigma2 = self.config, self.model.sigma2
        q, means, variances = beliefs
        u = self._nodes
        log_goal = (-0.5 * np.square(x + np.outer(u, means) - c.goal_mean) / c.goal_variance
                    + np.log(np.maximum(q, 1e-300)))
        log_weights = (c.control_prior_shape - 1.0) * np.log1p(-np.square(u)) + logsumexp(log_goal, axis=1)
        weights = np.exp(log_weights - log_weights.max())
        u2 = float(np.sum(weights * np.square(u)) / weights.sum())
        spread = float(q @ variances) * u2
        scale = math.sqrt(u2)
        floors = {}
        if "gain" in names:
            floors["gain"] = 0.5 * math.log((sigma2 + spread) / (sigma2 + c.goal_variance))
        if "mode" in names:
            floors["mode"] = rate_distortion(means * scale, sigma2 + variances * u2, q,
                                             sigma2 + spread + c.goal_variance)
        if "total" in names:
            floors["total"] = rate_distortion(means * scale, sigma2 + variances * u2, q,
                                              sigma2 + c.goal_variance)
        if c.chance_bound is not None:
            risk = self.risk(u, x, beliefs)
            safe = risk <= c.chance_level
            candidates = u[safe] if safe.any() else u[[int(np.argmin(risk))]]
            for name in floors:
                best = float(np.max(self.information(name, candidates, beliefs)))
                floors[name] = min(floors[name], best - c.floor_margin)
        return floors

    def _term_derivatives(self, name: str, u: float, beliefs, x: float) -> Tuple[float, float, float]:
        """Value, first and second derivative of an information or risk term."""
        h = 1e-4
        points = np.array([u - h, u, u + h])
        if name == "chance":
            values = self.risk(points, x, beliefs)
        else:
            values = self.information(name, points, beliefs)
        return (float(values[1]), float((values[2] - values[0]) / (2 * h)),
                float((values[2] - 2 * values[1] + values[0]) / h**2))

    # ----- planning -----

    def objective(self, plan: np.ndarray, weights: Dict[str, np.ndarray], x: float, beliefs) -> float:
        """J of the module docstring; ``weights[name]`` holds one weight per
        input (positive for information, the multiplier eta for risk)."""
        c = self.config
        q, means, variances = beliefs
        bbar = float(q @ means)
        vb = float(q @ (variances + means**2)) - bbar**2
        states = x + bbar * np.cumsum(plan)
        value = float(np.sum(-(c.control_prior_shape - 1.0) * np.log1p(-np.square(plan))
                             + (np.square(states - c.goal_mean) + vb * np.square(plan))
                             / (2.0 * c.goal_variance)))
        for name, w in weights.items():
            active = w > 0.0
            if not active.any():
                continue
            if name == "chance":
                value += float(w[0]) * float(self.risk(plan[0], x, beliefs))
            else:
                value -= float(np.sum(w[active] * self.information(name, plan[active], beliefs)))
        return value

    def _sweeps(self, plan: np.ndarray, weights, x: float, beliefs) -> Tuple[np.ndarray, float]:
        """Gauss-Newton message passing on the plan (iterative LQR on the
        linear dynamics x_k = x_{k-1} + bbar u_k)."""
        c = self.config
        q, means, variances = beliefs
        bbar = float(q @ means)
        vb = float(q @ (variances + means**2)) - bbar**2
        a1 = c.control_prior_shape - 1.0
        S = c.goal_variance
        K = len(plan)
        bound = 1.0 - c.bound_margin
        plan = np.clip(plan, -bound, bound)
        value = self.objective(plan, weights, x, beliefs)
        for _ in range(c.max_sweeps):
            states = x + bbar * np.cumsum(plan)
            previous = np.concatenate([[x], states[:-1]])
            # local derivatives of the input terms
            gu = 2.0 * a1 * plan / (1.0 - plan**2) + vb * plan / S
            guu = 2.0 * a1 * (1.0 + plan**2) / (1.0 - plan**2)**2 + vb / S
            for name, w in weights.items():
                for k in np.nonzero(w > 0.0)[0]:
                    _, d1, d2 = self._term_derivatives(name, float(plan[k]), beliefs, x)
                    sign = 1.0 if name == "chance" else -1.0
                    gu[k] += sign * w[k] * d1
                    guu[k] += max(sign * w[k] * d2, 0.0)
            # backward messages on the state (information form: precision, gradient)
            vx, vxx = 0.0, 0.0
            feedforward, feedback = np.zeros(K), np.zeros(K)
            for k in range(K - 1, -1, -1):
                wx = (states[k] - c.goal_mean) / S + vx
                wxx = 1.0 / S + vxx
                qu = gu[k] + bbar * wx
                quu = guu[k] + bbar**2 * wxx
                qux = bbar * wxx
                feedforward[k] = -qu / quu
                feedback[k] = -qux / quu
                vx = wx - qux * qu / quu
                vxx = wxx - qux**2 / quu
            # forward update with a backtracking line search
            step, improved = 1.0, False
            for _ in range(20):
                new = np.zeros(K)
                state = x
                for k in range(K):
                    new[k] = plan[k] + step * feedforward[k] + feedback[k] * (state - previous[k])
                    state = state + bbar * new[k]
                if np.all(np.abs(new) < bound):
                    new_value = self.objective(new, weights, x, beliefs)
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

    def _solve(self, weights, x, beliefs, start: Optional[np.ndarray]) -> Tuple[np.ndarray, float]:
        """Mode of the plan for fixed weights, from the warm start and, with a
        positive weight, from the constant plans at +-0.5."""
        self.last_plans += 1
        K = self.config.horizon
        starts = [np.zeros(K) if start is None else np.array(start, float)]
        if any(np.any(w > 0.0) for name, w in weights.items() if name != "chance"):
            starts += [np.full(K, 0.5), np.full(K, -0.5)]
        best, best_value = None, np.inf
        for initial in starts:
            plan, value = self._sweeps(initial, weights, x, beliefs)
            if value < best_value:
                best, best_value = plan, value
        return best, best_value

    def _weights(self, multipliers: Dict[str, float]) -> Dict[str, np.ndarray]:
        K = self.config.horizon
        out = {}
        for name, value in multipliers.items():
            w = np.zeros(K)
            w[0] = value
            out[name] = w
        return out

    def _residual(self, name: str, u: float, x: float, beliefs, floors) -> float:
        if name == "chance":
            return self.config.chance_level - float(self.risk(u, x, beliefs)) + 1e-12
        return float(self.information(name, u, beliefs)) - floors[name] + 1e-12

    def _root(self, name, multipliers, x, beliefs, floors, plan) -> Tuple[float, np.ndarray]:
        """Smallest multiplier >= 0 for which constraint ``name`` holds at the
        first input of the mode, with the other multipliers fixed."""
        c = self.config
        trial = dict(multipliers)

        def evaluate(value, start):
            trial[name] = value
            new_plan, _ = self._solve(self._weights(trial), x, beliefs, start)
            return self._residual(name, new_plan[0], x, beliefs, floors), new_plan

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

    def plan(self, x: float) -> float:
        """The next input at the state x."""
        c = self.config
        beliefs = self.filter.beliefs()
        self.last_plans = 0
        warm = (None if self._previous_plan is None
                else np.concatenate([self._previous_plan[1:], self._previous_plan[-1:]]))
        if c.weights is not None:
            K = c.horizon
            weights = {"mode": np.full(K, c.weights[0]), "gain": np.full(K, c.weights[1])}
            plan, _ = self._solve(weights, x, beliefs, warm)
            self._previous_plan = plan
            return float(plan[0])
        names = {"layers": ["mode", "gain"], "total": ["total"], "none": []}[c.constraints]
        floors = self.floors(x, beliefs, names) if names else {}
        names = [name for name in names if floors[name] > 0.0]
        if c.chance_bound is not None:
            names = names + ["chance"]
        self.last_floors = {name: floors.get(name, 0.0) for name in ("mode", "gain", "total")}
        multipliers = {name: 0.0 for name in names}
        plan, _ = self._solve(self._weights(multipliers), x, beliefs, warm)
        for _ in range(6):
            changed = False
            for name in names:
                if self._residual(name, plan[0], x, beliefs, floors) >= 0.0 and multipliers[name] == 0.0:
                    continue
                value, plan = self._root(name, multipliers, x, beliefs, floors, plan)
                if abs(value - multipliers[name]) > 1e-3 * (1.0 + multipliers[name]):
                    changed = True
                multipliers[name] = value
            if not changed:
                break
        self.last_multipliers = {name: multipliers.get(name, 0.0) for name in ("mode", "gain", "total", "chance")}
        self._previous_plan = plan
        return float(plan[0])
