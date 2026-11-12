"""CBFE agent for a quadcopter in wind (main.tex, Sections 2-4).

The model
---------
An inner loop tracks the commanded collective thrust and attitude within a
step, so the planner acts on the translational dynamics with the input
u in [-1, 1]^3: the thrust per mass tau = g (1 + alpha u_1), the roll
phi = phi_max u_2 and the pitch theta = phi_max u_3 (yaw zero). The thrust
axis is n(u) = (cos phi sin theta, -sin phi, cos phi cos theta), and
P(u) = I - n n^T projects onto the rotor plane.

The wind w acts through rotor drag, which grows with the thrust and the
relative airspeed in the rotor plane (Martin and Salauen, 2010; Faessler et
al., 2018). With the drag coefficient k, the specific force of step k is

    f_k = tau n - k tau P (v_{k-1} - w_k) + eta_k
        = tau n - k tau P v_{k-1} + tau P c_k + eta_k ,    c_k = k w_k ,

with the unmodelled force eta_k ~ N(0, sigma_eta^2 I), and

    v_k = v_{k-1} + dt (f_k - g e_3) ,   p_k = p_{k-1} + dt v_{k-1} + dt^2 / 2 (f_k - g e_3) .

The state is x = (p, v, f), and the outputs are a position fix and the IMU,

    y_k = (p_k + n_p, f_k + n_f) ,   n_p ~ N(0, sigma_p^2 I) ,  n_f ~ N(0, sigma_f^2 I) .

The controller knows this model and the noise levels, but neither the drag
coefficient k nor the wind. It estimates theta = (c, k), where the wind force
c follows a random walk with steps N(0, q_c I). The model is linear in theta
given the input and the velocity, and the input sets how strongly theta acts:
the thrust scales the drag, and the attitude sets which wind components lie
in the rotor plane.

Filter and planning model
-------------------------
An extended Kalman filter over (x, theta) approximates the posterior outside
the planning model, and the planner takes the marginals q(x_t) and q(theta)
as belief nodes. Under the factorization constraints, every transition node
draws x_{k-1} and theta independently, so with the Jacobians A_k and B_k of
the step at the predicted means,

    m_k = step(m_{k-1}, theta_hat, u_k) ,
    S_k = A_k S_{k-1} A_k^T + B_k P_k B_k^T + Q_x ,   P_k = P + k diag(q_c, q_c, q_c, 0) .

The goal prior N(p_*, S_* I) acts on the position output, whose predicted
covariance is Sigma_k = S_k[p, p] + sigma_p^2 I. The plan of K inputs minimises

    J(u) = sum_k [ -(c - 1) sum_i ln(1 - u_{k,i}^2) + |m_{p,k} - p_*|^2 / (2 S_*)
                   + tr(Sigma_k) / (2 S_*) ] - gamma I(u_1) ,

where the trace term is optional (``caution``). Since the goal acts only on
the position output, the information constraint and its floor eq. (10)
use that output:

    I(u)  = 1/2 ln |Sigma_1(u) (sigma_p^2 I)^{-1}| ,
    beta  = 1/2 ln |Sigma_bar ((sigma_p^2 + S_*) I)^{-1}| ,
    Sigma_bar = E_p~[Sigma_1(u)] ,   p~(u) ~ p(u) N(m_{p,1}(u); p_*, S_* I) ,

over the 4096 input nodes of ``agents.common.input_nodes``, with the floor
capped at the largest information minus ``floor_margin``. The KKT multiplier
is zero if the plan at gamma = 0 meets the floor, and otherwise the smallest
multiplier at which the first input of the plan meets it: a bracket that
grows fourfold from 1, then regula falsi (Illinois). CasADi builds J, its
gradient and the one-step moments, and L-BFGS-B minimises J.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Tuple

import casadi as ca
import numpy as np
from scipy.optimize import minimize

from agents.common import input_nodes

GRAVITY = 9.81
NX, NT = 9, 7          # state (p, v, f) and parameters (c, k, a, b, eta)


@dataclass
class Config:
    """Model, goal and constraint of the agent.

    dt, horizon          step [s] and planning horizon K
    thrust_range         alpha of tau = g (1 + alpha u_1)
    tilt_max             phi_max of the roll and the pitch [rad]
    position_noise       sigma_p of the position fix [m]
    imu_noise            sigma_f of the specific force of the IMU [m/s^2]
    force_noise          sigma_eta of the unmodelled specific force [m/s^2]
    wind_step            sqrt(q_c), step of the random walk of the wind force c
    gain_step            step of the random walk of the thrust gains (a, b, eta)
    goal, goal_std       target p_* [m] and sqrt(S_*) [m]
    control_prior_shape  c of the Beta(c, c) control prior of every input
    constrained          False: no information constraint (gamma = 0)
    caution              False: J drops the trace term (certainty equivalence)
    """
    dt: float = 0.5
    horizon: int = 6
    thrust_range: float = 0.5
    tilt_max: float = 0.4
    position_noise: float = 0.05
    imu_noise: float = 0.2
    force_noise: float = 0.2
    wind_step: float = 0.025
    gain_step: float = 0.0
    goal: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal_std: float = 0.1
    control_prior_shape: float = 2.0
    constrained: bool = True
    caution: bool = True
    max_multiplier: float = 1e6
    root_tolerance: float = 1e-3
    floor_margin: float = 0.01
    bound_margin: float = 1e-6


def _step(x, theta, u, dt: float, thrust_range: float, tilt_max: float):
    """Mean of the next state for the state x = (p, v, f), theta = (c, k, a, b, eta)
    and u. The thrust gains rotate and scale the horizontal thrust by
    [[a, -b], [b, a]] (a heading misalignment and a thrust loss) and scale the
    vertical thrust by eta."""
    p, v = x[0:3], x[3:6]
    c, k, a, b, eta = theta[0:3], theta[3], theta[4], theta[5], theta[6]
    tau = GRAVITY * (1.0 + thrust_range * u[0])
    phi, pitch = tilt_max * u[1], tilt_max * u[2]
    n = ca.vertcat(ca.cos(phi) * ca.sin(pitch), -ca.sin(phi), ca.cos(phi) * ca.cos(pitch))
    rotor_plane = ca.SX.eye(3) - n @ n.T
    thrust = tau * ca.vertcat(a * n[0] - b * n[1], b * n[0] + a * n[1], eta * n[2])
    f = thrust - k * tau * (rotor_plane @ v) + tau * (rotor_plane @ c)
    a = f - ca.vertcat(0.0, 0.0, GRAVITY)
    return ca.vertcat(p + dt * v + 0.5 * dt**2 * a, v + dt * a, f)


@lru_cache(maxsize=8)
def functions(dt: float, horizon: int, thrust_range: float, tilt_max: float, shape: float):
    """CasADi functions of the model: the step with its Jacobians, the planning
    objective with its gradient and the information of the first input, and
    the one-step moments of the position output."""
    x, th, u = ca.SX.sym("x", NX), ca.SX.sym("theta", NT), ca.SX.sym("u", 3)
    mean = _step(x, th, u, dt, thrust_range, tilt_max)
    step = ca.Function("step", [x, th, u], [mean, ca.jacobian(mean, x), ca.jacobian(mean, th)])

    U = ca.SX.sym("U", 3 * horizon)
    mx, Sx = ca.SX.sym("mx", NX), ca.SX.sym("Sx", NX, NX)
    mt, Pt = ca.SX.sym("mt", NT), ca.SX.sym("Pt", NT, NT)
    goal, gamma = ca.SX.sym("goal", 3), ca.SX.sym("gamma")
    noise = ca.SX.sym("noise", 6)     # sigma_p^2, S_*, sigma_eta^2, q_c, trace weight, q_gain
    r_p, s_goal, q_eta, q_c, trace, q_g = (noise[i] for i in range(6))
    lift = ca.vertcat(0.5 * dt**2 * ca.SX.eye(3), dt * ca.SX.eye(3), ca.SX.eye(3))
    Qx = q_eta * (lift @ lift.T)
    walk = ca.diag(ca.vertcat(q_c, q_c, q_c, 0.0, q_g, q_g, q_g))

    def advance(m, S, uk, k):
        m_next, A, B = step(m, mt, uk)
        S_next = A @ S @ A.T + B @ (Pt + (k + 1) * walk) @ B.T + Qx
        return m_next, S_next

    cost, m, S, info = 0.0, mx, Sx, 0.0
    for k in range(horizon):
        uk = U[3 * k:3 * k + 3]
        m, S = advance(m, S, uk, k)
        Sigma = S[0:3, 0:3] + r_p * ca.SX.eye(3)
        error = m[0:3] - goal
        cost += (-(shape - 1.0) * ca.sum1(ca.log(1.0 - uk**2)) + ca.dot(error, error) / (2.0 * s_goal)
                 + trace * ca.trace(Sigma) / (2.0 * s_goal))
        if k == 0:
            info = 0.5 * (ca.log(ca.det(Sigma)) - 3.0 * ca.log(r_p))
    J = cost - gamma * info
    objective = ca.Function("J", [U, mx, Sx, mt, Pt, goal, gamma, noise], [J, ca.gradient(J, U), info])

    m1, S1 = advance(mx, Sx, u, 0)
    moments = ca.Function("moments", [u, mx, Sx, mt, Pt, noise],
                          [m1[0:3], ca.reshape(S1[0:3, 0:3], 9, 1)])
    return step, objective, moments


@dataclass
class Agent:
    """Extended Kalman filter over (x, theta) and the planner of the module
    docstring. ``observe(y)`` conditions on outputs of the current state,
    ``plan()`` returns the next input, and ``update(u, y)`` adds the outputs
    that followed it."""
    config: Config
    state_mean: np.ndarray = field(default_factory=lambda: np.zeros(NX))
    state_variance: np.ndarray = field(default_factory=lambda: np.ones(NX))
    parameter_mean: np.ndarray = field(default_factory=lambda: np.zeros(NT))
    parameter_variance: np.ndarray = field(default_factory=lambda: np.ones(NT))

    def __post_init__(self):
        c = self.config
        self.step, self.objective_fn, moments = functions(c.dt, c.horizon, c.thrust_range, c.tilt_max,
                                                          c.control_prior_shape)
        self._nodes = input_nodes(3)
        self._moments = moments.map(len(self._nodes))
        self._log_prior_nodes = (c.control_prior_shape - 1.0) * np.log1p(-np.square(self._nodes)).sum(axis=1)
        self.mean = np.concatenate([self.state_mean, self.parameter_mean]).astype(float)
        self.covariance = np.diag(np.concatenate([self.state_variance, self.parameter_variance])).astype(float)
        self.last_multiplier, self.last_floor, self.last_information, self.last_plans = 0.0, np.nan, np.nan, 0
        self._previous_plan: Optional[np.ndarray] = None

    # ----- filter -----

    def _noise(self) -> np.ndarray:
        c = self.config
        return np.array([c.position_noise**2, c.goal_std**2, c.force_noise**2, c.wind_step**2,
                         1.0 if c.caution else 0.0, c.gain_step**2])

    def observe(self, position: Optional[np.ndarray] = None, force: Optional[np.ndarray] = None) -> None:
        """Condition the belief on a position fix and/or an IMU sample."""
        c = self.config
        rows, values, variances = [], [], []
        if position is not None:
            rows += [0, 1, 2]; values += list(position); variances += [c.position_noise**2] * 3
        if force is not None:
            rows += [6, 7, 8]; values += list(force); variances += [c.imu_noise**2] * 3
        if not rows:
            return
        H = np.zeros((len(rows), NX + NT))
        H[np.arange(len(rows)), rows] = 1.0
        innovation = H @ self.covariance @ H.T + np.diag(variances)
        gain = np.linalg.solve(innovation, H @ self.covariance).T
        self.mean = self.mean + gain @ (np.array(values) - H @ self.mean)
        self.covariance = self.covariance - gain @ innovation @ gain.T
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def update(self, control_input: np.ndarray, position: np.ndarray, force: np.ndarray) -> None:
        """Predict through the input, linearised at the mean, and condition on
        the outputs that followed."""
        c = self.config
        x, th = self.mean[:NX], self.mean[NX:]
        mean, A, B = (np.array(v) for v in self.step(x, th, control_input))
        F = np.eye(NX + NT)
        F[:NX, :NX], F[:NX, NX:] = A, B
        F[NX:, :NX] = 0.0
        lift = np.vstack([0.5 * c.dt**2 * np.eye(3), c.dt * np.eye(3), np.eye(3)])
        Q = np.zeros((NX + NT, NX + NT))
        Q[:NX, :NX] = c.force_noise**2 * lift @ lift.T
        Q[NX:NX + 3, NX:NX + 3] = c.wind_step**2 * np.eye(3)
        Q[NX + 4:, NX + 4:] = c.gain_step**2 * np.eye(3)
        self.mean = np.concatenate([mean.ravel(), th])
        self.covariance = F @ self.covariance @ F.T + Q
        self.observe(position, force)

    def set_parameters(self, theta: np.ndarray) -> None:
        """Known parameters: the reference controller that knows the wind."""
        self.mean[NX:] = theta
        self.covariance[NX:, :] = 0.0
        self.covariance[:, NX:] = 0.0

    def beliefs(self):
        return (self.mean[:NX].copy(), self.covariance[:NX, :NX].copy(), self.mean[NX:].copy(),
                self.covariance[NX:, NX:].copy())

    # ----- constraint -----

    def floor(self, beliefs) -> Tuple[float, float]:
        """Eq. (10) on the position output over the goal-reaching inputs, and
        the largest information on the input nodes."""
        c = self.config
        mx, Sx, mt, Pt = beliefs
        means, covs = self._moments(self._nodes.T, mx, Sx, mt, Pt, self._noise())
        means, covs = np.array(means), np.array(covs).T.reshape(-1, 3, 3)
        covs = covs + c.position_noise**2 * np.eye(3)
        log_w = self._log_prior_nodes - 0.5 * np.sum((means.T - np.asarray(c.goal))**2, axis=1) / c.goal_std**2
        w = np.exp(log_w - log_w.max())
        spread = np.einsum("n,nij->ij", w / w.sum(), covs)
        floor = 0.5 * (np.linalg.slogdet(spread)[1] - 3.0 * math.log(c.position_noise**2 + c.goal_std**2))
        largest = 0.5 * float(np.max(np.linalg.slogdet(covs)[1])) - 1.5 * math.log(c.position_noise**2)
        return min(floor, largest - c.floor_margin), largest

    # ----- planning -----

    def _solve(self, multiplier: float, beliefs, start: np.ndarray) -> Tuple[np.ndarray, float]:
        """Mode of the plan for a fixed multiplier, and the information of its
        first input."""
        self.last_plans += 1
        c = self.config
        bound = 1.0 - c.bound_margin
        args = (*beliefs, np.asarray(c.goal, float), multiplier, self._noise())

        def fun(U):
            value, grad, _ = self.objective_fn(U, *args)
            return float(value), np.array(grad).ravel()

        starts = [np.clip(start, -bound, bound)]
        if multiplier > 0.0:
            starts.append(np.zeros(3 * c.horizon))
        best, best_value = None, np.inf
        for initial in starts:
            result = minimize(fun, initial, jac=True, method="L-BFGS-B", bounds=[(-bound, bound)] * len(initial),
                              options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 300})
            if result.fun < best_value:
                best, best_value = result.x, float(result.fun)
        info = float(self.objective_fn(best, *args)[2])
        return best, info

    def _multiplier(self, beliefs, floor: float, start: np.ndarray) -> Tuple[float, np.ndarray, float]:
        c = self.config
        plan, info = self._solve(0.0, beliefs, start)
        if info - floor >= 0.0:
            return 0.0, plan, info
        low, trial = (0.0, info - floor), 1.0
        while True:
            plan, info = self._solve(trial, beliefs, plan)
            if info - floor >= 0.0:
                high = (trial, info - floor, plan, info)
                break
            low = (trial, info - floor)
            if trial >= c.max_multiplier:
                return trial, plan, info
            trial = min(4.0 * trial, c.max_multiplier)
        if high[1] < c.root_tolerance:
            return high[0], high[2], high[3]
        side = 0
        for _ in range(15):
            (t_low, r_low), (t_high, r_high) = low, high[:2]
            trial = t_high - r_high * (t_high - t_low) / (r_high - r_low)
            plan, info = self._solve(trial, beliefs, high[2])
            residual = info - floor
            if abs(residual) < c.root_tolerance:
                return trial, plan, info
            if residual > 0.0:
                high = (trial, residual, plan, info)
                if side == 1:
                    low = (low[0], 0.5 * low[1])
                side = 1
            else:
                low = (trial, residual)
                if side == -1:
                    high = (high[0], 0.5 * high[1], high[2], high[3])
                side = -1
        return high[0], high[2], high[3]

    def plan(self) -> np.ndarray:
        """The next input: the first input of the mode of the plan."""
        c = self.config
        beliefs = self.beliefs()
        self.last_plans = 0
        start = (np.zeros(3 * c.horizon) if self._previous_plan is None
                 else np.concatenate([self._previous_plan[3:], self._previous_plan[-3:]]))
        if c.constrained:
            floor, _ = self.floor(beliefs)
            multiplier, plan, info = self._multiplier(beliefs, floor, start)
        else:
            floor, multiplier = np.nan, 0.0
            plan, info = self._solve(0.0, beliefs, start)
        self.last_floor, self.last_multiplier, self.last_information = floor, float(multiplier), info
        self._previous_plan = plan
        return plan[:3].copy()
