"""Light-dark positioning with direction-dependent actuator gains (main.tex,
Section V-C):

    x_k = x_{k-1} + b_s u_k + w_k ,   s = + if u_k >= 0, else - ,

with unknown gains b_+ and b_- and the light-dark measurement noise of
``agent.py``. A move into the light reveals b_+, but the return to a target
in the dark needs b_-, so the agent must calibrate b_- on purpose before it
leaves the light.

``DirectionalGridFilter`` computes the exact posterior over (x, b_+, b_-) on a
grid. The planning model takes its moments as one belief node of the state and
the gains: means (m, beta_+, beta_-), the state variance V, the gain
covariance P (2x2) and the state-gain covariances C = (C_+, C_-). With the
beliefs fixed within the plan,

    m_k = m_{k-1} + beta_{s_k} u_k ,
    V_k = V + k q + sum_{j<=k} P_{s_j} u_j^2 + 2 C_{s_1} u_1 ,

and the chain rule splits the information of the next output into the gain
layer and the state layer,

    I_par(u)   = 1/2 ln(1 + G(u) / (V_c + q + r)) ,   G(u) = V - V_c + P_s u^2 + 2 C_s u ,
    I_state(u) = 1/2 ln(1 + (V_c + q) / r) ,          V_c  = V - C P^-1 C^T ,

with r at the predicted mean m + beta_s u. G(u) is the spread that the gains
add to the next state for an input of sign s, so the goal-reaching inputs of
the floor,

    beta_par   = 1/2 ln((A + E_p~[G(u)]) / (A + S_*)) ,   beta_state = 1/2 ln(A / S_*) ,
    A = V_c + q ,

count the spread of the gain that the goal needs: b_- for a return to a target
at lower x. The planner, the KKT search, the dual MPC and the belief-space
planner are those of ``gain.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import convolve1d

from .agent import LightDarkModel
from .gain import GainConfig, GainLightDarkAgent


def _shift(array: np.ndarray, delta: float, spacing: float) -> np.ndarray:
    """Shift ``array`` along its last axis by delta (linear interpolation,
    zero padding): out(x) = in(x - delta)."""
    steps = delta / spacing
    n = int(math.floor(steps))
    f = steps - n
    out = np.zeros_like(array)
    N = array.shape[-1]

    def place(source_shift: int, weight: float):
        if weight == 0.0:
            return
        if source_shift >= 0:
            if source_shift < N:
                out[..., source_shift:] += weight * array[..., :N - source_shift]
        else:
            if -source_shift < N:
                out[..., :N + source_shift] += weight * array[..., -source_shift:]

    place(n, 1.0 - f)
    place(n + 1, f)
    return out


@dataclass
class DirectionalGridFilter:
    """Exact posterior over (b_+, b_-, x) on a grid."""
    model: LightDarkModel
    x_mean: float = 0.0
    x_variance: float = 0.25
    b_mean: float = 1.0
    b_variance: float = 0.09
    x_range: Tuple[float, float, int] = (-2.5, 4.5, 701)
    b_range: Tuple[float, float, int] = (0.1, 1.9, 61)

    def __post_init__(self):
        self.x = np.linspace(*self.x_range)
        self.b = np.linspace(*self.b_range)
        self.dx = self.x[1] - self.x[0]
        px = np.exp(-0.5 * (self.x - self.x_mean)**2 / self.x_variance)
        pb = np.exp(-0.5 * (self.b - self.b_mean)**2 / self.b_variance)
        self.p = pb[:, None, None] * pb[None, :, None] * px[None, None, :]
        self.p /= self.p.sum()
        half = max(1, int(math.ceil(6.0 * math.sqrt(self.model.q) / self.dx)))
        offsets = self.dx * np.arange(-half, half + 1)
        kernel = np.exp(-0.5 * offsets**2 / self.model.q)
        self.kernel = kernel / kernel.sum()

    def update(self, control_input: float, output: float) -> None:
        predicted = np.empty_like(self.p)
        u = float(control_input)
        if u >= 0.0:
            for j, gain in enumerate(self.b):
                predicted[j] = _shift(self.p[j], gain * u, self.dx)
        else:
            for j, gain in enumerate(self.b):
                predicted[:, j] = _shift(self.p[:, j], gain * u, self.dx)
        predicted = convolve1d(predicted, self.kernel, axis=-1, mode="constant")
        r = self.model.noise(self.x)
        log_likelihood = -0.5 * ((output - self.x)**2 / r + np.log(r))
        posterior = predicted * np.exp(log_likelihood - log_likelihood.max())[None, None, :]
        self.p = posterior / posterior.sum()

    def moments(self):
        """Means (m, beta_+, beta_-) and the covariance of (x, b_+, b_-)."""
        px = self.p.sum(axis=(0, 1))
        pp = self.p.sum(axis=(1, 2))
        pm = self.p.sum(axis=(0, 2))
        m, bp, bm = float(px @ self.x), float(pp @ self.b), float(pm @ self.b)
        dx, db_p, db_m = self.x - m, self.b - bp, self.b - bm
        V = float(px @ dx**2)
        Pp, Pm = float(pp @ db_p**2), float(pm @ db_m**2)
        Cp = float(np.einsum("ijk,i,k->", self.p, db_p, dx))
        Cm = float(np.einsum("ijk,j,k->", self.p, db_m, dx))
        Ppm = float(np.einsum("ijk,i,j->", self.p, db_p, db_m))
        mean = np.array([m, bp, bm])
        cov = np.array([[V, Cp, Cm], [Cp, Pp, Ppm], [Cm, Ppm, Pm]])
        return mean, cov


@dataclass
class DirectionalAgent(GainLightDarkAgent):
    """The agent of ``gain.py`` with direction-dependent gains."""

    def __post_init__(self):
        super().__post_init__()
        self.filter = DirectionalGridFilter(self.model, self.x_mean, self.x_variance, self.b_mean, self.b_variance)

    def _beliefs(self, moments):
        mean, cov = moments
        if not self.config.joint_belief:
            cov = cov.copy()
            cov[0, 1:] = cov[1:, 0] = 0.0
        return mean, cov

    @staticmethod
    def _parts(beliefs):
        """m, (beta_+, beta_-), V, (P_+, P_-), (C_+, C_-), V_c."""
        mean, cov = beliefs
        Ptheta = cov[1:, 1:]
        Cx = cov[0, 1:]
        Vc = float(cov[0, 0] - Cx @ np.linalg.solve(Ptheta + 1e-12 * np.eye(2), Cx))
        return (float(mean[0]), mean[1:], float(cov[0, 0]), np.diag(Ptheta).copy(), Cx.copy(), max(Vc, 1e-12))

    def _starts(self, m: float, b_hat: float, start: Optional[np.ndarray]) -> list:
        return super()._starts(m, b_hat, start)

    def objective(self, plan: np.ndarray, weights: Dict[str, np.ndarray], beliefs):
        c, q = self.config, self.model.q
        m, betas, V, Ps, Cs, Vc = self._parts(beliefs)
        K = len(plan)
        a1 = c.control_prior_shape - 1.0
        S = c.goal_variance
        side = (plan < 0.0).astype(int)                 # 0: +, 1: -
        beta = betas[side]
        P = Ps[side]
        means = m + np.cumsum(beta * plan)
        caution = V + q * np.arange(1, K + 1) + np.cumsum(P * plan**2) + 2.0 * Cs[side[0]] * plan[0]
        error = means - c.goal_mean
        value = float(np.sum(-a1 * np.log1p(-plan**2) + (error**2 + caution) / (2.0 * S)))
        dm = error / S
        extra_u = np.zeros(K)
        wp, ws = weights["parameter"], weights["state"]
        if np.any(wp > 0.0) or np.any(ws > 0.0):
            A = Vc + q * np.arange(1, K + 1)
            r = self.model.noise(means)
            dr = self.model.noise_derivative(means)
            G = P * plan**2
            G[0] += (V - Vc) + 2.0 * Cs[side[0]] * plan[0]
            dG = 2.0 * P * plan
            dG[0] += 2.0 * Cs[side[0]]
            value -= float(np.sum(wp * 0.5 * np.log1p(G / (A + r))))
            value -= float(np.sum(ws * 0.5 * np.log1p(A / r)))
            extra_u = -wp * 0.5 * dG / (A + r + G)
            dm = dm + wp * 0.5 * G * dr / ((A + r) * (A + r + G))
            dm = dm + ws * 0.5 * A * dr / (r * (r + A))
        tail_m = np.cumsum(dm[::-1])[::-1]
        counts = K - np.arange(K)
        gradient = (2.0 * a1 * plan / (1.0 - plan**2) + beta * tail_m + P * plan * counts / S + extra_u)
        gradient[0] += K * Cs[side[0]] / S
        return value, gradient

    def bsp_objective(self, plan: np.ndarray, moments) -> float:
        c, q = self.config, self.model.q
        mean, cov = moments
        mean, cov = mean.copy(), cov.copy()
        a1 = c.control_prior_shape - 1.0
        value = 0.0
        for u in plan:
            F = np.eye(3)
            F[0, 1 if u >= 0.0 else 2] = u
            mean = F @ mean
            cov = F @ cov @ F.T
            cov[0, 0] += q
            value += -a1 * math.log1p(-u * u) + ((mean[0] - c.goal_mean)**2 + cov[0, 0]) / (2.0 * c.goal_variance)
            r = float(self.model.noise(mean[0]))
            gain = cov[:, 0] / (cov[0, 0] + r)
            cov = cov - np.outer(gain, cov[0, :])
        return value

    def _solve_bsp(self, moments, start) -> np.ndarray:
        from scipy.optimize import minimize
        bound = 1.0 - self.config.bound_margin
        mean, _ = moments
        best = None
        for initial in self._starts(float(mean[0]), float(mean[1]), start):
            result = minimize(self.bsp_objective, np.clip(initial, -bound, bound), args=(moments,),
                              method="L-BFGS-B", bounds=[(-bound, bound)] * self.config.horizon,
                              options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 300})
            if best is None or result.fun < best.fun:
                best = result
        return best.x

    def _solve(self, weights, beliefs, start):
        from scipy.optimize import minimize
        bound = 1.0 - self.config.bound_margin
        m, betas = float(beliefs[0][0]), beliefs[0][1:]
        best = None
        for initial in self._starts(m, float(betas[0]), start):
            result = minimize(self.objective, np.clip(initial, -bound, bound), args=(weights, beliefs), jac=True,
                              method="L-BFGS-B", bounds=[(-bound, bound)] * self.config.horizon,
                              options={"ftol": 1e-13, "gtol": 1e-10, "maxiter": 400})
            if best is None or result.fun < best.fun:
                best = result
        return best.x, float(best.fun)

    def _gain_spread(self, u, beliefs):
        m, betas, V, Ps, Cs, Vc = self._parts(beliefs)
        side = (np.asarray(u) < 0.0).astype(int)
        return (V - Vc) + Ps[side] * np.square(u) + 2.0 * Cs[side] * u

    def floors(self, beliefs) -> Dict[str, float]:
        c, q = self.config, self.model.q
        m, betas, V, Ps, Cs, Vc = self._parts(beliefs)
        u = self._nodes
        side = (u < 0.0).astype(int)
        predicted = m + betas[side] * u
        log_weights = ((c.control_prior_shape - 1.0) * np.log1p(-u**2)
                       - 0.5 * (predicted - c.goal_mean)**2 / c.goal_variance)
        weights = np.exp(log_weights - log_weights.max())
        G = self._gain_spread(u, beliefs)
        spread = float(np.sum(weights * G) / weights.sum())
        A = Vc + q
        floors = {"parameter": 0.5 * math.log((A + spread) / (A + c.goal_variance)),
                  "state": 0.5 * math.log(A / c.goal_variance)}
        r = self.model.noise(predicted)
        best_parameter = float(np.max(0.5 * np.log1p(G / (A + r))))
        best_state = float(np.max(0.5 * np.log1p(A / r)))
        floors["parameter"] = min(floors["parameter"], best_parameter - c.floor_margin)
        floors["state"] = min(floors["state"], best_state - c.floor_margin)
        return floors

    def _residual(self, name: str, u: float, beliefs, floors) -> float:
        m, betas, V, Ps, Cs, Vc = self._parts(beliefs)
        A = Vc + self.model.q
        predicted = m + betas[int(u < 0.0)] * u
        r = float(self.model.noise(predicted))
        if name == "parameter":
            value = 0.5 * math.log1p(float(self._gain_spread(np.array(u), beliefs)) / (A + r))
        else:
            value = 0.5 * math.log1p(A / r)
        return value - floors[name] + 1e-12
