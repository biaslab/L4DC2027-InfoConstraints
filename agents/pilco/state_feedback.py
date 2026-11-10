"""PILCO for a directly observed state.

``pilco.py`` implements PILCO for the thermal stage of the TAC 2026
project, where a Luenberger observer and a thermal Kalman filter estimate
the state. This module keeps the algorithm of that controller for systems
whose state is measured directly, such as an arm with joint encoders:

  * one GP per state coordinate models the increment s_{t+1} - s_t from
    (s_t, a_t), with standardised inputs and an ARD RBF kernel;
  * the policy is linear in the error coordinates (position - target,
    velocity) and squashed by tanh into the action box [-1, 1]. The
    thermal-stage policy is linear in (s, target) with independent
    weights, whose equilibrium coincides with the target only if the
    target weights cancel the position weights exactly; in error
    coordinates, every target is an equilibrium of the policy with zero
    bias;
  * at every new target, and every ``refit_every`` steps within a target,
    the GPs are refitted and IPOPT minimises the cost of a GP-mean rollout
    over ``horizon`` steps, within a trust region around the previous
    policy and with a soft pull towards a PD anchor, which is also the
    initial policy.

As in ``pilco.py``, the rollouts use the GP means only (no moment
matching), so the controller is risk-neutral. The state is
s = (position, velocity), both of dimension ``pos_dim``, and the targets
refer to the position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import casadi as ca
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from .pilco import _ca_gp_mean, _gp_extract


@dataclass
class StatePILCOConfig:
    pos_dim: int = 2              # position coordinates; the velocity has the same dimension
    act_dim: int = 2
    horizon: int = 15             # rollout steps in the policy NLP
    refit_every: int = 30         # steps between refits within one target
    n_max_data: int = 300         # GP training points per refit (random subset)
    cost_position: float = 1.0
    cost_velocity: float = 0.05
    action_penalty: float = 0.01
    kp: float = 1.5               # PD anchor gains in action units
    kd: float = 0.8
    theta_max: float = 5.0
    theta_step: float = 0.3
    theta_anchor: float = 0.3
    gp_alpha: float = 1e-4
    gp_n_restarts: int = 1
    ipopt_max_iter: int = 100
    ipopt_tol: float = 1e-4
    seed: int = 0


class StatePILCO:
    """PILCO with a measured state s = (position, velocity).

    ``act(s, target)`` returns an action in [-1, 1]^act_dim and records the
    transition from the previous call, so the caller only has to call
    ``reset()`` at the start of an episode and ``new_target()`` when the
    target changes. ``add_data`` supplies prior transitions, e.g. from a
    nominal simulator.
    """

    def __init__(self, cfg: StatePILCOConfig = StatePILCOConfig()):
        self.cfg = cfg
        n, m = cfg.pos_dim, cfg.act_dim
        self.D_s, self.D_u = 2 * n, m
        self.D_z, self.D_phi = self.D_s + m, self.D_s
        K0 = np.zeros((m, self.D_phi))
        K0[:, :n] = -cfg.kp * np.eye(m, n)            # position error
        K0[:, n:] = -cfg.kd * np.eye(m, n)            # velocity
        self.theta_anchor = np.concatenate([K0.ravel(), np.zeros(m)])
        self.theta = self.theta_anchor.copy()
        self.rng = np.random.default_rng(cfg.seed)
        self._Z: list = []
        self._dS: list = []
        self._gp_params: Optional[list] = None
        self._prev = None
        self._since_fit = 0
        self._fit_due = True
        self.n_fits = 0

    # -- data and episode bookkeeping ---------------------------------------

    def add_data(self, Z, dS) -> None:
        """Transitions z = (s, a) -> s_next - s."""
        self._Z.extend(np.atleast_2d(Z))
        self._dS.extend(np.atleast_2d(dS))
        self._fit_due = True

    def reset(self) -> None:
        """New episode: the data and the policy persist."""
        self._prev = None

    def new_target(self) -> None:
        self._fit_due = True

    # -- policy -------------------------------------------------------------

    def _unpack(self, theta):
        m = self.D_u
        return theta[:m * self.D_phi].reshape(m, self.D_phi), theta[m * self.D_phi:]

    def _features(self, s, target):
        n = self.cfg.pos_dim
        return np.r_[s[:n] - target, s[n:]]

    def policy(self, s, target) -> np.ndarray:
        K, b = self._unpack(self.theta)
        return np.tanh(K @ self._features(s, target) + b)

    def _ca_policy(self, theta, s, target):
        """CasADi counterpart of ``policy`` for a symbolic theta. CasADi
        reshapes column-major, so the gains are reshaped transposed."""
        n, m = self.cfg.pos_dim, self.D_u
        K = ca.reshape(theta[:m * self.D_phi], self.D_phi, m).T
        return ca.tanh(K @ ca.vertcat(s[:n] - target, s[n:]) + theta[m * self.D_phi:])

    def act(self, s, target) -> np.ndarray:
        s, target = np.asarray(s, float), np.asarray(target, float)
        if self._prev is not None:
            s_prev, a_prev = self._prev
            self._Z.append(np.r_[s_prev, a_prev])
            self._dS.append(s - s_prev)
        if self._fit_due or self._since_fit >= self.cfg.refit_every:
            self._refit_and_optimise(s, target)
        a = self.policy(s, target)
        self._prev = (s.copy(), a.copy())
        self._since_fit += 1
        return a

    # -- learning -----------------------------------------------------------

    def _fit_gps(self) -> bool:
        Z, dS = np.asarray(self._Z), np.asarray(self._dS)
        if len(Z) < 10:
            return False
        if len(Z) > self.cfg.n_max_data:
            idx = self.rng.choice(len(Z), self.cfg.n_max_data, replace=False)
            Z, dS = Z[idx], dS[idx]
        x_mean, x_std = Z.mean(axis=0), np.maximum(Z.std(axis=0), 1e-3)
        params = []
        for d in range(self.D_s):
            kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                      * RBF(length_scale=[1.0] * self.D_z, length_scale_bounds=(1e-2, 1e2))
                      + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1)))
            gp = GaussianProcessRegressor(kernel=kernel, alpha=self.cfg.gp_alpha,
                                          n_restarts_optimizer=self.cfg.gp_n_restarts,
                                          normalize_y=True, random_state=self.cfg.seed)
            gp.fit((Z - x_mean) / x_std, dS[:, d])
            p = _gp_extract(gp)
            p["x_mean"], p["x_std"] = x_mean.copy(), x_std.copy()
            params.append(p)
        self._gp_params = params
        return True

    def _refit_and_optimise(self, s0, target) -> None:
        self._fit_due, self._since_fit = False, 0
        try:
            if not self._fit_gps():
                return
        except Exception:      # a failed GP fit keeps the previous model and policy
            return
        self.theta = self._optimise_policy(s0, target)
        self.n_fits += 1

    def _optimise_policy(self, s0, target) -> np.ndarray:
        cfg, n = self.cfg, self.cfg.pos_dim
        opti = ca.Opti()
        theta = opti.variable(len(self.theta))
        s, tgt = ca.DM(s0), ca.DM(target)
        cost = 0
        for _ in range(cfg.horizon):
            a = self._ca_policy(theta, s, tgt)
            z = ca.vertcat(s, a)
            s = s + ca.vertcat(*[_ca_gp_mean(z, p) for p in self._gp_params])
            err = s[:n] - tgt
            cost = (cost + cfg.cost_position * ca.dot(err, err)
                    + cfg.cost_velocity * ca.dot(s[n:], s[n:]) + cfg.action_penalty * ca.dot(a, a))
        cost = cost + cfg.theta_anchor * ca.sumsqr(theta - ca.DM(self.theta_anchor))
        opti.minimize(cost)
        lo = np.maximum(self.theta - cfg.theta_step, -cfg.theta_max)
        hi = np.minimum(self.theta + cfg.theta_step, cfg.theta_max)
        opti.subject_to(opti.bounded(lo, theta, hi))
        opti.set_initial(theta, np.clip(self.theta, lo, hi))
        opti.solver("ipopt", {"print_time": 0, "ipopt.print_level": 0, "ipopt.sb": "yes"},
                    {"max_iter": cfg.ipopt_max_iter, "tol": cfg.ipopt_tol})
        try:
            new = np.asarray(opti.solve().value(theta)).reshape(-1)
        except RuntimeError:
            new = np.asarray(opti.debug.value(theta)).reshape(-1)
        # Keep the previous policy unless the new one lowers the predicted cost.
        J = ca.Function("J", [theta], [cost])
        return new if float(J(new)) < float(J(self.theta)) - 1e-9 else self.theta.copy()
