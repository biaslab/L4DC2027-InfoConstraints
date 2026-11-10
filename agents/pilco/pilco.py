"""
PILCO baseline: Probabilistic Inference for Learning Control
(Deisenroth & Rasmussen, 2011).

Model-based RL: a GP dynamics model is fit from interaction data and a
parameterised policy is improved by gradient descent on the predicted
trajectory cost.

This implementation differs from the canonical PILCO in two ways made for
practical reasons:

  * Trajectory rollout uses GP-mean predictions only (no moment matching).
    Moment matching propagates Gaussian state distributions analytically and
    is the original paper's signature contribution; we drop it for
    implementation simplicity. The GP posterior variance does not feed back
    into the cost here, so the agent is risk-neutral.

  * Direct state access is not available -- the env exposes only a noisy
    stage-frame position and a noisy thermal sensor. We run a Luenberger
    observer for (p, v) and a scalar Kalman filter for T (using the same
    nominal plant parameters as the LQR+Kalman baseline) and PILCO sees
    that 5D estimate as state. The agent does not know alpha; it only has
    the AlphaPrior mean, identical to what LQR+Kalman uses.

Schedule (read as canonical-PILCO trials even though the surrounding
driver runs as one continuous rollout -- trial boundaries are induced
by ``policy_update_every``):

  per fast step
    y_stage, T_obs           -> observer: p_hat, v_hat, T_hat
    step < n_warmup_steps    -> random (held-for-K-steps) excitation
                                (this is "trial 1" -- pure exploration)
    otherwise                -> policy(p_hat, v_hat, T_hat, target) -> u
                                (trial k > 1 -- learned policy execution)
    push transition into dataset

  at every trial boundary (step % policy_update_every == 0, post warmup,
  or when the active waypoint changes)
    subsample dataset to <= n_max_data
    refit 5 GPs (one per output dim of state delta)
    re-optimise policy params via IPOPT on a horizon-H rollout cost,
    constrained to a trust region around the previous policy
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import casadi as ca

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ConstantKernel, WhiteKernel,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PILCOConfig:
    # Action bounds (squashed via tanh in the policy)
    u_max: float = 5.0

    # Observer for (p, v)  -- nominal plant, same defaults as LQRConfig
    mass_nominal:    float = 1.0
    damping_nominal: float = 0.5
    obs_pole:        float = 0.85

    # Thermal KF -- same defaults as KalmanThermalConfig
    kappa_nominal:     float = 0.02
    eta_nominal:       float = 1.0
    P_process_nominal: float = 1.0
    T_amb:             float = 0.0
    T_process_var:     float = 1e-3
    T_sensor_var:      float = 0.05 ** 2
    T0_var:            float = 1.0

    # Prior on alpha (same as the AlphaPrior the env exposes to all agents
    # that get one). Used only inside the cost: workpiece target is
    #   target_stage = target_wp - alpha_hat * (T_hat - T_amb)
    alpha_hat: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.03]))

    # Discretisation / schedule.
    #
    # Read these as canonical-PILCO trial lengths even though the
    # surrounding driver runs as a single continuous rollout:
    #   trial 1 (excitation):    n_warmup_steps fast steps, no policy.
    #   trial k > 1 (deployment): policy_update_every steps, with a
    #                              GP refit + policy re-optimisation
    #                              at every trial boundary and at any
    #                              waypoint switch.
    # Defaults: 20 s of excitation, then ~20 s trial cadence over our
    # 400 s budget (=> ~20 PILCO trials per run).
    dt:                  float = 0.01
    horizon:             int   = 30      # rollout depth inside the policy NLP
    n_warmup_steps:      int   = 2000    # trial 1 length (20 s)
    policy_update_every: int   = 2000    # trial boundary cadence (20 s)
    n_max_data:          int   = 500     # max GP training points per refit
    excitation_hold:     int   = 5

    # Cost: quadratic position tracking + velocity penalty + action penalty.
    # We deviate from canonical PILCO (saturating cost) because the saturating
    # cost has no gradient far from the target; with deterministic rollouts
    # and no moment matching, that leaves IPOPT with nothing to descend on
    # whenever the agent starts more than a few sigma away. The velocity
    # penalty also matters: without it, IPOPT happily picks aggressive K
    # that overshoots and oscillates on the GP-mean dynamics.
    cost_position: float = 1.0
    cost_velocity: float = 0.05
    action_penalty: float = 0.01

    # Trust region around the previous policy at every NLP solve:
    #   |theta_i - theta_prev_i| <= theta_step
    # plus an outer absolute box |theta_i| <= theta_max. Without a trust
    # region IPOPT happily picks "optimal" policies on the GP-mean model
    # that diverge in reality (no moment matching => the optimiser sees no
    # cost to leaving the data manifold). The init theta is a stabilising
    # PD baseline; theta_anchor adds a soft pull back to the original PD
    # baseline so accumulated drift cannot push theta into an unstable
    # region.
    #
    # The previous defaults (theta_step=0.07, theta_anchor=1.5) were too
    # restrictive for our 400 s thermal-coupled task: the policy could
    # only drift ~theta_step * (#trials) per parameter from the PD
    # baseline, which was nowhere near enough to compensate for the
    # workpiece offset induced by thermal expansion. The defaults below
    # let one trial-boundary update move the policy ~4x further while
    # keeping a lighter PD anchor as a stability backstop.
    theta_max:    float = 5.0
    theta_step:   float = 0.30
    theta_anchor: float = 0.3

    # GP fit
    gp_alpha:       float = 1e-4
    gp_n_restarts:  int   = 2

    # Policy NLP
    ipopt_max_iter: int   = 100
    ipopt_tol:      float = 1e-4
    ipopt_print:    int   = 0

    seed: int = 0


# ---------------------------------------------------------------------------
# Internal state estimator: Luenberger observer + scalar thermal KF
# ---------------------------------------------------------------------------

class _MechanicalObserver:
    """Luenberger observer for the per-axis second-order stage. Identical to
    the one inside LQRWithObserver, lifted out so PILCO can reuse it."""

    def __init__(self, cfg: PILCOConfig):
        m, beta, dt, rho = (cfg.mass_nominal, cfg.damping_nominal,
                            cfg.dt, cfg.obs_pole)
        # A - LC = [[1 - L_p, dt], [-L_v, 1 - dt*beta/m]]
        # trace = 2*rho, det = rho^2  ->  closed-form L
        L_p = 2.0 - 2.0 * rho - dt * beta / m
        det_target = rho * rho
        det_left = (1.0 - L_p) * (1.0 - dt * beta / m)
        L_v = (det_target - det_left) / dt
        self.L = np.array([L_p, L_v])
        self.a_decay = 1.0 - dt * beta / m
        self.a_in = dt / m
        self.dt = dt
        self.p_hat = np.zeros(2)
        self.v_hat = np.zeros(2)
        self._init = False

    def update(self, y_stage: np.ndarray, u_prev: np.ndarray):
        if not self._init:
            self.p_hat = y_stage.copy()
            self.v_hat = np.zeros(2)
            self._init = True
            return
        innov = y_stage - self.p_hat
        v_pred = self.a_decay * self.v_hat + self.a_in * u_prev
        p_pred = self.p_hat + self.dt * self.v_hat
        self.p_hat = p_pred + self.L[0] * innov
        self.v_hat = v_pred + self.L[1] * innov


class _ThermalKF:
    """Scalar Kalman filter on the thermometer reading using the nominal
    thermal model. Mirrors KalmanThermalCompensator but exposed here as an
    estimator only."""

    def __init__(self, cfg: PILCOConfig):
        self.cfg = cfg
        self.T_mean = cfg.T_amb
        self.T_var = cfg.T0_var

    def predict(self, process_flag: bool):
        c = self.cfg
        P = c.P_process_nominal if process_flag else 0.0
        a = 1.0 - c.dt * c.kappa_nominal
        b = c.dt * (c.kappa_nominal * c.T_amb + c.eta_nominal * P)
        self.T_mean = a * self.T_mean + b
        self.T_var = a * self.T_var * a + c.T_process_var

    def correct(self, T_obs: float):
        c = self.cfg
        S = self.T_var + c.T_sensor_var
        K = self.T_var / S
        self.T_mean = self.T_mean + K * (T_obs - self.T_mean)
        self.T_var = (1.0 - K) * self.T_var


# ---------------------------------------------------------------------------
# GP dynamics model
# ---------------------------------------------------------------------------

#  state s = (p_x, p_y, v_x, v_y, T)              D_s = 5
#  input z = (s, u)                                D_z = 7
#  target y_d = s_{t+1} - s_t                      D_y = 5
#
#  one independent GP per output dimension; isotropic-per-dim ARD RBF kernel.

_D_S = 5
_D_U = 2
_D_Z = _D_S + _D_U


def _make_gp(cfg: PILCOConfig) -> GaussianProcessRegressor:
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * RBF(length_scale=[1.0] * _D_Z,
                    length_scale_bounds=(1e-2, 1e2))
              + WhiteKernel(noise_level=1e-4,
                            noise_level_bounds=(1e-8, 1e-1)))
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=cfg.gp_alpha,
        n_restarts_optimizer=cfg.gp_n_restarts,
        normalize_y=True,
    )


def _gp_extract(gp: GaussianProcessRegressor):
    """Pull the bits needed for a CasADi mean prediction from a fitted GP.

    Returns a dict with keys:
        sigma_f_sq : float
        l_inv_sq   : (D_z,) ndarray, 1 / length_scale^2
        X_train    : (N, D_z) ndarray
        alpha      : (N,) ndarray, the (K + sigma_n^2 I)^{-1} y_normalised
        y_mean     : float
        y_std      : float
    """
    k = gp.kernel_
    constant = k.k1.k1.constant_value
    length = np.atleast_1d(k.k1.k2.length_scale).astype(float)
    return {
        "sigma_f_sq": float(constant),
        "l_inv_sq":   1.0 / (length ** 2),
        "X_train":    gp.X_train_.astype(float),
        "alpha":      gp.alpha_.reshape(-1).astype(float),
        "y_mean":     float(gp._y_train_mean),
        "y_std":      float(gp._y_train_std),
    }


class GPDynamics:
    """Five independent GPs. Predicts state delta given (state, action).

    Inputs are standardised to zero mean / unit std before fitting so that
    a single ARD length-scale optimisation does not blow up on dimensions
    that span very different physical scales (T ~ 0.1, u ~ 5)."""

    def __init__(self, cfg: PILCOConfig):
        self.cfg = cfg
        self.gps: list[GaussianProcessRegressor] = []
        self.params: list[dict] = []   # extracted hyperparams + data
        self.x_mean = np.zeros(_D_Z)
        self.x_std = np.ones(_D_Z)
        self._fitted = False

    def fit(self, Z: np.ndarray, dS: np.ndarray):
        """Z: (N, D_z), dS: (N, D_s). Refits all 5 GPs."""
        self.x_mean = Z.mean(axis=0)
        self.x_std = np.maximum(Z.std(axis=0), 1e-3)
        Z_n = (Z - self.x_mean) / self.x_std

        self.gps = []
        self.params = []
        for d in range(_D_S):
            gp = _make_gp(self.cfg)
            gp.fit(Z_n, dS[:, d])
            self.gps.append(gp)
            params = _gp_extract(gp)
            # carry the input standardiser through the params dict; the
            # CasADi mean function applies (z - mean)/std before evaluating
            params["x_mean"] = self.x_mean.copy()
            params["x_std"] = self.x_std.copy()
            self.params.append(params)
        self._fitted = True

    def mean_predict(self, z: np.ndarray) -> np.ndarray:
        """Numpy single-point prediction; only used for diagnostics."""
        z_n = (z - self.x_mean) / self.x_std
        out = np.zeros(_D_S)
        for d in range(_D_S):
            mu = self.gps[d].predict(z_n.reshape(1, -1))
            out[d] = float(np.asarray(mu).reshape(-1)[0])
        return out


# ---------------------------------------------------------------------------
# Policy (linear feedback, squashed via tanh)
# ---------------------------------------------------------------------------

#  inputs to policy: phi = [p_x, p_y, v_x, v_y, T, target_x, target_y]   D_phi = 7
#  output: u = u_max * tanh(K phi + b)                                    D_u = 2

_D_PHI = _D_S + 2     # state + 2D target
_D_THETA = _D_U * _D_PHI + _D_U   # 16


def _theta_pack(K: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate([K.reshape(-1), b.reshape(-1)])


def _theta_unpack_np(theta: np.ndarray):
    K = theta[: _D_U * _D_PHI].reshape(_D_U, _D_PHI)
    b = theta[_D_U * _D_PHI:]
    return K, b


def _theta_unpack_ca(theta):
    """CasADi counterpart of _theta_unpack_np. CasADi reshapes column-major,
    so the gains are reshaped transposed and transposed back. (The version
    copied from the TAC 2026 project reshaped directly, so the NLP optimised
    a policy with permuted gains, not the one the controller applies.)"""
    K = ca.reshape(theta[: _D_U * _D_PHI], _D_PHI, _D_U).T
    return K, theta[_D_U * _D_PHI:]


def _policy_apply_np(theta: np.ndarray, phi: np.ndarray,
                     u_max: float) -> np.ndarray:
    K, b = _theta_unpack_np(theta)
    return u_max * np.tanh(K @ phi + b)


# ---------------------------------------------------------------------------
# CasADi rollout + IPOPT optimisation
# ---------------------------------------------------------------------------

def _ca_gp_mean(z, params: dict):
    """Symbolic GP mean at z (CasADi (D_z, 1)) given fitted params dict.

    Applies the input standardiser ((z - x_mean) / x_std) before evaluating
    the kernel. X_train inside `params` is already in standardised space.
    """
    X = ca.DM(params["X_train"])              # (N, D_z)
    alpha = ca.DM(params["alpha"])            # (N,)
    l_inv_sq = ca.DM(params["l_inv_sq"])      # (D_z,)
    x_mean = ca.DM(params["x_mean"])          # (D_z,)
    x_std = ca.DM(params["x_std"])            # (D_z,)
    sigma_f_sq = params["sigma_f_sq"]
    y_mean = params["y_mean"]
    y_std = params["y_std"]

    z_n = (z - x_mean) / x_std
    diff = X - ca.repmat(z_n.T, X.shape[0], 1)
    sq = ca.mtimes(diff ** 2, l_inv_sq)        # (N, 1)
    k = sigma_f_sq * ca.exp(-0.5 * sq)
    mu_norm = ca.dot(k, alpha)
    return mu_norm * y_std + y_mean


def _build_and_solve_policy_nlp(
    cfg: PILCOConfig,
    gp_params: list[dict],
    theta_init: np.ndarray,
    theta_anchor_vec: np.ndarray,
    s0: np.ndarray,
    target_wp: np.ndarray,
    T_amb: float,
) -> np.ndarray:
    """Set up and solve the H-step PILCO NLP. Returns optimised theta."""

    H = cfg.horizon
    opti = ca.Opti()
    theta = opti.variable(_D_THETA)
    K_var, b_var = _theta_unpack_ca(theta)

    s = ca.DM(s0)                              # (5, 1)
    target = ca.DM(target_wp)                  # (2, 1)

    cost = 0
    for _ in range(H):
        phi = ca.vertcat(s, target)
        u = cfg.u_max * ca.tanh(K_var @ phi + b_var)
        z = ca.vertcat(s, u)

        # GP-mean state delta
        ds = ca.vertcat(*[_ca_gp_mean(z, gp_params[d]) for d in range(_D_S)])
        s_next = s + ds

        # Workpiece-frame stage target depends on T (state[4])
        T_hat = s_next[4]
        stage_target = target - ca.DM(cfg.alpha_hat) * (T_hat - T_amb)
        err = s_next[0:2] - stage_target
        v_next = s_next[2:4]
        cost = cost + cfg.cost_position * ca.dot(err, err)
        cost = cost + cfg.cost_velocity * ca.dot(v_next, v_next)
        cost = cost + cfg.action_penalty * ca.dot(u, u)

        s = s_next

    delta_anchor = theta - ca.DM(theta_anchor_vec)
    cost = cost + cfg.theta_anchor * ca.dot(delta_anchor, delta_anchor)

    opti.minimize(cost)
    opti.subject_to(opti.bounded(-cfg.theta_max, theta, cfg.theta_max))
    lo = np.maximum(theta_init - cfg.theta_step, -cfg.theta_max)
    hi = np.minimum(theta_init + cfg.theta_step,  cfg.theta_max)
    opti.subject_to(opti.bounded(lo, theta, hi))
    opti.set_initial(theta, np.clip(theta_init, lo, hi))
    opti.solver(
        "ipopt",
        {"print_time": cfg.ipopt_print, "ipopt.print_level": cfg.ipopt_print,
         "ipopt.sb": "yes"},
        {"max_iter": cfg.ipopt_max_iter, "tol": cfg.ipopt_tol},
    )
    try:
        sol = opti.solve()
        new_theta = np.asarray(sol.value(theta)).reshape(-1)
    except RuntimeError:
        # IPOPT failure: keep previous theta
        return theta_init.copy()

    # Reject the update if it does not actually improve the predicted cost
    # at theta_init -- IPOPT can return a (boundary) local minimum that is
    # worse than the warm-start when the GP model is highly non-convex.
    cost_fn = ca.Function("J", [theta], [cost])
    j_init = float(cost_fn(theta_init))
    j_new = float(cost_fn(new_theta))
    if j_new > j_init - 1e-6:
        return theta_init.copy()
    return new_theta


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class PILCOController:
    """Drop-in replacement for the env's controller protocol.

    Driver is expected to call:
        u = controller(y_stage, target_workpiece)
        controller.observe_thermal(T_obs)
        controller.notify_process_flag(process_flag)   # optional but recommended

    At each call:
        * advance observers using the previous u and current obs
        * either emit excitation (warmup) or policy(state, target)
        * if it's a refit step, retrain GPs and reoptimise theta

    Internal data buffer: every step appends (z=(s, u), ds=s_next - s) to a
    growing list. We don't have access to s_{t+1} until the next call (one
    step later), so the buffer push is deferred by one tick.
    """

    def __init__(self, cfg: PILCOConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.obs = _MechanicalObserver(cfg)
        self.kf = _ThermalKF(cfg)
        self.dyn = GPDynamics(cfg)

        # Linear policy parameters: hand-rolled stabilising PD on (p_err, v),
        # used both as theta_init for IPOPT and as the warmup-time controller
        # underneath the random excitation. Stored separately as the
        # anchor for the soft regulariser so accumulated drift cannot
        # push the policy into an unstable region.
        #   phi = [p_x, p_y, v_x, v_y, T, target_x, target_y]
        K0 = np.zeros((_D_U, _D_PHI))
        K0[0, 0] = -3.0             # pull p_x toward target_x (stiff PD)
        K0[1, 1] = -3.0
        K0[0, 2] = -1.0             # damp v_x
        K0[1, 3] = -1.0
        K0[0, _D_S]     = 3.0       # weight on target_x
        K0[1, _D_S + 1] = 3.0
        b0 = np.zeros(_D_U)
        self.theta_anchor = _theta_pack(K0, b0)
        self.theta = self.theta_anchor.copy()

        # Buffers: store only completed (z, ds) pairs.
        self._Z_buf: list[np.ndarray] = []
        self._dS_buf: list[np.ndarray] = []

        # Previous-step state used to construct the (z, ds) pair on this step
        self._prev_s: Optional[np.ndarray] = None
        self._prev_u: Optional[np.ndarray] = None

        # Excitation state
        self._exc_target: Optional[np.ndarray] = None
        self._exc_hold = 0

        # Bookkeeping
        self._step = 0
        self._last_process = False
        self._last_target: Optional[np.ndarray] = None
        self._last_T_obs: float = cfg.T_amb

    # ----- env-side hooks ---------------------------------------------------

    def observe_thermal(self, T_obs: float):
        self._last_T_obs = float(T_obs)

    def notify_process_flag(self, process: bool):
        self._last_process = bool(process)

    # ----- internal helpers ------------------------------------------------

    def _current_state(self) -> np.ndarray:
        return np.array(
            [self.obs.p_hat[0], self.obs.p_hat[1],
             self.obs.v_hat[0], self.obs.v_hat[1],
             self.kf.T_mean],
            dtype=float,
        )

    def _emit_excitation(self, target: np.ndarray) -> np.ndarray:
        """Stabilising PD with held random perturbations on top.

        Pure random forcing makes the system run away from the workspace
        before the GP has any data near the target neighbourhood. Mixing in
        the hand-tuned PD keeps the data localised; the perturbation
        excites the dynamics for system identification.
        """
        if self._exc_target is None or self._exc_hold == 0:
            self._exc_target = self.rng.uniform(
                -0.6 * self.cfg.u_max, 0.6 * self.cfg.u_max, size=2)
            self._exc_hold = self.cfg.excitation_hold
        self._exc_hold -= 1
        u_pd = self._emit_policy(target)
        return u_pd + self._exc_target

    def _emit_policy(self, target: np.ndarray) -> np.ndarray:
        s = self._current_state()
        phi = np.concatenate([s, target])
        return _policy_apply_np(self.theta, phi, self.cfg.u_max)

    def _push_dataset(self, s_now: np.ndarray):
        if self._prev_s is None or self._prev_u is None:
            return
        z = np.concatenate([self._prev_s, self._prev_u])
        ds = s_now - self._prev_s
        self._Z_buf.append(z)
        self._dS_buf.append(ds)

    def _maybe_subsample(self):
        N = len(self._Z_buf)
        if N <= self.cfg.n_max_data:
            return np.asarray(self._Z_buf), np.asarray(self._dS_buf)
        idx = self.rng.choice(N, size=self.cfg.n_max_data, replace=False)
        Z = np.asarray(self._Z_buf)[idx]
        dS = np.asarray(self._dS_buf)[idx]
        return Z, dS

    def _refit_and_reoptimise(self, target: np.ndarray):
        Z, dS = self._maybe_subsample()
        if Z.shape[0] < 10:
            return
        try:
            self.dyn.fit(Z, dS)
        except Exception:
            return
        s0 = self._current_state()
        self.theta = _build_and_solve_policy_nlp(
            self.cfg, self.dyn.params, self.theta, self.theta_anchor,
            s0=s0, target_wp=target, T_amb=self.cfg.T_amb,
        )

    # ----- main entrypoint -------------------------------------------------

    def __call__(self, y_stage: np.ndarray, target_workpiece: np.ndarray,
                 T_estimate: float = 0.0, T_amb: float = 0.0) -> np.ndarray:
        # Run estimators (uses the previous applied u via self._prev_u)
        u_prev = self._prev_u if self._prev_u is not None else np.zeros(_D_U)
        self.obs.update(y_stage, u_prev)
        self.kf.predict(self._last_process)
        self.kf.correct(self._last_T_obs)

        s_now = self._current_state()
        self._push_dataset(s_now)

        target_changed = (
            self._last_target is None
            or np.any(self._last_target != target_workpiece)
        )
        do_refit = (
            self._step >= self.cfg.n_warmup_steps
            and (self._step % self.cfg.policy_update_every == 0
                 or (target_changed and self.dyn._fitted))
        )
        if do_refit:
            self._refit_and_reoptimise(target_workpiece)

        if self._step < self.cfg.n_warmup_steps or not self.dyn._fitted:
            u = self._emit_excitation(target_workpiece)
        else:
            u = self._emit_policy(target_workpiece)
        u = np.clip(u, -self.cfg.u_max, self.cfg.u_max)

        self._prev_s = s_now
        self._prev_u = u.copy()
        self._last_target = target_workpiece.copy()
        self._step += 1
        return u
