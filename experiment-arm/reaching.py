"""Reaching protocol for the agents on the two-link arm.

The arm starts at rest in the configuration q = (0, π/2), with the end
effector at (0.5, 0.5), and must reach a sequence of end-effector targets.
Each target stays active for a fixed time, and the agent acts at every
control step. ``run_episode`` drives any controller with the interface of
``controllers.py`` and ``summarise`` scores the episode. ``run_agent``
builds the agent from a specification (CBFE-ARX, dual MPC, PILCO, TD-MPC2
or a fixed PD law; see ``make_controller``), and ``run_many`` runs many
episodes in parallel processes. ``Settings`` holds the defaults;
``FIRST_STUDY`` and ``PAYLOAD_STUDY`` pin the settings of the earlier
notebooks.

Observation. The ARX model regresses the next output on the current input
and the previous output. The agent observes the joint angles from encoders
with a 0.001 rad noise level, and forms the output y = (q, v) with the
finite-difference velocity v over one control period. Two choices follow
from the ARX model class:

- Joint space, not end-effector space. A torque moves the end effector in
  a direction that rotates with the shoulder angle, which no linear ARX
  model over Cartesian outputs can represent across the workspace. In joint
  space, the gain from torque to acceleration, M⁻¹(q₂), keeps its sign
  structure. The end-effector targets therefore become joint-space goals
  through inverse kinematics (elbow-up branch).
- Encoders, not a camera. With 0.01 rad of noise, the finite-difference
  velocity is as noisy as the motion, and the ARX regression attenuates
  the input gain it learns from data by up to a factor of three (errors in
  variables). With the informative prior, the agent still reaches the
  targets at this noise level, but holds them at about 2 cm instead of
  1.2 cm.

The agent acts every ``substeps`` plant steps under a zero-order hold, so
the input has leverage on the next position.

Prior. The agent's prior comes from a nominal simulator, i.e., the arm with
θ at its prior mean (``ArmModel.params``): closed-loop excitation data from
the nominal arm, counted with a weight below one (``tempered_posterior``),
give an informative matrix-normal Wishart prior with the confidence of
``prior_weight * prior_steps`` observations. The prior covariance of the
ARX coefficients scales with 1 / prior_weight.

Payload change. ``payload_change`` repeats the target sequence and changes
the plant's payload at the start of the second round, so that every target
is scored once before and once after the change (``summarise`` with
``segments``). ``run_episode`` applies any such schedule of parameter
changes, indexed by control step.

Unknown polarity. With ``Settings.polarity_known = False``, each motor of
the plant may be reversed (``POLARITIES``, cycled over the seeds), which is
the unknown control direction problem of adaptive control. The ARX agents
then keep the nominal prior of the passive dynamics but forget the
coefficients that change sign with a motor (``forget_polarity``), so that
their prior is symmetric in both polarities: without probing, every input
only adds predicted spread, and a cautious agent never moves.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from agents.cbfe_arx import Agent
from agents.common import (ActionScaler, MultivariateNormal, predictive_noise_covariance,
                           tempered_posterior)
from controllers import ARXController, PDController, PILCOController, TDMPC2Controller
from envs.robot_arm import (
    ArmConfig, ArmModel, ArmParams, RobotArm, forward_kinematics, inverse_kinematics,
)

TARGETS = np.array([[0.7, 0.2], [0.2, 0.75], [-0.45, 0.45], [0.35, -0.35]])
REACH_TOL = 0.02           # [m] a target counts as reached within 2 cm
HOLD_TIME = 1.0            # [s] the last second of each segment measures holding

# Joint-angle outputs from encoders. The torque limit and the RK4 substeps
# of the plant come from the Settings (``arm_config``).
ARM_CONFIG = ArmConfig(output="joints", sigma_y=1e-3)


@dataclass
class Settings:
    """Everything that defines an ARX agent on the arm and its actuators,
    except β."""
    substeps: int = 2                  # plant steps per control step (0.1 s)
    seconds_per_target: float = 3.0
    # Actuators: the torque limit was doubled relative to the first study
    # (2 N m) so that the arm stays reachable with a 1.5 kg payload; four
    # RK4 substeps keep the plant accurate at the higher speeds.
    u_max: float = 4.0                 # torque limit per joint [N m]
    rk4_substeps: int = 4              # RK4 steps per plant step (0.05 s)
    # Goal prior: the variances were halved relative to the first study
    # (0.02 rad, 0.2 rad/s), so the standard deviations carry a factor 1/√2.
    pos_std: float = 0.02 / np.sqrt(2)   # goal prior std of the joint angles [rad]
    vel_std: float = 0.2 / np.sqrt(2)    # goal prior std of the joint velocities [rad/s]
    horizon: int = 5                   # planning horizon [control steps]
    # CBFE-ARX planner: "messages" passes backward messages through the
    # linearised model until convergence (at most max_sweeps sweeps);
    # "sweeps", the planner of the earlier studies, solves one-step
    # programs towards intermediate goals (n_iters sweeps).
    planner: str = "messages"
    n_iters: int = 2                   # forward-backward sweeps of the "sweeps" planner
    max_sweeps: int = 100              # cap on the sweeps of the "messages" planner
    # Floor of the information constraint: "task" compares the
    # parameter-induced spread of the predictions with the goal prior;
    # "negentropy" is -H[p_*], the floor of the earlier studies.
    info_floor: str = "task"
    # Nodes of the information constraint: "all" shares one multiplier, from
    # the one-step program, among all nodes; "next" constrains the node of
    # the next input and solves the multiplier on the plan.
    constraint_nodes: str = "all"
    N_basis: int = 1                   # harmonic orders per input coordinate
    # False omits the constant and linear features, so that the ARX model has
    # the Fourier terms only (main.tex since 2026-11-08)
    polynomial_features: bool = True
    L_basis: tuple = (1.0, 1.0, np.pi, np.pi, 6.0, 6.0)   # half-periods for (u, q, v)
    Lambda0_scale: float = 0.1
    Omega0_scale: float = 0.1
    beta_prior_a: float = 2.0          # symmetric Beta control prior
    beta_max: float = 50.0
    prior_steps: int = 600             # nominal excitation data for the prior
    # Weight of each nominal data point. The prior covariance of the ARX
    # coefficients scales with 1 / prior_weight; 0.15 doubles it relative
    # to the first study (0.3).
    prior_weight: float = 0.15
    prior_seed: int = 0
    # Unknown control direction: with polarity_known = False, the motor
    # polarities of the plant follow the seed (``POLARITIES``), and the ARX
    # agents replace their prior over the coefficients that change sign
    # with a motor by a zero-mean prior (``forget_polarity``).
    polarity_known: bool = True

    @property
    def dt(self) -> float:
        return self.substeps * ARM_CONFIG.dt

    @property
    def steps_per_target(self) -> int:
        return int(round(self.seconds_per_target / self.dt))


# The settings of the first study (cbfe_arx_arm.ipynb) and of the payload
# study (cbfe_arx_payload.ipynb), which doubled the horizon as well. Both
# used the one-step sweeps of the CBFE-ARX planner and the floor -H[p_*].
FIRST_STUDY = Settings(pos_std=0.02, vel_std=0.2, prior_weight=0.3, u_max=2.0, horizon=5,
                       rk4_substeps=1, planner="sweeps", info_floor="negentropy")
PAYLOAD_STUDY = Settings(horizon=10, planner="sweeps", info_floor="negentropy")


def arm_config(s: Settings) -> ArmConfig:
    """Plant configuration of the settings: ARM_CONFIG with their torque
    limit and RK4 substeps."""
    return replace(ARM_CONFIG, u_max=s.u_max, substeps=s.rk4_substeps)


# Motor polarities of the unknown-polarity condition, cycled over the seeds
# so that every pair occurs equally often.
POLARITIES = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))


def plant_config(s: Settings, seed: int) -> ArmConfig:
    """Configuration of the plant of an episode: ``arm_config``, with the
    motor polarities POLARITIES[seed % 4] if the polarity is unknown."""
    config = arm_config(s)
    return config if s.polarity_known else replace(config, polarity=POLARITIES[seed % 4])


def polarity_features(estimator) -> np.ndarray:
    """Indices of the features that change sign with an input, i.e. whose
    coefficients flip with the polarity of that input's motor."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.9, 0.9, estimator.regressor_dim)
    phi = estimator.feature_map(x)
    odd = set()
    for j in range(estimator.input_dim):
        x_flip = x.copy()
        x_flip[j] = -x_flip[j]
        odd |= set(np.flatnonzero((np.abs(phi + estimator.feature_map(x_flip)) < 1e-12)
                                  & (np.abs(phi) > 1e-12)))
    return np.array(sorted(odd))


def forget_polarity(estimator, width: float = 2.0) -> None:
    """Replace the prior over the coefficients of ``polarity_features`` by
    an independent zero-mean prior, symmetric in every motor's polarity,
    whose standard deviation is at least ``width`` times the largest of
    those coefficients. The other coefficients keep their marginal prior
    (the Schur complement of their precision)."""
    post = estimator.posterior
    odd = polarity_features(estimator)
    rest = np.setdiff1d(np.arange(post.feature_dim), odd)
    Lam, M = post.coefficient_precision, post.coefficient_mean.copy()
    L_rr = Lam[np.ix_(rest, rest)] - Lam[np.ix_(rest, odd)] @ np.linalg.solve(
        Lam[np.ix_(odd, odd)], Lam[np.ix_(odd, rest)])
    noise = float(np.min(np.diag(predictive_noise_covariance(post))))
    precision = noise / (width * np.abs(M[odd]).max()) ** 2
    new = np.zeros_like(Lam)
    new[np.ix_(rest, rest)] = 0.5 * (L_rr + L_rr.T)
    new[np.ix_(odd, odd)] = precision * np.eye(len(odd))
    M[odd] = 0.0
    estimator.posterior = type(post)(coefficient_mean=M, coefficient_precision=new,
                                     noise_scale=post.noise_scale,
                                     degrees_of_freedom=post.degrees_of_freedom)


def joint_targets(targets=TARGETS, params: ArmParams = ArmParams()) -> np.ndarray:
    """Elbow-up joint angles of the end-effector targets."""
    return inverse_kinematics(np.asarray(targets, float), params, sign=1)


def joint_goal(q_target, s: Settings) -> MultivariateNormal:
    """Goal prior over y = (q, v): at the target configuration, at rest."""
    var = np.r_[np.full(2, s.pos_std**2), np.full(2, s.vel_std**2)]
    return MultivariateNormal(mean=np.r_[q_target, 0.0, 0.0], covariance=np.diag(var))


def nominal_data(s: Settings, params: ArmParams, config: Optional[ArmConfig] = None):
    """Closed-loop excitation of the nominal arm: a PD controller to random
    joint waypoints (a new one every second) plus uniform noise. Returns
    the ARX inputs (u, q, v) and outputs (q, v) in agent units."""
    config = arm_config(s) if config is None else config
    plant = RobotArm(params, config, seed=s.prior_seed)
    rng = np.random.default_rng(s.prior_seed)
    p = plant.observe()
    y = np.r_[p, 0.0, 0.0]
    X, Y, q_ref = [], [], plant.x[:2].copy()
    for k in range(s.prior_steps):
        if k % 10 == 0:
            q_ref = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(0.3, 2.8)])
        a = np.clip(1.5 * (q_ref - y[:2]) - 0.8 * y[2:] + rng.uniform(-0.5, 0.5, 2), -1, 1)
        for _ in range(s.substeps):
            p_next = plant.step(config.u_max * a)
        y_next = np.r_[p_next, (p_next - p) / s.dt]
        X.append(np.r_[a, y])
        Y.append(y_next)
        p, y = p_next, y_next
    return np.array(X), np.array(Y)


def payload_change(new_payload: float, s: Settings, targets=TARGETS, rounds: int = 2):
    """Targets for ``rounds`` passes through ``targets``, and the schedule
    that sets the payload to ``new_payload`` at the start of the second
    pass. Returns (targets, changes) for ``run_episode``."""
    targets = np.asarray(targets, float)
    change_step = len(targets) * s.steps_per_target
    return np.tile(targets, (rounds, 1)), {change_step: {"m_p": float(new_payload)}}


def build_agent(s: Settings, fixed_beta: Optional[float], nominal: ArmParams,
                goal: MultivariateNormal, config: Optional[ArmConfig] = None) -> Agent:
    """CBFE-ARX agent with the informative prior from the nominal arm.
    fixed_beta = 0 removes the information constraint, None solves β by
    KKT, and any other value fixes it."""
    agent = Agent.build(input_dim=2, output_dim=4, input_memory=0, output_memory=1,
                        goal_prior=goal, horizon=s.horizon, num_harmonics=s.N_basis,
                        half_period=np.asarray(s.L_basis),
                        prior_precision_scale=s.Lambda0_scale, prior_noise_scale=s.Omega0_scale,
                        control_prior_shape=s.beta_prior_a,
                        forward_backward_iterations=s.n_iters,
                        fixed_multiplier=fixed_beta, max_multiplier=s.beta_max,
                        planner_type=s.planner, max_sweeps=s.max_sweeps,
                        information_floor_type=s.info_floor,
                        constraint_nodes=s.constraint_nodes,
                        polynomial_features=s.polynomial_features)
    set_nominal_prior(agent.estimator, s, nominal, config)
    return agent


def set_nominal_prior(estimator, s: Settings, nominal: ArmParams,
                      config: Optional[ArmConfig] = None) -> None:
    """Replace the estimator's posterior by the tempered posterior after the
    excitation data of the nominal arm (no change for prior_weight = 0),
    and forget the motor polarities if they are unknown."""
    if s.prior_weight > 0:
        X, Y = nominal_data(s, nominal, config)
        Phi = np.array([estimator.feature_map(x) for x in X])
        estimator.posterior = tempered_posterior(estimator.posterior, Phi, Y, s.prior_weight)
    if not s.polarity_known:
        forget_polarity(estimator)


def build_dual_mpc(s: Settings, lambda_dual: float, nominal: ArmParams, goal: MultivariateNormal,
                   R_scalar: float = 2.0, config: Optional[ArmConfig] = None):
    """Dual MPC on the same ARX model, feature map and prior as the CBFE-ARX
    agent. With the default tracking weight Q = S_*⁻¹, the effort weight
    R = 2 and the probing weight λ = 2β make its cost twice the negative log
    policy of the CBFE-ARX agent with multiplier β, up to the shape of the
    control prior (quadratic here, symmetric Beta there)."""
    from agents.dual_mpc import DualMPC
    agent = DualMPC.build(input_dim=2, output_dim=4, input_memory=0, output_memory=1,
                          goal_prior=goal, horizon=s.horizon, probing_weight=lambda_dual,
                          effort_weight=R_scalar, num_harmonics=s.N_basis,
                          half_period=np.asarray(s.L_basis),
                          prior_precision_scale=s.Lambda0_scale,
                          prior_noise_scale=s.Omega0_scale,
                          polynomial_features=s.polynomial_features)
    set_nominal_prior(agent.estimator, s, nominal, config)
    return agent


def build_mpcx(s: Settings, gamma: float, nominal: ArmParams, goal: MultivariateNormal,
               R_scalar: float = 2.0, config: Optional[ArmConfig] = None, **overrides):
    """MPC with integrated experiment design (Larsson et al., ECC 2013) on
    the same ARX model, feature map and prior as the CBFE-ARX agent, with
    the cost of the dual MPC without caution and probing. ``gamma`` is the
    accuracy of the application set (degradation at most 1/gamma); the
    ``confidence`` and the ``experiment_length`` of ``MPCX.build`` can be
    overridden."""
    from agents.mpcx import MPCX
    agent = MPCX.build(input_dim=2, output_dim=4, input_memory=0, output_memory=1,
                       goal_prior=goal, horizon=s.horizon, accuracy=gamma,
                       effort_weight=R_scalar, num_harmonics=s.N_basis,
                       half_period=np.asarray(s.L_basis),
                       prior_precision_scale=s.Lambda0_scale, prior_noise_scale=s.Omega0_scale,
                       input_prior_shape=s.beta_prior_a,
                       polynomial_features=s.polynomial_features, **overrides)
    set_nominal_prior(agent.estimator, s, nominal, config)
    return agent


def build_pilco(s: Settings, nominal: ArmParams, config: Optional[ArmConfig] = None, **overrides):
    """PILCO whose initial data are a random subset of the nominal arm's
    excitation data, as many transitions as the ARX prior counts
    (prior_weight * prior_steps)."""
    from agents.pilco import StatePILCO, StatePILCOConfig
    agent = StatePILCO(StatePILCOConfig(**overrides))
    X, Y = nominal_data(s, nominal, config)          # X = (a, q, v), Y = (q', v')
    n = int(round(s.prior_weight * s.prior_steps))
    idx = np.random.default_rng(s.prior_seed).choice(len(X), n, replace=False)
    state, action = X[idx, 2:], X[idx, :2]
    agent.add_data(np.c_[state, action], Y[idx] - state)
    return agent


def load_tdmpc2(checkpoint, threads: int = 1):
    """TD-MPC2 agent from a checkpoint of ``train_tdmpc2.py``."""
    import torch
    torch.set_num_threads(threads)
    from agents.tdmpc2 import TDMPC2
    return TDMPC2.load(checkpoint, device="cpu")


def make_controller(spec: dict, s: Settings, nominal: ArmParams, q_goal0):
    """Controller for an agent specification:

    {"kind": "cbfe", "fixed_beta": 0.0 or None, and optionally "info_floor",
     "constraint_nodes" and "beta_max", which override the settings}
    {"kind": "dual", "lambda_dual": λ, "R_scalar": 2.0, "input_cost": "quadratic" or "beta",
     "rest_tolerance": None or tol}
    {"kind": "mpcx", "gamma": γ, **MPCX.build overrides}
    {"kind": "pilco", **StatePILCOConfig overrides}
    {"kind": "tdmpc2", "checkpoint": path, "learn_online": True}
    {"kind": "pd", "kp": 1.5, "kd": 0.8}
    """
    spec = dict(spec)
    kind = spec.pop("kind")
    goal_fn = lambda q: joint_goal(q, s)          # noqa: E731
    if kind == "cbfe":
        overrides = {key: spec[key] for key in ("info_floor", "constraint_nodes", "beta_max")
                     if key in spec}
        agent = build_agent(replace(s, **overrides), spec.get("fixed_beta"), nominal,
                            goal_fn(q_goal0))
        return ARXController(agent, goal_fn, "cbfe")
    if kind == "dual":
        agent = build_dual_mpc(s, spec["lambda_dual"], nominal, goal_fn(q_goal0),
                               spec.get("R_scalar", 2.0))
        agent.config.input_cost = spec.get("input_cost", "quadratic")
        agent.config.input_prior_shape = s.beta_prior_a
        agent.config.rest_tolerance = spec.get("rest_tolerance")
        return ARXController(agent, goal_fn, "dual")
    if kind == "mpcx":
        agent = build_mpcx(s, spec.pop("gamma"), nominal, goal_fn(q_goal0), **spec)
        return ARXController(agent, goal_fn, "mpcx")
    if kind == "pilco":
        return PILCOController(build_pilco(s, nominal, **spec))
    if kind == "pd":
        return PDController(**spec)
    if kind == "tdmpc2":
        return TDMPC2Controller(load_tdmpc2(spec["checkpoint"], spec.get("threads", 1)),
                                learn_online=spec.get("learn_online", True))
    raise ValueError(f"unknown agent kind '{kind}'")


@dataclass
class Episode:
    params: ArmParams
    dt: float
    u_max: float
    xs: np.ndarray            # (N+1, 4) true states at the control steps
    ys: np.ndarray            # (N, 4) agent outputs (q, v) after each step
    actions: np.ndarray       # (N, 2) agent actions in [-1, 1]
    targets: np.ndarray       # (N, 2) end-effector target of each step
    beta: np.ndarray          # (N,) information multiplier of each plan
    info: np.ndarray          # (N,) I(Θ; y | u) at the chosen action [nats]
    floor: np.ndarray         # (N,) floor of the planner's information constraint [nats];
                              # nan without one. The task floor is compared with
                              # (D_y/2) ln(1 + leverage), the floor -H[p_*] with info
    leverage: np.ndarray      # (N,)
    surprise: np.ndarray      # (N,) -ln p(y_{k+1} | u_k, D_k) under the predictive
    pred_mean: np.ndarray     # (N, 4) predictive mean of y_{k+1} given u_k
    plan_time: np.ndarray     # (N,) [s]
    payload: np.ndarray       # (N,) true payload mass during step k [kg]

    @property
    def torques(self):
        return self.u_max * self.actions

    @property
    def ee(self):
        """True end-effector positions at the control steps, (N+1, 2)."""
        return forward_kinematics(self.xs[:, :2], self.params)[1]

    @property
    def distance(self):
        """Distance of the end effector after step k to the target of step k."""
        return np.linalg.norm(self.ee[1:] - self.targets, axis=1)

    @property
    def pred_error(self):
        """One-step prediction error of the joint velocities, (N, 2) [rad/s]."""
        return self.ys[:, 2:] - self.pred_mean[:, 2:]

    @property
    def velocity_change(self):
        """Realised and predicted change of the joint velocities in each
        step, two arrays of shape (N, 2) [rad/s]. The episode starts at rest."""
        v_prev = np.vstack([np.zeros((1, 2)), self.ys[:-1, 2:]])
        return self.ys[:, 2:] - v_prev, self.pred_mean[:, 2:] - v_prev


def run_episode(controller, plant: RobotArm, s: Settings, targets=TARGETS, changes=None,
                warmup_steps: int = 0, warmup_amp: float = 0.3, seed: int = 0) -> Episode:
    """Run the target sequence in closed loop. The plant is reset, the
    controller keeps whatever it has learned.

    controller is any object with the interface of ``controllers.py``; a
    CBFE-ARX agent or dual MPC passed directly is wrapped automatically.
    changes maps a control step to the plant parameters that change at that
    step, e.g. {120: {"m_p": 1.5}}; the plant gets its original parameters
    back at the end. ``warmup_steps`` random actions of amplitude
    ``warmup_amp`` precede the controller (the system-ID soft start of the
    thermal-stage experiments; zero by default)."""
    import time
    if hasattr(controller, "estimator") and hasattr(controller, "plan"):
        controller = ARXController(controller, lambda q: joint_goal(q, s))
    params0 = plant.params
    changes = {} if changes is None else dict(changes)
    plant.reset()
    rng = np.random.default_rng(10_000 + seed)
    scaler = ActionScaler.symmetric(plant.config.u_max, 2)
    q_goals = joint_targets(targets, plant.params)
    p = plant.observe()
    y = np.r_[p, 0.0, 0.0]
    controller.start(y)
    rec = {k: [] for k in ("ys", "actions", "targets", "beta", "info", "floor", "leverage",
                           "surprise", "pred_mean", "plan_time", "payload")}
    xs, k = [plant.x.copy()], 0
    for target, q_goal in zip(np.asarray(targets, float), q_goals):
        controller.set_target(q_goal)
        for _ in range(s.steps_per_target):
            if k in changes:
                plant.params = replace(plant.params, **changes[k])
            t0 = time.perf_counter()
            if k < warmup_steps:
                a = rng.uniform(-warmup_amp, warmup_amp, 2)
            else:
                a = controller.act(y)
            plan_time = time.perf_counter() - t0
            diag = controller.diagnostics(a)
            pred = diag.get("predictive")
            for _ in range(s.substeps):
                p_next = plant.step(scaler.to_physical(a))
            y_next = np.r_[p_next, (p_next - p) / s.dt]
            surprise = -pred.log_pdf(y_next) if pred is not None else np.nan
            controller.learn(y_next, a)
            beta = np.nan if k < warmup_steps else diag.get("beta", np.nan)
            for key, val in (("ys", y_next), ("actions", a), ("targets", target), ("beta", beta),
                             ("info", diag.get("info", np.nan)), ("floor", diag.get("floor", np.nan)),
                             ("leverage", diag.get("leverage", np.nan)), ("surprise", surprise),
                             ("pred_mean", pred.mean if pred is not None else np.full(4, np.nan)),
                             ("plan_time", plan_time), ("payload", plant.params.m_p)):
                rec[key].append(val)
            xs.append(plant.x.copy())
            p, y, k = p_next, y_next, k + 1
    plant.params = params0
    return Episode(params0, s.dt, plant.config.u_max, np.array(xs),
                   **{key: np.array(val) for key, val in rec.items()})


def summarise(ep: Episode, s: Settings, segments=slice(None)) -> dict:
    """Scores of one episode, or of the target segments selected by
    ``segments`` (a slice or index array over the segments).

    reach_time   mean over targets of the first time within REACH_TOL [s]
                 (the segment length if a target is never reached)
    reached      fraction of targets reached
    hold_error   mean end-effector distance over the last HOLD_TIME of each
                 segment [cm]
    rms_torque   root mean square of the applied torques [N m]
    beta_mean    mean information multiplier
    active       fraction of steps with β > 0
    info_gain    mean information about Θ per step at the chosen action [nats]
    surprise     mean surprise of the realised outputs [nats]
    pred_rmse    RMS one-step prediction error of the joint velocities [rad/s]
    response_ratio  least-squares slope of the realised on the predicted
                 velocity change: 1 for a calibrated model, below 1 when the
                 model expects more response than the arm delivers
    """
    n = s.steps_per_target
    seg = np.arange(len(ep.distance) // n)[segments]
    steps = (seg[:, None] * n + np.arange(n)).ravel()
    d = ep.distance.reshape(-1, n)[seg]
    hit = d < REACH_TOL
    first = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, n)
    hold = int(round(HOLD_TIME / ep.dt))
    beta = ep.beta[steps]
    beta = beta[np.isfinite(beta)]          # empty for agents without a multiplier
    return {"reach_time": float(np.mean(first) * ep.dt),
            "reached": float(hit.any(axis=1).mean()),
            "hold_error": float(100 * d[:, -hold:].mean()),
            "rms_torque": float(np.sqrt(np.mean(ep.torques[steps]**2))),
            "beta_mean": float(beta.mean()) if beta.size else np.nan,
            "active": float((beta > 0).mean()) if beta.size else np.nan,
            "info_gain": float(np.mean(ep.info[steps])),
            "surprise": float(np.mean(ep.surprise[steps])),
            "pred_rmse": float(np.sqrt(np.mean(ep.pred_error[steps]**2))),
            "response_ratio": response_ratio(ep, steps)}


def response_ratio(ep: Episode, steps=slice(None)) -> float:
    """Least-squares slope of the realised on the predicted change of the
    joint velocities over the given steps (see ``summarise``)."""
    real, pred = (dv[steps] for dv in ep.velocity_change)
    if not np.all(np.isfinite(pred)):
        return np.nan                      # agents without a predictive model
    return float(np.sum(real * pred) / np.sum(pred * pred))


def run_seed(fixed_beta, seed: int, s: Settings = Settings(), truth: ArmParams = ArmParams(),
             targets=TARGETS, nominal: Optional[ArmParams] = None, **episode_kwargs) -> Episode:
    """One episode of a fresh CBFE-ARX agent with plant noise seed ``seed``.
    The agent's prior comes from simulating the arm ``nominal``; by default
    the arm of the default physical prior means (θ one standard deviation
    below the truth)."""
    if nominal is None:
        nominal = ArmModel(truth, ARM_CONFIG).params
    q_goal = joint_targets(targets[:1], truth)[0]
    agent = build_agent(s, fixed_beta, nominal, joint_goal(q_goal, s))
    return run_episode(agent, RobotArm(truth, plant_config(s, seed), seed=seed), s, targets,
                       seed=seed, **episode_kwargs)


def run_agent(agent: dict, seed: int, s: Settings = Settings(), truth: ArmParams = ArmParams(),
              targets=TARGETS, nominal: Optional[ArmParams] = None, **episode_kwargs) -> Episode:
    """One episode of a fresh agent built from the specification ``agent``
    (see ``make_controller``) with plant noise seed ``seed``. The prior
    knowledge comes from simulating ``nominal``, by default the arm at the
    prior means of its uncertain parameters."""
    if nominal is None:
        nominal = ArmModel(truth, ARM_CONFIG).params
    controller = make_controller(agent, s, nominal, joint_targets(targets[:1], truth)[0])
    return run_episode(controller, RobotArm(truth, plant_config(s, seed), seed=seed), s, targets,
                       seed=seed, **episode_kwargs)


def run_job(job: dict) -> Episode:
    """``run_agent`` (if the job has an "agent" entry) or ``run_seed`` with
    its keyword arguments in one dict (picklable)."""
    return run_agent(**job) if "agent" in job else run_seed(**job)


def run_many(jobs, processes: Optional[int] = None) -> list:
    """Run ``run_job`` for every job, a dict of keyword arguments, in
    parallel processes, and return the episodes in the order of the jobs.
    Set the BLAS thread variables to 1 before importing numpy: the planner
    works with tiny matrices, so threads only add overhead."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    jobs = list(jobs)
    processes = processes or min(len(jobs), os.cpu_count() or 1)
    if processes <= 1:
        return [run_job(job) for job in jobs]
    with ProcessPoolExecutor(processes, mp_context=get_context("forkserver")) as pool:
        return list(pool.map(run_job, jobs))
