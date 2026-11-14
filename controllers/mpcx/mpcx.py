"""MPC with integrated experiment design (MPC-X), after Larsson, Annergren,
Hjalmarsson, Rojas, Bombois, Mesbah and Modén, "Model predictive control
with integrated experiment design for output error systems", ECC 2013.

MPC-X adds an application-oriented experiment-design constraint to a
nominal MPC. The constraint asks the information matrix of the data that
will exist at the end of the prediction horizon to dominate the curvature
of an application cost, scaled so that, with confidence alpha, the
identified parameters lie in the set where the application degrades by at
most 1/gamma (Larsson et al., eqs. 15-17):

    I_1^{t+K}(theta)  >=  kappa(t) (gamma chi2_alpha(n_theta) / 2) V''_app(theta),
    kappa(t) = min(N, t + K) / N.

The ramp kappa(t) spreads the requirement over an experiment of N steps.
At each call this agent solves

    min_U  sum_k (mu_k - y_*)^T Q (mu_k - y_*) + R ||u_k||^2       [nominal MPC]
    s.t.   u_k in [-1, 1]^{D_u},
           Lambda_t + sum_{k in tau} phi_k phi_k^T  >=  c_t rho_t Phibar_t  [experiment design]

on the ARX model and posterior that the CBFE-ARX agent and the dual MPC
use, with point-mass buffers at the predicted means. The constraint
follows from the three choices below.

Information matrix. For y = A^T phi(x) + e, e ~ N(0, W^-1), the Fisher
information of vec(A) is W kron sum phi phi^T. We use the Bayesian
information matrix of the posterior, W kron Lambda_t with W = R_t^-1, so
that the prior counts as data, as it does for the other ARX controllers, and
add the regressors of the plan.

Application cost. Larsson et al. leave V_app to the application; their
example uses the output deviation of a step response under the MPC. Here
V_app is the goal-weighted deviation of the one-step output at the
goal-reaching inputs p~(u) of the CBFE floor (``goal_reaching_moments``),

    V_app(A) = 1/2 E_p~ ||(A - M_t)^T phi||^2_{S_*^-1},
    V''_app = S_*^-1 kron Phibar_t,  Phibar_t = E_p~[phi phi^T],

evaluated at the current estimate (their Remark). With these choices the
matrix inequality W kron Lambda >= c S_*^-1 kron Phibar holds if and only
if Lambda >= c rho_t Phibar, with rho_t = lambda_max(S_*^-1 R_t): one
D_phi x D_phi LMI, with c_t = kappa(t) gamma chi2_alpha(n_theta) / 2 and
n_theta = D_phi D_y.

Solution. For output-error models, Larsson et al. lift the inputs into
U = u u^T and solve a semidefinite relaxation. The features here are
nonlinear in u, so the program is solved locally with IPOPT instead. The
LMI is whitened with the Cholesky factor Lambda_t = C C^T,

    F(U) = I - c_t rho_t C^-1 Phibar_t C^-T + sum_k (C^-1 phi_k)(C^-1 phi_k)^T >= 0,

so that its eigenvalues measure information relative to the information
already present, and enters through a Cholesky factor, F(U) + s I = L L^T.
The agent first solves the nominal MPC. If the plan meets the LMI, the
constraint is inactive and the plan stands. Otherwise phase 1 finds the
smallest shortfall s* of the minimum eigenvalue of F, and phase 2
minimises the MPC cost subject to F(U) + max(s*, 0) I >= 0. When the
requirement is attainable (s* <= 0), phase 2 is the MPC-X program;
otherwise the agent keeps the largest attainable margin, which softens
the constraint lexicographically.

The cost is the dual-MPC cost without its caution and probing terms, with
an effort penalty on u instead of the input increments of Larsson et al.,
so that MPC-X and the dual MPC differ only in how they excite.
``planner.last_multiplier`` records 1 when the experiment-design constraint
shaped the input and 0 otherwise, so the multiplier log of an episode shows
when the constraint is active.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional, Sequence

import casadi as ca
import numpy as np
from scipy.stats import chi2

from controllers.common import (
    MultivariateNormal, ParameterEstimator, buffer_with_head, flatten_buffer,
    goal_reaching_moments, input_nodes, ipopt_options, predictive_noise_covariance)


@dataclass
class MPCXConfig:
    horizon: int = 5
    effort_weight: float = 0.1                     # R per input dimension
    tracking_weights: Optional[np.ndarray] = None  # diagonal of Q; None: inverse goal variances
    accuracy: float = 3e-3                         # gamma: V_app <= 1 / gamma
    confidence: float = 0.99                       # alpha of the identification ellipsoid
    experiment_length: int = 120                   # N of the ramp kappa(t)
    input_prior_shape: float = 2.0                 # Beta(a, a) weights of the inputs in p~
    bound_margin: float = 1e-3
    ipopt_options: Optional[dict] = None


@dataclass
class MPCX:
    """MPC with an application-oriented experiment-design constraint on the
    shared ARX model (see the module docstring).

    Exposes the online-agent interface of the CBFE-ARX agent: ``estimator``,
    ``plan(latest_output)``, ``update(output, control_input)``, a settable
    ``goal_prior``, ``reset_planner_cache()`` and a ``planner`` handle.
    """
    estimator: ParameterEstimator
    goal_prior: MultivariateNormal
    config: MPCXConfig = field(default_factory=MPCXConfig)

    _solvers: Optional[dict] = field(default=None, init=False, repr=False)
    _previous_plan: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _input_nodes: Optional[tuple] = field(default=None, init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        if self.config.ipopt_options is None:
            self.config.ipopt_options = ipopt_options(max_iter=300, max_cpu_time=5.0, tol=1e-6)
        num_parameters = self.estimator.feature_dim * self.estimator.output_dim
        self.chi_square_quantile = float(chi2.ppf(self.config.confidence, num_parameters))
        # last_multiplier: 1 if the experiment-design constraint shaped the
        # input, 0 if the nominal plan met it; last_shortfall: s* > 0 when
        # no plan meets it; last_requirement: c_t rho_t.
        self.planner = SimpleNamespace(last_multiplier=0.0, last_shortfall=0.0,
                                       last_requirement=0.0, reset=lambda: None)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def update(self, output: np.ndarray, control_input: np.ndarray) -> None:
        self.estimator.update(output, control_input)

    def reset_planner_cache(self) -> None:
        """Clear the warm start (at a new target). The experiment, and with
        it the ramp kappa(t), continues."""
        self._previous_plan = None

    def plan(self, latest_output: Optional[np.ndarray]) -> np.ndarray:
        if self._solvers is None:
            self._build_solvers()
        input_dim, horizon = self.estimator.input_dim, self.config.horizon
        margin = self.config.bound_margin
        lower_bounds = np.full(input_dim * horizon, -1.0 + margin)
        upper_bounds = np.full(input_dim * horizon, 1.0 - margin)
        initial_plan = (np.zeros(input_dim * horizon) if self._previous_plan is None
                        else np.clip(self._previous_plan, lower_bounds, upper_bounds))

        base_parameters = self._base_parameters(latest_output)
        nominal_plan = np.array(self._solvers["nominal"](
            x0=initial_plan, p=base_parameters, lbx=lower_bounds, ubx=upper_bounds)["x"]).flatten()

        lmi_offset, whitening, requirement = self._requirement(latest_output)
        self._step += 1
        nominal_lmi = (lmi_offset + whitening @ self._plan_information(nominal_plan, latest_output)
                       @ whitening.T)
        nominal_margin = float(np.linalg.eigvalsh(nominal_lmi)[0])
        self.planner.last_requirement = requirement
        if nominal_margin >= 0.0:
            plan = nominal_plan
            self.planner.last_multiplier, self.planner.last_shortfall = 0.0, 0.0
        else:
            plan = self._constrained(nominal_plan, nominal_lmi, -nominal_margin, base_parameters,
                                     lmi_offset, whitening, latest_output,
                                     lower_bounds, upper_bounds)
            self.planner.last_multiplier = 1.0
        self._previous_plan = np.concatenate([plan[input_dim:], plan[-input_dim:]])
        return plan[:input_dim]

    # ------------------------------------------------------------------
    # Experiment-design constraint
    # ------------------------------------------------------------------

    def _requirement(self, latest_output):
        """The LMI without the plan's information,
        G = I - c_t rho_t C^-1 Phibar_t C^-T, the whitening C^-1, and c_t rho_t."""
        estimator, config, goal_prior = self.estimator, self.config, self.goal_prior
        if self._input_nodes is None:
            nodes = input_nodes(estimator.input_dim)
            self._input_nodes = (nodes, (config.input_prior_shape - 1.0)
                                 * np.sum(np.log1p(-nodes**2), axis=1))
        feature_moments = goal_reaching_moments(estimator, goal_prior, *self._input_nodes,
                                                latest_output)
        noise_covariance = predictive_noise_covariance(estimator.posterior)
        largest_noise_ratio = float(np.max(np.linalg.eigvals(
            np.linalg.solve(goal_prior.covariance, noise_covariance)).real))
        ramp = (min(config.experiment_length, self._step + config.horizon)
                / config.experiment_length)
        requirement = (ramp * config.accuracy * self.chi_square_quantile / 2.0
                       * largest_noise_ratio)
        whitening = np.linalg.inv(np.linalg.cholesky(estimator.posterior.coefficient_precision))
        lmi_offset = (np.eye(estimator.feature_dim)
                      - requirement * whitening @ feature_moments @ whitening.T)
        return lmi_offset, whitening, requirement

    def _plan_information(self, plan: np.ndarray, latest_output) -> np.ndarray:
        """sum_k phi_k phi_k^T of the plan, with point-mass buffers."""
        estimator = self.estimator
        input_dim, output_dim = estimator.input_dim, estimator.output_dim
        input_memory, output_memory = estimator.input_memory, estimator.output_memory
        output_buffer = flatten_buffer(buffer_with_head(estimator.output_buffer, latest_output,
                                                        output_memory),
                                       output_memory, output_dim)
        input_buffer = flatten_buffer(estimator.input_buffer, input_memory, input_dim)
        past_outputs = [output_buffer[i * output_dim:(i + 1) * output_dim]
                        for i in range(output_memory)]
        past_inputs = [input_buffer[i * input_dim:(i + 1) * input_dim]
                       for i in range(input_memory)]
        information = np.zeros((estimator.feature_dim, estimator.feature_dim))
        for k in range(len(plan) // input_dim):
            control_input = plan[k * input_dim:(k + 1) * input_dim]
            features = estimator.feature_map(np.concatenate([control_input] + past_inputs
                                                            + past_outputs))
            information += np.outer(features, features)
            past_outputs = ([estimator.posterior.coefficient_mean.T @ features]
                            + past_outputs)[:output_memory]
            past_inputs = ([control_input] + past_inputs)[:input_memory]
        return information

    def _lmi(self, plan, lmi_offset, whitening, latest_output) -> np.ndarray:
        """F(U) of the plan."""
        return lmi_offset + whitening @ self._plan_information(plan, latest_output) @ whitening.T

    def _constrained(self, nominal_plan, nominal_lmi, nominal_shortfall, base_parameters,
                     lmi_offset, whitening, latest_output, lower_bounds, upper_bounds):
        """Phase 1 (largest margin) and phase 2 (MPC cost at that margin)."""
        plan_length = len(nominal_plan)
        num_cholesky_entries = self.estimator.feature_dim * (self.estimator.feature_dim + 1) // 2
        lmi_parameters = np.concatenate([base_parameters, lmi_offset.flatten(order="F"),
                                         whitening.flatten(order="F")])
        unbounded = np.full(num_cholesky_entries, np.inf)
        # Phase 1: minimise s subject to F(U) + s I = L L^T. The information
        # is quadratic in the inputs, so a plan at rest is a stationary
        # point; starts shifted by ±0.3 let the excitation grow.
        best_plan, best_lmi, best_shortfall = nominal_plan, nominal_lmi, nominal_shortfall
        for start in (nominal_plan, np.clip(nominal_plan + 0.3, lower_bounds, upper_bounds),
                      np.clip(nominal_plan - 0.3, lower_bounds, upper_bounds)):
            start_lmi = self._lmi(start, lmi_offset, whitening, latest_output)
            start_shortfall = -float(np.linalg.eigvalsh(start_lmi)[0])
            start_slack = start_shortfall + 1e-3 * (1.0 + abs(start_shortfall))
            initial_point = np.concatenate([
                start, self._cholesky_vector(start_lmi + start_slack * np.eye(len(start_lmi))),
                [start_slack]])
            solution = self._solvers["phase1"](
                x0=initial_point, p=lmi_parameters,
                lbx=np.concatenate([lower_bounds, -unbounded, [-np.inf]]),
                ubx=np.concatenate([upper_bounds, unbounded, [np.inf]]), lbg=0.0, ubg=0.0)
            candidate_plan = np.array(solution["x"]).flatten()[:plan_length]
            candidate_lmi = self._lmi(candidate_plan, lmi_offset, whitening, latest_output)
            candidate_shortfall = -float(np.linalg.eigvalsh(candidate_lmi)[0])
            if candidate_shortfall < best_shortfall:
                best_plan, best_lmi, best_shortfall = (candidate_plan, candidate_lmi,
                                                       candidate_shortfall)
        required_slack = (max(best_shortfall, 0.0) * (1.0 + 1e-3)
                          + (1e-6 if best_shortfall > 0.0 else 0.0))
        self.planner.last_shortfall = max(best_shortfall, 0.0)
        # Phase 2: minimise the MPC cost subject to F(U) + s_req I = L L^T.
        cholesky_lift = (max(required_slack, 0.0)
                         + max(0.0, -float(np.linalg.eigvalsh(best_lmi)[0]) - required_slack)
                         + 1e-9)
        initial_point = np.concatenate([
            best_plan, self._cholesky_vector(best_lmi + cholesky_lift * np.eye(len(best_lmi)))])
        solution = self._solvers["phase2"](
            x0=initial_point, p=np.concatenate([lmi_parameters, [required_slack]]),
            lbx=np.concatenate([lower_bounds, -unbounded]),
            ubx=np.concatenate([upper_bounds, unbounded]), lbg=0.0, ubg=0.0)
        phase2_plan = np.array(solution["x"]).flatten()[:plan_length]
        phase2_lmi = self._lmi(phase2_plan, lmi_offset, whitening, latest_output)
        # Fall back to the phase-1 plan if IPOPT left the constraint violated.
        if float(np.linalg.eigvalsh(phase2_lmi)[0]) + required_slack < -1e-6:
            return best_plan
        return phase2_plan

    @staticmethod
    def _cholesky_vector(matrix: np.ndarray) -> np.ndarray:
        """Lower-triangular Cholesky factor of a matrix as a row-major vector."""
        factor = np.linalg.cholesky(0.5 * (matrix + matrix.T))
        return factor[np.tril_indices(len(matrix))]

    # ------------------------------------------------------------------
    # Cached parametric NLPs
    # ------------------------------------------------------------------

    def _build_solvers(self) -> None:
        estimator, config = self.estimator, self.config
        input_dim, output_dim = estimator.input_dim, estimator.output_dim
        input_memory, output_memory = estimator.input_memory, estimator.output_memory
        feature_dim, horizon = estimator.feature_dim, config.horizon
        num_cholesky_entries = feature_dim * (feature_dim + 1) // 2

        plan = ca.SX.sym("U", input_dim * horizon)
        output_buffer = ca.SX.sym("output_buffer", output_dim * output_memory)
        input_buffer = ca.SX.sym("input_buffer", input_dim * input_memory)
        coefficient_mean = ca.SX.sym("M", feature_dim, output_dim)
        goal_mean = ca.SX.sym("m_star", output_dim)
        tracking_weights = ca.SX.sym("Q", output_dim)
        lmi_offset = ca.SX.sym("G", feature_dim, feature_dim)
        whitening = ca.SX.sym("T", feature_dim, feature_dim)
        required_slack = ca.SX.sym("s_req")
        cholesky_entries = ca.SX.sym("L", num_cholesky_entries)
        slack = ca.SX.sym("s")

        past_outputs = [output_buffer[i * output_dim:(i + 1) * output_dim]
                        for i in range(output_memory)]
        past_inputs = [input_buffer[i * input_dim:(i + 1) * input_dim]
                       for i in range(input_memory)]
        cost = ca.SX(0.0)
        whitened_information = ca.SX.zeros(feature_dim, feature_dim)
        for k in range(horizon):
            control_input = plan[k * input_dim:(k + 1) * input_dim]
            features = estimator.feature_map.casadi(
                ca.vertcat(*([control_input] + past_inputs + past_outputs)))
            predicted_mean = ca.mtimes(coefficient_mean.T, features)
            goal_error = predicted_mean - goal_mean
            cost += (ca.dot(tracking_weights, goal_error * goal_error)
                     + float(config.effort_weight) * ca.dot(control_input, control_input))
            whitened_features = ca.mtimes(whitening, features)
            whitened_information += ca.mtimes(whitened_features, whitened_features.T)
            past_outputs = ([predicted_mean] + past_outputs)[:output_memory]
            past_inputs = ([control_input] + past_inputs)[:input_memory]

        cholesky_factor = ca.SX.zeros(feature_dim, feature_dim)
        lower_rows, lower_cols = np.tril_indices(feature_dim)
        for i, (row, col) in enumerate(zip(lower_rows, lower_cols)):
            cholesky_factor[int(row), int(col)] = cholesky_entries[i]
        base_parameters = ca.vertcat(output_buffer, input_buffer, ca.vec(coefficient_mean),
                                     goal_mean, tracking_weights)
        lmi_parameters = ca.vertcat(base_parameters, ca.vec(lmi_offset), ca.vec(whitening))

        def lower_triangle(matrix):
            return ca.vertcat(*[matrix[int(row), int(col)]
                                for row, col in zip(lower_rows, lower_cols)])

        options = config.ipopt_options
        phase1_residual = (lmi_offset + whitened_information + slack * ca.SX.eye(feature_dim)
                           - ca.mtimes(cholesky_factor, cholesky_factor.T))
        phase2_residual = (lmi_offset + whitened_information
                           + required_slack * ca.SX.eye(feature_dim)
                           - ca.mtimes(cholesky_factor, cholesky_factor.T))
        self._solvers = {
            "nominal": ca.nlpsol("mpcx_nominal", "ipopt",
                                 {"x": plan, "f": cost, "p": base_parameters}, options),
            "phase1": ca.nlpsol("mpcx_phase1", "ipopt",
                                {"x": ca.vertcat(plan, cholesky_entries, slack), "f": slack,
                                 "p": lmi_parameters, "g": lower_triangle(phase1_residual)},
                                options),
            "phase2": ca.nlpsol("mpcx_phase2", "ipopt",
                                {"x": ca.vertcat(plan, cholesky_entries), "f": cost,
                                 "p": ca.vertcat(lmi_parameters, required_slack),
                                 "g": lower_triangle(phase2_residual)}, options),
        }

    def _base_parameters(self, latest_output) -> np.ndarray:
        estimator, goal_prior = self.estimator, self.goal_prior
        tracking_weights = (np.asarray(self.config.tracking_weights, float)
                            if self.config.tracking_weights is not None
                            else 1.0 / np.diag(np.asarray(goal_prior.covariance, float)))
        return np.concatenate([
            flatten_buffer(buffer_with_head(estimator.output_buffer, latest_output,
                                            estimator.output_memory),
                           estimator.output_memory, estimator.output_dim),
            flatten_buffer(estimator.input_buffer, estimator.input_memory, estimator.input_dim),
            np.asarray(estimator.posterior.coefficient_mean, float).flatten(order="F"),
            np.asarray(goal_prior.mean, float).flatten(),
            tracking_weights,
        ])

    # ------------------------------------------------------------------
    # Construction (mirrors Agent.build so the model is identical)
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, input_dim: int, output_dim: int, input_memory: int, output_memory: int,
              goal_prior: MultivariateNormal,
              horizon: int = 5,
              accuracy: float = 3e-3,
              confidence: float = 0.99,
              experiment_length: int = 120,
              effort_weight: float = 0.1,
              tracking_weights: Optional[Sequence[float]] = None,
              input_prior_shape: float = 2.0,
              num_harmonics: int = 2,
              half_period: float | np.ndarray = 1.0,
              prior_degrees_of_freedom: Optional[float] = None,
              prior_precision_scale: float = 1e-2,
              prior_noise_scale: float = 1.0,
              prior_mean_scale: float = 0.1,
              polynomial_features: bool = True) -> "MPCX":
        estimator = ParameterEstimator.from_dimensions(
            input_dim=input_dim, output_dim=output_dim,
            input_memory=input_memory, output_memory=output_memory,
            num_harmonics=num_harmonics, half_period=half_period,
            prior_degrees_of_freedom=prior_degrees_of_freedom,
            prior_precision_scale=prior_precision_scale,
            prior_noise_scale=prior_noise_scale,
            prior_mean_scale=prior_mean_scale,
            polynomial_features=polynomial_features)
        config = MPCXConfig(horizon=horizon, effort_weight=effort_weight, accuracy=accuracy,
                            confidence=confidence, experiment_length=experiment_length,
                            tracking_weights=(None if tracking_weights is None
                                              else np.asarray(tracking_weights, float)),
                            input_prior_shape=input_prior_shape)
        return cls(estimator=estimator, goal_prior=goal_prior, config=config)
