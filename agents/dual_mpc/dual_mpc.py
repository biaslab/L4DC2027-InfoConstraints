"""Explicit dual-control MPC baseline (paper Sec. V-A, Eq. dualmpc).

Dual control (Feldbaum 1960; Bar-Shalom & Tse 1974) recognises that under
parameter uncertainty every action serves two purposes at once: it
*regulates* the plant toward the goal under the current parameter estimate,
and it *probes* the plant to reduce parameter uncertainty for future
control. The optimal trade-off solves an intractable dynamic program over
the joint state-and-posterior hyperstate; practical "explicit dual"
controllers instead optimise a surrogate cost in which the probing benefit
appears as an additive bonus with a *fixed* weight.

This baseline is deliberately built on the *same* Bayesian autoregressive
model and matrix-normal Wishart posterior as the CBFE-ARX agent (same
``ParameterEstimator``, same feature map, same priors). It therefore
isolates the single design choice our contribution concerns: how the
weight on probing is set.

    Dual MPC:   fixed probing weight λ, tuned offline (this file)
    CBFE-ARX:   per-step KKT multiplier β_t (active only on demand)

At each control step the dual MPC solves the receding-horizon program

    u_1..u_K = argmin  sum_k  (mu_k - y_*)^T Q (mu_k - y_*)          [track]
                              + (1 + ell_k) tr(Q Omega_eff)          [cautious]
                              + u_k^T R u_k                          [effort]
                              - λ (D_y/2) ln(1 + ell_k)              [probe]
               s.t.  u_k in [-1, 1]^{D_u},

where mu_k and ell_k are the posterior-predictive mean and leverage of the
shared model, unrolled forward with point-mass buffers at the predicted
means, as in the forward pass of the CBFE-ARX planners. The [track] +
[cautious] sum is exactly E_q[(y_k - y_*)^T Q (y_k - y_*)] under the
Student-t predictive, so the tracking cost is the cautious,
uncertainty-inflated one. The [probe] term is the expected parameter
information gain of Lemma 1. Setting λ = 0 recovers certainty-equivalence
MPC (cautious control, no probing).

The tracking weights Q default to the inverse goal covariance, so the dual
MPC and the CBFE-ARX agent share the same effective tracking objective and
differ only in the probing weight; R is a light effort penalty
(tie-breaker). λ is the quantity to grid-search on held-out trials.

With λ > 0, the probe term makes the program nonconvex, and under a
posterior that is symmetric in the sign of an input's effect, a plan at
rest is a stationary point from which IPOPT does not move. The planner then
also solves from the constant plan at ±0.5 per input with the lowest cost
and keeps the better solution, as the CBFE-ARX message planner does.

With ``rest_tolerance`` set, the plan at rest stands whenever the solver's
plan improves on it by less than that tolerance, as the message planner
keeps its warm start on ties. Without it, round-off can decide symmetric
cases: at λ = 2 with the goal covariance equal to the noise covariance, the
caution and probing terms cancel to second order around rest, and IPOPT
returns inputs of order 1e-8 that move the gain estimate off zero, after
which the inputs grow within a few steps.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from types import SimpleNamespace
from typing import List, Optional, Sequence
import numpy as np
import casadi as ca

from agents.common import (
    MultivariateNormal, ParameterEstimator, buffer_with_head, flatten_buffer,
    ipopt_options, predictive_noise_covariance)


@dataclass
class DualMPCConfig:
    horizon: int = 5
    probing_weight: float = 1.0                    # λ ≥ 0
    effort_weight: float = 0.1                     # R per input dimension
    tracking_weights: Optional[np.ndarray] = None  # diagonal of Q; None: inverse goal variances
    cautious: bool = True                          # include the (1 + ell) tr(Q Omega_eff) term
    # "quadratic": R ||u||^2 on the box; "beta": -R (a - 1) sum_i ln(1 - u_i^2), twice the
    # negative log-density of the symmetric Beta(a, a) prior of the CBFE-ARX agent for
    # R = 2, which equals R ||u||^2 near u = 0 and grows without bound at the limits
    input_cost: str = "quadratic"
    input_prior_shape: float = 2.0                 # a of the "beta" input cost
    bound_margin: float = 1e-3
    ipopt_options: Optional[dict] = None
    # If set, the plan at rest replaces the solver's plan whenever its cost is within
    # rest_tolerance (1 + |J|) of it, so that round-off cannot move the plan away from
    # a symmetric optimum at rest (None: the solver's plan always stands)
    rest_tolerance: Optional[float] = None


@dataclass
class DualMPC:
    """Explicit dual-control MPC over the shared ARX model.

    Exposes the online-agent interface of the CBFE-ARX agent: ``estimator``,
    ``plan(latest_output)``, ``update(output, control_input)``, a settable
    ``goal_prior``, ``reset_planner_cache()``, and a ``planner`` handle whose
    ``last_multiplier`` is the fixed probing weight, so that the multiplier
    log of an episode records λ.
    """
    estimator: ParameterEstimator
    goal_prior: MultivariateNormal
    config: DualMPCConfig = field(default_factory=DualMPCConfig)

    _solver: object = field(default=None, init=False, repr=False)
    _cost: object = field(default=None, init=False, repr=False)
    _previous_plan: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.config.ipopt_options is None:
            self.config.ipopt_options = ipopt_options(max_iter=200, max_cpu_time=2.0, tol=1e-4)
        self.planner = SimpleNamespace(last_multiplier=float(self.config.probing_weight),
                                       reset=lambda: None)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def update(self, output: np.ndarray, control_input: np.ndarray) -> None:
        self.estimator.update(output, control_input)

    def reset_planner_cache(self) -> None:
        self._previous_plan = None

    def plan(self, latest_output: Optional[np.ndarray]) -> np.ndarray:
        if self._solver is None:
            self._build_solver()
        input_dim, horizon = self.estimator.input_dim, self.config.horizon

        margin = self.config.bound_margin
        lower_bounds = np.full(input_dim * horizon, -1.0 + margin)
        upper_bounds = np.full(input_dim * horizon, 1.0 - margin)

        if self._previous_plan is not None:
            initial_plan = np.clip(self._previous_plan, lower_bounds, upper_bounds)
        else:
            initial_plan = np.zeros(input_dim * horizon)

        parameter_values = self._parameter_vector(latest_output, self.goal_prior)
        solution = self._solver(x0=initial_plan, p=parameter_values,
                                lbx=lower_bounds, ubx=upper_bounds)
        best_cost, best_plan = float(solution["f"]), np.array(solution["x"]).flatten()
        if self.config.probing_weight > 0.0:
            # Symmetry-breaking start: the constant plan with the lowest cost.
            starts = [np.tile(0.5 * np.asarray(signs), horizon)
                      for signs in product((-1.0, 1.0), repeat=input_dim)]
            costs = [float(self._cost(start, parameter_values)) for start in starts]
            solution = self._solver(x0=starts[int(np.argmin(costs))], p=parameter_values,
                                    lbx=lower_bounds, ubx=upper_bounds)
            if float(solution["f"]) < best_cost:
                best_cost, best_plan = float(solution["f"]), np.array(solution["x"]).flatten()
        if self.config.rest_tolerance is not None:
            # Under a posterior that is symmetric in the sign of an input's effect, the
            # plan at rest can be optimal while IPOPT returns inputs of order 1e-8 whose
            # cost differs from it only by round-off; the plan at rest then stands.
            rest_plan = np.zeros_like(best_plan)
            rest_cost = float(self._cost(rest_plan, parameter_values))
            if rest_cost <= best_cost + self.config.rest_tolerance * (1.0 + abs(best_cost)):
                best_cost, best_plan = rest_cost, rest_plan
        # Warm start of the next call: the plan shifted one step forward.
        self._previous_plan = np.concatenate([best_plan[input_dim:], best_plan[-input_dim:]])
        return best_plan[:input_dim]

    # ------------------------------------------------------------------
    # Cached parametric NLP
    # ------------------------------------------------------------------

    def _build_solver(self) -> None:
        estimator = self.estimator
        input_dim, output_dim = estimator.input_dim, estimator.output_dim
        input_memory, output_memory = estimator.input_memory, estimator.output_memory
        feature_dim = estimator.feature_map.feature_dim
        horizon = self.config.horizon

        tracking_weights = ca.DM(np.asarray(self._tracking_weights(), dtype=float).reshape(-1, 1))
        effort_weight = float(self.config.effort_weight)
        probing_weight = float(self.config.probing_weight)

        plan = ca.MX.sym("U", input_dim * horizon)

        coefficient_mean = ca.MX.sym("M", feature_dim, output_dim)
        coefficient_covariance = ca.MX.sym("Lambda_inv", feature_dim, feature_dim)
        noise_covariance = ca.MX.sym("Omega_eff", output_dim, output_dim)
        goal_mean = ca.MX.sym("m_star", output_dim)
        output_buffer_length = output_dim * output_memory
        input_buffer_length = input_dim * input_memory
        output_buffer = (ca.MX.sym("output_buffer", output_buffer_length)
                         if output_buffer_length > 0 else None)
        input_buffer = (ca.MX.sym("input_buffer", input_buffer_length)
                        if input_buffer_length > 0 else None)

        # Symbolic buffers, most recent first, as in ParameterEstimator.regressor.
        past_outputs = ([output_buffer[i * output_dim:(i + 1) * output_dim]
                         for i in range(output_memory)] if output_buffer is not None else [])
        past_inputs = ([input_buffer[i * input_dim:(i + 1) * input_dim]
                        for i in range(input_memory)] if input_buffer is not None else [])

        cost = ca.MX(0.0)
        for k in range(horizon):
            control_input = plan[k * input_dim:(k + 1) * input_dim]
            regressor = ([control_input] + list(past_inputs[:input_memory])
                         + list(past_outputs[:output_memory]))
            features = estimator.feature_map.casadi(ca.vertcat(*regressor))

            predicted_mean = ca.mtimes(coefficient_mean.T, features)
            leverage = ca.mtimes(features.T, ca.mtimes(coefficient_covariance, features))

            goal_error = predicted_mean - goal_mean
            cost += ca.dot(tracking_weights, goal_error * goal_error)                # track
            if self.config.cautious:
                # (1 + ell) tr(Q Omega_eff) = (1 + ell) sum_i Q_i (Omega_eff)_ii
                weighted_noise = ca.dot(tracking_weights, ca.diag(noise_covariance))
                cost += (1.0 + leverage) * weighted_noise                            # cautious
            if self.config.input_cost == "beta":
                cost += -effort_weight * (self.config.input_prior_shape - 1.0) * ca.sum1(
                    ca.log(1.0 - control_input * control_input))                     # effort
            else:
                cost += effort_weight * ca.dot(control_input, control_input)         # effort
            cost += -probing_weight * (output_dim / 2.0) * ca.log(1.0 + leverage)    # probe

            # Point-mass buffers: the predicted mean feeds the next slice.
            past_outputs = [predicted_mean] + past_outputs[:max(output_memory - 1, 0)]
            past_inputs = [control_input] + past_inputs[:max(input_memory - 1, 0)]

        parameters = []
        if output_buffer is not None:
            parameters.append(output_buffer)
        if input_buffer is not None:
            parameters.append(input_buffer)
        parameters += [ca.vec(coefficient_mean), ca.vec(coefficient_covariance),
                       ca.vec(noise_covariance), goal_mean]
        nlp = {"x": plan, "f": cost, "p": ca.vertcat(*parameters)}
        self._solver = ca.nlpsol("dualmpc", "ipopt", nlp, self.config.ipopt_options)
        self._cost = ca.Function("cost", [plan, nlp["p"]], [cost])

    def _parameter_vector(self, latest_output: Optional[np.ndarray],
                          goal_prior: MultivariateNormal) -> np.ndarray:
        estimator = self.estimator
        posterior = estimator.posterior
        pieces: List[np.ndarray] = []
        if estimator.output_dim * estimator.output_memory > 0:
            output_buffer = buffer_with_head(estimator.output_buffer, latest_output,
                                             estimator.output_memory)
            pieces.append(flatten_buffer(output_buffer, estimator.output_memory,
                                         estimator.output_dim))
        if estimator.input_dim * estimator.input_memory > 0:
            pieces.append(flatten_buffer(estimator.input_buffer, estimator.input_memory,
                                         estimator.input_dim))
        pieces += [
            np.asarray(posterior.coefficient_mean, float).flatten(order="F"),
            np.asarray(posterior.coefficient_covariance, float).flatten(order="F"),
            np.asarray(predictive_noise_covariance(posterior), float).flatten(order="F"),
            np.asarray(goal_prior.mean, float).flatten(),
        ]
        return np.concatenate(pieces)

    def _tracking_weights(self) -> np.ndarray:
        if self.config.tracking_weights is not None:
            return np.asarray(self.config.tracking_weights, float)
        # Default: the inverse goal variances, the tracking objective of the
        # CBFE-ARX agent, so that only the probing weight differs.
        return 1.0 / np.diag(np.asarray(self.goal_prior.covariance, float))

    # ------------------------------------------------------------------
    # Construction (mirrors Agent.build so the model is identical)
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, input_dim: int, output_dim: int, input_memory: int, output_memory: int,
              goal_prior: MultivariateNormal,
              horizon: int = 5,
              probing_weight: float = 1.0,
              effort_weight: float = 0.1,
              tracking_weights: Optional[Sequence[float]] = None,
              cautious: bool = True,
              num_harmonics: int = 2,
              half_period: float | np.ndarray = 1.0,
              prior_degrees_of_freedom: Optional[float] = None,
              prior_precision_scale: float = 1e-2,
              prior_noise_scale: float = 1.0,
              prior_mean_scale: float = 0.1,
              polynomial_features: bool = True) -> "DualMPC":
        estimator = ParameterEstimator.from_dimensions(
            input_dim=input_dim, output_dim=output_dim,
            input_memory=input_memory, output_memory=output_memory,
            num_harmonics=num_harmonics, half_period=half_period,
            prior_degrees_of_freedom=prior_degrees_of_freedom,
            prior_precision_scale=prior_precision_scale,
            prior_noise_scale=prior_noise_scale,
            prior_mean_scale=prior_mean_scale,
            polynomial_features=polynomial_features)
        config = DualMPCConfig(
            horizon=horizon, probing_weight=probing_weight, effort_weight=effort_weight,
            tracking_weights=(None if tracking_weights is None
                              else np.asarray(tracking_weights, float)),
            cautious=cautious)
        return cls(estimator=estimator, goal_prior=goal_prior, config=config)
