"""Planner: finds the MAP input by minimising the constrained Bethe free energy.

The control posterior of one slice is

    q*(u) ∝ p(u) (1 + ℓ(u))^(β D_y / 2) exp(-∫ λ(y) p(y | u, D_t) dy),

where p(u) is the control prior, ℓ(u) the leverage, β the KKT multiplier of
the information constraint (sign-restricted to β ≥ 0), and
λ(y) = -ln N(y | m_*, S_*) makes the goal-seeking term the cross-entropy
from the predictive to the goal prior. The planner finds the MAP input
û = argmax_u ln q*(u) on the box of the control prior with CasADi's IPOPT
interface.

The KKT conditions determine the multiplier: β = 0 if the constraint holds
without it (inactive), and otherwise the unique positive root of the
active-constraint equation on [0, β_max].

The constraint requires the information that the next output carries about
the coefficients A, (D_y / 2) ln(1 + ℓ(u)), to reach the floor

    1/2 ln|ℓ̄_t R_t S_*⁻¹| = H[N(0, ℓ̄_t R_t)] - H[p_*],

where R_t is the noise covariance and ℓ̄_t the expected leverage of the
goal-reaching inputs p̃(u) ∝ p(u) N(m_* | Mᵀφ(u), S_*). The floor is the
entropy by which the parameter-induced spread of the predictions at those
inputs exceeds the goal prior. It is positive while the model is too
uncertain for the goal and non-positive, which switches the constraint
off, once that spread fits inside the goal (``information_floor_type="task"``).

The task floor can exceed the information that one output can carry when
the goal is tighter than the noise. The predictive floor

    1/2 ln|(R_t + ℓ̄_t R_t)(R_t + S_*)⁻¹| = H[N(0, R_t + ℓ̄_t R_t)] - H[N(0, R_t + S_*)]

(``information_floor_type="predictive"``) compares predictive covariances
that both include the noise, as the information does. It switches off at
the same point for one output, and it stays below the information
(D_y / 2) ln(1 + max_u ℓ(u)) of the most informative input, so the KKT
multiplier is always finite.
The earlier floor -H[p_*] is available as
``information_floor_type="negentropy"``; it is compared with the full
information about (A, W).

A single IPOPT NLP is constructed lazily on first use and reused across all
calls; only its parameter vector (posterior moments, goal prior, multiplier,
buffers) changes from call to call, which avoids the dominant cost of
repeated solver instantiation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from typing import Optional, Tuple, List
import numpy as np
import casadi as ca

from controllers.common import (
    MultivariateNormal, ParameterEstimator, buffer_with_head, flatten_buffer,
    goal_reaching_moments, input_nodes, ipopt_options, predictive_noise_covariance)


@dataclass
class Planner:
    """MAP planner with a cached parametric CasADi NLP.

    estimator            the ARX model: posterior, feature map and buffers
    control_prior        prior on the inputs with ``lower_bounds``,
                         ``upper_bounds`` and ``log_pdf_casadi``
                         (``SymmetricBeta`` or ``Exponential``)
    bound_margin         the inputs stay this far inside the prior's box
    fixed_multiplier     if set, the multiplier β of every plan instead of
                         the KKT solution (0 removes the constraint)
    max_multiplier       upper end of the KKT search; β saturates here when
                         the information floor is unattainable
    multistart           also start every MAP solve from the corners of the
                         input box; with harmonic features the objective
                         can be multimodal. Solves with a positive
                         multiplier use these starts regardless, because
                         the probing term makes an input at rest under a
                         symmetric posterior a stationary point
    information_floor_type  "task" (1/2 ln|ℓ̄ R S_*⁻¹|), "predictive"
                         (1/2 ln|(R + ℓ̄ R)(R + S_*)⁻¹|) or "negentropy"
                         (-H[p_*], the earlier studies)
    """
    estimator: ParameterEstimator
    control_prior: object
    bound_margin: float = 1e-3
    ipopt_options: Optional[dict] = None
    fixed_multiplier: Optional[float] = None
    max_multiplier: float = 50.0
    multistart: bool = False
    information_floor_type: str = "task"

    # Cached parametric solver state (populated lazily on first solve).
    _solver: object = field(default=None, init=False, repr=False)
    _layout: Optional[dict] = field(default=None, init=False, repr=False)
    _last_planned_input: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    # Input nodes and their control-prior log-densities for ℓ̄_t.
    _input_nodes: Optional[tuple] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.ipopt_options is None:
            self.ipopt_options = ipopt_options(max_iter=100, max_cpu_time=1.0, tol=1e-4)
        if self.information_floor_type not in ("task", "predictive", "negentropy"):
            raise ValueError(f"unknown information_floor_type '{self.information_floor_type}'")
        # The floor of the last KKT search (nan with a fixed multiplier).
        self.last_information_floor = float("nan")

    # ----- Cached parametric NLP construction -----

    def _build_solver(self) -> None:
        """Build the IPOPT NLP once, with every changing quantity as a
        parameter; later calls only update the parameter vector."""
        estimator = self.estimator
        input_dim, output_dim = estimator.input_dim, estimator.output_dim
        feature_dim = estimator.feature_map.feature_dim

        input_symbol = ca.MX.sym("u", input_dim)

        # Buffer parameters are omitted where the memory is zero.
        input_buffer_length = input_dim * estimator.input_memory
        output_buffer_length = output_dim * estimator.output_memory
        input_buffer_symbol = (ca.MX.sym("input_buffer", input_buffer_length)
                               if input_buffer_length > 0 else None)
        output_buffer_symbol = (ca.MX.sym("output_buffer", output_buffer_length)
                                if output_buffer_length > 0 else None)

        coefficient_mean = ca.MX.sym("M", feature_dim, output_dim)
        coefficient_covariance = ca.MX.sym("Lambda_inv", feature_dim, feature_dim)
        # Ω_eff (``predictive_noise_covariance``) is computed in Python, so
        # that the symbolic objective stays linear in the parameters.
        noise_covariance = ca.MX.sym("Omega_eff", output_dim, output_dim)
        goal_precision = ca.MX.sym("S_star_inv", output_dim, output_dim)
        goal_mean = ca.MX.sym("m_star", output_dim)
        multiplier = ca.MX.sym("beta")

        # Regressor x = (u, input buffer, output buffer).
        regressor_parts = [input_symbol]
        if input_buffer_symbol is not None:
            regressor_parts.append(input_buffer_symbol)
        if output_buffer_symbol is not None:
            regressor_parts.append(output_buffer_symbol)
        features = estimator.feature_map.casadi(ca.vertcat(*regressor_parts))

        predicted_mean = ca.mtimes(coefficient_mean.T, features)
        leverage = ca.mtimes(features.T, ca.mtimes(coefficient_covariance, features))

        log_control_prior = self.control_prior.log_pdf_casadi(input_symbol)
        log_information = multiplier * (output_dim / 2.0) * ca.log(1.0 + leverage)

        mean_error_outer = ca.mtimes(predicted_mean - goal_mean, (predicted_mean - goal_mean).T)
        cross_entropy = 0.5 * ca.trace(
            ca.mtimes(goal_precision, noise_covariance * (1.0 + leverage) + mean_error_outer))

        log_goal = -cross_entropy
        log_posterior = log_control_prior + log_information + log_goal

        # The order of the parameters is mirrored in _parameter_vector.
        parameters = []
        if output_buffer_symbol is not None:
            parameters.append(output_buffer_symbol)
        if input_buffer_symbol is not None:
            parameters.append(input_buffer_symbol)
        parameters.extend([
            ca.vec(coefficient_mean),
            ca.vec(coefficient_covariance),
            ca.vec(noise_covariance),
            ca.vec(goal_precision),
            goal_mean,
            multiplier,
        ])
        nlp = {"x": input_symbol, "f": -log_posterior, "p": ca.vertcat(*parameters)}
        self._solver = ca.nlpsol("map_solver", "ipopt", nlp, self.ipopt_options)
        self._layout = {"input_buffer_length": input_buffer_length,
                        "output_buffer_length": output_buffer_length}

    # ----- Parameter packing -----

    def _parameter_vector(self, latest_output: Optional[np.ndarray],
                          goal_prior: MultivariateNormal,
                          multiplier: float) -> np.ndarray:
        """The parameter vector of the cached solver, with ``latest_output``
        (if given) as the newest entry of the output buffer."""
        estimator = self.estimator
        posterior = estimator.posterior
        pieces: List[np.ndarray] = []
        if self._layout["output_buffer_length"] > 0:
            output_buffer = buffer_with_head(estimator.output_buffer, latest_output,
                                             estimator.output_memory)
            pieces.append(flatten_buffer(output_buffer, estimator.output_memory,
                                         estimator.output_dim))
        if self._layout["input_buffer_length"] > 0:
            pieces.append(flatten_buffer(estimator.input_buffer, estimator.input_memory,
                                         estimator.input_dim))
        pieces.extend([
            np.asarray(posterior.coefficient_mean, dtype=float).flatten(order='F'),
            np.asarray(posterior.coefficient_covariance, dtype=float).flatten(order='F'),
            np.asarray(predictive_noise_covariance(posterior), dtype=float).flatten(order='F'),
            np.asarray(np.linalg.inv(goal_prior.covariance), dtype=float).flatten(order='F'),
            np.asarray(goal_prior.mean, dtype=float).flatten(),
            np.array([float(multiplier)]),
        ])
        return np.concatenate(pieces)

    # ----- MAP input for a fixed multiplier -----

    def _solve_map(self, latest_output: Optional[np.ndarray],
                   goal_prior: MultivariateNormal,
                   multiplier: float,
                   initial_input: Optional[np.ndarray] = None,
                   multistart: Optional[bool] = None) -> np.ndarray:
        """argmax_u ln q*(u) on the box of the control prior for the given
        multiplier. The corners of the input box are added as starts if
        ``multistart`` (default: the ``multistart`` field)."""
        if self._solver is None:
            self._build_solver()

        lower_bounds = (np.asarray(self.control_prior.lower_bounds, dtype=float)
                        + self.bound_margin)
        upper_bounds = (np.asarray(self.control_prior.upper_bounds, dtype=float)
                        - self.bound_margin)

        if initial_input is not None:
            start = np.asarray(initial_input, dtype=float).flatten()
        elif self._last_planned_input is not None:
            start = self._last_planned_input.copy()
        else:
            start = 0.5 * (lower_bounds + upper_bounds)
        starts = [np.clip(start, lower_bounds, upper_bounds)]
        if self.multistart if multistart is None else multistart:
            for corner in product((0.2, 0.8), repeat=self.estimator.input_dim):
                starts.append(np.clip(
                    lower_bounds + np.asarray(corner) * (upper_bounds - lower_bounds),
                    lower_bounds, upper_bounds))

        parameter_values = self._parameter_vector(latest_output, goal_prior, multiplier)
        best_input, best_objective = None, np.inf
        for start in starts:
            solution = self._solver(x0=start, p=parameter_values,
                                    lbx=lower_bounds, ubx=upper_bounds)
            objective = float(solution["f"])
            if objective < best_objective:
                best_objective = objective
                best_input = np.array(solution["x"]).flatten()
        return best_input

    # ----- KKT determination of the multiplier -----

    def information_floor(self, latest_output: Optional[np.ndarray],
                          goal_prior: MultivariateNormal) -> float:
        """Floor of the information constraint on the next output.

        "task": 1/2 ln|ℓ̄ R S_*⁻¹|, with ℓ̄ = tr(Λ⁻¹ E[φφᵀ]) the expected
        leverage of the goal-reaching inputs, averaged over input nodes
        weighted by the control prior. "predictive":
        1/2 ln|(1 + ℓ̄) R (R + S_*)⁻¹|, the same comparison between
        predictive covariances that include the noise R. "negentropy":
        -H[p_*].
        """
        if self.information_floor_type == "negentropy":
            return goal_prior.negative_entropy()
        estimator = self.estimator
        if self._input_nodes is None:
            nodes = input_nodes(estimator.input_dim)
            self._input_nodes = (nodes, np.array([self.control_prior.log_pdf(node)
                                                  for node in nodes]))
        nodes, log_prior_density = self._input_nodes
        feature_moments = goal_reaching_moments(estimator, goal_prior, nodes,
                                                log_prior_density, latest_output)
        expected_leverage = float(np.sum(estimator.posterior.coefficient_covariance
                                         * feature_moments))
        noise_covariance = predictive_noise_covariance(estimator.posterior)
        if self.information_floor_type == "predictive":
            _, log_det_noise = np.linalg.slogdet(noise_covariance)
            _, log_det_tolerated = np.linalg.slogdet(noise_covariance + goal_prior.covariance)
            return 0.5 * (estimator.output_dim * np.log1p(expected_leverage)
                          + log_det_noise - log_det_tolerated)
        _, log_det = np.linalg.slogdet(noise_covariance @ np.linalg.inv(goal_prior.covariance))
        return 0.5 * (estimator.output_dim * np.log(expected_leverage) + log_det)

    def information(self, control_input: np.ndarray,
                    latest_output: Optional[np.ndarray]) -> float:
        """Left side of the information constraint at an input: the
        information about the coefficients, (D_y / 2) ln(1 + ℓ), for the
        task and predictive floors, and the information about (A, W) for
        -H[p_*]."""
        estimator = self.estimator
        if self.information_floor_type == "negentropy":
            return estimator.mutual_information(control_input, latest_output=latest_output)
        return 0.5 * estimator.output_dim * float(
            np.log1p(estimator.leverage(control_input, latest_output=latest_output)))

    def _information_at_multiplier(self, latest_output: Optional[np.ndarray],
                                   goal_prior: MultivariateNormal,
                                   multiplier: float,
                                   initial_input: Optional[np.ndarray] = None,
                                   multistart: Optional[bool] = None
                                   ) -> Tuple[float, np.ndarray]:
        """The MAP input for the multiplier and its information."""
        planned_input = self._solve_map(latest_output, goal_prior, multiplier, initial_input,
                                        multistart=multistart)
        return self.information(planned_input, latest_output), planned_input

    def _solve_multiplier(self, latest_output: Optional[np.ndarray],
                          goal_prior: MultivariateNormal,
                          max_multiplier: float = 50.0,
                          tolerance: float = 1e-3,
                          max_iterations: int = 15) -> Tuple[float, np.ndarray]:
        """The smallest multiplier β ≥ 0 for which the information of the
        MAP input reaches the floor of ``information_floor``.

        Three regimes follow from the KKT conditions:
          - inactive: the constraint holds at β = 0, so β = 0 (the leverage
            factor drops out of q*);
          - saturated: even max_multiplier falls short, so β = max_multiplier;
          - active: the unique root at which the information equals the
            floor, found by regula falsi (Illinois) on [0, max_multiplier]
            to |information - floor| < tolerance. Monotonicity in β gives
            uniqueness.

        The solve at β = 0 starts from the previous MAP input. The solve at
        max_multiplier also starts from the corners of the input box,
        because the probing term makes an input at rest under a symmetric
        posterior a stationary point, and the root search starts each
        solve from the MAP input of the bracket end with the more
        informative input, so that it follows the probing solution.
        """
        floor = self.information_floor(latest_output, goal_prior)
        self.last_information_floor = float(floor)

        # Inactive regime: try β = 0.
        information_low, input_low = self._information_at_multiplier(
            latest_output, goal_prior, 0.0, initial_input=self._last_planned_input)
        if information_low >= floor:
            self._last_planned_input = input_low
            return 0.0, input_low

        # Saturated regime: check max_multiplier.
        information_high, input_high = self._information_at_multiplier(
            latest_output, goal_prior, max_multiplier, initial_input=input_low, multistart=True)
        if information_high <= floor:
            self._last_planned_input = input_high
            return max_multiplier, input_high

        # Active regime: Illinois on f(β) = information - floor. If the
        # information jumps between solution branches, the search ends at
        # the feasible end of the bracket.
        multiplier_low, residual_low = 0.0, information_low - floor
        multiplier_high, residual_high = max_multiplier, information_high - floor
        last_side = 0
        for _ in range(max_iterations):
            multiplier = (multiplier_high - residual_high * (multiplier_high - multiplier_low)
                          / (residual_high - residual_low))
            information, planned_input = self._information_at_multiplier(
                latest_output, goal_prior, multiplier, initial_input=input_high)
            residual = information - floor
            if abs(residual) < tolerance:
                self._last_planned_input = planned_input
                return multiplier, planned_input
            if residual > 0:
                multiplier_high, residual_high, input_high = multiplier, residual, planned_input
                if last_side == 1:
                    residual_low *= 0.5
                last_side = 1
            else:
                multiplier_low, residual_low = multiplier, residual
                if last_side == -1:
                    residual_high *= 0.5
                last_side = -1
        self._last_planned_input = input_high
        return multiplier_high, input_high

    def _multiplier(self, latest_output: Optional[np.ndarray],
                    goal_prior: MultivariateNormal) -> float:
        """The multiplier of a plan: the fixed one, or the KKT solution."""
        if self.fixed_multiplier is not None:
            return float(self.fixed_multiplier)
        multiplier, _ = self._solve_multiplier(latest_output, goal_prior,
                                               max_multiplier=self.max_multiplier)
        return multiplier

    # ----- Cache management -----

    def reset(self) -> None:
        """Clear the warm start (call between unrelated trials)."""
        self._last_planned_input = None

    # ----- Public API -----

    def plan_one_step(self, latest_output: Optional[np.ndarray],
                      goal_prior: MultivariateNormal) -> Tuple[np.ndarray, float]:
        """The MAP input of a single slice and its KKT multiplier, given the
        most recent output (the head of the output buffer)."""
        if self.fixed_multiplier is not None:
            multiplier = float(self.fixed_multiplier)
            planned_input = self._solve_map(latest_output, goal_prior, multiplier,
                                            initial_input=self._last_planned_input,
                                            multistart=True if multiplier > 0 else None)
            self._last_planned_input = planned_input
        else:
            multiplier, planned_input = self._solve_multiplier(
                latest_output, goal_prior, max_multiplier=self.max_multiplier)
        self.last_multiplier = float(multiplier)
        return planned_input, multiplier

    def plan_horizon(self, latest_output: Optional[np.ndarray],
                     goal_prior: MultivariateNormal,
                     horizon: int = 1,
                     forward_backward_iterations: int = 3
                     ) -> Tuple[List[np.ndarray], List[float], List[MultivariateNormal]]:
        """Plan ``horizon`` inputs by forward-backward iteration.

        The multiplier is solved once per call, against the goal prior, and
        held fixed across all passes: it is a quantity of the time step,
        not of the horizon step. Each of the ``forward_backward_iterations``
        sweeps consists of
          - a forward pass: the MAP input of every slice, with the current
            intermediate goal of that slice as its goal;
          - a backward pass: the forward predictive of a slice times the
            intermediate goal of the next slice, a Gaussian product, becomes
            the intermediate goal of the slice.

        Returns the planned inputs, the multiplier of every slice (the same
        value ``horizon`` times) and the intermediate goals (the last is the
        goal prior).
        """
        if horizon == 1:
            planned_input, multiplier = self.plan_one_step(latest_output, goal_prior)
            return [planned_input], [multiplier], [goal_prior]

        multiplier = self._multiplier(latest_output, goal_prior)
        self.last_multiplier = float(multiplier)

        output_dim = self.estimator.output_dim
        intermediate_goals = [
            MultivariateNormal(mean=goal_prior.mean.copy(), covariance=goal_prior.covariance.copy())
            for _ in range(horizon)]
        planned_inputs = [np.zeros(self.estimator.input_dim) for _ in range(horizon)]
        predicted_means = [np.zeros(output_dim) for _ in range(horizon)]
        predicted_covariances = [np.eye(output_dim) for _ in range(horizon)]

        for _ in range(forward_backward_iterations):
            # Forward pass, with point-mass buffers at the predicted means.
            output = latest_output
            for step in range(horizon):
                planned_input = self._solve_map(output, intermediate_goals[step], multiplier,
                                                initial_input=self._last_planned_input)
                self._last_planned_input = planned_input
                planned_inputs[step] = planned_input
                predictive = self.estimator.predictive(planned_input, latest_output=output)
                predicted_means[step] = predictive.mean
                predicted_covariances[step] = predictive.covariance
                output = predictive.mean

            # Backward pass; the last intermediate goal is the goal prior.
            intermediate_goals[-1] = MultivariateNormal(
                mean=goal_prior.mean.copy(), covariance=goal_prior.covariance.copy())
            for step in range(horizon - 2, -1, -1):
                forward_precision = np.linalg.inv(predicted_covariances[step]
                                                  + 1e-8 * np.eye(output_dim))
                next_goal = intermediate_goals[step + 1]
                next_goal_precision = np.linalg.inv(next_goal.covariance)
                # Gaussian approximation of the product of the Student-t
                # predictive and the next intermediate goal.
                combined_covariance = np.linalg.inv(forward_precision + next_goal_precision)
                combined_mean = combined_covariance @ (
                    forward_precision @ predicted_means[step]
                    + next_goal_precision @ next_goal.mean)
                intermediate_goals[step] = MultivariateNormal(mean=combined_mean,
                                                              covariance=combined_covariance)

        return planned_inputs, [multiplier] * horizon, intermediate_goals
