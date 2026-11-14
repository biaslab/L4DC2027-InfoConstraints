"""Planner that passes backward messages through the predictive nodes.

With point-mass buffers, the negative logarithm of the surrogate control
posterior of the paper (Corollary 1) is, up to a constant,

    F(u) = sum_k  1/2 ||mu_k - m_*||^2_{S_*^-1}           tracking
                + 1/2 (1 + ell_k) tr(S_*^-1 Omega_eff)     caution
                - (beta D_y / 2) ln(1 + ell_k)            probing
                + c_u(u_k)                                 control prior

over the policy u = (u_1, ..., u_K), with mu_k = M^T phi(x_k) and
ell_k = phi(x_k)^T Lambda^-1 phi(x_k). The regressor x_k = (u_k, s_{k-1})
contains the buffer s_{k-1} of past inputs and outputs, whose future
entries are the predicted means of the earlier slices. F is the cost of
the dual MPC (up to a factor of two and the shape of the control prior).

``MessagePlanner`` finds the mode of the posterior by Gaussian message
passing on the factor graph of the planning model. A sweep consists of

  forward messages   point masses at the predicted means, rolled out
                     through the nonlinear model (messages 1 and 2 of the
                     paper's factor graph);
  backward messages  Gaussian messages on the buffers, in information form
                     (precision P_k, gradient p_k), passed from the last
                     slice to the first through the predictive nodes,
                     linearised at the forward messages (message 4). At
                     every slice, the goal message (3) joins the backward
                     message, and the input of the slice is marginalised
                     under the Gaussian message of the control prior and a
                     local Gaussian of the caution and probing terms;
  input update       each input becomes the mean of its conditional
                     posterior given the incoming buffer, an affine function
                     u_k + alpha k_k + K_k (s_{k-1} - s_k_nominal) of the
                     buffer (feedforward k_k, feedback gain K_k, step size
                     alpha), and the nonlinear model is rolled out again.

The control prior enters the backward messages through its Laplace
approximation at the current input. For the symmetric Beta, the precision
2(a-1)(1+u^2)/(1-u^2)^2 of that message grows without bound towards the
actuator limits, so a saturated input carries no feedback gain, and the
backward messages account for the limited braking authority of the
inputs. A backtracking line search on alpha keeps the inputs inside the
support of the prior and makes every sweep decrease F. The sweeps repeat
until the predicted decrease falls below a tolerance. Linearising the
predictive nodes (Gauss-Newton) makes this the iterative LQR algorithm,
and at convergence the plan is a stationary point of F.

The goal message attaches to every slice, as in the planning model of the
paper, and the backward message on y_k, combined with the goal message of
slice k, is the intermediate goal g_k of that slice.

With a parameter node in the estimator (``RandomWalk`` or
``ExponentialForgetting``), the equality nodes of the parameter chain
become that node, and slice k uses the belief over the parameters of its
own step: the posterior propagated through the node k - 1 times
(``ParameterEstimator.parameter_beliefs``). M, Lambda^-1 and Omega_eff in
F then depend on k. Without a node, every slice uses the posterior.

The information constraint applies either to the node of the next input
only (``constraint_nodes="next"``) or to every node of the horizon with one
shared multiplier (``"all"``, the earlier planner). With "next", the
probing term of slice 1 alone carries the multiplier, and the KKT search
solves it on the first input of the plan itself: it replans for every
trial multiplier, expands the bracket until the information of the first
input reaches the floor, and then runs regula falsi. The other control
beliefs, and hence the messages into the first input, do not carry the
multiplier, as Lemma 4.5 of the paper assumes. With "all", the multiplier
comes from the one-step program of ``Planner`` and weights every slice. A
fixed multiplier weights every slice in both cases, so that beta = 1 is
the expected free energy of the whole plan.

With beta > 0, the probing term makes F nonconvex. Under a posterior that
is symmetric in the sign of an input's effect, a plan at rest is then a
stationary point that Gauss-Newton sweeps cannot leave, since the gradient
of F vanishes there. The planner therefore also sweeps from the constant
plan at +-0.5 per input with the lowest F and keeps the better of the two
results (``symmetric_starts``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from controllers.common import (
    MultivariateNormal, buffer_with_head, flatten_buffer, predictive_noise_covariance)
from .planner import Planner


@dataclass
class MessagePlanner(Planner):
    """Mode of the control posterior over the horizon by forward and
    backward Gaussian messages (see the module docstring).

    The multiplier β comes from the KKT search of ``Planner`` (one per call,
    from the first slice), or from ``fixed_multiplier``. A horizon of one
    step uses the one-step program of ``Planner``. The control prior must
    provide ``cost`` and ``cost_derivatives`` (``SymmetricBeta`` does).

    max_sweeps        upper bound on the sweeps
    tolerance         the sweeps stop once the decrease of F predicted for
                      a full step is below tolerance (1 + |F|)
    symmetric_starts  with β > 0, also sweep from the best constant plan at
                      ±0.5 per input (see the module docstring)
    propagate_parameters  with a parameter node, every slice uses the belief
                      over the parameters of its own step; if False, every
                      slice uses the posterior, as the dual MPC does
    constraint_nodes  "next": the KKT multiplier weights the information of
                      the next input only and is solved on the plan; "all":
                      one multiplier from the one-step program weights every
                      slice (see the module docstring)
    initial_bracket   first trial multiplier of the bracket expansion
                      ("next"); the bracket grows fourfold up to
                      ``max_multiplier``. The expansion starts there at
                      every step, because an upward scan finds the smallest
                      root, whereas a bracket that starts at the previous
                      multiplier can end on another solution branch
    root_tolerance    the KKT search stops once |information - floor| is
                      below this value [nats]
    """
    max_sweeps: int = 100
    tolerance: float = 1e-7
    symmetric_starts: bool = True
    propagate_parameters: bool = True
    constraint_nodes: str = "all"
    initial_bracket: float = 1.0
    root_tolerance: float = 1e-3

    _previous_plan: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if not hasattr(self.control_prior, "cost_derivatives"):
            raise TypeError(f"{type(self.control_prior).__name__} has no cost_derivatives; "
                            "the message planner needs the Laplace approximation of the prior")
        estimator = self.estimator
        input_dim, output_dim = estimator.input_dim, estimator.output_dim
        self._input_buffer_length = input_dim * estimator.input_memory
        self._output_buffer_length = output_dim * estimator.output_memory
        buffer_length = self._input_buffer_length + self._output_buffer_length
        # Jacobian of the next buffer with respect to x = (u, buffer), apart
        # from the rows of the new output: the shift of both buffers.
        shift = np.zeros((buffer_length, input_dim + buffer_length))
        if estimator.input_memory > 0:
            shift[:input_dim, :input_dim] = np.eye(input_dim)
            for i in range(1, estimator.input_memory):
                shift[i * input_dim:(i + 1) * input_dim,
                      i * input_dim:(i + 1) * input_dim] = np.eye(input_dim)
        offset = self._input_buffer_length
        for j in range(1, estimator.output_memory):
            shift[offset + j * output_dim:offset + (j + 1) * output_dim,
                  input_dim + offset + (j - 1) * output_dim:
                  input_dim + offset + j * output_dim] = np.eye(output_dim)
        self._buffer_shift = shift
        if self.constraint_nodes not in ("next", "all"):
            raise ValueError(f"unknown constraint_nodes '{self.constraint_nodes}'")
        self.last_sweeps = 0
        self.last_converged = True
        self.last_root_plans = 0          # plans of the last KKT search ("next")
        self.last_saturated = False       # the last KKT search stopped at max_multiplier

    # ----- Cache management -----

    def reset(self) -> None:
        """Clear the warm starts (call at a new target)."""
        super().reset()
        self._previous_plan = None

    # ----- Public API -----

    def plan_horizon(self, latest_output: Optional[np.ndarray],
                     goal_prior: MultivariateNormal,
                     horizon: int = 1,
                     forward_backward_iterations: int = 3
                     ) -> Tuple[List[np.ndarray], List[float], List[MultivariateNormal]]:
        """Plan ``horizon`` inputs. Returns the inputs, the multiplier of
        every slice (one per call) and the intermediate goals, i.e. the
        goal message of each slice times the backward message on its
        output. ``forward_backward_iterations`` is not used: the sweeps run
        until convergence, at most ``max_sweeps`` times."""
        if horizon == 1:
            return super().plan_horizon(latest_output, goal_prior, horizon=1)
        if self.fixed_multiplier is None and self.constraint_nodes == "next":
            multiplier, plan, goals = self._plan_with_next_constraint(
                latest_output, goal_prior, horizon)
            multipliers = [multiplier] + [0.0] * (horizon - 1)
        else:
            multiplier = self._multiplier(latest_output, goal_prior)
            model = self._slice_model(latest_output, goal_prior, multiplier)
            plan, goals = self._plan(model, self._warm_start(horizon), multiplier > 0.0)
            multipliers = [multiplier] * horizon
        self.last_multiplier = float(multiplier)
        self._previous_plan = plan
        return [control_input.copy() for control_input in plan], multipliers, goals

    def _plan(self, model, start, symmetric: bool):
        """Sweeps from ``start``, and also from the symmetric starts if
        ``symmetric`` (and ``symmetric_starts``)."""
        if symmetric and self.symmetric_starts:
            return self._sweeps_with_starts(model, start)
        return self._sweeps(model, start)

    def _plan_with_next_constraint(self, latest_output, goal_prior: MultivariateNormal,
                                   horizon: int):
        """KKT multiplier of the information constraint on the node of the
        next input, solved on the plan. Returns the multiplier, the plan and
        the intermediate goals.

        The plan at multiplier 0 meets the floor (inactive constraint), or
        the bracket grows from ``initial_bracket`` fourfold until the first
        input of the plan reaches the floor. If even ``max_multiplier``
        falls short, the search stops there (``last_saturated``). Otherwise
        regula falsi (Illinois) brackets the smallest multiplier at which
        the information of the first input reaches the floor. The
        information of the mode is nondecreasing in the multiplier but can
        jump between solution branches; the search then ends at the
        feasible end of the bracket."""
        floor = self.information_floor(latest_output, goal_prior)
        self.last_information_floor = float(floor)
        self.last_saturated = False
        plans = 0

        def evaluate(multiplier, start, symmetric):
            nonlocal plans
            plans += 1
            model = self._slice_model(latest_output, goal_prior, multiplier, probing_slices=1)
            plan, goals = self._plan(model, start, symmetric and multiplier > 0.0)
            return self.information(plan[0], latest_output) - floor, plan, goals

        residual, plan, goals = evaluate(0.0, self._warm_start(horizon), False)
        if residual >= 0.0:
            self.last_root_plans = plans
            return 0.0, plan, goals
        low = (0.0, residual)
        multiplier = min(self.initial_bracket, self.max_multiplier)
        rest_plan = plan
        while True:
            residual, plan, goals = evaluate(multiplier, rest_plan, True)
            if residual >= 0.0:
                high = (multiplier, residual, plan, goals)
                break
            low = (multiplier, residual)
            if multiplier >= self.max_multiplier:
                self.last_saturated = True
                self.last_root_plans = plans
                return multiplier, plan, goals
            multiplier = min(4.0 * multiplier, self.max_multiplier)
        if high[1] < self.root_tolerance:
            self.last_root_plans = plans
            return high[0], high[2], high[3]
        last_side = 0
        for _ in range(15):
            (multiplier_low, residual_low), (multiplier_high, residual_high) = low, high[:2]
            multiplier = (multiplier_high - residual_high * (multiplier_high - multiplier_low)
                          / (residual_high - residual_low))
            residual, plan, goals = evaluate(multiplier, high[2], False)
            if abs(residual) < self.root_tolerance:
                self.last_root_plans = plans
                return multiplier, plan, goals
            if residual > 0.0:
                high = (multiplier, residual, plan, goals)
                if last_side == 1:
                    low = (low[0], 0.5 * low[1])
                last_side = 1
            else:
                low = (multiplier, residual)
                if last_side == -1:
                    high = (high[0], 0.5 * high[1], high[2], high[3])
                last_side = -1
        self.last_root_plans = plans
        return high[0], high[2], high[3]

    # ----- Model of one slice -----

    def _slice_model(self, latest_output, goal_prior: MultivariateNormal,
                     multiplier: float, probing_slices: Optional[int] = None) -> SimpleNamespace:
        """The quantities of the slice terms of F: goal, initial buffer and
        probing weight, and the parameter moments of every slice
        (``_slice_parameters``). The probing weight applies to the first
        ``probing_slices`` slices, or to all of them if None."""
        estimator = self.estimator
        posterior = estimator.posterior
        goal_precision = np.linalg.inv(goal_prior.covariance)
        initial_buffer = np.concatenate([
            flatten_buffer(estimator.input_buffer, estimator.input_memory, estimator.input_dim),
            flatten_buffer(buffer_with_head(estimator.output_buffer, latest_output,
                                            estimator.output_memory),
                           estimator.output_memory, estimator.output_dim)])
        model = SimpleNamespace(goal_precision=goal_precision, goal_mean=goal_prior.mean,
                                initial_buffer=initial_buffer,
                                probing_weight=float(multiplier) * estimator.output_dim / 2.0,
                                probing_slices=probing_slices,
                                parameter_belief=posterior, slices=[])
        self._slice_parameters(model, 0)
        return model

    def _slice_parameters(self, model, k: int) -> SimpleNamespace:
        """Coefficient mean, coefficient covariance and caution weight of
        slice k (k = 0 is the next step), from the belief over the
        parameters of that step. The beliefs are propagated through the
        parameter node on first use and cached in the model."""
        dynamics = self.estimator.parameter_dynamics if self.propagate_parameters else None
        while len(model.slices) <= k:
            if model.slices and dynamics is None:
                model.slices.append(model.slices[-1])
                continue
            if model.slices:
                model.parameter_belief = dynamics(model.parameter_belief)
            belief = model.parameter_belief
            model.slices.append(SimpleNamespace(
                coefficient_mean=belief.coefficient_mean,
                coefficient_covariance=belief.coefficient_covariance,
                caution_weight=0.5 * float(np.trace(model.goal_precision
                                                    @ predictive_noise_covariance(belief)))))
        return model.slices[k]

    @staticmethod
    def _probing_weight(model, k: int) -> float:
        """Weight of the probing term of slice k (k = 0 is the next step)."""
        if model.probing_slices is None or k < model.probing_slices:
            return model.probing_weight
        return 0.0

    def _slice_cost(self, model, k: int, control_input, buffer) -> Tuple[float, np.ndarray]:
        """Term of slice k in F and the predicted output mu_k."""
        parameters = self._slice_parameters(model, k)
        probing_weight = self._probing_weight(model, k)
        features = self.estimator.feature_map(np.concatenate([control_input, buffer]))
        predicted_output = parameters.coefficient_mean.T @ features
        leverage = features @ parameters.coefficient_covariance @ features
        goal_error = predicted_output - model.goal_mean
        cost = (0.5 * goal_error @ model.goal_precision @ goal_error
                + parameters.caution_weight * (1.0 + leverage)
                - probing_weight * np.log1p(leverage)
                + self.control_prior.cost(control_input))
        return cost, predicted_output

    def _next_buffer(self, buffer, control_input, predicted_output) -> np.ndarray:
        """Buffer of the next slice: the new input and output in front."""
        input_dim, output_dim = self.estimator.input_dim, self.estimator.output_dim
        input_length, output_length = self._input_buffer_length, self._output_buffer_length
        parts = []
        if input_length:
            parts += [control_input, buffer[:input_length - input_dim]]
        if output_length:
            parts += [predicted_output,
                      buffer[input_length:input_length + output_length - output_dim]]
        return np.concatenate(parts) if parts else np.zeros(0)

    def _linearise(self, model, k: int, control_input, buffer):
        """Gradient and Gauss-Newton Hessian of the term of slice k with
        respect to x = (u, s), and the Jacobian of the next buffer.
        Curvature of the caution and probing terms that is not positive
        semidefinite is dropped, so that every backward message is a proper
        Gaussian."""
        estimator = self.estimator
        input_dim = estimator.input_dim
        parameters = self._slice_parameters(model, k)
        probing_weight = self._probing_weight(model, k)
        regressor = np.concatenate([control_input, buffer])
        features = estimator.feature_map(regressor)
        feature_jacobian = estimator.feature_map.jacobian(regressor)
        mean_jacobian = parameters.coefficient_mean.T @ feature_jacobian   # d mu / dx
        weighted_features = parameters.coefficient_covariance @ features
        leverage = features @ weighted_features
        leverage_gradient = 2.0 * feature_jacobian.T @ weighted_features   # d ell / dx
        # F depends on the leverage through c1 (1 + ell) - c2 ln(1 + ell).
        leverage_slope = parameters.caution_weight - probing_weight / (1.0 + leverage)
        probing_curvature = probing_weight / (1.0 + leverage)**2
        cost_gradient = (mean_jacobian.T @ (model.goal_precision
                                            @ (parameters.coefficient_mean.T @ features
                                               - model.goal_mean))
                         + leverage_slope * leverage_gradient)
        cost_hessian = (mean_jacobian.T @ model.goal_precision @ mean_jacobian
                        + max(leverage_slope, 0.0) * 2.0 * feature_jacobian.T
                        @ parameters.coefficient_covariance @ feature_jacobian
                        + probing_curvature * np.outer(leverage_gradient, leverage_gradient))
        prior_gradient, prior_curvature = self.control_prior.cost_derivatives(control_input)
        cost_gradient[:input_dim] += prior_gradient
        cost_hessian[:input_dim, :input_dim] += np.diag(prior_curvature)
        buffer_jacobian = self._buffer_shift.copy()
        if self._output_buffer_length:
            buffer_jacobian[self._input_buffer_length:
                            self._input_buffer_length + estimator.output_dim] = mean_jacobian
        return cost_gradient, cost_hessian, buffer_jacobian

    # ----- Sweeps -----

    def _warm_start(self, horizon: int) -> np.ndarray:
        """The previous plan shifted by one step, or zeros."""
        if self._previous_plan is None or len(self._previous_plan) != horizon:
            return np.zeros((horizon, self.estimator.input_dim))
        return np.vstack([self._previous_plan[1:], self._previous_plan[-1:]])

    def _rollout(self, model, plan):
        """F and the buffers s_0, ..., s_K of the plan."""
        buffer, buffers, objective = model.initial_buffer, [model.initial_buffer], 0.0
        for k, control_input in enumerate(plan):
            cost, predicted_output = self._slice_cost(model, k, control_input, buffer)
            objective += cost
            buffer = self._next_buffer(buffer, control_input, predicted_output)
            buffers.append(buffer)
        return objective, buffers

    def _backward(self, linearisations, damping: float):
        """Backward messages from the last slice to the first. Returns the
        feedforward and feedback terms of the input updates, the decrease
        of F predicted for a full step (linear and quadratic part), and the
        backward message (precision, gradient) on the buffer after every
        slice, or None if the input precision of a slice is not positive
        definite (or not finite)."""
        input_dim = self.estimator.input_dim
        buffer_length = self._input_buffer_length + self._output_buffer_length
        message_precision = np.zeros((buffer_length, buffer_length))
        message_gradient = np.zeros(buffer_length)
        horizon = len(linearisations)
        feedforward, feedback = [None] * horizon, [None] * horizon
        messages = [None] * horizon
        predicted_decrease = np.zeros(2)
        for k in reversed(range(horizon)):
            cost_gradient, cost_hessian, buffer_jacobian = linearisations[k]
            messages[k] = (message_precision, message_gradient)
            # Gradient and Hessian of the cost-to-go with respect to x = (u, s).
            gradient = cost_gradient + buffer_jacobian.T @ message_gradient
            hessian = cost_hessian + buffer_jacobian.T @ message_precision @ buffer_jacobian
            gradient_input, gradient_buffer = gradient[:input_dim], gradient[input_dim:]
            hessian_input = hessian[:input_dim, :input_dim]
            hessian_input_buffer = hessian[:input_dim, input_dim:]
            hessian_buffer = hessian[input_dim:, input_dim:]
            try:
                cholesky = cho_factor(hessian_input + damping * np.eye(input_dim))
            except (np.linalg.LinAlgError, ValueError):     # not PD, or not finite
                return None
            feedforward[k] = -cho_solve(cholesky, gradient_input)
            feedback[k] = -cho_solve(cholesky, hessian_input_buffer)
            gain, step = feedback[k], feedforward[k]
            message_precision = (hessian_buffer + gain.T @ hessian_input @ gain
                                 + gain.T @ hessian_input_buffer + hessian_input_buffer.T @ gain)
            message_precision = 0.5 * (message_precision + message_precision.T)
            message_gradient = (gradient_buffer + gain.T @ hessian_input @ step
                                + gain.T @ gradient_input + hessian_input_buffer.T @ step)
            predicted_decrease += [step @ gradient_input, 0.5 * step @ hessian_input @ step]
        return feedforward, feedback, predicted_decrease, messages

    def _forward(self, model, plan, buffers, feedforward, feedback, step_size: float):
        """Roll out the updated inputs. Returns F, the inputs and the
        buffers, with F infinite if an input leaves the prior's support."""
        buffer, new_buffers, new_plan = model.initial_buffer, [model.initial_buffer], []
        objective = 0.0
        for k in range(len(plan)):
            control_input = (plan[k] + step_size * feedforward[k]
                             + feedback[k] @ (buffer - buffers[k]))
            cost, predicted_output = self._slice_cost(model, k, control_input, buffer)
            if not np.isfinite(cost):
                return np.inf, None, None
            objective += cost
            buffer = self._next_buffer(buffer, control_input, predicted_output)
            new_plan.append(control_input)
            new_buffers.append(buffer)
        return objective, np.array(new_plan), new_buffers

    def _sweeps(self, model, plan):
        objective, buffers = self._rollout(model, plan)
        if not np.isfinite(objective):
            plan = np.zeros_like(plan)
            objective, buffers = self._rollout(model, plan)
        damping, converged, sweeps = 0.0, False, 0
        backward = None
        while sweeps < self.max_sweeps:
            sweeps += 1
            linearisations = [self._linearise(model, k, plan[k], buffers[k])
                              for k in range(len(plan))]
            backward = self._backward(linearisations, damping)
            while backward is None and damping < 1e10:
                damping = max(10.0 * damping, 1e-6)
                backward = self._backward(linearisations, damping)
            if backward is None:
                break
            feedforward, feedback, predicted_decrease, _ = backward
            if -(predicted_decrease[0] + predicted_decrease[1]) < self.tolerance * (1.0 + abs(objective)):
                converged = True
                break
            for step_size in 0.5 ** np.arange(12):
                new_objective, new_plan, new_buffers = self._forward(
                    model, plan, buffers, feedforward, feedback, step_size)
                if new_objective < objective + 1e-4 * (step_size * predicted_decrease[0]
                                                       + step_size**2 * predicted_decrease[1]):
                    plan, buffers, objective = new_plan, new_buffers, new_objective
                    damping = 0.0 if damping < 1e-5 else damping / 10.0
                    break
            else:
                damping = max(10.0 * damping, 1e-6)
                if damping > 1e10:
                    break
        if not converged:
            linearisations = [self._linearise(model, k, plan[k], buffers[k])
                              for k in range(len(plan))]
            backward = self._backward(linearisations, max(damping, 1e-6))
        self.last_sweeps, self.last_converged, self.last_objective = sweeps, converged, objective
        goals = self._intermediate_goals(model, buffers,
                                         backward[3] if backward is not None else None)
        return plan, goals

    def _sweeps_with_starts(self, model, plan):
        """Sweeps from the warm start and from the constant plan at ±0.5 per
        input with the lowest F; returns the plan with the lower F after the
        sweeps."""
        warm_plan, warm_goals = self._sweeps(model, plan)
        warm_run = (self.last_objective, self.last_sweeps, self.last_converged)
        starts = [np.tile(0.5 * np.asarray(signs), (len(plan), 1))
                  for signs in product((-1.0, 1.0), repeat=self.estimator.input_dim)]
        start = starts[int(np.argmin([self._rollout(model, constant_plan)[0]
                                      for constant_plan in starts]))]
        symmetric_plan, symmetric_goals = self._sweeps(model, start)
        if warm_run[0] <= self.last_objective:
            self.last_objective, self.last_sweeps, self.last_converged = warm_run
            return warm_plan, warm_goals
        return symmetric_plan, symmetric_goals

    def _intermediate_goals(self, model, buffers, messages) -> List[MultivariateNormal]:
        """g_k ∝ p_*(y_k) times the backward message on y_k, with the
        older buffer entries at their predicted values."""
        goal = MultivariateNormal(mean=model.goal_mean.copy(),
                                  covariance=np.linalg.inv(model.goal_precision))
        if messages is None or not self._output_buffer_length:
            return [goal] * (len(buffers) - 1)
        output_dim, offset = self.estimator.output_dim, self._input_buffer_length
        output_slice = slice(offset, offset + output_dim)
        goals = []
        for k, (message_precision, message_gradient) in enumerate(messages):
            output_precision = message_precision[output_slice, output_slice]
            output_gradient = message_gradient[output_slice]
            predicted_output = buffers[k + 1][output_slice]
            covariance = np.linalg.inv(model.goal_precision + output_precision)
            goals.append(MultivariateNormal(
                mean=covariance @ (model.goal_precision @ model.goal_mean
                                   + output_precision @ predicted_output - output_gradient),
                covariance=0.5 * (covariance + covariance.T)))
        return goals
