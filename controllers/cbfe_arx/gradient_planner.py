"""Planner that descends the gradient of the joint control posterior.

The message planner (``MessagePlanner``) finds the mode of the surrogate
control posterior by passing messages between the factors of the planning
graph: one control belief at a time, through Gaussian backward messages.
This planner treats the posterior of the whole policy instead as one joint
density over all inputs,

    q(u_1, ..., u_K) ∝ exp(-F(u)),

where F is the objective of the message planner (the same slice model,
point-mass buffers, multiplier and control prior), and ascends its
log-density with gradient steps on all inputs at once,

    u ← u - α ∇F(u),

with an Armijo backtracking line search on α. The search starts every
iteration from ``step_growth`` times the last accepted step, keeps the
inputs inside the support of the control prior (F is infinite outside it)
and makes every iteration decrease F.

The gradient comes from reverse-mode differentiation of F through the
rollout of the plan. Reverse mode passes back the gradient of the
cost-to-go with the later inputs held fixed (the adjoint), a vector on the
buffer. The backward messages of the message planner carry the gradient
and the precision of the cost-to-go minimised over the later inputs, from
which every input update follows by a local Gauss-Newton step. The two
planners therefore differ only in how they use the backward pass: one
gradient step on the joint density, or one Gaussian message per factor.

The descent stops when F reaches ``target_objective`` (if set; used to time
the planner to a given accuracy), when the gradient norm falls below
``gradient_tolerance``, when the line search stalls, or after
``max_iterations`` iterations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from controllers.common import MultivariateNormal
from .message_planner import MessagePlanner


@dataclass
class GradientPlanner(MessagePlanner):
    """Mode of the joint control posterior over the horizon by gradient
    descent (see the module docstring). The slice model, the warm start,
    the multiplier and the symmetric starts are those of ``MessagePlanner``;
    only the sweeps differ.

    max_iterations      upper bound on the gradient steps per plan
    gradient_tolerance  the descent stops once the largest entry of the
                        gradient is below this value
    target_objective    if set, the descent stops once F is at or below it
    armijo              sufficient-decrease constant of the line search
    step_growth         every line search starts from this multiple of the
                        last accepted step
    backtracking        factor by which the line search shortens the step
    record_trace        if True, keep the processor time, F and the first
                        input after every iteration in ``last_trace``
    """
    max_iterations: int = 100_000
    gradient_tolerance: float = 1e-6
    target_objective: Optional[float] = None
    armijo: float = 1e-4
    step_growth: float = 2.0
    backtracking: float = 0.5
    record_trace: bool = False

    _step_size: float = field(default=1.0, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self.last_trace = None
        self.last_gradient_norm = np.nan
        self.last_gradient_evaluations = 0
        self.last_objective_evaluations = 0

    def reset(self) -> None:
        """Clear the warm starts and the step size (call at a new target)."""
        super().reset()
        self._step_size = 1.0

    # ----- Gradient of F -----

    def _slice_gradient(self, model, k: int, control_input, buffer):
        """Gradient of the term of slice k with respect to x = (u, s), and
        the Jacobian of the next buffer (the first-order part of
        ``MessagePlanner._linearise``)."""
        estimator = self.estimator
        parameters = self._slice_parameters(model, k)
        regressor = np.concatenate([control_input, buffer])
        features = estimator.feature_map(regressor)
        feature_jacobian = estimator.feature_map.jacobian(regressor)
        mean_jacobian = parameters.coefficient_mean.T @ feature_jacobian   # d mu / dx
        weighted_features = parameters.coefficient_covariance @ features
        leverage = features @ weighted_features
        leverage_slope = parameters.caution_weight - model.probing_weight / (1.0 + leverage)
        cost_gradient = (mean_jacobian.T @ (model.goal_precision
                                            @ (parameters.coefficient_mean.T @ features
                                               - model.goal_mean))
                         + leverage_slope * 2.0 * feature_jacobian.T @ weighted_features)
        cost_gradient[:estimator.input_dim] += self.control_prior.cost_derivatives(control_input)[0]
        buffer_jacobian = self._buffer_shift.copy()
        if self._output_buffer_length:
            buffer_jacobian[self._input_buffer_length:
                            self._input_buffer_length + estimator.output_dim] = mean_jacobian
        return cost_gradient, buffer_jacobian

    def _gradient(self, model, plan, buffers) -> np.ndarray:
        """∇F with respect to every input of the plan, by reverse-mode
        differentiation through the rollout: the adjoint on the buffer
        after slice k is the gradient of the later slice terms with the
        later inputs held fixed."""
        input_dim = self.estimator.input_dim
        adjoint = np.zeros(self._input_buffer_length + self._output_buffer_length)
        gradient = np.empty_like(plan)
        for k in reversed(range(len(plan))):
            cost_gradient, buffer_jacobian = self._slice_gradient(model, k, plan[k], buffers[k])
            total = cost_gradient + buffer_jacobian.T @ adjoint
            gradient[k] = total[:input_dim]
            adjoint = total[input_dim:]
        return gradient

    # ----- Descent -----

    def _sweeps(self, model, plan):
        """Gradient descent from ``plan``; the counterpart of the message
        sweeps, with ``last_sweeps`` the number of gradient steps."""
        objective, buffers = self._rollout(model, plan)
        if not np.isfinite(objective):
            plan = np.zeros_like(plan)
            objective, buffers = self._rollout(model, plan)
        step_size = self._step_size
        iterations, gradient_evaluations, objective_evaluations = 0, 0, 1
        converged, gradient_norm = False, np.nan
        trace = [(time.process_time(), objective, float(plan[0, 0]))] if self.record_trace else None
        while True:
            if self.target_objective is not None and objective <= self.target_objective:
                converged = True
                break
            gradient = self._gradient(model, plan, buffers)
            gradient_evaluations += 1
            gradient_norm = float(np.max(np.abs(gradient)))
            if gradient_norm <= self.gradient_tolerance:
                converged = True
                break
            if iterations >= self.max_iterations:
                break
            squared_norm = float(np.sum(gradient * gradient))
            step_size *= self.step_growth
            while step_size > 1e-20:
                candidate = plan - step_size * gradient
                new_objective, new_buffers = self._rollout(model, candidate)
                objective_evaluations += 1
                if new_objective <= objective - self.armijo * step_size * squared_norm:
                    break
                step_size *= self.backtracking
            else:                                   # the line search stalled
                step_size = self._step_size
                break
            plan, buffers, objective = candidate, new_buffers, new_objective
            iterations += 1
            if trace is not None:
                trace.append((time.process_time(), objective, float(plan[0, 0])))
        self._step_size = step_size
        self.last_sweeps, self.last_converged, self.last_objective = iterations, converged, objective
        self.last_gradient_norm = gradient_norm
        self.last_gradient_evaluations = gradient_evaluations
        self.last_objective_evaluations = objective_evaluations
        self.last_trace = np.array(trace) if trace is not None else None
        # Without backward messages, the goal of every slice is the goal prior.
        goal = MultivariateNormal(mean=model.goal_mean.copy(),
                                  covariance=np.linalg.inv(model.goal_precision))
        return plan, [goal] * len(plan)
