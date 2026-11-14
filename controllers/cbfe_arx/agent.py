"""The CBFE-ARX agent: one planner over one horizon, towards the goal prior.

    Agent.update(output, control_input)   closed-form posterior update
    Agent.plan(latest_output)             the next input
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from controllers.common import MultivariateNormal, ParameterEstimator
from .distributions import SymmetricBeta
from .message_planner import MessagePlanner
from .planner import Planner


@dataclass
class Agent:
    """Active inference agent with information-constrained Bethe planning.

    estimator       the ARX model: posterior, feature map and buffers
    control_prior   prior on the inputs (``SymmetricBeta`` or
                    ``Exponential``)
    goal_prior      the goal prior p_* over the outputs; settable between
                    calls, e.g. at a new target
    horizon         number of slices the planner plans over
    forward_backward_iterations  sweeps of the one-step ``Planner`` over the
                    horizon (the message planner sweeps until convergence)
    planner         ``Planner`` or ``MessagePlanner``; by default a
                    ``Planner`` on the estimator and the control prior
    receding_horizon  if True, ``plan`` executes the cached plan and only
                    replans when the cache is exhausted or the one-step
                    prediction error exceeds ``replan_threshold``; if False,
                    it replans at every call
    replan_threshold  output-space distance between the latest output and
                    its prediction that triggers a replan (None: replan only
                    when the cache is exhausted, every ``horizon`` steps)
    """
    estimator: ParameterEstimator
    control_prior: object
    goal_prior: MultivariateNormal
    horizon: int = 3
    forward_backward_iterations: int = 1
    planner: Optional[Planner] = None
    receding_horizon: bool = False
    replan_threshold: Optional[float] = None

    _cached_plan: List[np.ndarray] = field(default_factory=list, init=False)
    _last_predicted_mean: Optional[np.ndarray] = field(default=None, init=False)

    def __post_init__(self):
        if self.planner is None:
            self.planner = Planner(estimator=self.estimator, control_prior=self.control_prior)

    # ============================================================
    # Public API
    # ============================================================

    def update(self, output: np.ndarray, control_input: np.ndarray) -> None:
        """Closed-form Bayesian update of the matrix-normal Wishart
        posterior with the output that followed the input."""
        self.estimator.update(output, control_input)

    def plan(self, latest_output: Optional[np.ndarray]) -> np.ndarray:
        """The next input, given the most recent output."""
        if not self.receding_horizon:
            return self._plan_horizon(latest_output)[0]

        # Receding horizon: execute the next cached input; replan when the
        # cache is exhausted or the one-step prediction error is too large.
        if self._needs_replan(latest_output):
            self._cached_plan = list(self._plan_horizon(latest_output))
            self._last_predicted_mean = None
        control_input = self._cached_plan.pop(0)
        if self.replan_threshold is not None:
            self._last_predicted_mean = self.estimator.predictive(
                control_input, latest_output=latest_output).mean
        return control_input

    def reset_planner_cache(self) -> None:
        """Clear the cached plan and the planner's warm starts (e.g. at a
        new target)."""
        self._cached_plan = []
        self._last_predicted_mean = None
        if self.planner is not None:
            self.planner.reset()

    # ============================================================
    # Planning
    # ============================================================

    def _plan_horizon(self, latest_output: Optional[np.ndarray]) -> List[np.ndarray]:
        planned_inputs, _, _ = self.planner.plan_horizon(
            latest_output=latest_output,
            goal_prior=self.goal_prior,
            horizon=self.horizon,
            forward_backward_iterations=self.forward_backward_iterations)
        return planned_inputs

    def _needs_replan(self, latest_output: Optional[np.ndarray]) -> bool:
        """Cache exhaustion, or a prediction error above the threshold."""
        if not self._cached_plan:
            return True
        if (self.replan_threshold is None
                or self._last_predicted_mean is None
                or latest_output is None):
            return False
        prediction_error = np.asarray(latest_output) - self._last_predicted_mean
        return float(np.linalg.norm(prediction_error)) > self.replan_threshold

    # ============================================================
    # Convenience constructor
    # ============================================================

    @classmethod
    def build(cls,
              input_dim: int,
              output_dim: int,
              input_memory: int,
              output_memory: int,
              goal_prior: MultivariateNormal,
              horizon: int = 3,
              num_harmonics: int = 2,
              half_period: float | np.ndarray = 1.0,
              control_prior_shape: float = 2.0,
              prior_degrees_of_freedom: Optional[float] = None,
              prior_precision_scale: float = 1e-2,
              prior_noise_scale: float = 1.0,
              prior_mean_scale: float = 0.1,
              forward_backward_iterations: int = 3,
              fixed_multiplier: Optional[float] = None,
              max_multiplier: float = 50.0,
              receding_horizon: bool = False,
              replan_threshold: Optional[float] = None,
              control_prior: Optional[object] = None,
              planner_type: str = "messages",
              max_sweeps: int = 100,
              information_floor_type: str = "task",
              constraint_nodes: str = "all",
              parameter_dynamics=None,
              polynomial_features: bool = True) -> "Agent":
        """Agent with a weakly informative prior.

        input_dim, output_dim        dimensions of u and y
        input_memory, output_memory  past inputs and outputs in the regressor
        goal_prior                   goal prior over the outputs
        horizon                      planning horizon [slices]
        num_harmonics                harmonic orders of the Fourier features
                                     (0: affine features)
        half_period                  half-period of the features, one value
                                     or one per regressor coordinate
        control_prior_shape          shape of the symmetric Beta control prior
        prior_degrees_of_freedom     ν of the Wishart prior (default
                                     output_dim + 2)
        prior_precision_scale, prior_noise_scale, prior_mean_scale
                                     Λ_0 = scale I, Ω_0 = scale I and the
                                     standard deviation of the random M_0
        forward_backward_iterations  sweeps of the one-step planner
        fixed_multiplier             if set, the multiplier β of every plan
                                     instead of the KKT solution (0 removes
                                     the information constraint)
        max_multiplier               upper end of the KKT search, where β
                                     saturates when the floor is unattainable
        planner_type                 "messages" (default) passes Gaussian
                                     backward messages through the linearised
                                     model (``MessagePlanner``); "sweeps"
                                     solves one-step programs towards
                                     intermediate goals (``Planner``)
        max_sweeps                   cap on the sweeps of the message planner
        information_floor_type       "task" compares the parameter-induced
                                     spread of the predictions at the
                                     goal-reaching inputs with the goal
                                     prior and switches the constraint off
                                     once that spread fits inside the goal;
                                     "predictive" makes the same comparison
                                     between predictive covariances that
                                     include the noise, so that the floor
                                     stays attainable; "negentropy" is
                                     -H[p_*], the floor of the earlier
                                     studies
        polynomial_features          False omits the constant and linear
                                     features, leaving the Fourier terms
        constraint_nodes             "next": the information constraint
                                     applies to the node of the next input,
                                     with the multiplier solved on the plan;
                                     "all": one multiplier, from the
                                     one-step program, weights every node
                                     (message planner only)
        parameter_dynamics           parameter node between consecutive
                                     steps (``RandomWalk`` or
                                     ``ExponentialForgetting`` of
                                     ``controllers.common``); None keeps the
                                     parameters constant. The message
                                     planner propagates the parameter
                                     belief through the node over the
                                     horizon; the sweeps planner holds it
                                     fixed
        """
        estimator = ParameterEstimator.from_dimensions(
            input_dim=input_dim, output_dim=output_dim,
            input_memory=input_memory, output_memory=output_memory,
            num_harmonics=num_harmonics, half_period=half_period,
            prior_degrees_of_freedom=prior_degrees_of_freedom,
            prior_precision_scale=prior_precision_scale,
            prior_noise_scale=prior_noise_scale,
            prior_mean_scale=prior_mean_scale,
            parameter_dynamics=parameter_dynamics,
            polynomial_features=polynomial_features)
        if control_prior is None:
            control_prior = SymmetricBeta(dimension=input_dim, shape=control_prior_shape)

        if planner_type == "sweeps":
            planner = Planner(estimator=estimator, control_prior=control_prior,
                              fixed_multiplier=fixed_multiplier,
                              max_multiplier=max_multiplier,
                              information_floor_type=information_floor_type)
        elif planner_type == "messages":
            planner = MessagePlanner(estimator=estimator, control_prior=control_prior,
                                     fixed_multiplier=fixed_multiplier,
                                     max_multiplier=max_multiplier,
                                     information_floor_type=information_floor_type,
                                     max_sweeps=max_sweeps,
                                     constraint_nodes=constraint_nodes)
        else:
            raise ValueError(f"unknown planner_type '{planner_type}'")

        return cls(estimator=estimator,
                   control_prior=control_prior,
                   goal_prior=goal_prior,
                   horizon=horizon,
                   forward_backward_iterations=forward_backward_iterations,
                   planner=planner,
                   receding_horizon=receding_horizon,
                   replan_threshold=replan_threshold)
