"""Constrained Bethe free energy agent on a Bayesian ARX model (CBFE-ARX).

The agent regresses the next output on the current input and a buffer of
past inputs and outputs through a fixed feature map, keeps a matrix-normal
Wishart posterior over the regression coefficients and the noise
precision, and selects inputs by minimising a Bethe free energy with an
information constraint on the expected parameter information gain. The
floor of that constraint is the entropy by which the parameter-induced
spread of the predictions at the goal-reaching inputs exceeds the goal
prior, so the constraint switches off once the model is accurate enough
for the goal. The KKT multiplier β of the constraint is solved per
planning call, or fixed with ``fixed_multiplier`` (``fixed_multiplier=0``
removes the constraint).

The agent has a single level: one planner over one horizon towards the
goal prior. The model (posterior, feature map, estimator, goal
distribution) lives in ``controllers.common``, which this agent shares with the
dual MPC and MPC-X. This package adds the planners, the agent cycle and
the control priors. Three planners find the mode of the control posterior
over the horizon:

``MessagePlanner``  (default) Gaussian backward messages through the
                    linearised predictive nodes, with the control prior
                    entering through its Laplace approximation; it sweeps
                    until the plan is a stationary point of the horizon
                    objective.
``Planner``         one-step programs towards intermediate goals that
                    combine the forward message of a slice with the goal of
                    the next one (``planner_type="sweeps"``, the planner of
                    the earlier arm studies).
``GradientPlanner`` gradient descent on the joint control posterior of all
                    inputs, with the objective of the message planner; the
                    baseline of the cost comparison in
                    experiment-computation (build it on the estimator and
                    control prior of an agent).

Origin: the HARxI package of the TAC 2026 / NeurIPS 2026 projects, without
its hierarchical agent.

    from controllers.cbfe_arx import Agent
    from controllers.common import MultivariateNormal

    goal = MultivariateNormal(mean=np.zeros(2), covariance=1e-2 * np.eye(2))
    agent = Agent.build(input_dim=2, output_dim=2, input_memory=0, output_memory=1,
                        goal_prior=goal, horizon=3)
    for t in range(T):
        u = agent.plan(y)             # input in [-1, 1]^input_dim
        y_next = system.step(u)
        agent.update(y_next, u)
        y = y_next
"""

from .distributions import Exponential, SymmetricBeta
from .planner import Planner
from .message_planner import MessagePlanner
from .gradient_planner import GradientPlanner
from .agent import Agent

__all__ = ["Agent", "Planner", "MessagePlanner", "GradientPlanner", "SymmetricBeta", "Exponential"]
