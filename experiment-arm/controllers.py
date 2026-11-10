"""One control interface for every agent of the comparison on the arm.

``reaching.run_episode`` drives a controller through the reaching protocol
with five calls:

    start(y0)          beginning of an episode, with the first output
    set_target(q)      a new joint-space target
    act(y)             action in [-1, 1]^2 for the latest output
    diagnostics(a)     optional per-step quantities, before learning
    learn(y_next, a)   the outcome of the action

All controllers see the same information: the joint angles measured by the
encoders, the finite-difference joint velocities, the joint-space targets,
and the torque limit through the action scaling. Prior knowledge enters
only through the nominal simulator, i.e., the arm with its uncertain
parameters at their prior means.

``ARXController``     CBFE-ARX agent, dual MPC or MPC-X (shared ARX interface)
``PILCOController``   PILCO with a directly observed state
``TDMPC2Controller``  TD-MPC2 with a goal-relative observation
``PDController``      fixed joint-space PD law, a reference without learning

The agent packages are imported lazily, so a run that uses one agent does
not need the dependencies of the others.
"""

from __future__ import annotations

import numpy as np

JOINT_TOL = 0.02          # [rad] arrival tolerance in the TD-MPC2 reward


class ARXController:
    """Adapter for agents with the CBFE-ARX interface: ``plan(latest_output)``,
    ``update(output, control_input)``, a settable ``goal_prior``,
    ``reset_planner_cache()`` and an ``estimator`` (the CBFE-ARX agent, the
    dual MPC and MPC-X)."""

    def __init__(self, agent, goal_fn, name: str = "arx"):
        self.agent = agent
        self.goal_fn = goal_fn          # joint-space target -> MultivariateNormal goal
        self.name = name

    def start(self, y0):
        self.agent.estimator.output_buffer = [np.asarray(y0, float).copy()]
        self.agent.estimator.input_buffer = []

    def set_target(self, q_goal):
        self.agent.goal_prior = self.goal_fn(q_goal)
        self.agent.reset_planner_cache()

    def act(self, y):
        return np.clip(self.agent.plan(y), -1.0, 1.0)

    def diagnostics(self, a):
        est = self.agent.estimator
        planner = getattr(self.agent, "planner", None)
        return {"beta": getattr(planner, "last_multiplier", np.nan),
                "info": est.mutual_information(a),
                "floor": getattr(planner, "last_information_floor", np.nan),
                "leverage": est.leverage(a),
                "predictive": est.predictive(a)}

    def learn(self, y_next, a):
        self.agent.update(y_next, a)


class PDController:
    """Fixed PD law a = clip(kp (q* - q) - kd v) in action units. The default
    gains are those of the excitation controller of the nominal data and of
    PILCO's anchor policy. It learns nothing and serves as a reference."""

    def __init__(self, kp: float = 1.5, kd: float = 0.8, name: str = "pd"):
        self.kp, self.kd, self.name = kp, kd, name

    def start(self, y0):
        pass

    def set_target(self, q_goal):
        self.q_goal = np.asarray(q_goal, float)

    def act(self, y):
        y = np.asarray(y, float)
        return np.clip(self.kp * (self.q_goal - y[:2]) - self.kd * y[2:4], -1.0, 1.0)

    def diagnostics(self, a):
        return {}

    def learn(self, y_next, a):
        pass


class PILCOController:
    """Adapter for ``agents.pilco.StatePILCO``: the state is the output
    (q, v) itself."""

    def __init__(self, agent, name: str = "pilco"):
        self.agent = agent
        self.name = name

    def start(self, y0):
        self.agent.reset()

    def set_target(self, q_goal):
        self.q_goal = np.asarray(q_goal, float)
        self.agent.new_target()

    def act(self, y):
        return np.clip(self.agent.act(np.asarray(y, float), self.q_goal), -1.0, 1.0)

    def diagnostics(self, a):
        return {}

    def learn(self, y_next, a):
        pass      # StatePILCO records the transition at its next act() call


def joint_error(q, q_goal) -> np.ndarray:
    """Joint-angle error wrapped to [-π, π). The task is defined by the
    end-effector position, so whole turns of a joint do not count."""
    return (np.asarray(q, float) - q_goal + np.pi) % (2 * np.pi) - np.pi


def tdmpc2_obs(y, q_goal) -> np.ndarray:
    """Goal-relative observation (q - q*, v, sin q2, cos q2), with the
    wrapped joint error. The dynamics in the horizontal plane depend on the
    elbow angle but not on q1."""
    y = np.asarray(y, float)
    return np.r_[joint_error(y[:2], q_goal), y[2:4], np.sin(y[1]), np.cos(y[1])].astype(np.float32)


def tdmpc2_reward(y_next, a, q_goal) -> float:
    """Dense reward: squared (wrapped) joint error, a velocity and an effort
    penalty, and a bonus of 1 within JOINT_TOL of the target."""
    err = joint_error(y_next[:2], q_goal)
    v, a = np.asarray(y_next[2:4], float), np.asarray(a, float)
    return float(-err @ err - 0.1 * v @ v - 0.01 * a @ a + (np.linalg.norm(err) < JOINT_TOL))


class TDMPC2Controller:
    """Adapter for ``agents.tdmpc2.TDMPC2``. The agent acts with the MPPI
    mean (``eval_mode``) and, with ``learn_online``, keeps storing
    transitions and updating its networks, as the ARX agents keep updating
    their posteriors."""

    def __init__(self, agent, learn_online: bool = True, eval_mode: bool = True,
                 name: str = "tdmpc2"):
        self.agent = agent
        self.learn_online = learn_online
        self.eval_mode = eval_mode
        self.name = name

    def start(self, y0):
        self.agent.buffer.new_episode()
        self.agent.reset_plan()

    def set_target(self, q_goal):
        self.q_goal = np.asarray(q_goal, float)
        self.agent.reset_plan()

    def act(self, y):
        self.obs = tdmpc2_obs(y, self.q_goal)
        return np.clip(np.asarray(self.agent.act(self.obs, eval_mode=self.eval_mode), float), -1.0, 1.0)

    def diagnostics(self, a):
        return {}

    def learn(self, y_next, a):
        if self.learn_online:
            self.agent.store(self.obs, np.asarray(a, np.float32), tdmpc2_reward(y_next, a, self.q_goal),
                             tdmpc2_obs(y_next, self.q_goal))
