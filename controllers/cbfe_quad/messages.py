"""Message passing planner of the quadcopter CBFE agent (main.tex, Section 4.3).

At every step k = 1, ..., K of the horizon, the planning model (3) has a
transition node f_x = p(x_k | x_{k-1}, theta, u_k), a measurement node
f_y = p(y_k | x_k), a control prior f_u = p(u_k) and a goal node
f_* = p(y_k | phi_*) on the position output (Fig. 1). The form constraints
fix the beliefs q(x_t) = N(m_0, S_0) and q(theta) = N(theta_hat, P_t) to the
filter, and the random walk of the parameters widens the parameter message
of step k to P_k = P_t + k W. Every control belief is represented by its
mode u_k, a point mass. The messages, numbered as in Fig. 1, are hard-coded:

  3, 5  forward state messages N(m_k, S_k). The transition node is
        linearised at its incoming means (m_{k-1}, theta_hat, u_k), as in
        the extended Kalman filter,
            m_k = F(m_{k-1}, theta_hat, u_k) ,
            S_k = A_k S_{k-1} A_k^T + B_k P_k B_k^T + Q_x ,
        with the Jacobians A_k = dF/dx and B_k = dF/dtheta. The factorization
        constraint draws x_{k-1} and theta independently in every node.
  4     parameter message N(theta_hat, P_k).
  7     forward output message N(H m_k, H S_k H^T + R) on the position output.
  8     goal message N(p_*, S_*) on the position output.
  6     backward state messages, the exponentiated multipliers of the
        marginalization constraints of the states. Under the Gaussian forward
        messages, the free energy to go from x_k depends on the forward
        message N(m, S) of x_k as v_k(m) + tr(Omega_k S) / 2, so the backward
        message consists of the Gaussian exp(-v_k(x)) in information form,
        v_k(x) = x^T W_k x / 2 - w_k^T x, and the caution weight Omega_k of the
        covariance. The measurement node turns the goal message into
        G = H^T S_*^-1 H and g = H^T S_*^-1 p_*, the equality node of x_k adds
        these to the message of the next transition node, and that node maps
        the message of x_{k+1} back to the incoming mean,
            v_k(x) = |H x - p_*|^2_{S_*^-1} / 2 + v_{k+1}(F(x, theta_hat, u_{k+1}))
                     + tr(Omega_{k+1} B(x) P_{k+1} B(x)^T) / 2 ,
            Omega_k = G + A_{k+1}^T Omega_{k+1} A_{k+1} ,
        from v_K(x) = |H x - p_*|^2_{S_*^-1} / 2 and Omega_K = G. The node is
        affine in the state for fixed parameters and in the parameters for a
        fixed state, so v_k is exactly quadratic. Its precision W_k contains
        the term of B(x), which grows with the velocity through the drag
        coefficient. The controller without caution drops the traces.
  2     message of the transition node to its control edge, eq. (21): the
        free energy to go at the forward message 5 of the input value u,
            -ln nu_k(u) = v_k(m(u)) + tr(Omega_k S(u)) / 2 ,
        where m(u) and S(u) linearise the node at (m_{k-1}, theta_hat, u).
  1     control prior p(u) = prod_i Beta((u_i + 1) / 2; c, c) on (-1, 1)^3.

The control belief of step k is the mode of p(u) nu_k(u) exp(gamma_k I_k(u))
(Theorem 5). Only the next input carries the information
I_1(u) = 1/2 ln |Sigma_1(u) (sigma_p^2 I)^-1| of the position output, with
gamma_1 = gamma and gamma_k = 0 for later inputs. A damped Newton method on
(-1, 1)^3 finds the mode from the current value of the input.

Schedule (Section 4.3). A forward pass computes the messages 3-5 and 7 of all
steps. The backward pass then computes the messages 8, 6 and 2 from the last
step to the first, and it updates every control mode on its way. When it
updates u_k, the inputs before it have not changed since the forward pass,
so their forward messages are current, and the backward message of x_k
already carries the updated later inputs. Every update therefore minimises
the free energy J below over one control belief, given the others, and the
sweeps decrease J monotonically. They repeat until no input moves by more
than ``tolerance``. The free energy of the plan,

    J(u) = sum_k [ -ln p(u_k) - E_{q(y_k)}[ln p(y_k | phi_*)] ] - gamma I_1(u_1)
         = sum_k [ -(c - 1) sum_i ln(1 - u_{k,i}^2) + |m_{p,k} - p_*|^2 / (2 S_*)
                   + tr(Sigma_k) / (2 S_*) ] - gamma I_1(u_1) + const ,

follows from the forward messages. It is the objective of the gradient
planner (``agent.functions``), and it selects between the two starts of a
plan, as there. A fixed point of the sweeps minimises J over every input
given the others, so it is a stationary point of J, and both planners solve
the same problem.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import casadi as ca
import numpy as np

from .agent import NX, NT, _step


@lru_cache(maxsize=8)
def message_functions(dt: float, thrust_range: float, tilt_max: float, shape: float):
    """CasADi functions of one transition node: its forward message (mean and
    Jacobians of the step), and the negative logarithm of the control belief
    p(u) nu(u) exp(gamma I(u)) with its gradient and Hessian in u."""
    u = ca.SX.sym("u", 3)
    m, S = ca.SX.sym("m", NX), ca.SX.sym("S", NX, NX)
    th, P = ca.SX.sym("theta", NT), ca.SX.sym("P", NT, NT)
    W, w, Om = ca.SX.sym("W", NX, NX), ca.SX.sym("w", NX), ca.SX.sym("Omega", NX, NX)
    gamma = ca.SX.sym("gamma")
    noise = ca.SX.sym("noise", 6)     # sigma_p^2, S_*, sigma_eta^2, q_c, trace weight, q_gain
    r_p, q_eta, trace = noise[0], noise[2], noise[4]

    mean = _step(m, th, u, dt, thrust_range, tilt_max)
    A, B = ca.jacobian(mean, m), ca.jacobian(mean, th)
    forward = ca.Function("forward", [m, th, u], [mean, A, B])

    # message 6: the node maps the free energy to go of x_{k+1} back to the
    # incoming state mean m. It is quadratic in m, because the node is affine
    # in the state, and B is affine in it, so value and derivatives are exact.
    to_go = 0.5 * ca.dot(mean, W @ mean) - ca.dot(w, mean) + 0.5 * trace * ca.trace(Om @ B @ P @ B.T)
    to_go_hessian, to_go_gradient = ca.hessian(to_go, m)
    backward = ca.Function("backward", [m, th, u, P, W, w, Om, noise], [to_go_hessian, to_go_gradient, A])

    # message 2 and the control belief
    lift = ca.vertcat(0.5 * dt**2 * ca.SX.eye(3), dt * ca.SX.eye(3), ca.SX.eye(3))
    S_next = A @ S @ A.T + B @ P @ B.T + q_eta * (lift @ lift.T)
    Sigma = S_next[0:3, 0:3] + r_p * ca.SX.eye(3)
    info = 0.5 * (ca.log(ca.det(Sigma)) - 3.0 * ca.log(r_p))
    neg_log_nu = 0.5 * ca.dot(mean, W @ mean) - ca.dot(w, mean) + 0.5 * trace * ca.trace(Om @ S_next)
    neg_log_belief = -(shape - 1.0) * ca.sum1(ca.log(1.0 - u**2)) + neg_log_nu - gamma * info
    hessian, gradient = ca.hessian(neg_log_belief, u)
    edge = ca.Function("control_edge", [u, m, S, th, P, W, w, Om, gamma, noise],
                       [neg_log_belief, gradient, hessian, info])
    return forward, backward, edge


@dataclass
class MessagePlanner:
    """Mode of the control posterior over the horizon by the hard-coded
    messages of the module docstring.

    max_sweeps     upper bound on the forward-backward sweeps of one plan
    tolerance      the sweeps stop once no input moves by more than this
    newton_steps   upper bound on the Newton steps of one control update
    """
    config: object
    max_sweeps: int = 2000
    tolerance: float = 1e-7
    newton_steps: int = 50

    def __post_init__(self):
        c = self.config
        self.forward_fn, self.backward_fn, self.edge_fn = message_functions(c.dt, c.thrust_range, c.tilt_max,
                                                                            c.control_prior_shape)
        lift = np.vstack([0.5 * c.dt**2 * np.eye(3), c.dt * np.eye(3), np.eye(3)])
        self.Q = c.force_noise**2 * lift @ lift.T
        self.walk = np.diag([c.wind_step**2] * 3 + [0.0] + [c.gain_step**2] * 3)
        H = np.zeros((3, NX))
        H[:, :3] = np.eye(3)
        self.G = H.T @ H / c.goal_std**2                           # measurement node: H^T S_*^-1 H
        self.g = H.T @ np.asarray(c.goal, float) / c.goal_std**2   # and H^T S_*^-1 p_*
        self.bound = 1.0 - c.bound_margin
        self.last_sweeps = 0
        self.last_converged = True

    # ----- messages -----

    def node(self, m: np.ndarray, theta: np.ndarray, u: np.ndarray):
        """The transition node linearised at (m, theta, u): mean, A and B."""
        mean, A, B = self.forward_fn(m, theta, u)
        return np.array(mean).ravel(), np.array(A), np.array(B)

    def forward_pass(self, plan: np.ndarray, beliefs):
        """Forward state messages (3, 5) of every step: means, covariances."""
        mx, Sx, mt, Pt = beliefs
        K = self.config.horizon
        means, covs = [mx], [Sx]
        for k in range(K):
            mean, A, B = self.node(means[-1], mt, plan[3 * k:3 * k + 3])
            covs.append(A @ covs[-1] @ A.T + B @ (Pt + (k + 1) * self.walk) @ B.T + self.Q)
            means.append(mean)
        return means, covs

    def free_energy(self, plan: np.ndarray, means, covs, multiplier: float) -> Tuple[float, float]:
        """J of the module docstring from the forward messages, and the
        information I_1 of the next position output."""
        c = self.config
        U = plan.reshape(-1, 3)
        r_p, s = c.position_noise**2, c.goal_std**2
        trace = 1.0 if c.caution else 0.0
        J = -(c.control_prior_shape - 1.0) * float(np.sum(np.log1p(-U**2)))
        for k in range(1, len(means)):
            error = means[k][:3] - np.asarray(c.goal, float)
            J += float(error @ error) / (2.0 * s) + trace * float(np.trace(covs[k][:3, :3]) + 3.0 * r_p) / (2.0 * s)
        info = 0.5 * (np.linalg.slogdet(covs[1][:3, :3] + r_p * np.eye(3))[1] - 3.0 * np.log(r_p))
        return J - multiplier * info, float(info)

    def control_update(self, u: np.ndarray, args) -> np.ndarray:
        """Mode of p(u) nu(u) exp(gamma I(u)) by damped Newton steps from u.
        Negative curvature is mirrored, so that every step descends, and a
        backtracking line search keeps u inside the support of the prior."""
        u = np.clip(u, -self.bound, self.bound)
        value, grad, hess, _ = self.edge_fn(u, *args)
        value, grad, hess = float(value), np.array(grad).ravel(), np.array(hess)
        for _ in range(self.newton_steps):
            eigenvalues, vectors = np.linalg.eigh(0.5 * (hess + hess.T))
            eigenvalues = np.maximum(np.abs(eigenvalues), 1e-8 * max(1.0, np.max(np.abs(eigenvalues))))
            direction = -vectors @ ((vectors.T @ grad) / eigenvalues)
            slope = float(grad @ direction)
            if -slope < 1e-14 * max(1.0, abs(value)):
                break
            with np.errstate(divide="ignore"):
                room = np.where(direction > 0, (self.bound - u) / direction,
                                np.where(direction < 0, (-self.bound - u) / direction, np.inf))
            step = min(1.0, 0.99 * float(np.min(room)))
            while step > 1e-12:
                trial = u + step * direction
                outputs = self.edge_fn(trial, *args)
                if float(outputs[0]) <= value + 1e-4 * step * slope:
                    break
                step *= 0.5
            else:
                break
            u = trial
            value, grad, hess = float(outputs[0]), np.array(outputs[1]).ravel(), np.array(outputs[2])
        return u

    def sweep(self, plan: np.ndarray, beliefs, multiplier: float, means, covs, noise) -> np.ndarray:
        """Backward pass: messages 8, 6 and 2 from the last step to the
        first, updating every control mode on its way."""
        mx, Sx, mt, Pt = beliefs
        K = self.config.horizon
        plan = plan.copy()
        W, w, Om = self.G.copy(), self.g.copy(), self.G.copy()     # message 6 of the last step: its goal
        for k in range(K - 1, -1, -1):
            P = Pt + (k + 1) * self.walk
            # message 2 to u_{k+1} and its control update
            args = (means[k], covs[k], mt, P, W, w, Om, multiplier if k == 0 else 0.0, noise)
            plan[3 * k:3 * k + 3] = self.control_update(plan[3 * k:3 * k + 3], args)
            if k == 0:
                break
            # message 6 to x_k through the transition node of the updated input,
            # joined with the goal of x_k at its equality node
            hessian, gradient, A = (np.array(v) for v in
                                    self.backward_fn(means[k], mt, plan[3 * k:3 * k + 3], P, W, w, Om, noise))
            W = self.G + 0.5 * (hessian + hessian.T)
            w = self.g + hessian @ means[k] - gradient.ravel()
            Om = self.G + A.T @ Om @ A
        return plan

    def solve(self, multiplier: float, beliefs, start: np.ndarray, noise: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """Sweeps from ``start`` until no input moves by more than the
        tolerance. Returns the plan, its free energy J and the sweeps."""
        plan = np.clip(start, -self.bound, self.bound)
        means, covs = self.forward_pass(plan, beliefs)
        sweeps, converged = 0, False
        while sweeps < self.max_sweeps:
            sweeps += 1
            updated = self.sweep(plan, beliefs, multiplier, means, covs, noise)
            moved = float(np.max(np.abs(updated - plan)))
            plan = updated
            means, covs = self.forward_pass(plan, beliefs)
            if moved < self.tolerance:
                converged = True
                break
        self.last_converged = self.last_converged and converged
        J, _ = self.free_energy(plan, means, covs, multiplier)
        return plan, J, sweeps

    def plan(self, multiplier: float, beliefs, start: np.ndarray, noise: np.ndarray) -> Tuple[np.ndarray, float]:
        """Mode of the plan for a fixed multiplier and the information of its
        first input. As the gradient planner, it starts from ``start`` and,
        for a positive multiplier, also from the zero plan, and the plan with
        the lower free energy stands."""
        starts = [start] + ([np.zeros_like(start)] if multiplier > 0.0 else [])
        best, best_J = None, np.inf
        for initial in starts:
            plan, J, sweeps = self.solve(multiplier, beliefs, initial, noise)
            self.last_sweeps += sweeps
            if J < best_J:
                best, best_J = plan, J
        means, covs = self.forward_pass(best, beliefs)
        _, info = self.free_energy(best, means, covs, multiplier)
        return best, info
