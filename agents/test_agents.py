"""Checks for the agents in ``agents/``.

The first check enforces the package rule: an agent imports from
``agents.common`` but never from another agent. The CBFE-ARX checks cover
the feature map (CasADi against numpy) and its Jacobian, the conjugate
update (sequential against batch), the mutual information (closed form
against the entropy of the predictive Student-t), the KKT regimes of the
information multiplier, the objective and the optimum of the message
planner, the gradient and the optimum of the gradient planner, and closed
loops on a scalar integrator. The remaining checks run
the shared utilities, the dual MPC, MPC-X, PILCO, the light-dark agent, the
message planner of the quadcopter against L-BFGS-B and, when torch is
available, TD-MPC2.

    python agents/test_agents.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.cbfe_arx import Agent  # noqa: E402
from agents.common import (  # noqa: E402
    ActionScaler, ExponentialForgetting, HarmonicFourierBasis, MatrixNormalWishart,
    MultivariateNormal, ParameterEstimator, RandomWalk, buffer_with_head, flatten_buffer,
    predictive_noise_covariance, tempered_posterior,
)

RNG = np.random.default_rng(0)


def test_agents_import_only_from_common():
    import ast
    packages = [p for p in (REPO_ROOT / "agents").iterdir() if (p / "__init__.py").exists()]
    for pkg in packages:
        for path in pkg.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.level > 0:
                    depth = len(path.relative_to(pkg).parts) - 1
                    assert node.level <= depth + 1, f"{path}: relative import leaves {pkg.name}"
                    continue
                if isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                    if node.module == "agents":
                        names = [f"agents.{a.name}" for a in node.names]
                elif isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                else:
                    continue
                for name in names:
                    if name.split(".")[0] == "agents":
                        assert name == "agents.common", \
                            f"{path.relative_to(REPO_ROOT)} imports {name}"


def regression_data(n, feature_dim=4, output_dim=2, noise=0.3, rng=RNG):
    features = rng.normal(size=(n, feature_dim))
    outputs = (features @ rng.normal(size=(feature_dim, output_dim))
               + noise * rng.normal(size=(n, output_dim)))
    return features, outputs


def test_features_casadi_matches_numpy():
    import casadi as ca
    feature_map = HarmonicFourierBasis(regressor_dim=3, num_harmonics=2,
                                       half_periods=np.array([1.0, 2.0, 3.0]))
    regressor = RNG.uniform(-1, 1, 3)
    regressor_symbol = ca.MX.sym("x", 3)
    casadi_features = np.array(ca.Function("f", [regressor_symbol],
                                           [feature_map.casadi(regressor_symbol)])(regressor)).flatten()
    assert casadi_features.shape == (feature_map.feature_dim,) == (1 + 3 + 2 * 2 * 3,)
    assert np.allclose(feature_map(regressor), casadi_features, atol=1e-12)


def test_feature_jacobian_matches_differences():
    feature_map = HarmonicFourierBasis(regressor_dim=3, num_harmonics=2,
                                       half_periods=np.array([1.0, 2.0, 3.0]))
    regressor = RNG.uniform(-1, 1, 3)
    differences = np.column_stack([(feature_map(regressor + 1e-6 * e)
                                    - feature_map(regressor - 1e-6 * e)) / 2e-6
                                   for e in np.eye(3)])
    assert np.allclose(feature_map.jacobian(regressor), differences, atol=1e-8)


def test_sequential_update_matches_batch():
    prior = MatrixNormalWishart.isotropic_prior(feature_dim=4, output_dim=2,
                                                precision_scale=0.1, noise_scale=0.1)
    features, outputs = regression_data(30)
    posterior = prior
    for feature_vector, output in zip(features, outputs):
        posterior = posterior.update(feature_vector, output)
    batch = tempered_posterior(prior, features, outputs, 1.0)
    for name in ("coefficient_mean", "coefficient_precision", "noise_scale", "degrees_of_freedom"):
        assert np.allclose(getattr(posterior, name), getattr(batch, name)), name


def test_tempered_posterior_counts_weighted_data():
    prior = MatrixNormalWishart.isotropic_prior(feature_dim=4, output_dim=2,
                                                precision_scale=0.1, noise_scale=0.1)
    features, outputs = regression_data(40)
    half = tempered_posterior(prior, features, outputs, 0.5)
    assert np.isclose(half.degrees_of_freedom, prior.degrees_of_freedom + 20)
    assert np.allclose(half.coefficient_precision,
                       prior.coefficient_precision + 0.5 * features.T @ features)


def test_random_walk_node_is_the_kalman_filter_of_a_drifting_gain():
    # With the intercept, the pole and the noise level known, the conjugate
    # update followed by the random-walk node is the Kalman filter of the
    # gain b' = phi b + rho v of Astrom and Helmersson (1986), Eq. (5.2).
    sigma, rho = 0.1, 0.2
    phi = np.sqrt(1.0 - rho**2)
    posterior = MatrixNormalWishart(coefficient_mean=[[0.0], [0.3], [1.0]],
                                    coefficient_precision=np.diag([1e9, sigma**2 / 0.5, 1e9]),
                                    noise_scale=[[sigma**2 * (1e6 - 2.0)]],
                                    degrees_of_freedom=1e6)
    node = RandomWalk(process_covariance=np.diag([0.0, rho**2 / sigma**2, 0.0]),
                      transition=np.diag([1.0, phi, 1.0]))
    bhat, P, y = 0.3, 0.5, 0.0
    rng = np.random.default_rng(4)
    for _ in range(20):
        u = rng.uniform(-1.0, 1.0)
        y_next = y + 0.8 * u + sigma * rng.normal()
        posterior = node(posterior.update([1.0, u, y], [y_next]))
        s = sigma**2 + u * u * P
        bhat, P = bhat + P * u * (y_next - y - bhat * u) / s, P * sigma**2 / s
        bhat, P = phi * bhat, phi**2 * P + rho**2
        y = y_next
    assert np.isclose(posterior.coefficient_mean[1, 0], bhat, rtol=1e-6), \
        (posterior.coefficient_mean[1, 0], bhat)
    assert np.isclose(posterior.coefficient_covariance[1, 1] * sigma**2, P, rtol=1e-6)


def test_forgetting_node_is_a_geometric_mixture():
    # q' ∝ q^λ p_0^(1-λ): log q' - λ log q - (1-λ) log p_0 is the same at
    # every (A, W), and likewise q' ∝ q^λ without a reference. Plain
    # forgetting keeps the mean and scales the coefficient precision and
    # the noise scale by λ; λ = 1 changes nothing, and repeated stabilised
    # forgetting relaxes to the reference.
    from scipy.stats import matrix_normal, wishart
    feature_dim, output_dim, factor = 4, 2, 0.7
    rng = np.random.default_rng(5)

    def random_belief(dof):
        root = rng.normal(size=(feature_dim, feature_dim))
        scale_root = rng.normal(size=(output_dim, output_dim))
        return MatrixNormalWishart(rng.normal(size=(feature_dim, output_dim)),
                                   root @ root.T + feature_dim * np.eye(feature_dim),
                                   scale_root @ scale_root.T + output_dim * np.eye(output_dim), dof)

    def log_density(belief, A, W):
        return (matrix_normal(mean=belief.coefficient_mean, rowcov=belief.coefficient_covariance,
                              colcov=np.linalg.inv(W)).logpdf(A)
                + wishart(df=belief.degrees_of_freedom,
                          scale=np.linalg.inv(belief.noise_scale)).logpdf(W))

    q, reference = random_belief(7.0), random_belief(5.0)
    stabilised = ExponentialForgetting(factor, reference=reference)(q)
    plain = ExponentialForgetting(factor)(q)
    gaps_stabilised, gaps_plain = [], []
    for _ in range(4):
        W = wishart(df=6, scale=np.eye(output_dim)).rvs(random_state=rng)
        A = rng.normal(size=(feature_dim, output_dim))
        gaps_stabilised.append(log_density(stabilised, A, W) - factor * log_density(q, A, W)
                               - (1 - factor) * log_density(reference, A, W))
        gaps_plain.append(log_density(plain, A, W) - factor * log_density(q, A, W))
    assert np.ptp(gaps_stabilised) < 1e-8 and np.ptp(gaps_plain) < 1e-8, \
        (gaps_stabilised, gaps_plain)
    assert np.allclose(plain.coefficient_mean, q.coefficient_mean)
    assert np.allclose(plain.coefficient_precision, factor * q.coefficient_precision)
    assert np.allclose(plain.noise_scale, factor * q.noise_scale)
    same = ExponentialForgetting(1.0)(q)
    assert np.allclose(same.coefficient_mean, q.coefficient_mean)
    assert np.allclose(same.noise_scale, q.noise_scale)
    assert np.isclose(same.degrees_of_freedom, q.degrees_of_freedom)
    relaxed = q
    for _ in range(300):
        relaxed = ExponentialForgetting(0.9, reference=reference)(relaxed)
    for name in ("coefficient_mean", "coefficient_precision", "noise_scale", "degrees_of_freedom"):
        assert np.allclose(getattr(relaxed, name), getattr(reference, name)), name


def test_parameter_node_widens_the_belief_over_the_horizon():
    # The estimator passes its posterior through the node after every
    # measurement. Over the horizon, slice k of the message planner uses
    # the belief over the parameters of its own step, whose covariance
    # grows by Q per step; without a node, every slice uses the posterior.
    agent = symmetric_gain_agent(horizon=4, planner_type="messages")
    estimator, planner = agent.estimator, agent.planner
    process_covariance = np.diag([0.0, 0.5, 0.0])
    estimator.parameter_dynamics = RandomWalk(process_covariance)
    prior = estimator.posterior
    estimator.update(np.array([0.3]), np.array([0.5]))
    expected = RandomWalk(process_covariance)(prior.update(estimator.feature_map(
        np.array([0.5, 0.0])), np.array([0.3])))
    assert np.allclose(estimator.posterior.coefficient_precision, expected.coefficient_precision)
    covariance = estimator.posterior.coefficient_covariance
    beliefs = estimator.parameter_beliefs(4)
    model = planner._slice_model(np.array([0.3]), agent.goal_prior, 0.0)
    for k in range(4):
        assert np.allclose(beliefs[k].coefficient_covariance, covariance + k * process_covariance)
        assert np.allclose(planner._slice_parameters(model, k).coefficient_covariance,
                           covariance + k * process_covariance)
    estimator.parameter_dynamics = None
    model = planner._slice_model(np.array([0.3]), agent.goal_prior, 0.0)
    assert np.allclose(planner._slice_parameters(model, 3).coefficient_covariance, covariance)


def test_mutual_information_matches_predictive_entropy():
    # I(Θ; y | φ) = H[p(y | φ, D)] - E[H[p(y | Θ, φ)]], with the first term the
    # entropy of the predictive Student-t. The information vanishes with data.
    prior = MatrixNormalWishart.isotropic_prior(feature_dim=4, output_dim=2,
                                                precision_scale=0.1, noise_scale=0.1)
    informations = []
    for n in (5, 50, 500):
        posterior = tempered_posterior(prior, *regression_data(n))
        features = RNG.normal(size=4)
        likelihood_entropy = (0.5 * 2 * np.log(2 * np.pi * np.e)
                              - 0.5 * posterior.expected_log_det_noise_precision())
        expected = posterior.posterior_predictive(features).entropy() - likelihood_entropy
        informations.append(posterior.mutual_information(features))
        assert np.isclose(informations[-1], expected), (n, informations[-1], expected)
    assert informations[0] > informations[1] > informations[2] > 0 and informations[2] < 0.05, \
        informations


def test_shared_utilities():
    posterior = tempered_posterior(MatrixNormalWishart.isotropic_prior(feature_dim=4, output_dim=2),
                                   *regression_data(20))
    assert np.allclose(predictive_noise_covariance(posterior),
                       posterior.noise_scale / (posterior.predictive_degrees_of_freedom - 2))
    buffer = buffer_with_head([np.ones(2), 2 * np.ones(2)], np.zeros(2), length=2)
    assert np.allclose(flatten_buffer(buffer, 3, 2), [0, 0, 1, 1, 0, 0])
    scaler = ActionScaler.symmetric(2.0, 2)
    physical_input = np.array([1.5, -0.4])
    assert np.allclose(scaler.to_physical(scaler.to_agent(physical_input)), physical_input)
    assert np.allclose(scaler.to_physical([1.0, -1.0]), [2.0, -2.0])


def scalar_agent(goal_std, fixed_multiplier=None, n_data=40, max_multiplier=50.0, horizon=1,
                 planner_type="sweeps", information_floor_type="task"):
    """Agent on y_t = y_{t-1} + 0.5 u_t + noise, after n_data observations."""
    goal = MultivariateNormal(mean=np.array([0.5]), covariance=np.array([[goal_std**2]]))
    agent = Agent.build(input_dim=1, output_dim=1, input_memory=0, output_memory=1,
                        goal_prior=goal, horizon=horizon, num_harmonics=0,
                        prior_precision_scale=0.1, prior_noise_scale=0.1,
                        fixed_multiplier=fixed_multiplier, max_multiplier=max_multiplier,
                        planner_type=planner_type, information_floor_type=information_floor_type)
    rng = np.random.default_rng(1)
    y = 0.0
    agent.estimator.output_buffer = [np.array([y])]
    for _ in range(n_data):
        u = rng.uniform(-1, 1, 1)
        y = y + 0.5 * u[0] + 0.01 * rng.normal()
        agent.update(np.array([y]), u)
    return agent, y


def test_kkt_regimes_of_the_information_multiplier():
    # A broad goal tolerates the parameter-induced spread, so the task floor
    # is negative and β = 0. A tight goal demands more information than any
    # action can give (β = β_max). Fixing β = 0 removes the constraint.
    broad, y = scalar_agent(goal_std=10.0)
    broad.plan(np.array([y]))
    assert broad.planner.last_information_floor < 0.0
    assert broad.planner.last_multiplier == 0.0
    tight, y = scalar_agent(goal_std=1e-3, max_multiplier=20.0)
    tight.plan(np.array([y]))
    assert tight.planner.last_multiplier == 20.0
    off, y = scalar_agent(goal_std=1e-3, fixed_multiplier=0.0)
    off.plan(np.array([y]))
    assert off.planner.last_multiplier == 0.0
    # The earlier floor -H[p_*] gives an interior root here, at which the
    # information of the MAP action about (A, W) equals the floor.
    root, y = scalar_agent(goal_std=0.085, n_data=3, information_floor_type="negentropy")
    u = root.plan(np.array([y]))
    information = root.estimator.mutual_information(u, latest_output=np.array([y]))
    assert 0.0 < root.planner.last_multiplier < 50.0, root.planner.last_multiplier
    assert abs(information - root.goal_prior.negative_entropy()) < 0.01
    assert root.planner.last_information_floor == root.goal_prior.negative_entropy()


def symmetric_gain_posterior(sigma=0.05):
    """Posterior over the features (1, u, y) that knows the intercept, the
    pole and the noise level, with the symmetric prior b ~ N(0, 1) on the gain."""
    noise_variance = sigma**2
    return MatrixNormalWishart(
        coefficient_mean=np.array([[0.0], [0.0], [1.0]]),
        coefficient_precision=np.diag([1e6, noise_variance, 1e6]),
        noise_scale=np.array([[18.0 * noise_variance]]), degrees_of_freedom=20.0)


def symmetric_gain_agent(goal_std=0.1, sigma=0.05, horizon=1, planner_type="sweeps",
                         fixed_multiplier=None):
    """Agent on y_t = y_{t-1} + b u_t + noise with ``symmetric_gain_posterior``."""
    goal = MultivariateNormal(mean=np.array([1.0]), covariance=np.array([[goal_std**2]]))
    agent = Agent.build(input_dim=1, output_dim=1, input_memory=0, output_memory=1,
                        goal_prior=goal, horizon=horizon, num_harmonics=0,
                        planner_type=planner_type, fixed_multiplier=fixed_multiplier)
    agent.estimator.posterior = symmetric_gain_posterior(sigma)
    agent.estimator.output_buffer = [np.array([0.0])]
    return agent


def test_task_floor_switches_probing_on_and_off():
    # Under the symmetric prior, the mean model cannot rule out any input,
    # and the parameter-induced spread of those inputs exceeds the goal: the
    # floor is positive and the multiplier probes, with the information of
    # the MAP action at the floor. Once the outputs have revealed the gain,
    # the spread fits inside the goal and the constraint switches off.
    agent = symmetric_gain_agent()
    rng = np.random.default_rng(3)
    y, b = 0.0, 0.8
    for k in range(3):
        u = agent.plan(np.array([y]))
        if k == 0:
            assert agent.planner.last_information_floor > 0.0
            assert 0.0 < agent.planner.last_multiplier < 50.0, agent.planner.last_multiplier
            assert abs(u[0]) > 0.1
            information = agent.planner.information(u, np.array([y]))
            assert abs(information - agent.planner.last_information_floor) < 0.01
        y = y + b * u[0] + 0.05 * rng.normal()
        agent.update(np.array([y]), u)
    assert agent.planner.last_information_floor < 0.0
    assert agent.planner.last_multiplier == 0.0


def test_predictive_floor_is_attainable_and_switches_with_the_task_floor():
    # Under the symmetric prior, the goal-reaching inputs are the control
    # prior, so the expected leverage is E_p[u^2] / sigma^2 (R = sigma^2,
    # P = 1). The predictive floor 1/2 ln((1 + l) R / (R + S_*)) matches its
    # closed form, stays below the information of the most informative
    # input when the goal is tighter than the noise, where the task floor
    # exceeds it, and switches sign where the task floor does (one output).
    sigma = 0.05
    for goal_std in (0.01, 0.05, 0.3, 2.0):
        floors = {}
        for floor_type in ("task", "predictive"):
            agent = symmetric_gain_agent(goal_std=goal_std, sigma=sigma)
            agent.planner.information_floor_type = floor_type
            floors[floor_type] = agent.planner.information_floor(np.array([0.0]), agent.goal_prior)
        planner = agent.planner
        nodes, log_density = planner._input_nodes
        weights = np.exp(log_density - log_density.max())
        mean_square = float(np.sum(weights * nodes[:, 0] ** 2) / weights.sum())
        leverage = mean_square / sigma**2 + 1e-6
        assert np.isclose(floors["predictive"],
                          0.5 * np.log((1 + leverage) * sigma**2 / (sigma**2 + goal_std**2)),
                          atol=1e-4)
        assert np.isclose(floors["task"], 0.5 * np.log(leverage * sigma**2 / goal_std**2), atol=1e-4)
        assert (floors["predictive"] > 0) == (floors["task"] > 0)
        maximal_information = planner.information(np.array([1.0]), np.array([0.0]))
        assert floors["predictive"] < maximal_information
        if goal_std == 0.01:
            assert floors["task"] > maximal_information


def test_next_node_constraint_is_solved_on_the_plan():
    # With constraint_nodes = "next", only the first slice carries the
    # multiplier, and the KKT search finds the multiplier at which the
    # first input of the plan carries the floor. With the predictive floor,
    # a goal tighter than the noise still gives an interior root, whereas
    # the task floor saturates.
    y = np.array([0.0])
    for floor_type, goal_std in (("task", 0.1), ("predictive", 0.1), ("predictive", 0.01)):
        agent = symmetric_gain_agent(goal_std=goal_std, horizon=5, planner_type="messages")
        planner = agent.planner
        planner.constraint_nodes, planner.information_floor_type = "next", floor_type
        planner.max_multiplier = 1e4
        plans, multipliers, _ = planner.plan_horizon(y, agent.goal_prior, horizon=5)
        assert 0.0 < multipliers[0] < 1e4 and multipliers[1:] == [0.0] * 4, multipliers
        assert not planner.last_saturated
        information = planner.information(plans[0], y)
        assert -planner.root_tolerance < information - planner.last_information_floor < 0.05
        model = planner._slice_model(y, agent.goal_prior, multipliers[0], probing_slices=1)
        assert planner._probing_weight(model, 0) > 0.0 and planner._probing_weight(model, 1) == 0.0
    tight = symmetric_gain_agent(goal_std=0.01, horizon=5, planner_type="messages")
    tight.planner.constraint_nodes, tight.planner.max_multiplier = "next", 50.0
    tight.plan(y)
    assert tight.planner.last_saturated and tight.planner.last_multiplier == 50.0


def test_hidden_state_filter_is_exact():
    # The integrator with a hidden state is linear in (x, b) given the
    # inputs, so the Kalman filter of the agent equals the conditional of the
    # joint Gaussian of (x_n, b, y_1, ..., y_n).
    from agents.cbfe_ssm import HiddenStateAgent, HiddenStateConfig
    q, r, n = 0.01, 0.04, 4
    agent = HiddenStateAgent(HiddenStateConfig(q=q, r=r), gain_mean=0.3, gain_variance=1.0,
                             state_mean=0.1, state_variance=0.02)
    rng = np.random.default_rng(0)
    inputs, outputs = rng.uniform(-1, 1, n), rng.normal(0, 0.3, n)
    for u, y in zip(inputs, outputs):
        agent.update(u, y)
    # latent vector z = (x_0, b, w_1..w_n, v_1..v_n)
    dim = 2 + 2 * n
    mean = np.zeros(dim); mean[:2] = [0.1, 0.3]
    cov = np.diag([0.02, 1.0] + [q] * n + [r] * n)
    rows = []
    for k in range(1, n + 1):
        row = np.zeros(dim); row[0], row[1] = 1.0, inputs[:k].sum(); row[2:2 + k] = 1.0
        rows.append(row)
    states = np.array(rows)
    observations = states.copy()
    for k in range(n):
        observations[k, 2 + n + k] = 1.0
    target = np.vstack([states[-1], np.eye(dim)[1]])
    gain = target @ cov @ observations.T @ np.linalg.inv(observations @ cov @ observations.T)
    posterior_mean = target @ mean + gain @ (outputs - observations @ mean)
    posterior_cov = target @ cov @ target.T - gain @ observations @ cov @ target.T
    assert np.allclose(agent.mean, posterior_mean, atol=1e-10)
    assert np.allclose(agent.covariance, posterior_cov, atol=1e-10)


def test_state_and_parameter_information_bind_in_complementary_regimes():
    # Under the symmetric prior, the parameter information demands the larger
    # first input with a precise sensor and the state information with a
    # noisy one; without either constraint the input is zero. The chance
    # constraint caps the risk of the next state at its level.
    from agents.cbfe_ssm import HiddenStateAgent, HiddenStateConfig

    def first_input(r, parameter, state, chance_bound=None):
        config = HiddenStateConfig(q=0.01, r=r, parameter_information=parameter,
                                   state_information=state, chance_bound=chance_bound)
        agent = HiddenStateAgent(config, gain_mean=0.0, gain_variance=1.0,
                                 state_mean=0.0, state_variance=0.01)
        return abs(agent.plan()), agent

    for r, larger in ((0.0025, "parameter"), (0.04, "state")):
        parameter, _ = first_input(r, True, False)
        state, _ = first_input(r, False, True)
        assert (parameter > state) == (larger == "parameter"), (r, parameter, state)
        assert first_input(r, False, False)[0] < 1e-6
    u, agent = first_input(0.04, True, True, chance_bound=0.3)
    m, V, b_hat, P = agent.marginals()
    assert u > 0.05 and agent.risk(u, m, V, b_hat, P) <= agent.config.chance_level + 1e-3


def test_information_constraint_tracks_a_drifting_gain():
    # The gain drifts as b' = phi b + rho v from b_0 = 0.9 (Astrom and
    # Helmersson, 1986), and the agents start from the stationary prior
    # N(0, 1) with the goal N(0, sigma^2). With the random-walk node of the
    # drift and the information constraint, the multiplier probes first,
    # the belief then follows the gain, and the constraint switches on
    # again whenever the drift widens the belief. Without the constraint,
    # rest is optimal under the symmetric belief, so the agent never acts
    # and its belief stays at N(0, 1). With a constant gain in the model,
    # the belief contracts around a gain that no longer holds and the
    # constraint never switches on again.
    sigma, rho, steps = 0.05, 0.1, 60
    phi = np.sqrt(1.0 - rho**2)
    rng = np.random.default_rng(4)
    gain = [0.9]
    for _ in range(steps - 1):
        gain.append(phi * gain[-1] + rho * rng.normal())
    gain, noise = np.array(gain), sigma * rng.normal(size=steps)
    node = RandomWalk(process_covariance=np.diag([0.0, rho**2 / sigma**2, 0.0]),
                      transition=np.diag([1.0, phi, 1.0]))

    def episode(fixed_multiplier, parameter_dynamics):
        agent = symmetric_gain_agent(sigma=sigma, horizon=3, planner_type="messages",
                                     fixed_multiplier=fixed_multiplier)
        agent.goal_prior = MultivariateNormal(mean=np.zeros(1), covariance=[[sigma**2]])
        agent.estimator.parameter_dynamics = parameter_dynamics
        y, record = 0.0, []
        for t in range(steps):
            posterior = agent.estimator.posterior
            u = agent.plan(np.array([y]))
            record.append((posterior.coefficient_mean[1, 0],
                           posterior.coefficient_covariance[1, 1] * sigma**2,
                           agent.planner.last_multiplier, u[0]))
            y = y + gain[t] * u[0] + noise[t]
            agent.update(np.array([y]), u)
        estimate, variance, multiplier, inputs = np.array(record).T
        covered = np.mean(np.abs(estimate - gain) < 2.0 * np.sqrt(variance))
        return estimate, variance, multiplier, inputs, covered

    estimate, variance, multiplier, _, covered = episode(None, node)
    assert multiplier[0] > 0.0 and np.any(multiplier[10:] > 0.0), multiplier
    assert np.mean(np.abs(estimate[30:] - gain[30:])) < 0.2 and covered >= 0.9, covered
    estimate, variance, multiplier, inputs, _ = episode(0.0, node)
    assert np.all(inputs == 0.0) and np.all(np.abs(estimate) < 1e-9)
    assert np.allclose(variance, 1.0)
    _, variance, multiplier, _, covered = episode(None, None)
    assert np.all(multiplier[10:] == 0.0) and covered < 0.5 and variance[-1] < 0.01, covered


def test_information_grows_with_the_multiplier():
    # With β > 0 the MAP action trades goal cross-entropy for leverage.
    informations = []
    for multiplier in (0.0, 5.0, 50.0):
        agent, y = scalar_agent(goal_std=0.05, fixed_multiplier=multiplier)
        u = agent.plan(np.array([y]))
        informations.append(agent.estimator.mutual_information(u, latest_output=np.array([y])))
    assert informations[0] <= informations[1] + 1e-9 <= informations[2] + 2e-9, informations


def test_agent_reaches_goal_on_integrator():
    agent, y = scalar_agent(goal_std=0.02, fixed_multiplier=0.0, n_data=10)
    rng = np.random.default_rng(2)
    for _ in range(15):
        u = agent.plan(np.array([y]))
        y = y + 0.5 * u[0] + 0.01 * rng.normal()
        agent.update(np.array([y]), u)
    assert abs(y - 0.5) < 0.05, y


def test_message_planner_minimises_the_horizon_objective():
    # The objective of the message planner is the negative log surrogate
    # posterior: goal cross-entropy of the point-mass predictive plus the
    # input cost (β = 0). For a linear model it is convex, and the plan
    # matches a bounded quasi-Newton solution.
    from scipy.optimize import minimize
    horizon = 4
    agent, y = scalar_agent(goal_std=0.05, fixed_multiplier=0.0, n_data=10, horizon=horizon,
                            planner_type="messages")
    planner, estimator, goal = agent.planner, agent.estimator, agent.goal_prior
    y = np.array([y])
    plan = np.array(planner.plan_horizon(y, goal, horizon=horizon)[0])

    def objective(plan):
        total, output = 0.0, y
        for control_input in np.reshape(plan, (-1, 1)):
            predictive = estimator.predictive(control_input, latest_output=output)
            total += (predictive.cross_entropy_to_normal(goal.mean, goal.covariance)
                      + agent.control_prior.cost(control_input))
            output = predictive.mean
        return total
    constant = horizon * 0.5 * (np.log(2 * np.pi) + np.linalg.slogdet(goal.covariance)[1])
    assert np.isclose(objective(plan) - constant,
                      planner._rollout(planner._slice_model(y, goal, 0.0), plan)[0])
    reference = minimize(objective, np.zeros(horizon), method="L-BFGS-B",
                         bounds=[(-1 + 1e-9, 1 - 1e-9)] * horizon,
                         options={"ftol": 1e-15, "gtol": 1e-12})
    # the planner stops once the predicted decrease is below 1e-7 (1 + |F|)
    assert objective(plan) <= reference.fun + 1e-7 * (1 + abs(reference.fun)), \
        (objective(plan), reference.fun)
    assert np.allclose(plan.ravel(), reference.x, atol=1e-4), (plan.ravel(), reference.x)


def gradient_planner(agent, **options):
    """Gradient planner on the estimator and control prior of ``agent``."""
    from agents.cbfe_arx.gradient_planner import GradientPlanner
    return GradientPlanner(estimator=agent.estimator, control_prior=agent.control_prior,
                           fixed_multiplier=agent.planner.fixed_multiplier,
                           symmetric_starts=False, **options)


def test_gradient_planner_gradient_matches_differences():
    # Reverse-mode differentiation of the horizon objective, with the
    # caution and probing terms of β = 1, against central differences.
    horizon = 5
    agent, y = scalar_agent(goal_std=0.05, fixed_multiplier=1.0, n_data=10, horizon=horizon,
                            planner_type="messages")
    planner = gradient_planner(agent)
    model = planner._slice_model(np.array([y]), agent.goal_prior, 1.0)
    plan = np.random.default_rng(3).uniform(-0.6, 0.6, (horizon, 1))
    gradient = planner._gradient(model, plan, planner._rollout(model, plan)[1])
    differences = np.zeros_like(plan)
    for k in range(horizon):
        step = np.zeros_like(plan)
        step[k] = 1e-6
        differences[k] = (planner._rollout(model, plan + step)[0]
                          - planner._rollout(model, plan - step)[0]) / 2e-6
    assert np.allclose(gradient, differences, rtol=1e-6, atol=1e-6), (gradient, differences)


def test_gradient_planner_reaches_the_message_plan():
    # Both planners minimise the same objective; for a linear model at
    # β = 1 it is strictly convex, so gradient descent ends at the plan of
    # the message planner. The first inputs sit close to the actuator
    # limit, where the barrier of the control prior makes the objective
    # ill-conditioned: about 5000 gradient steps against 9 sweeps.
    horizon = 6
    agent, y = scalar_agent(goal_std=0.05, fixed_multiplier=1.0, n_data=10, horizon=horizon,
                            planner_type="messages")
    agent.planner.symmetric_starts = False
    y = np.array([y])
    message_plan = np.array(agent.planner.plan_horizon(y, agent.goal_prior, horizon=horizon)[0])
    planner = gradient_planner(agent, gradient_tolerance=1e-4)
    plan = np.array(planner.plan_horizon(y, agent.goal_prior, horizon=horizon)[0])
    assert planner.last_converged
    objective = agent.planner.last_objective
    assert planner.last_objective <= objective + 1e-7 * (1 + abs(objective)), \
        (planner.last_objective, objective)
    assert np.allclose(plan, message_plan, atol=1e-3), (plan.ravel(), message_plan.ravel())
    assert planner.last_sweeps > 100 * agent.planner.last_sweeps
    # with a target objective, the descent stops once it reaches the target
    planner.reset()
    planner.target_objective = objective + 1e-3 * (1 + abs(objective))
    planner.plan_horizon(y, agent.goal_prior, horizon=horizon)
    assert planner.last_converged and planner.last_objective <= planner.target_objective


def test_agent_with_backward_messages_reaches_goal_on_integrator():
    agent, y = scalar_agent(goal_std=0.02, fixed_multiplier=0.0, n_data=10, horizon=3,
                            planner_type="messages")
    rng = np.random.default_rng(2)
    for _ in range(15):
        u = agent.plan(np.array([y]))
        assert np.all(np.abs(u) < 1)
        y = y + 0.5 * u[0] + 0.01 * rng.normal()
        agent.update(np.array([y]), u)
    assert abs(y - 0.5) < 0.05, y


def test_dual_mpc_rest_tolerance_keeps_a_symmetric_optimum():
    # Integrator with an unknown gain of prior mean zero and a goal as tight as
    # the noise: at λ = 2, caution and probing cancel to second order around
    # rest, so the plan at rest is optimal, and with rest_tolerance the
    # round-off of IPOPT cannot leave it. At λ = 10 the probing term wins.
    from agents.dual_mpc import DualMPC
    sigma = 0.1
    goal = MultivariateNormal(mean=np.zeros(1), covariance=np.array([[sigma**2]]))
    agents = {}
    for weight in (2.0, 10.0):
        agent = DualMPC.build(input_dim=1, output_dim=1, input_memory=0, output_memory=1,
                              goal_prior=goal, horizon=5, probing_weight=weight,
                              effort_weight=2.0, num_harmonics=0)
        agent.config.rest_tolerance = 1e-7
        agent.estimator.posterior = MatrixNormalWishart(
            coefficient_mean=np.array([[0.0], [0.0], [1.0]]),
            coefficient_precision=np.diag([1e6, sigma**2, 1e6]),
            noise_scale=np.array([[sigma**2 * (1e4 - 2.0)]]), degrees_of_freedom=1e4)
        agent.estimator.output_buffer, agent.estimator.input_buffer = [np.array([0.0])], []
        agents[weight] = agent
    rng = np.random.default_rng(4)
    y = 0.1
    for _ in range(8):
        u = agents[2.0].plan(np.array([y]))
        assert np.all(u == 0.0), u
        y_next = y + sigma * rng.normal()
        agents[2.0].update(np.array([y_next]), u)
        y = y_next
    assert np.all(np.abs(agents[10.0].plan(np.array([y]))) > 0.05)


def test_dual_mpc_plans_within_bounds():
    from agents.dual_mpc import DualMPC
    goal = MultivariateNormal(mean=np.array([0.5, 0.0]), covariance=np.diag([1e-3, 1e-2]))
    agent = DualMPC.build(input_dim=1, output_dim=2, input_memory=0, output_memory=1,
                          goal_prior=goal, horizon=4, probing_weight=2.0, num_harmonics=0,
                          prior_precision_scale=0.1, prior_noise_scale=0.1)
    y = np.zeros(2)
    agent.estimator.output_buffer = [y.copy()]
    for _ in range(5):
        u = agent.plan(y)
        assert u.shape == (1,) and np.all(np.abs(u) <= 1)
        y = np.array([y[0] + 0.3 * u[0], 0.3 * u[0]])
        agent.update(y, u)


def test_planners_leave_rest_under_symmetric_prior():
    # Under the symmetric prior, an input at rest is a stationary point of
    # every probing objective. The symmetry-breaking starts let the message
    # planner, the KKT search and the dual MPC probe; without probing, the
    # message planner keeps the input at rest (the turn-off).
    from agents.dual_mpc import DualMPC
    y = np.array([0.0])
    probing = symmetric_gain_agent(horizon=5, planner_type="messages", fixed_multiplier=1.0)
    assert abs(probing.plan(y)[0]) > 0.05
    plain = symmetric_gain_agent(horizon=5, planner_type="messages", fixed_multiplier=1.0)
    plain.planner.symmetric_starts = False
    assert abs(plain.plan(y)[0]) < 1e-6
    cautious = symmetric_gain_agent(horizon=5, planner_type="messages", fixed_multiplier=0.0)
    assert abs(cautious.plan(y)[0]) < 1e-6
    kkt = symmetric_gain_agent(horizon=5, planner_type="messages")
    assert abs(kkt.plan(y)[0]) > 0.05 and kkt.planner.last_multiplier > 0.0
    goal = MultivariateNormal(mean=np.array([1.0]), covariance=np.array([[0.01]]))
    dual = DualMPC.build(input_dim=1, output_dim=1, input_memory=0, output_memory=1,
                         goal_prior=goal, horizon=5, probing_weight=2.0, effort_weight=2.0,
                         num_harmonics=0)
    dual.estimator.posterior = symmetric_gain_posterior()
    dual.estimator.output_buffer = [y.copy()]
    assert abs(dual.plan(y)[0]) > 0.05


def symmetric_gain_mpcx(accuracy, sigma=0.05):
    """MPC-X on y_t = y_{t-1} + b u_t + noise with ``symmetric_gain_posterior``."""
    from agents.mpcx import MPCX
    goal = MultivariateNormal(mean=np.array([1.0]), covariance=np.array([[0.01]]))
    agent = MPCX.build(input_dim=1, output_dim=1, input_memory=0, output_memory=1,
                       goal_prior=goal, horizon=3, accuracy=accuracy, experiment_length=1,
                       num_harmonics=0)
    agent.estimator.posterior = symmetric_gain_posterior(sigma)
    agent.estimator.output_buffer = [np.array([0.0])]
    return agent


def test_mpcx_excites_only_when_the_requirement_binds():
    # Under the symmetric prior the nominal MPC keeps u = 0, which teaches
    # nothing about the gain. A lax accuracy accepts that plan; a strict one
    # makes the experiment-design constraint excite the input, and the plan
    # meets the whitened LMI.
    lax = symmetric_gain_mpcx(accuracy=1e-6)
    u = lax.plan(np.array([0.0]))
    assert lax.planner.last_multiplier == 0.0 and abs(u[0]) < 1e-3
    strict = symmetric_gain_mpcx(accuracy=1.0)
    y = np.array([0.0])
    u = strict.plan(y)
    assert strict.planner.last_multiplier == 1.0 and strict.planner.last_shortfall == 0.0
    assert abs(u[0]) > 0.05 and abs(u[0]) <= 1.0
    strict._step -= 1                                       # the requirement of that call
    lmi_offset, whitening, _ = strict._requirement(y)
    plan = np.concatenate([u, strict._previous_plan[:-1]])  # the plan before the shift
    assert np.linalg.eigvalsh(strict._lmi(plan, lmi_offset, whitening, y))[0] > -1e-6


def test_pilco_runs():
    from agents.pilco import PILCOConfig, PILCOController
    cfg = PILCOConfig(n_warmup_steps=15, policy_update_every=15, horizon=4, n_max_data=40,
                      gp_n_restarts=0)
    controller = PILCOController(cfg)
    y = np.zeros(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(20):
            u = controller(y, np.array([0.5, 0.5]))
            y = y + 0.01 * u
    assert controller.dyn._fitted and np.all(np.abs(u) <= cfg.u_max)


def test_pilco_policy_layouts_agree():
    # The CasADi policy inside each NLP must be the policy the controller
    # applies; a column-major reshape would permute the gains.
    import casadi as ca
    from agents.pilco.pilco import _D_PHI, _D_THETA, _policy_apply_np, _theta_unpack_ca
    from agents.pilco import StatePILCO
    theta, phi = RNG.normal(size=_D_THETA), RNG.normal(size=_D_PHI)
    K, b = _theta_unpack_ca(ca.DM(theta))
    assert np.allclose(np.array(ca.tanh(K @ ca.DM(phi) + b)).ravel(), _policy_apply_np(theta, phi, 1.0))
    agent = StatePILCO()
    agent.theta = RNG.normal(size=len(agent.theta))
    s, target = RNG.normal(size=4), RNG.normal(size=2)
    sym = ca.MX.sym("theta", len(agent.theta))
    f = ca.Function("pi", [sym], [agent._ca_policy(sym, ca.DM(s), ca.DM(target))])
    assert np.allclose(np.array(f(agent.theta)).ravel(), agent.policy(s, target))


def test_state_pilco_runs():
    # Double integrator per coordinate, with prior data; the policy starts
    # as the PD anchor and stays within the action box.
    from agents.pilco import StatePILCO, StatePILCOConfig
    agent = StatePILCO(StatePILCOConfig(horizon=4, refit_every=5, n_max_data=60, gp_n_restarts=0))
    rng = np.random.default_rng(4)
    S = rng.normal(size=(40, 4))
    A = rng.uniform(-1, 1, size=(40, 2))
    dS = np.c_[0.1 * S[:, 2:], 0.2 * A]
    agent.add_data(np.c_[S, A], dS)
    s, target = np.zeros(4), np.array([0.5, -0.3])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(12):
            a = agent.act(s, target)
            s = s + np.r_[0.1 * s[2:], 0.2 * a]
    assert agent.n_fits >= 2 and np.all(np.abs(a) <= 1)


def test_protocol_runs_a_pd_controller():
    # The episode runner drives any controller; a joint-space PD law reaches
    # the targets of the horizontal arm.
    sys.path.insert(0, str(REPO_ROOT / "experiment-arm"))
    import reaching as R
    s = R.Settings()
    ep = R.run_agent({"kind": "pd"}, 0, s, targets=R.TARGETS[:2])
    assert R.summarise(ep, s)["reached"] == 1.0
    assert np.isnan(ep.beta).all() and ep.actions.shape == (2 * s.steps_per_target, 2)


def test_unknown_polarity_prior_is_symmetric():
    # Under unknown polarity, the plant reverses the motors of the seed, and
    # the prior of the ARX agents has zero mean on every coefficient that
    # changes sign with a motor (u_j and sin(pi u_j)), with a prior std of at
    # least twice the nominal magnitude of those coefficients.
    sys.path.insert(0, str(REPO_ROOT / "experiment-arm"))
    import reaching as R
    from envs.robot_arm import ArmModel, ArmParams
    s_known = R.Settings()
    s = R.replace(s_known, polarity_known=False)
    assert R.plant_config(s_known, 3).polarity == (1.0, 1.0)
    assert R.plant_config(s, 3).polarity == R.POLARITIES[3]
    nominal = ArmModel(ArmParams(), R.ARM_CONFIG).params
    known = R.build_agent(s_known, 0.0, nominal, R.joint_goal(np.zeros(2), s_known)).estimator.posterior
    est = R.build_agent(s, 0.0, nominal, R.joint_goal(np.zeros(2), s)).estimator
    odd = R.polarity_features(est)
    assert odd.tolist() == [1, 2, 8, 10]
    posterior = est.posterior
    assert np.all(posterior.coefficient_mean[odd] == 0.0)
    std = np.sqrt(np.diag(posterior.coefficient_covariance)[odd, None]
                  * np.diag(predictive_noise_covariance(posterior))[None, :])
    assert np.all(std >= 2.0 * np.abs(known.coefficient_mean[odd]).max() - 1e-9)


def test_integrator_dual_dp():
    # V_1 is the loss of the cautious law, 1 + eta^2 / (1 + beta^2); with a
    # well-known gain V_k tends to k; V_2(0, 0) = 2.9694 by adaptive
    # quadrature; and, as in Fig. 1 of Astrom and Helmersson (1986), the
    # optimal input at eta = 0 with three steps to go jumps from probing to
    # zero at beta = 0.39.
    sys.path.insert(0, str(REPO_ROOT / "experiment-integrator"))
    import integrator as I
    dp = I.DualDP(T=3, n_x=61, n_z=81).solve(processes=1)
    E, B = np.meshgrid(dp.x, dp.x, indexing="ij")
    assert np.allclose(dp.V[1], 1 + E**2 / (1 + B**2), atol=1e-9)
    assert abs(dp.V[3][0, -1] - 3.0) < 0.01
    assert abs(float(dp.value(2, 0.0, 0.0)) - 2.9694) < 2e-3
    assert abs(dp.nu(3, 0.0, 0.37)) > 0.3 and abs(dp.nu(3, 0.0, 0.41)) < 1e-6
    assert abs(dp.nu(3, 2.0, 0.0)) > 0.5
    bench = I.Benchmark(T=5)
    assert np.all(I.run_episode({"kind": "cautious"}, bench, seed=0).u == 0.0)   # b0 = 0: turns off
    u = I.run_episode({"kind": "ce"}, bench, seed=0).u
    assert np.all(np.isfinite(u)) and np.any(u != 0.0)                          # full input at bhat = 0


def test_tdmpc2_runs_if_torch_available():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        print("     (torch not installed: TD-MPC2 skipped)")
        return
    from agents.tdmpc2 import TDMPC2, TDMPC2Config
    cfg = TDMPC2Config(obs_dim=4, act_dim=2, latent_dim=32, mlp_dim=64, batch_size=16,
                       buffer_capacity=1000, seed_steps=30, num_samples=64, num_pi_trajs=8,
                       num_elites=16, mppi_iterations=2, device="cpu")
    agent, obs = TDMPC2(cfg), np.zeros(4, np.float32)
    for _ in range(40):
        a = agent.act(obs)
        nxt = (obs + 0.1 * np.r_[a, a]).astype(np.float32)
        agent.store(obs, a, -float(np.sum(nxt**2)), nxt)
        obs = nxt
    assert a.shape == (2,) and np.all(np.abs(a) <= 1)


def test_fault_filter_is_exact():
    """The fault filter equals brute-force enumeration of the fault hypotheses."""
    import math
    from agents.cbfe_fault import FaultFilter, FaultModel
    model = FaultModel(hazard=0.1)
    rng = np.random.default_rng(3)
    T = 6
    u, x = rng.uniform(-1, 1, T), np.zeros(T + 1)
    gains = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]
    filt = FaultFilter(model)
    for k in range(T):
        x[k + 1] = x[k] + gains[k] * u[k] + 0.1 * rng.standard_normal()
        filt.update(u[k], x[k], x[k + 1])

    def log_likelihood(steps, mean, var):
        total = 0.0
        for k in steps:
            pv = var * u[k]**2 + model.sigma2
            r = x[k + 1] - x[k] - mean * u[k]
            total += -0.5 * (r * r / pv + math.log(2 * math.pi * pv))
            g = var * u[k] / pv
            mean, var = mean + g * r, var - g * u[k] * var
        return total

    log_weights, modes = [T * math.log(1 - model.hazard) + log_likelihood(range(T), 1.0, 0.01)], [0]
    for tau in range(T):
        for F in (1, 2):
            log_weights.append(tau * math.log(1 - model.hazard) + math.log(model.hazard / 2)
                               + log_likelihood(range(tau), 1.0, 0.01)
                               + log_likelihood(range(tau, T), model.means[F], model.variances[F]))
            modes.append(F)
    w = np.exp(np.array(log_weights) - max(log_weights))
    posterior = np.bincount(modes, weights=w / w.sum(), minlength=3)
    assert np.allclose(filt.current_modes(), posterior, atol=1e-10)


def test_fault_floor_is_rate_distortion_of_the_prediction():
    """For one Gaussian, the rate-distortion floor equals eq. (10); the mode
    information of the mixture matches Monte Carlo."""
    import math
    from agents.cbfe_fault import mixture_entropy, rate_distortion
    for v, D in [(1.0, 0.5), (1.0, 0.1), (0.26, 0.02)]:
        assert abs(rate_distortion([0.0], [v], [1.0], D) - 0.5 * math.log(v / D)) < 2e-3
    assert rate_distortion([0.0, 0.1], [0.01, 0.01], [0.5, 0.5], 0.05) == 0.0
    rng = np.random.default_rng(0)
    mu, var, w = np.array([0.0, 0.3, -1.0]), np.array([0.01, 0.05, 0.02]), np.array([0.5, 0.3, 0.2])
    c = rng.choice(3, 200_000, p=w)
    y = mu[c] + np.sqrt(var[c]) * rng.standard_normal(c.size)
    log_p = np.log(np.sum(w * np.exp(-0.5 * (y[:, None] - mu)**2 / var) / np.sqrt(2 * np.pi * var), axis=1))
    assert abs(float(mixture_entropy(mu, var, w)) + log_p.mean()) < 5e-3


def test_fault_planner_matches_quasi_newton():
    """The Gauss-Newton message planner reaches the minimum that L-BFGS-B finds."""
    from scipy.optimize import minimize
    from agents.cbfe_fault import FaultAgent, FaultConfig
    agent = FaultAgent(FaultConfig())
    beliefs = (np.array([0.5, 0.1, 0.4]), np.array([1.0, 0.5, -1.0]), np.array([0.01, 0.04, 0.09]))
    for weights in ({"mode": np.array([3.0, 0, 0, 0, 0]), "gain": np.array([1.0, 0, 0, 0, 0])},
                    {"mode": np.full(5, 2.0), "gain": np.full(5, 2.0)}, {}):
        plan, value = agent._solve(weights, 0.4, beliefs, None)
        reference = min((minimize(lambda z: agent.objective(z, weights, 0.4, beliefs), s,
                                  method="L-BFGS-B", bounds=[(-0.999999, 0.999999)] * 5)
                         for s in (np.zeros(5), np.full(5, 0.5), np.full(5, -0.5))), key=lambda r: r.fun)
        assert value <= reference.fun + 1e-6


def test_lightdark_grid_filter_matches_kalman_in_the_dark():
    """With a constant measurement noise, the grid filter is the Kalman filter."""
    from agents.cbfe_lightdark import GridFilter, LightDarkModel
    model = LightDarkModel(r0=0.5, r_dark=0.5)
    filt = GridFilter(model, 0.0, 0.25)
    m, V = 0.0, 0.25
    rng = np.random.default_rng(1)
    for _ in range(5):
        u, y = rng.uniform(-0.5, 0.5), rng.normal()
        filt.update(u, y)
        m, V = m + u, V + model.q
        gain = V / (V + 0.5)
        m, V = m + gain * (y - m), V - gain * V
    fm, fV = filt.moments()
    assert abs(fm - m) < 2e-3 and abs(fV - V) < 2e-3


def test_lightdark_planner_and_floor():
    """The message planner reaches the L-BFGS-B minimum, the floor vanishes once
    the state is known to within the goal tolerance, and CBFE then plans as the
    cautious planner."""
    from scipy.optimize import minimize
    from agents.cbfe_lightdark import LightDarkAgent, LightDarkConfig
    agent = LightDarkAgent(LightDarkConfig(goal_variance=0.01))
    for weights in (np.array([20.0, 0, 0, 0, 0]), np.full(5, 30.0), np.zeros(5)):
        _, value = agent._solve(weights, 0.0, 0.25, None)
        reference = min((minimize(lambda z: agent.objective(z, weights, 0.0, 0.25), s, method="L-BFGS-B",
                                  bounds=[(-0.999999, 0.999999)] * 5)
                         for s in (np.zeros(5), np.array([0.9, 0, 0, 0, 0]))), key=lambda r: r.fun)
        assert value <= reference.fun + 1e-6
    assert agent.floor(0.0, 0.005) < 0.0 < agent.floor(0.0, 0.25)


def test_quadcopter_message_planner_reaches_the_gradient_minimum():
    """The hard-coded messages of agents/cbfe_quad/messages.py: the free energy
    of the forward messages equals the objective J of the gradient planner,
    and the coordinate updates reach the minimum of J that a long L-BFGS-B
    run finds, without and with a multiplier on the next information."""
    from scipy.optimize import minimize
    from agents.cbfe_quad.agent import Agent as QuadAgent, Config as QuadConfig, GRAVITY
    config = QuadConfig(goal=(3.0, 2.0, 1.0), goal_std=0.05, position_noise=0.3, imu_noise=1.0, planner="message")
    agent = QuadAgent(config, state_mean=np.r_[np.zeros(6), 0.0, 0.0, GRAVITY],
                      state_variance=np.r_[[0.09] * 3, [0.0025] * 3, [1.0] * 3],
                      parameter_mean=np.array([0.1, -0.05, 0.0, 0.04, 0.2, 0.5, 1.0]),
                      parameter_variance=np.array([0.02, 0.02, 3e-4, 1e-4, 0.4, 0.6, 0.0025]))
    beliefs, noise, goal = agent.beliefs(), agent._noise(), np.asarray(config.goal, float)
    planner, start = agent._message, np.zeros(3 * config.horizon)
    for multiplier in (0.0, 1e3):
        plan, _ = planner.plan(multiplier, beliefs, start, noise)

        def J(U, multiplier=multiplier):
            value, gradient, _ = agent.objective_fn(U, *beliefs, goal, multiplier, noise)
            return float(value), np.array(gradient).ravel()
        means, covs = planner.forward_pass(plan, beliefs)
        assert np.isclose(planner.free_energy(plan, means, covs, multiplier)[0], J(plan)[0], rtol=1e-12)
        reference = minimize(J, start, jac=True, method="L-BFGS-B", bounds=[(-1 + 1e-6, 1 - 1e-6)] * len(start),
                             options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 20000})
        assert J(plan)[0] <= reference.fun + 1e-9 * (1 + abs(reference.fun)), (J(plan)[0], reference.fun)
        assert np.allclose(plan, reference.x, atol=1e-4), np.max(np.abs(plan - reference.x))


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {exc!r}")
        else:
            print(f"ok   {t.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
