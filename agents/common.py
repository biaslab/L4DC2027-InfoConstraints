"""Code shared by the agents in ``agents/``.

Agents never import from each other: whatever two agents share lives here.

The CBFE-ARX agent and the dual-MPC and MPC-X baselines plan on the same
Bayesian ARX model,

    y_t ~ N(Aᵀ φ(x_t), W⁻¹),    x_t = (u_t, u_{t-1}, ..., y_{t-1}, ...),

with a fixed feature map φ and a matrix-normal Wishart posterior over the
coefficients A and the noise precision W, updated in closed form. They
compile one CasADi NLP and feed it, at every call, a parameter vector with
the autoregressive buffers and the posterior moments.

The names follow the paper: u is the (control) input, y the output, x the
regressor, φ(x) the features, and the posterior has the coefficient mean M,
the coefficient precision Λ (row precision of A), the noise scale Ω and the
degrees of freedom ν.

Model
    MultivariateNormal         goal priors
    MultivariateT              posterior predictive
    MatrixNormalWishart        parameter posterior, leverage, information
    RandomWalk                 parameter node: Gaussian drift of the coefficients
    ExponentialForgetting      parameter node: (stabilised) exponential forgetting
    HarmonicFourierBasis       feature map φ
    ParameterEstimator         posterior plus autoregressive buffers
    tempered_posterior         posterior from a batch of weighted data
    save_posterior, load_posterior

Planning utilities
    ipopt_options              default IPOPT settings for the cached NLPs
    predictive_noise_covariance  Ω_eff, the covariance scale of the predictive
    buffer_with_head           buffer with a hypothetical newest output in front
    flatten_buffer             most-recent-first buffer as a zero-padded vector
    input_nodes                nodes on the input box for averages over inputs
    goal_reaching_moments      E[φφᵀ] under the inputs that reach the goal
    ActionScaler               agent actions in [-1, 1] <-> physical inputs

The model classes come from the HARxI package of the TAC 2026 project.
``MatrixNormalWishart.mutual_information`` carries a correction (see the
comment in the method).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from scipy.special import digamma, gammaln


# ==========================================================================
# Distributions
# ==========================================================================

@dataclass
class MultivariateNormal:
    """Multivariate normal distribution N(mean, covariance)."""
    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        self.mean = np.atleast_1d(np.asarray(self.mean, dtype=float))
        self.covariance = np.atleast_2d(np.asarray(self.covariance, dtype=float))
        if self.mean.shape[0] != self.covariance.shape[0]:
            raise ValueError(
                f"Dimension mismatch: mean is {self.mean.shape}, "
                f"covariance is {self.covariance.shape}")

    @property
    def dimension(self) -> int:
        return self.mean.shape[0]

    @property
    def precision(self) -> np.ndarray:
        return np.linalg.inv(self.covariance)

    def log_pdf(self, value: np.ndarray) -> float:
        """Log-density at ``value``."""
        value = np.atleast_1d(np.asarray(value, dtype=float))
        deviation = value - self.mean
        sign, log_det = np.linalg.slogdet(self.covariance)
        if sign <= 0:
            raise ValueError("Covariance is not positive definite")
        cholesky_factor = np.linalg.cholesky(self.covariance)
        whitened = np.linalg.solve(cholesky_factor, deviation)
        squared_distance = float(np.dot(whitened, whitened))
        return -0.5 * (self.dimension * np.log(2 * np.pi) + log_det + squared_distance)

    def entropy(self) -> float:
        """Differential entropy H = 1/2 ln|2 pi e Sigma|."""
        _, log_det = np.linalg.slogdet(self.covariance)
        return 0.5 * (self.dimension * np.log(2 * np.pi * np.e) + log_det)

    def negative_entropy(self) -> float:
        """Negative differential entropy -H."""
        return -self.entropy()

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.multivariate_normal(self.mean, self.covariance)

    def __repr__(self) -> str:
        return (f"MultivariateNormal(mean={self.mean.tolist()}, "
                f"covariance_diagonal={np.diag(self.covariance).tolist()})")


@dataclass
class MultivariateT:
    """Multivariate location-scale Student-t distribution T_eta(mean, scale)
    with eta = ``degrees_of_freedom``,

        T_eta(y | mean, scale) ∝ |scale|^{-1/2}
            (1 + (1/eta) (y - mean)ᵀ scale⁻¹ (y - mean))^{-(eta + D)/2}.

    ``scale`` is not the covariance; the covariance is scale eta / (eta - 2)
    for eta > 2.
    """
    mean: np.ndarray
    scale: np.ndarray
    degrees_of_freedom: float

    def __post_init__(self):
        self.mean = np.atleast_1d(np.asarray(self.mean, dtype=float))
        self.scale = np.atleast_2d(np.asarray(self.scale, dtype=float))
        self.degrees_of_freedom = float(self.degrees_of_freedom)

    @property
    def dimension(self) -> int:
        return self.mean.shape[0]

    @property
    def covariance(self) -> np.ndarray:
        if self.degrees_of_freedom <= 2:
            raise ValueError("Covariance undefined: degrees of freedom <= 2")
        return self.scale * (self.degrees_of_freedom / (self.degrees_of_freedom - 2))

    def log_pdf(self, value: np.ndarray) -> float:
        value = np.atleast_1d(np.asarray(value, dtype=float))
        dof, dimension = self.degrees_of_freedom, self.dimension
        deviation = value - self.mean
        symmetric_scale = 0.5 * (self.scale + self.scale.T)
        try:
            cholesky_factor = np.linalg.cholesky(symmetric_scale)
        except np.linalg.LinAlgError:
            # Numerical safeguard: jitter escalation for near-singular
            # scale matrices (can occur under extreme leverage).
            jitter = 1e-10 * max(np.trace(symmetric_scale) / dimension, 1e-12)
            for _ in range(12):
                try:
                    cholesky_factor = np.linalg.cholesky(
                        symmetric_scale + jitter * np.eye(dimension))
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10.0
            else:
                return -np.inf
        whitened = np.linalg.solve(cholesky_factor, deviation)
        squared_distance = float(np.dot(whitened, whitened))
        log_normaliser = (gammaln(0.5 * (dof + dimension))
                          - gammaln(0.5 * dof)
                          - 0.5 * dimension * np.log(dof * np.pi)
                          - np.sum(np.log(np.diag(cholesky_factor))))
        log_kernel = -0.5 * (dof + dimension) * np.log(1 + squared_distance / dof)
        return log_normaliser + log_kernel

    def entropy(self) -> float:
        """Differential entropy of the Student-t distribution."""
        dimension, dof = self.dimension, self.degrees_of_freedom
        sign, log_det = np.linalg.slogdet(self.scale)
        half = 0.5 * (dof + dimension)
        constant_part = (-gammaln(half) + gammaln(0.5 * dof)
                         + 0.5 * dimension * np.log(dof * np.pi)
                         + 0.5 * log_det)
        digamma_part = half * (digamma(half) - digamma(0.5 * dof))
        return constant_part + digamma_part

    def cross_entropy_to_normal(self, goal_mean: np.ndarray,
                                goal_covariance: np.ndarray) -> float:
        """E_T[-ln N(y | goal_mean, goal_covariance)] in closed form."""
        goal_mean = np.atleast_1d(np.asarray(goal_mean, dtype=float))
        goal_covariance = np.atleast_2d(np.asarray(goal_covariance, dtype=float))
        goal_precision = np.linalg.inv(goal_covariance)
        sign, log_det = np.linalg.slogdet(goal_covariance)
        deviation = self.mean - goal_mean
        deviation_outer = np.outer(deviation, deviation)
        dof = self.degrees_of_freedom
        scale_factor = dof / (dof - 2) if dof > 2 else 10.0
        return 0.5 * (self.dimension * np.log(2 * np.pi)
                      + log_det
                      + np.trace(goal_precision @ (self.scale * scale_factor + deviation_outer)))

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        # A Gaussian draw divided by the square root of a scaled chi^2 draw.
        chi_square = rng.chisquare(self.degrees_of_freedom) / self.degrees_of_freedom
        gaussian = rng.multivariate_normal(np.zeros(self.dimension), self.scale)
        return self.mean + gaussian / np.sqrt(chi_square)

    def __repr__(self) -> str:
        return (f"MultivariateT(degrees_of_freedom={self.degrees_of_freedom:.2f}, "
                f"mean={self.mean.tolist()})")


@dataclass
class MatrixNormalWishart:
    """Matrix-normal Wishart distribution over the coefficients A and the
    noise precision W,

        A | W ~ MN(M, Λ⁻¹, W⁻¹),    W ~ Wishart(Ω⁻¹, ν),

    with M = ``coefficient_mean`` (feature_dim x output_dim),
    Λ = ``coefficient_precision`` (the row precision of A, feature_dim x
    feature_dim), Ω = ``noise_scale`` (the inverse scale of the Wishart,
    output_dim x output_dim) and ν = ``degrees_of_freedom``.
    """
    coefficient_mean: np.ndarray
    coefficient_precision: np.ndarray
    noise_scale: np.ndarray
    degrees_of_freedom: float

    def __post_init__(self):
        self.coefficient_mean = np.atleast_2d(np.asarray(self.coefficient_mean, dtype=float))
        self.coefficient_precision = np.atleast_2d(
            np.asarray(self.coefficient_precision, dtype=float))
        self.noise_scale = np.atleast_2d(np.asarray(self.noise_scale, dtype=float))
        self.degrees_of_freedom = float(self.degrees_of_freedom)

    @property
    def feature_dim(self) -> int:
        return self.coefficient_mean.shape[0]

    @property
    def output_dim(self) -> int:
        return self.coefficient_mean.shape[1]

    @property
    def predictive_degrees_of_freedom(self) -> float:
        """Degrees of freedom eta = ν - output_dim + 1 of the posterior
        predictive Student-t distribution."""
        return self.degrees_of_freedom - self.output_dim + 1

    @property
    def coefficient_covariance(self) -> np.ndarray:
        """Λ⁻¹, the row covariance of A."""
        return np.linalg.inv(self.coefficient_precision)

    def update(self, features: np.ndarray, output: np.ndarray) -> "MatrixNormalWishart":
        """Posterior after one more pair (φ, y) of features and output:

            ν'  = ν + 1
            Λ'  = Λ + φ φᵀ
            M'  = Λ'⁻¹ (Λ M + φ yᵀ)
            Ω'  = Ω + y yᵀ + Mᵀ Λ M - M'ᵀ Λ' M'
        """
        features = np.atleast_1d(np.asarray(features, dtype=float))
        output = np.atleast_1d(np.asarray(output, dtype=float))
        if features.shape[0] != self.feature_dim:
            raise ValueError(
                f"features have dimension {features.shape[0]}, expected {self.feature_dim}")
        if output.shape[0] != self.output_dim:
            raise ValueError(
                f"output has dimension {output.shape[0]}, expected {self.output_dim}")

        features_outer = np.outer(features, features)
        features_output = np.outer(features, output)
        new_precision = self.coefficient_precision + features_outer
        new_covariance = np.linalg.inv(new_precision)
        new_mean = new_covariance @ (self.coefficient_precision @ self.coefficient_mean
                                     + features_output)
        new_degrees_of_freedom = self.degrees_of_freedom + 1
        new_noise_scale = (self.noise_scale + np.outer(output, output)
                           + self.coefficient_mean.T @ self.coefficient_precision
                           @ self.coefficient_mean
                           - new_mean.T @ new_precision @ new_mean)
        # Symmetrise against numerical asymmetry.
        new_noise_scale = 0.5 * (new_noise_scale + new_noise_scale.T)
        return MatrixNormalWishart(
            coefficient_mean=new_mean, coefficient_precision=new_precision,
            noise_scale=new_noise_scale, degrees_of_freedom=new_degrees_of_freedom)

    def posterior_predictive(self, features: np.ndarray) -> MultivariateT:
        """The likelihood marginalised over the posterior: the Student-t
        T_eta(y | Mᵀφ, (1/eta) Ω (1 + φᵀ Λ⁻¹ φ))."""
        features = np.atleast_1d(np.asarray(features, dtype=float))
        predicted_mean = self.coefficient_mean.T @ features
        leverage = float(features @ self.coefficient_covariance @ features)
        dof = self.predictive_degrees_of_freedom
        scale = (1.0 / dof) * self.noise_scale * (1 + leverage)
        return MultivariateT(mean=predicted_mean, scale=scale, degrees_of_freedom=dof)

    def leverage(self, features: np.ndarray) -> float:
        """Statistical leverage ℓ = φᵀ Λ⁻¹ φ."""
        features = np.atleast_1d(np.asarray(features, dtype=float))
        return float(features @ self.coefficient_covariance @ features)

    def expected_log_det_noise_precision(self) -> float:
        """E[ln |W|] under Wishart(Ω⁻¹, ν):

            E[ln |W|] = D_y ln 2 - ln |Ω| + sum_{i=1}^{D_y} digamma((ν + 1 - i) / 2).
        """
        output_dim = self.output_dim
        sign, log_det_noise_scale = np.linalg.slogdet(self.noise_scale)
        return (output_dim * np.log(2) - log_det_noise_scale
                + sum(digamma(0.5 * (self.degrees_of_freedom + 1 - i))
                      for i in range(1, output_dim + 1)))

    def mutual_information(self, features: np.ndarray) -> float:
        """Mutual information I(Θ; y | φ) between the parameters and the
        next output,

            I = H(y | φ, D) - E_{p(W | D)}[H(y | A, W, φ)],

        which equals (D_y / 2) ln(1 + ℓ) plus a term that does not depend on φ.
        """
        leverage = self.leverage(features)
        output_dim = self.output_dim
        dof = self.predictive_degrees_of_freedom

        half = 0.5 * (dof + output_dim)
        sign, log_det_noise_scale = np.linalg.slogdet(self.noise_scale)

        # H(y | φ, D): entropy of the predictive Student-t with scale
        # Ω (1 + ℓ) / eta. Its log-determinant contributes
        # 0.5 ln|Ω| + 0.5 D_y ln(1 + ℓ) - 0.5 D_y ln(eta), and the
        # -0.5 D_y ln(eta) cancels the eta in the 0.5 D_y ln(eta pi) of the
        # Student-t entropy. (The version copied from the HARxI package
        # omitted the 1/eta of the scale and overstated I by (D_y / 2) ln(eta).)
        predictive_entropy = (0.5 * output_dim * np.log(np.pi)
                              + 0.5 * log_det_noise_scale
                              + 0.5 * output_dim * np.log(1 + leverage)
                              - gammaln(half) + gammaln(0.5 * dof)
                              + half * (digamma(half) - digamma(0.5 * dof)))

        # E[H(y | A, W, φ)] = (D_y / 2) ln(2 pi e) - 0.5 E[ln |W|]
        likelihood_entropy = (0.5 * output_dim * np.log(2 * np.pi * np.e)
                              - 0.5 * self.expected_log_det_noise_precision())

        return predictive_entropy - likelihood_entropy

    def __repr__(self) -> str:
        return (f"MatrixNormalWishart(feature_dim={self.feature_dim}, "
                f"output_dim={self.output_dim}, degrees_of_freedom={self.degrees_of_freedom:.2f})")

    @classmethod
    def isotropic_prior(cls, feature_dim: int, output_dim: int,
                        degrees_of_freedom: float | None = None,
                        precision_scale: float = 1e-2,
                        noise_scale: float = 1.0,
                        mean_scale: float = 0.1,
                        mean_seed: int = 0) -> "MatrixNormalWishart":
        """Weakly informative isotropic prior: Λ = precision_scale I,
        Ω = noise_scale I and ν = output_dim + 2 (the smallest value with
        a finite predictive covariance) unless given.

        The coefficient mean is a small Gaussian draw of standard deviation
        ``mean_scale``. A random draw, rather than zeros or a constant,
        avoids the degenerate symmetry in which every output has the same
        prior predictive Mᵀφ, which corrupts the planner gradient.
        """
        if degrees_of_freedom is None:
            degrees_of_freedom = output_dim + 2.0
        rng = np.random.default_rng(mean_seed)
        return cls(
            coefficient_mean=mean_scale * rng.standard_normal((feature_dim, output_dim)),
            coefficient_precision=precision_scale * np.eye(feature_dim),
            noise_scale=noise_scale * np.eye(output_dim),
            degrees_of_freedom=degrees_of_freedom)


# ==========================================================================
# Parameter dynamics: the node between the parameters of consecutive steps
# ==========================================================================

@dataclass
class RandomWalk:
    """Linear Gaussian random walk of the coefficients,

        A_k = Φ A_{k-1} + C + E_k,    E_k | W ~ MN(0, Q, W⁻¹),

    with a constant noise precision W. The node maps a matrix-normal
    Wishart belief over (A_{k-1}, W) to the exact belief over (A_k, W),

        M' = Φ M + C,    Λ'⁻¹ = Φ Λ⁻¹ Φᵀ + Q,    Ω' = Ω,    ν' = ν.

    Q = ``process_covariance`` is a row covariance in noise units, like
    Λ⁻¹, so the coefficients of output i drift with covariance
    Q (W⁻¹)_ii. Φ = ``transition`` defaults to the identity and
    C = ``offset`` to zero; Φ = φ I with C = (1 - φ) M_0 lets the
    coefficients revert to M_0.
    """
    process_covariance: np.ndarray
    transition: Optional[np.ndarray] = None
    offset: Optional[np.ndarray] = None

    def __post_init__(self):
        self.process_covariance = np.atleast_2d(np.asarray(self.process_covariance, dtype=float))
        if self.transition is not None:
            self.transition = np.atleast_2d(np.asarray(self.transition, dtype=float))
        if self.offset is not None:
            self.offset = np.atleast_2d(np.asarray(self.offset, dtype=float))

    def __call__(self, belief: MatrixNormalWishart) -> MatrixNormalWishart:
        mean, covariance = belief.coefficient_mean, belief.coefficient_covariance
        if self.transition is not None:
            mean = self.transition @ mean
            covariance = self.transition @ covariance @ self.transition.T
        if self.offset is not None:
            mean = mean + self.offset
        covariance = covariance + self.process_covariance
        precision = np.linalg.inv(0.5 * (covariance + covariance.T))
        return MatrixNormalWishart(
            coefficient_mean=mean, coefficient_precision=0.5 * (precision + precision.T),
            noise_scale=belief.noise_scale.copy(), degrees_of_freedom=belief.degrees_of_freedom)


@dataclass
class ExponentialForgetting:
    """Exponential forgetting with the factor λ = ``factor`` in (0, 1]. The
    node replaces the belief q over (A, W) by the normalised geometric
    mixture

        q'(A, W) ∝ q(A, W)^λ p_0(A, W)^(1 - λ)

    with the ``reference`` p_0 (stabilised forgetting, Kulhavý and Zarrop,
    1993), or by q^λ without a reference. Both stay matrix-normal Wishart,
    because the natural parameters (Λ, ΛM, Ω + MᵀΛM, ν + F - D - 1) of the
    family mix linearly (F features, D outputs). Without a reference, the
    node keeps M and scales Λ and Ω by λ: the coefficient covariance grows
    by 1/λ per step, without bound in the directions that the data stop
    exciting. With a reference, the belief relaxes towards p_0 instead,
    and its precision stays above (1 - λ) times that of p_0.
    """
    factor: float
    reference: Optional[MatrixNormalWishart] = None

    def __post_init__(self):
        if not 0.0 < self.factor <= 1.0:
            raise ValueError(f"the forgetting factor must lie in (0, 1], got {self.factor}")

    @staticmethod
    def _natural(belief: MatrixNormalWishart, dof_offset: float) -> tuple:
        weighted_mean = belief.coefficient_precision @ belief.coefficient_mean
        return (belief.coefficient_precision, weighted_mean,
                belief.noise_scale + belief.coefficient_mean.T @ weighted_mean,
                belief.degrees_of_freedom + dof_offset)

    def __call__(self, belief: MatrixNormalWishart) -> MatrixNormalWishart:
        dof_offset = belief.feature_dim - belief.output_dim - 1
        terms = [(self.factor, self._natural(belief, dof_offset))]
        if self.reference is not None:
            terms.append((1.0 - self.factor, self._natural(self.reference, dof_offset)))
        precision, weighted_mean, scatter, dof = (
            sum(weight * natural[i] for weight, natural in terms) for i in range(4))
        precision = 0.5 * (precision + precision.T)
        mean = np.linalg.solve(precision, weighted_mean)
        noise_scale = scatter - mean.T @ precision @ mean
        return MatrixNormalWishart(
            coefficient_mean=mean, coefficient_precision=precision,
            noise_scale=0.5 * (noise_scale + noise_scale.T),
            degrees_of_freedom=dof - dof_offset)


# ==========================================================================
# Feature map (harmonic Fourier basis with a linear pass-through)
# ==========================================================================

@dataclass
class HarmonicFourierBasis:
    """Harmonic Fourier feature map φ of a regressor x: the constant 1 if
    ``include_constant``, the scaled regressor x_j / L_j if
    ``include_linear``, and cos(n π x_j / L_j), sin(n π x_j / L_j) for
    n = 1, ..., ``num_harmonics`` and every coordinate j. The basis is
    periodic on [-L_j, L_j], with L = ``half_periods``.
    """
    regressor_dim: int
    num_harmonics: int
    half_periods: np.ndarray
    include_linear: bool = True
    include_constant: bool = True

    def __post_init__(self):
        self.half_periods = np.atleast_1d(np.asarray(self.half_periods, dtype=float))
        if self.half_periods.shape[0] != self.regressor_dim:
            if self.half_periods.shape[0] == 1:
                self.half_periods = self.half_periods.repeat(self.regressor_dim)
            else:
                raise ValueError(f"half_periods has shape {self.half_periods.shape} "
                                 f"but regressor_dim = {self.regressor_dim}")

    @property
    def feature_dim(self) -> int:
        return (int(self.include_constant) + (self.regressor_dim if self.include_linear else 0)
                + 2 * self.num_harmonics * self.regressor_dim)

    def __call__(self, regressor: np.ndarray) -> np.ndarray:
        """φ(x) with numpy."""
        regressor = np.atleast_1d(np.asarray(regressor, dtype=float))
        if regressor.shape[0] != self.regressor_dim:
            raise ValueError(
                f"regressor has shape {regressor.shape}, expected ({self.regressor_dim},)")
        features = [np.array([1.0])] if self.include_constant else []
        if self.include_linear:
            features.append(regressor / self.half_periods)
        for j in range(self.regressor_dim):
            for harmonic in range(1, self.num_harmonics + 1):
                angle = harmonic * np.pi * regressor[j] / self.half_periods[j]
                features.append(np.array([np.cos(angle), np.sin(angle)]))
        return np.concatenate(features)

    def batch(self, regressors: np.ndarray) -> np.ndarray:
        """φ of every row of ``regressors`` (N, regressor_dim), as an
        (N, feature_dim) array with the columns in the order of ``__call__``."""
        regressors = np.atleast_2d(np.asarray(regressors, dtype=float))
        if regressors.shape[1] != self.regressor_dim:
            raise ValueError(f"regressors have {regressors.shape[1]} columns, "
                             f"expected {self.regressor_dim}")
        features = [np.ones((len(regressors), 1))] if self.include_constant else []
        if self.include_linear:
            features.append(regressors / self.half_periods)
        for j in range(self.regressor_dim):
            for harmonic in range(1, self.num_harmonics + 1):
                angle = harmonic * np.pi * regressors[:, j] / self.half_periods[j]
                features.append(np.stack([np.cos(angle), np.sin(angle)], axis=1))
        return np.hstack(features)

    def jacobian(self, regressor: np.ndarray) -> np.ndarray:
        """Jacobian dφ/dx of shape (feature_dim, regressor_dim), with the rows
        in the order of ``__call__``."""
        regressor = np.atleast_1d(np.asarray(regressor, dtype=float))
        jacobian = np.zeros((self.feature_dim, self.regressor_dim))
        row = int(self.include_constant)
        if self.include_linear:
            jacobian[row:row + self.regressor_dim] = np.diag(1.0 / self.half_periods)
            row += self.regressor_dim
        for j in range(self.regressor_dim):
            for harmonic in range(1, self.num_harmonics + 1):
                frequency = harmonic * np.pi / self.half_periods[j]
                jacobian[row, j] = -frequency * np.sin(frequency * regressor[j])
                jacobian[row + 1, j] = frequency * np.cos(frequency * regressor[j])
                row += 2
        return jacobian

    def casadi(self, regressor):
        """φ(x) of a symbolic CasADi vector (SX or MX) of length regressor_dim."""
        import casadi as ca

        constant = ca.MX(1.0) if isinstance(regressor, ca.MX) else ca.SX(1.0)
        features = [constant] if self.include_constant else []
        if self.include_linear:
            for j in range(self.regressor_dim):
                features.append(regressor[j] / self.half_periods[j])
        for j in range(self.regressor_dim):
            for harmonic in range(1, self.num_harmonics + 1):
                angle = harmonic * np.pi * regressor[j] / self.half_periods[j]
                features.append(ca.cos(angle))
                features.append(ca.sin(angle))
        return ca.vertcat(*features)


# ==========================================================================
# Parameter estimation: posterior plus autoregressive buffers
# ==========================================================================

@dataclass
class ParameterEstimator:
    """Online Bayesian estimator of the ARX model with closed-form updates.

    The model is

        x_t = (u_t, u_{t-1}, ..., u_{t-M_u}, y_{t-1}, ..., y_{t-M_y}),
        y_t ~ N(Aᵀ φ(x_t), W⁻¹),

    with ``input_memory`` = M_u past inputs and ``output_memory`` = M_y past
    outputs in the regressor. The estimator keeps the matrix-normal Wishart
    posterior over (A, W) and the buffers of past inputs and outputs, most
    recent first. The regressor has dimension
    input_dim (M_u + 1) + output_dim M_y.

    Without ``parameter_dynamics`` the parameters are constant. A parameter
    node (``RandomWalk`` or ``ExponentialForgetting``) lets them change
    between steps: ``update`` passes the posterior through the node after
    every measurement, so that ``posterior`` is always the belief over the
    parameters of the next step, and ``parameter_beliefs`` propagates it
    over a planning horizon.
    """
    input_dim: int
    output_dim: int
    input_memory: int
    output_memory: int
    feature_map: HarmonicFourierBasis
    posterior: MatrixNormalWishart
    input_buffer: List[np.ndarray] = field(default_factory=list)
    output_buffer: List[np.ndarray] = field(default_factory=list)
    parameter_dynamics: Optional[Callable[[MatrixNormalWishart], MatrixNormalWishart]] = None

    @property
    def regressor_dim(self) -> int:
        return self.input_dim * (self.input_memory + 1) + self.output_dim * self.output_memory

    @property
    def feature_dim(self) -> int:
        return self.feature_map.feature_dim

    def regressor(self, control_input: np.ndarray,
                  input_buffer: Optional[List[np.ndarray]] = None,
                  output_buffer: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """The regressor x_t for the input u_t and the buffers (by default
        the estimator's; others give hypothetical predictions). Missing
        buffer entries are zero."""
        control_input = np.atleast_1d(np.asarray(control_input, dtype=float))
        if input_buffer is None:
            input_buffer = self.input_buffer
        if output_buffer is None:
            output_buffer = self.output_buffer

        parts = [control_input.flatten()]
        for i in range(self.input_memory):
            if i < len(input_buffer):
                parts.append(input_buffer[i].flatten())
            else:
                parts.append(np.zeros(self.input_dim))
        for i in range(self.output_memory):
            if i < len(output_buffer):
                parts.append(output_buffer[i].flatten())
            else:
                parts.append(np.zeros(self.output_dim))
        return np.concatenate(parts)

    def features(self, control_input: np.ndarray,
                 input_buffer: Optional[List[np.ndarray]] = None,
                 output_buffer: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """The features φ(x_t) of the regressor."""
        return self.feature_map(self.regressor(control_input, input_buffer, output_buffer))

    def batch_features(self, control_inputs: np.ndarray,
                       latest_output: Optional[np.ndarray] = None) -> np.ndarray:
        """Features of the regressors for every row of ``control_inputs``
        (N, input_dim), with ``latest_output`` as the newest output if given."""
        output_buffer = buffer_with_head(self.output_buffer, latest_output, self.output_memory)
        buffer_part = self.regressor(np.zeros(self.input_dim),
                                     output_buffer=output_buffer)[self.input_dim:]
        control_inputs = np.atleast_2d(np.asarray(control_inputs, dtype=float))
        return self.feature_map.batch(
            np.hstack([control_inputs, np.tile(buffer_part, (len(control_inputs), 1))]))

    def update(self, output: np.ndarray, control_input: np.ndarray) -> None:
        """Update the posterior with the output that followed the input,
        pass it through the parameter node (if any), and roll the buffers."""
        output = np.atleast_1d(np.asarray(output, dtype=float))
        control_input = np.atleast_1d(np.asarray(control_input, dtype=float))

        # The features use the buffers before the roll: they describe what
        # the system saw when it produced the output.
        features = self.features(control_input)
        self.posterior = self.posterior.update(features, output)
        if self.parameter_dynamics is not None:
            self.posterior = self.parameter_dynamics(self.posterior)

        self.input_buffer.insert(0, control_input.copy())
        self.output_buffer.insert(0, output.copy())
        self.input_buffer = self.input_buffer[:max(self.input_memory, 1)]
        self.output_buffer = self.output_buffer[:max(self.output_memory, 1)]

    def parameter_beliefs(self, horizon: int) -> List[MatrixNormalWishart]:
        """Beliefs over the parameters of the next ``horizon`` steps: the
        posterior, and its propagation through the parameter node (the
        posterior for every step if the parameters are constant)."""
        beliefs = [self.posterior]
        for _ in range(horizon - 1):
            beliefs.append(beliefs[-1] if self.parameter_dynamics is None
                           else self.parameter_dynamics(beliefs[-1]))
        return beliefs

    def _output_buffer(self, latest_output, output_buffer):
        """``output_buffer`` (by default the estimator's), with
        ``latest_output`` in front if given."""
        if output_buffer is None:
            output_buffer = self.output_buffer
        if latest_output is not None:
            output_buffer = ([np.atleast_1d(latest_output).copy()]
                             + output_buffer[:max(self.output_memory - 1, 0)])
        return output_buffer

    def predictive(self, control_input: np.ndarray,
                   latest_output: Optional[np.ndarray] = None,
                   input_buffer: Optional[List[np.ndarray]] = None,
                   output_buffer: Optional[List[np.ndarray]] = None) -> MultivariateT:
        """Posterior predictive of the next output for the input. A given
        ``latest_output`` goes in front of the output buffer (for multi-step
        planning, where the buffer holds hypothetical entries)."""
        output_buffer = self._output_buffer(latest_output, output_buffer)
        features = self.features(control_input, input_buffer=input_buffer,
                                 output_buffer=output_buffer)
        return self.posterior.posterior_predictive(features)

    def leverage(self, control_input: np.ndarray,
                 latest_output: Optional[np.ndarray] = None,
                 input_buffer: Optional[List[np.ndarray]] = None,
                 output_buffer: Optional[List[np.ndarray]] = None) -> float:
        output_buffer = self._output_buffer(latest_output, output_buffer)
        features = self.features(control_input, input_buffer=input_buffer,
                                 output_buffer=output_buffer)
        return self.posterior.leverage(features)

    def mutual_information(self, control_input: np.ndarray,
                           latest_output: Optional[np.ndarray] = None,
                           input_buffer: Optional[List[np.ndarray]] = None,
                           output_buffer: Optional[List[np.ndarray]] = None) -> float:
        output_buffer = self._output_buffer(latest_output, output_buffer)
        features = self.features(control_input, input_buffer=input_buffer,
                                 output_buffer=output_buffer)
        return self.posterior.mutual_information(features)

    @classmethod
    def from_dimensions(cls, input_dim: int, output_dim: int,
                        input_memory: int, output_memory: int,
                        num_harmonics: int = 2,
                        half_period: float | np.ndarray = 1.0,
                        prior_degrees_of_freedom: Optional[float] = None,
                        prior_precision_scale: float = 1e-2,
                        prior_noise_scale: float = 1.0,
                        prior_mean_scale: float = 0.1,
                        parameter_dynamics=None,
                        polynomial_features: bool = True) -> "ParameterEstimator":
        """Estimator with a harmonic Fourier feature map, the weakly
        informative prior of ``MatrixNormalWishart.isotropic_prior`` and
        the parameter node ``parameter_dynamics`` (None: constant). With
        ``polynomial_features`` False, the feature map omits the constant and
        the linear terms and keeps the Fourier terms only."""
        regressor_dim = input_dim * (input_memory + 1) + output_dim * output_memory
        if isinstance(half_period, (int, float)):
            half_period = float(half_period) * np.ones(regressor_dim)
        feature_map = HarmonicFourierBasis(regressor_dim=regressor_dim,
                                           num_harmonics=num_harmonics,
                                           half_periods=half_period,
                                           include_linear=polynomial_features,
                                           include_constant=polynomial_features)
        prior = MatrixNormalWishart.isotropic_prior(
            feature_dim=feature_map.feature_dim, output_dim=output_dim,
            degrees_of_freedom=prior_degrees_of_freedom,
            precision_scale=prior_precision_scale,
            noise_scale=prior_noise_scale,
            mean_scale=prior_mean_scale)
        return cls(input_dim=input_dim, output_dim=output_dim,
                   input_memory=input_memory, output_memory=output_memory,
                   feature_map=feature_map, posterior=prior,
                   parameter_dynamics=parameter_dynamics)


# ==========================================================================
# Persistence: the posterior of one phase is the prior of the next
# ==========================================================================

def save_posterior(path: Union[str, Path],
                   posterior: MatrixNormalWishart) -> None:
    """Save the four parameters of the posterior to an .npz file."""
    np.savez(str(path),
             coefficient_mean=posterior.coefficient_mean,
             coefficient_precision=posterior.coefficient_precision,
             noise_scale=posterior.noise_scale,
             degrees_of_freedom=np.array(posterior.degrees_of_freedom))


def load_posterior(path: Union[str, Path]) -> MatrixNormalWishart:
    """Load a posterior saved by :func:`save_posterior`."""
    with np.load(str(path)) as data:
        return MatrixNormalWishart(
            coefficient_mean=data["coefficient_mean"],
            coefficient_precision=data["coefficient_precision"],
            noise_scale=data["noise_scale"],
            degrees_of_freedom=float(data["degrees_of_freedom"]))


# ==========================================================================
# Planning utilities
# ==========================================================================

def ipopt_options(max_iter: int = 100, max_cpu_time: float = 1.0, tol: float = 1e-4) -> dict:
    """Silent IPOPT options for a cached ``casadi.nlpsol``."""
    return {
        "print_time": 0,
        "ipopt": {"print_level": 0, "sb": "yes", "max_iter": max_iter,
                  "max_cpu_time": max_cpu_time, "tol": tol},
    }


def predictive_noise_covariance(posterior) -> np.ndarray:
    """Ω_eff = Ω / (eta - 2), the covariance of the predictive Student-t at
    zero leverage. The predictive covariance at leverage ℓ is
    (1 + ℓ) Ω_eff. For eta ≤ 2 the covariance does not exist, and the
    planners fall back to the scale 10 Ω / eta."""
    dof = posterior.predictive_degrees_of_freedom
    scale_factor = dof / (dof - 2) if dof > 2 else 10.0
    return posterior.noise_scale * (scale_factor / dof)


def buffer_with_head(buffer: Sequence[np.ndarray], head: Optional[np.ndarray],
                     length: int) -> list:
    """The buffer with ``head`` as its newest entry, truncated to
    ``length``. Without a head, the buffer is returned unchanged."""
    buffer = list(buffer)
    if head is None:
        return buffer
    return [np.atleast_1d(np.asarray(head, dtype=float))] + buffer[:max(length - 1, 0)]


def flatten_buffer(buffer: Sequence[np.ndarray], length: int, entry_dim: int) -> np.ndarray:
    """The first ``length`` entries of a most-recent-first buffer as one
    vector, padded with zeros where the buffer is shorter."""
    parts = [np.asarray(buffer[i], dtype=float).flatten() if i < len(buffer)
             else np.zeros(entry_dim) for i in range(length)]
    return np.concatenate(parts) if parts else np.zeros(0)


def input_nodes(input_dim: int) -> np.ndarray:
    """Nodes on the open input box (-1, 1)^input_dim for averages over
    inputs: a grid of 399 points for one input, a 39 x 39 grid for two, and
    4096 uniform points (fixed seed) for more."""
    if input_dim <= 2:
        grid = np.linspace(-1.0, 1.0, 401 if input_dim == 1 else 41)[1:-1]
        return np.stack(np.meshgrid(*[grid] * input_dim, indexing="ij"),
                        axis=-1).reshape(-1, input_dim)
    return np.random.default_rng(0).uniform(-1.0, 1.0, (4096, input_dim))


def goal_reaching_moments(estimator, goal_prior, nodes: np.ndarray,
                          log_prior_density: np.ndarray,
                          latest_output: Optional[np.ndarray] = None) -> np.ndarray:
    """Second moment E[φφᵀ] of the features of the next regressor under the
    goal-reaching inputs

        p̃(u) ∝ p(u) N(y_* | M_tᵀ φ(u, ȳ_t), S_*),

    i.e. the inputs whose mean prediction lies at the goal, on the input
    ``nodes`` with the log-density ``log_prior_density`` of the control
    prior p(u) (up to a constant). Without a caution term, p̃ keeps every
    input that the mean model cannot rule out: a mean model that is flat in
    u gives p̃ = p. The expected leverage of these inputs is tr(Λ_t⁻¹ E[φφᵀ])."""
    features = estimator.batch_features(nodes, latest_output)
    goal_deviation = (features @ estimator.posterior.coefficient_mean
                      - np.asarray(goal_prior.mean, dtype=float))
    log_weights = np.asarray(log_prior_density, dtype=float) - 0.5 * np.einsum(
        "ij,jk,ik->i", goal_deviation, np.linalg.inv(goal_prior.covariance), goal_deviation)
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    return (features * weights[:, None]).T @ features


def tempered_posterior(prior, features: np.ndarray, outputs: np.ndarray, weight: float = 1.0):
    """Matrix-normal Wishart posterior after the ``features`` (N, feature_dim)
    and ``outputs`` (N, output_dim), with every data point counted ``weight``
    times.

    For weight = 1 this equals N sequential ``MatrixNormalWishart.update``
    calls. A weight below one tempers the likelihood (a power posterior),
    which turns data from a nominal simulator into an informative prior
    with the confidence of weight * N observations.
    """
    features, outputs = np.atleast_2d(features), np.atleast_2d(outputs)
    precision = prior.coefficient_precision + weight * features.T @ features
    mean = np.linalg.solve(precision, prior.coefficient_precision @ prior.coefficient_mean
                           + weight * features.T @ outputs)
    noise_scale = (prior.noise_scale + weight * outputs.T @ outputs
                   + prior.coefficient_mean.T @ prior.coefficient_precision @ prior.coefficient_mean
                   - mean.T @ precision @ mean)
    return type(prior)(coefficient_mean=mean, coefficient_precision=precision,
                       noise_scale=0.5 * (noise_scale + noise_scale.T),
                       degrees_of_freedom=prior.degrees_of_freedom + weight * len(outputs))


@dataclass
class ActionScaler:
    """Affine map between agent actions in [-1, 1]^D and physical inputs in
    the box [low, high]."""
    low: np.ndarray
    high: np.ndarray

    def __post_init__(self):
        self.low = np.asarray(self.low, dtype=float)
        self.high = np.asarray(self.high, dtype=float)

    @classmethod
    def symmetric(cls, limit, dimension: int) -> "ActionScaler":
        limit = np.broadcast_to(np.asarray(limit, dtype=float), (dimension,))
        return cls(-limit, limit)

    def to_physical(self, action) -> np.ndarray:
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        return self.low + 0.5 * (action + 1.0) * (self.high - self.low)

    def to_agent(self, physical_input) -> np.ndarray:
        physical_input = np.asarray(physical_input, dtype=float)
        return np.clip(2.0 * (physical_input - self.low) / (self.high - self.low) - 1.0,
                       -1.0, 1.0)
