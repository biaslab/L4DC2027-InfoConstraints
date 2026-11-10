"""Symmetric Beta distribution on [-1, 1]^D for bounded controls."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import betaln


@dataclass
class SymmetricBeta:
    """Symmetric Beta distribution on [-1, 1]^dimension. Each component
    u_i is independently distributed as

        p(u_i) ∝ (1 - u_i^2)^(shape - 1).

    For shape > 1 it is unimodal at 0 with soft bounds at ±1; for shape = 1
    it is uniform on [-1, 1].
    """
    dimension: int
    shape: float = 2.0

    def __post_init__(self):
        if self.shape <= 0:
            raise ValueError("shape must be positive")

    @property
    def lower_bounds(self) -> np.ndarray:
        return -np.ones(self.dimension)

    @property
    def upper_bounds(self) -> np.ndarray:
        return np.ones(self.dimension)

    def log_pdf_casadi(self, input_symbol):
        """Symbolic unnormalised log-density used by the planner NLP."""
        import casadi as ca
        return (self.shape - 1.0) * ca.sum1(ca.log(1.0 - input_symbol * input_symbol))

    def cost(self, control_input: np.ndarray) -> float:
        """Input cost c_u(u) = -ln p(u) up to a constant: the logarithmic
        barrier -(shape - 1) sum_i ln(1 - u_i^2), infinite outside (-1, 1)^D."""
        control_input = np.atleast_1d(np.asarray(control_input, dtype=float))
        if np.any(np.abs(control_input) >= 1.0):
            return np.inf
        return float(-(self.shape - 1.0) * np.sum(np.log1p(-control_input * control_input)))

    def cost_derivatives(self, control_input: np.ndarray):
        """Gradient and diagonal Hessian of ``cost`` at u inside (-1, 1)^D.
        They define the Laplace approximation of the prior at u, a Gaussian
        message with precision 2(shape-1)(1+u^2)/(1-u^2)^2, which grows
        without bound towards the actuator limits."""
        control_input = np.atleast_1d(np.asarray(control_input, dtype=float))
        distance_to_limits = 1.0 - control_input * control_input
        return (2.0 * (self.shape - 1.0) * control_input / distance_to_limits,
                2.0 * (self.shape - 1.0) * (1.0 + control_input * control_input)
                / distance_to_limits**2)

    def log_pdf(self, control_input: np.ndarray) -> float:
        control_input = np.atleast_1d(np.asarray(control_input, dtype=float))
        if np.any(np.abs(control_input) >= 1.0):
            return -np.inf
        # log normaliser per dimension: -log(2^(2 shape - 1) B(shape, shape))
        log_normaliser_per_dimension = -((2 * self.shape - 1) * np.log(2)
                                         + betaln(self.shape, self.shape))
        return float(np.sum((self.shape - 1) * np.log(1 - control_input**2))
                     + self.dimension * log_normaliser_per_dimension)

    def log_pdf_unnormalized(self, control_input: np.ndarray) -> float:
        """Log-density without the normalising constant."""
        control_input = np.atleast_1d(np.asarray(control_input, dtype=float))
        if np.any(np.abs(control_input) >= 1.0):
            return -np.inf
        return float(np.sum((self.shape - 1) * np.log(1 - control_input**2)))

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        # Beta(shape, shape) on [0, 1], shifted to [-1, 1]
        unit = rng.beta(self.shape, self.shape, size=self.dimension)
        return 2 * unit - 1

    def __repr__(self) -> str:
        return f"SymmetricBeta(dimension={self.dimension}, shape={self.shape})"
