"""Exponential distribution on [0, upper_bound]^D for non-negative controls."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Exponential:
    """Independent exponential distribution per component on [0, ∞),

        p(u_i) = rate exp(-rate u_i),    u_i >= 0.

    The mode is 0 (the prior pulls u towards 0) and the mean is 1/rate. The
    support is unbounded above, but IPOPT needs finite bounds, so the
    planner uses the action box [0, upper_bound]^dimension.
    """
    dimension: int
    rate: float = 1.0
    upper_bound: float = 50.0

    def __post_init__(self):
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        if self.upper_bound <= 0:
            raise ValueError("upper_bound must be positive")

    @property
    def lower_bounds(self) -> np.ndarray:
        return np.zeros(self.dimension)

    @property
    def upper_bounds(self) -> np.ndarray:
        return np.full(self.dimension, self.upper_bound)

    def log_pdf(self, control_input: np.ndarray) -> float:
        control_input = np.atleast_1d(np.asarray(control_input, dtype=float))
        if np.any(control_input < 0.0):
            return -np.inf
        return float(self.dimension * np.log(self.rate) - self.rate * np.sum(control_input))

    def log_pdf_unnormalized(self, control_input: np.ndarray) -> float:
        control_input = np.atleast_1d(np.asarray(control_input, dtype=float))
        if np.any(control_input < 0.0):
            return -np.inf
        return float(-self.rate * np.sum(control_input))

    def log_pdf_casadi(self, input_symbol):
        """Symbolic unnormalised log-density, used by the planner NLP, where
        additive constants are immaterial."""
        import casadi as ca
        return -self.rate * ca.sum1(input_symbol)

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.exponential(scale=1.0 / self.rate, size=self.dimension)

    def __repr__(self) -> str:
        return (f"Exponential(dimension={self.dimension}, rate={self.rate}, "
                f"upper_bound={self.upper_bound})")
