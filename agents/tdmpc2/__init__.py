"""TD-MPC2 baseline (Hansen, Su & Wang, ICLR 2024). Requires torch."""

from .tdmpc2 import TDMPC2, TDMPC2Config

__all__ = ["TDMPC2", "TDMPC2Config"]
