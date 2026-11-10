"""Explicit dual-control MPC on the CBFE-ARX model, with a fixed probing weight."""

from .dual_mpc import DualMPC, DualMPCConfig

__all__ = ["DualMPC", "DualMPCConfig"]
