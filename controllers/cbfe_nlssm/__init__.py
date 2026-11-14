"""CBFE agent for a nonlinear state-space model with unknown parameters and a
noisy measurement: a geared joint under gravity with unknown gravity torque
and motor gain (main.tex, Sections 2-4)."""
from .agent import Agent, Config

__all__ = ["Agent", "Config"]
