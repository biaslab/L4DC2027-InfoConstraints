"""CBFE agent for a quadcopter in wind: rotor drag that grows with the thrust,
a position fix and an IMU (main.tex, Sections 2-4)."""
from .agent import Agent, Config, functions

__all__ = ["Agent", "Config", "functions"]
