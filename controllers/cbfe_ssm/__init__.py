"""CBFE agent for a scalar state-space model with a hidden state, an unknown
input gain, separate state- and parameter-information constraints and a
chance constraint on the next state (main.tex, Section V-C)."""
from .agent import HiddenStateAgent, HiddenStateConfig

__all__ = ["HiddenStateAgent", "HiddenStateConfig"]
