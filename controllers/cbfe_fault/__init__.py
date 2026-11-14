"""CBFE agent for an integrator with actuator faults: an exact filter over the
fault hypotheses, information constraints on the mode and gain layers, and a
Gauss-Newton message planner (main.tex, Section V-C)."""
from .agent import (MODES, FaultAgent, FaultConfig, FaultFilter, FaultModel, mixture_entropy,
                    rate_distortion)

__all__ = ["MODES", "FaultAgent", "FaultConfig", "FaultFilter", "FaultModel", "mixture_entropy",
           "rate_distortion"]
