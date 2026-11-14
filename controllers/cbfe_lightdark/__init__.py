"""CBFE controllers for a light-dark positioning task: a hidden state whose
measurement noise depends on its position, with an information constraint on
the observation node and, with an unknown actuator gain, one on the
transition node as well (main.tex, Section V-C)."""
from .agent import GridFilter, LightDarkAgent, LightDarkConfig, LightDarkModel
from .gain import GainConfig, GainLightDarkAgent, JointGridFilter
from .directional import DirectionalAgent, DirectionalGridFilter

__all__ = ["DirectionalAgent", "DirectionalGridFilter", "GainConfig", "GainLightDarkAgent", "GridFilter", "JointGridFilter", "LightDarkAgent",
           "LightDarkConfig", "LightDarkModel"]
