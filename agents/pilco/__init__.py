"""PILCO baselines (Deisenroth & Rasmussen, 2011) with GP-mean rollouts.

``PILCOController`` is the controller of the thermal-stage environment of
the TAC 2026 project, whose state estimate combines a position/velocity
observer and a thermal Kalman filter with that plant's nominal parameters.
``StatePILCO`` runs the same algorithm on a directly measured state, such as
the joint angles and velocities of the arm.
"""

from .pilco import GPDynamics, PILCOConfig, PILCOController
from .state_feedback import StatePILCO, StatePILCOConfig

__all__ = ["GPDynamics", "PILCOConfig", "PILCOController", "StatePILCO", "StatePILCOConfig"]
