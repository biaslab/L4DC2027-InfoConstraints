"""Controllers of the ECC 2027 experiments.

``cbfe_arx``  constrained Bethe free energy agent on a Bayesian ARX model
``dual_mpc``  explicit dual-control MPC on the same model
``pilco``     PILCO with GP-mean rollouts
``tdmpc2``    TD-MPC2 (needs torch)
``common``    utilities shared by the controllers

The controllers load on first access, so importing one agent does not require
the dependencies of the others (scikit-learn for PILCO, torch for TD-MPC2).
"""

from importlib import import_module

_LOCATIONS = {
    "Agent": "controllers.cbfe_arx",
    "DualMPC": "controllers.dual_mpc",
    "PILCOController": "controllers.pilco",
    "TDMPC2": "controllers.tdmpc2",
}

__all__ = list(_LOCATIONS)


def __getattr__(name):
    if name in _LOCATIONS:
        return getattr(import_module(_LOCATIONS[name]), name)
    raise AttributeError(f"module 'controllers' has no attribute '{name}'")
