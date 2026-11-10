"""Agents of the ECC 2027 experiments.

``cbfe_arx``  constrained Bethe free energy agent on a Bayesian ARX model
``dual_mpc``  explicit dual-control MPC on the same model
``pilco``     PILCO with GP-mean rollouts
``tdmpc2``    TD-MPC2 (needs torch)
``common``    utilities shared by the agents

The agents load on first access, so importing one agent does not require
the dependencies of the others (scikit-learn for PILCO, torch for TD-MPC2).
"""

from importlib import import_module

_LOCATIONS = {
    "Agent": "agents.cbfe_arx",
    "DualMPC": "agents.dual_mpc",
    "PILCOController": "agents.pilco",
    "TDMPC2": "agents.tdmpc2",
}

__all__ = list(_LOCATIONS)


def __getattr__(name):
    if name in _LOCATIONS:
        return getattr(import_module(_LOCATIONS[name]), name)
    raise AttributeError(f"module 'agents' has no attribute '{name}'")
