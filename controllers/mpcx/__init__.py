"""MPC with integrated experiment design (MPC-X) after Larsson et al. (ECC
2013), on the ARX model of ``controllers.common``."""

from .mpcx import MPCX, MPCXConfig

__all__ = ["MPCX", "MPCXConfig"]
