"""Control priors of the CBFE-ARX agent.

Both implement ``lower_bounds``, ``upper_bounds`` and ``log_pdf_casadi``,
which is what the planner's NLP needs. The model distributions (normal, Student-t,
matrix-normal Wishart) are shared with the dual MPC and live in
``controllers.common``.
"""

from .beta import SymmetricBeta
from .exponential import Exponential

__all__ = ["SymmetricBeta", "Exponential"]
