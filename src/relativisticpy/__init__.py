r"""RelativisticPy: General Relativity for physicists in a hurry.

Write tensor equations the way they look on paper, or paste the LaTeX, and
compute them::

    >>> from relativisticpy import Workbook
    >>> wb = Workbook()
    >>> wb.expr(r'''
    ...     Coordinates := [t, r, \theta, \phi]
    ...     ds^2 = -(1 - \frac{2M}{r}) dt^2 + \frac{dr^2}{1 - \frac{2M}{r}} + r^2 d\Omega^2
    ...     R_{abcd} R^{abcd}
    ... ''')
    48*M**2/r**6
"""

from importlib.metadata import PackageNotFoundError, version

from relativisticpy import spacetimes
from relativisticpy.diffgeom import Metric
from relativisticpy.errors import (
    CompileError,
    EvaluationError,
    IndexNotationError,
    ParseError,
    RelativisticPyError,
)
from relativisticpy.tensors import Index, Indices, Tensor
from relativisticpy.tensors.equation import TensorEquation
from relativisticpy.workbook import Workbook, latex

try:
    __version__ = version("relativisticpy")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0+unknown"

__all__ = [
    "CompileError",
    "EvaluationError",
    "Index",
    "IndexNotationError",
    "Indices",
    "Metric",
    "ParseError",
    "RelativisticPyError",
    "Tensor",
    "TensorEquation",
    "Workbook",
    "__version__",
    "latex",
    "spacetimes",
]
