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

The API at a glance:

- :class:`Workbook` runs programs: ``wb.expr(source)`` gives the value of the last
  expression, ``wb.exe(source)`` every value, ``wb.check(document)`` checks a paper.
- Results are SymPy values, or :class:`Tensor` (and :class:`Metric`),
  :class:`TensorEquation`, :class:`IndexEquation` and :class:`IndexExpression`;
  a check gives a :class:`Report` of :class:`Finding` values, each with a
  :class:`Verdict`.
- :func:`holds` says whether any claim holds; :mod:`relativisticpy.display` shows any
  result in Unicode, text, LaTeX or as data; :func:`latex` typesets one.
- :mod:`relativisticpy.spacetimes` builds the known metrics, and
  :mod:`relativisticpy.numeric` turns symbols into numbers and integrates geodesics.
- Every error a user can cause is a :class:`RelativisticPyError`.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from relativisticpy import display, spacetimes
from relativisticpy.abstract import IndexEquation, IndexExpression
from relativisticpy.checking import Finding, Report, Verdict
from relativisticpy.diffgeom import Metric
from relativisticpy.display import latex
from relativisticpy.documents import Document
from relativisticpy.errors import (
    CompileError,
    DataError,
    EvaluationError,
    FileError,
    IndexNotationError,
    InternalError,
    MetricError,
    NotComputedError,
    ParseError,
    RelativisticPyError,
    UnknownNameError,
    UsageError,
)
from relativisticpy.tensors import Index, Indices, Space, Tensor
from relativisticpy.tensors.equation import TensorEquation, UndecidedError, holds
from relativisticpy.workbook import Step, Workbook

try:
    __version__ = _version("relativisticpy")
except _PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0+unknown"

__all__ = [
    "CompileError",
    "DataError",
    "Document",
    "EvaluationError",
    "FileError",
    "Finding",
    "Index",
    "IndexEquation",
    "IndexExpression",
    "IndexNotationError",
    "Indices",
    "InternalError",
    "Metric",
    "MetricError",
    "NotComputedError",
    "ParseError",
    "RelativisticPyError",
    "Report",
    "Space",
    "Step",
    "Tensor",
    "TensorEquation",
    "UndecidedError",
    "UnknownNameError",
    "UsageError",
    "Verdict",
    "Workbook",
    "__version__",
    "display",
    "holds",
    "latex",
    "spacetimes",
]
