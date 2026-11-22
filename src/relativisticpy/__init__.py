r"""RelativisticPy reads mathematics as it is written, in LaTeX, plain text or Unicode,
and computes it: General Relativity with SymPy, numerical data in index notation over any
space, and the equations of LaTeX papers, which it checks.

It is a language. A :class:`Workbook` runs programs written in it::

    >>> from relativisticpy import Workbook
    >>> wb = Workbook()
    >>> wb.expr(r'''
    ...     ds^2 = -(1 - \frac{2M}{r}) dt^2 + \frac{dr^2}{1 - \frac{2M}{r}} + r^2 d\Omega^2
    ...     R_{abcd} R^{abcd}
    ... ''')
    48*M**2/r**6

From the command line, ``relativisticpy FILE`` runs a program or a Markdown notebook,
and ``relativisticpy check paper.tex`` checks the equations of a paper.

Before writing a program:

- Learn the language offline: ``relativisticpy help reference`` prints all of it on one
  page, as :func:`relativisticpy.reference.markdown` gives it; ``relativisticpy help``
  lists the other pages of the documentation.
- ``:=`` always defines. ``=`` defines only what has no meaning yet: a new name,
  function or tensor, or a line element. Otherwise it is a claim, which is checked, or
  an equation to solve.
- Write LaTeX in Python in raw strings, ``r"\frac{1}{2}"``: in an ordinary string,
  Python turns ``\f``, ``\t``, ``\n`` and the like into control characters.

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
