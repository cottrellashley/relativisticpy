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
  page, as :func:`relativisticpy.reference.markdown` gives it, and
  ``relativisticpy help NAME`` one function in full; ``relativisticpy help`` lists the
  other pages of the documentation.
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

Each name is imported the first time it is asked for, so that importing a module of the
package, as the command line does to start, imports nothing else.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

    __version__: str

#: The module each name the package exports is defined in, other than its modules.
_DEFINED_IN: dict[str, str] = {
    "IndexEquation": "relativisticpy.abstract",
    "IndexExpression": "relativisticpy.abstract",
    "Finding": "relativisticpy.checking",
    "Report": "relativisticpy.checking",
    "Verdict": "relativisticpy.checking",
    "Metric": "relativisticpy.diffgeom",
    "latex": "relativisticpy.display",
    "Document": "relativisticpy.documents",
    "CompileError": "relativisticpy.errors",
    "DataError": "relativisticpy.errors",
    "EvaluationError": "relativisticpy.errors",
    "FileError": "relativisticpy.errors",
    "IndexNotationError": "relativisticpy.errors",
    "InternalError": "relativisticpy.errors",
    "MetricError": "relativisticpy.errors",
    "NotComputedError": "relativisticpy.errors",
    "ParseError": "relativisticpy.errors",
    "RelativisticPyError": "relativisticpy.errors",
    "UnknownNameError": "relativisticpy.errors",
    "UsageError": "relativisticpy.errors",
    "Index": "relativisticpy.tensors",
    "Indices": "relativisticpy.tensors",
    "Space": "relativisticpy.tensors",
    "Tensor": "relativisticpy.tensors",
    "TensorEquation": "relativisticpy.tensors.equation",
    "UndecidedError": "relativisticpy.tensors.equation",
    "holds": "relativisticpy.tensors.equation",
    "Step": "relativisticpy.workbook",
    "Workbook": "relativisticpy.workbook",
}

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


def __getattr__(name: str) -> Any:
    """A name the package exports, or a module of it, imported the first time it is asked
    for (PEP 562), and kept."""
    if name == "__version__":
        value: Any = _version()
    elif name in _DEFINED_IN:
        value = getattr(importlib.import_module(_DEFINED_IN[name]), name)
    elif name.isidentifier() and not name.startswith("__"):
        value = _module(name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def _module(name: str) -> Any:
    """The module of the package called ``name``, as ``relativisticpy.checking``; an
    attribute error where there is none."""
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as error:
        if error.name != f"{__name__}.{name}":
            raise  # a module that is there, which needs one that is not
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


def _version() -> str:
    """The version installed, or ``0+unknown`` in a source tree that is not installed."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("relativisticpy")
    except PackageNotFoundError:  # pragma: no cover - running from a source tree
        return "0+unknown"
