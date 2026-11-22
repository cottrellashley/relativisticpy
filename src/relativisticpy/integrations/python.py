r"""Python's own values: numbers, lists of numbers and functions.

A number is a scalar, exact where Python's is: an integer, or a fraction, stays one. A list
of numbers, or of lists of them, is a vector or a matrix, as the language's own lists are.
A function, or anything else Python calls, is called in a program, as
``\texttt{hann}(n, N)``: on numbers, and on data elementwise.
"""

from __future__ import annotations

import fractions
import numbers
from collections.abc import Sequence
from typing import Any

import sympy

from relativisticpy.errors import DataError
from relativisticpy.integrations import Array, Example, PythonFunction, Scalar, python_object

x = sympy.Symbol("x")


@python_object(
    "numbers",
    takes=[
        "builtins.int",
        "builtins.float",
        "builtins.complex",
        "builtins.bool",
        "fractions.Fraction",
        "decimal.Decimal",
        "numbers.Number",
    ],
    summary="a number: a scalar, exact where Python's is, as an integer or a fraction",
    examples=[
        Example(
            r"r_s := \frac{2 \texttt{G} \texttt{M}}{\texttt{c}^2}" "\n" "r_s",
            2 * 6.6743e-11 * 1.98847e30 / 299792458.0**2,
            "2GM/c² of the Sun, computed in Python",
            given=lambda: {"G": 6.6743e-11, "M": 1.98847e30, "c": 299792458.0},
        ),
        Example(
            r"\texttt{third} + \frac{2}{3}",
            1,
            "1/3 + 2/3, exactly",
            given=lambda: {"third": fractions.Fraction(1, 3)},
        ),
    ],
)
class Numbers:
    def becomes(self, value: Any) -> Scalar:
        if isinstance(value, bool):  # a truth value, as an equation that holds is
            return Scalar(sympy.true if value else sympy.false)
        return Scalar(sympy.sympify(value, strict=True))


@python_object(
    "lists",
    takes=["builtins.list", "builtins.tuple"],
    summary="a list or tuple of numbers, or of lists of them: a vector or a matrix, "
    "as the language's own lists are",
    examples=[
        Example(
            r"\det(\texttt{A})",
            -2,
            "1·4 − 2·3",
            given=lambda: {"A": [[1, 2], [3, 4]]},
        ),
        Example(
            r"\texttt{A}^{-1}",
            [[-2, 1], [sympy.Rational(3, 2), sympy.Rational(-1, 2)]],
            "the inverse of a 2 × 2 matrix, exactly",
            given=lambda: {"A": ((1, 2), (3, 4))},
        ),
    ],
)
class Lists:
    def becomes(self, value: Sequence[Any]) -> Array:
        _numbers_only(value)
        ragged = _ragged(value)
        if ragged:
            raise DataError(f"the object is ragged: {ragged}")
        return Array(sympy.ImmutableDenseNDimArray(_entries(value)))


@python_object(
    "functions",
    takes="__call__",
    summary="a function, or anything called: called in a program, on numbers, and on data "
    "elementwise",
    examples=[
        Example(
            r"\texttt{square}(3) + 1",
            10,
            "3² + 1",
            given=lambda: {"square": lambda number: number * number},
        ),
        Example(
            r"\frac{d}{dx} \texttt{cube}(x)",
            3 * x**2,
            "the derivative of x³",
            given=lambda: {"cube": lambda number: number**3},
        ),
    ],
)
class Functions:
    def becomes(self, value: Any) -> PythonFunction:
        return PythonFunction(value, getattr(value, "__name__", type(value).__name__))


def _numbers_only(value: Any) -> None:
    """Refuse what is no number, nor a list of them, anywhere in nested lists: text above
    all, which SymPy would read as Python, and run."""
    if isinstance(value, list | tuple):
        for item in value:
            _numbers_only(item)
        return
    if isinstance(value, str | bytes):
        raise DataError(
            f"the object holds text, {value[:40]!r}: read numbers",
            hint="to compute with an expression written as text, write it in the program",
        )
    if not isinstance(value, numbers.Number | sympy.Basic):
        raise DataError(f"the object holds a {type(value).__name__}: read numbers")


def _entries(value: Any) -> Any:
    """Nested lists of SymPy numbers, exact where Python's are."""
    if isinstance(value, list | tuple):
        return [_entries(item) for item in value]
    return sympy.sympify(value, strict=True)


def _ragged(value: Sequence[Any]) -> str:
    """Where nested lists are not rectangular, in words; empty where they are."""
    nested = [isinstance(item, list | tuple) for item in value]
    if not any(nested):
        return ""
    if not all(nested):
        return f"entry {nested.index(False) + 1} is a number, and others are lists"
    lengths = [len(item) for item in value]
    for position, length in enumerate(lengths):
        if length != lengths[0]:
            return f"entry {position + 1} has {length} entries, but entry 1 has {lengths[0]}"
    for position, item in enumerate(value):
        inner = _ragged(item)
        if inner:
            return f"in entry {position + 1}, {inner}"
    return ""
