r"""Expressions: SymPy's, and anything that makes itself one, through ``_sympy_``.

A SymPy expression is an expression of the program's, its symbols the program's by name, so
that ``\frac{d}{dx} \texttt{f}`` differentiates with respect to the x the program writes. A
SymPy matrix or array is a matrix, as the language's own lists are, and a ``Lambda`` is a
function, called as the program calls it.
"""

from __future__ import annotations

from typing import Any

import sympy

from relativisticpy.integrations import Array, Example, PythonFunction, Scalar, python_object

x, y = sympy.symbols("x y")


@python_object(
    "sympy",
    takes=["sympy.Basic", "sympy.matrices.MatrixBase", "sympy.NDimArray", "_sympy_"],
    summary="a SymPy expression, or anything with _sympy_: an expression, its symbols the "
    "program's by name; a SymPy matrix is a matrix",
    examples=[
        Example(
            r"\frac{d}{dx} \texttt{f}",
            2 * sympy.sin(x) * sympy.cos(x),
            "d/dx sin²x",
            given=lambda: {"f": sympy.sin(x) ** 2},
        ),
        Example(
            r"\det(\texttt{M}) - y",
            x**2 - 2 * y,
            "the determinant of [[x, y], [1, x]], less y",
            given=lambda: {"M": sympy.Matrix([[x, y], [1, x]])},
        ),
    ],
)
class Expressions:
    def becomes(self, value: Any) -> Scalar | Array | PythonFunction:
        if isinstance(value, sympy.MatrixBase | sympy.NDimArray):
            return Array(sympy.ImmutableDenseNDimArray(value.tolist()))
        if isinstance(value, sympy.Lambda):
            return PythonFunction(value, "lambda")
        expression = value if isinstance(value, sympy.Basic) else value._sympy_()
        return Scalar(expression)
