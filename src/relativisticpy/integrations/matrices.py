r"""Matrices.

Functions of a matrix written as a list of rows or as an environment, with SymPy: a
diagonal matrix, the determinant, the inverse, the transpose and the trace. Given a tensor
written with its indices, ``det``, ``tr``, ``inv`` and ``diag`` are linear algebra on it
instead, as the reference says.
"""

from __future__ import annotations

from typing import Any

import sympy

from relativisticpy.errors import EvaluationError
from relativisticpy.integrations import Example, map_function
from relativisticpy.tensors import Tensor

a, b, c, d = sympy.symbols("a b c d")


def _matrix(value: Any) -> sympy.Matrix:
    if isinstance(value, Tensor):
        value = value.components
    if isinstance(value, sympy.NDimArray):
        value = value.tolist()
    return sympy.Matrix(value)


def _square(value: Any, what: str) -> sympy.Matrix:
    """``value`` as a square matrix, for ``what`` needs one; an error otherwise."""
    matrix = _matrix(value)
    rows, columns = matrix.shape
    if rows != columns:
        flat = isinstance(value, list) and not any(isinstance(row, list) for row in value)
        flat = flat or (isinstance(value, sympy.NDimArray) and value.rank() == 1)
        written = f"a list of {len(matrix)}" if flat else f"a {rows}×{columns} matrix"
        raise EvaluationError(f"{what} needs a square matrix, but this is {written}")
    return matrix


@map_function(
    "diag",
    summary="`diag(a, b, ...)`, a diagonal matrix",
    examples=[
        Example("diag(1, 2)", [[1, 0], [0, 2]], "standard"),
        Example("diag(a, b, c)", [[a, 0, 0], [0, b, 0], [0, 0, c]], "standard"),
    ],
)
class Diagonal:
    def exact(self, first: Any, *rest: Any) -> Any:
        return sympy.ImmutableDenseNDimArray(sympy.diag(first, *rest).tolist())


@map_function(
    "det",
    summary="the determinant",
    examples=[
        Example("det([[1, 2], [3, 4]])", -2, "standard"),
        Example("det([[a, b], [c, d]])", a * d - b * c, "standard"),
    ],
)
class Determinant:
    def exact(self, matrix: Any) -> Any:
        """The determinant of a matrix; a scalar is its own determinant."""
        if isinstance(matrix, sympy.NDimArray | sympy.MatrixBase | list | Tensor):
            return sympy.simplify(_square(matrix, "a determinant").det())
        return matrix


@map_function(
    "inverse",
    aliases=["inv"],
    summary="the inverse of a matrix",
    examples=[
        Example("inverse([[1, 2], [3, 4]])",
                [[-2, 1], [sympy.Rational(3, 2), sympy.Rational(-1, 2)]], "standard"),
        Example("inv([[2, 0], [0, 4]])",
                [[sympy.Rational(1, 2), 0], [0, sympy.Rational(1, 4)]], "standard"),
    ],
)  # fmt: skip
class Inverse:
    def exact(self, matrix: Any) -> Any:
        square = _square(matrix, "an inverse")
        try:
            inverted = square.inv()
        except sympy.matrices.exceptions.NonInvertibleMatrixError:
            raise EvaluationError("this matrix has no inverse: its determinant is zero") from None
        return sympy.ImmutableDenseNDimArray(inverted.tolist())


@map_function(
    "transpose",
    summary="the transpose of a matrix",
    examples=[
        Example("transpose([[1, 2], [3, 4]])", [[1, 3], [2, 4]], "standard"),
        Example("transpose([[1, 2, 3]])", [[1], [2], [3]], "standard"),
    ],
)
class Transpose:
    def exact(self, matrix: Any) -> Any:
        return sympy.ImmutableDenseNDimArray(_matrix(matrix).T.tolist())


@map_function(
    "trace",
    summary="the trace of a matrix; of an equation between tensors, its contraction",
    examples=[
        Example("trace([[1, 2], [3, 4]])", 5, "standard"),
        Example("trace([[a, b], [c, d]])", a + d, "standard"),
    ],
    scalar=True,
)
class Trace:
    def exact(self, matrix: Any) -> Any:
        return _matrix(matrix).trace()
