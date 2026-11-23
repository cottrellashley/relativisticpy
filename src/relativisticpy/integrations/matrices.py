r"""Matrices.

Functions of a matrix written as a list of rows or as an environment, by the names books
give them: the trace ``\operatorname{tr} A``, the rank, the determinant, the inverse and
the pseudoinverse ``A^+``, the transpose ``A^T`` and the conjugate transpose
``A^\dagger``, the exponential ``e^{A}``, the logarithm and the square root, powers
``A^k``, the spectral radius ``\rho(A)``, the eigenvalues ``\lambda_{\max}(A)`` and
``\lambda_k(A)``, norms ``\|A\|_2`` and the condition number ``\kappa(A)``, the
adjugate, the null space ``\ker A`` and its dimension, ``\operatorname{vec}``, and the
Kronecker product ``A \otimes B`` and direct sum ``A \oplus B``. Exact with SymPy, and
numeric with NumPy and SciPy, or JAX for JAX's arrays.

Given a map written with its indices, as ``A^a{}_b``, a function of a matrix acts on
the map, and its value is over the same spaces. The notation that is no call, as
``A^T`` and ``e^{A t}``, is read in :mod:`relativisticpy.reading.matrices`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import sympy

from relativisticpy.errors import NO_INVERSE, EvaluationError
from relativisticpy.integrations import Example, map_function
from relativisticpy.tensors import Tensor
from relativisticpy.tensors.numerical import whole_number

a, b, c, d, k, t = sympy.symbols("a b c d k t")

#: Matrices the functions of a matrix are checked at: with positive eigenvalues, so that
#: each function has its principal value there, symmetric or not.
_SQUARE = [
    ([[2, 1], [1, 3]],),
    ([[4, 1], [2, 3]],),
    ([[5, 2, 0], [2, 5, 0], [0, 0, 4]],),
]


def _matrix(value: Any, what: str = "this") -> sympy.Matrix:
    """``value`` as a matrix: a list of rows, a list as a column, or an array; an error
    for anything else, as a number, saying that ``what`` takes a matrix."""
    if isinstance(value, Tensor):
        value = value.components
    if isinstance(value, sympy.NDimArray) and value.rank() == 2:
        return sympy.Matrix(*value.shape, sympy.flatten(value.tolist()))
    if isinstance(value, sympy.NDimArray):
        value = value.tolist()
    if isinstance(value, sympy.MatrixBase | list | tuple) or hasattr(value, "__array__"):
        return sympy.Matrix(value)
    raise EvaluationError(f"{what} takes a matrix, as [[1, 2], [3, 4]], not {value}")


def _square(value: Any, what: str) -> sympy.Matrix:
    """``value`` as a square matrix, for ``what`` needs one; an error otherwise."""
    matrix = _matrix(value, what)
    rows, columns = matrix.shape
    if rows != columns:
        flat = isinstance(value, list) and not any(isinstance(row, list) for row in value)
        flat = flat or (isinstance(value, sympy.NDimArray) and value.rank() == 1)
        written = f"a list of {len(matrix)}" if flat else f"a {rows}×{columns} matrix"
        raise EvaluationError(f"{what} needs a square matrix, but this is {written}")
    return matrix


def _array(matrix: sympy.MatrixBase) -> sympy.ImmutableDenseNDimArray:
    """A matrix as the language's values are, an array, of any shape, even n × 0."""
    return sympy.ImmutableDenseNDimArray(list(matrix), matrix.shape)


def _simplified(matrix: sympy.MatrixBase) -> sympy.ImmutableDenseNDimArray:
    """A matrix as an array, each entry simplified."""
    return _array(matrix.applyfunc(sympy.simplify))


def _of_decimals(
    matrix: sympy.MatrixBase, name: str, *rest: Any
) -> sympy.ImmutableDenseNDimArray | None:
    """The function of SciPy's linear algebra called ``name`` of a matrix of numbers among
    which is a decimal, in decimals, as SymPy's would give it, but at once, where SymPy's
    takes minutes for three rows; ``None`` for a matrix with symbols, or no decimal."""
    if matrix.free_symbols or not (matrix.has(sympy.Float) or rest):
        return None
    import numpy as np
    import scipy.linalg

    values = np.array(matrix.tolist(), dtype=complex)
    real = not values.imag.any()
    result = np.asarray(getattr(scipy.linalg, name)(values.real if real else values, *rest))
    if np.iscomplexobj(result) and np.allclose(result.imag, 0, atol=1e-12):
        result = result.real
    return sympy.ImmutableDenseNDimArray(
        [sympy.sympify(value) for value in result.ravel().tolist()], result.shape
    )


def _vector(matrix: sympy.MatrixBase) -> bool:
    """Whether a matrix is a vector, of one row or one column."""
    return 1 in matrix.shape


def _matrix_like(value: Any) -> bool:
    """Whether a value is written as a matrix, rather than as a number or a symbol."""
    return isinstance(value, sympy.NDimArray | sympy.MatrixBase | list | tuple | Tensor)


def _linalg(xp: Any) -> Any:
    """SciPy's linear algebra, or JAX's for JAX's arrays."""
    if xp.__name__.startswith("jax"):
        import jax.scipy.linalg

        return jax.scipy.linalg
    import scipy.linalg

    return scipy.linalg


def _through_eigenvalues(xp: Any, matrix: Any, f: Callable[[Any], Any]) -> Any:
    """``f`` of a diagonalisable matrix, through its eigenvalues: V f(Λ) V⁻¹."""
    values, vectors = xp.linalg.eig(matrix)
    return vectors @ xp.diag(f(values)) @ xp.linalg.inv(vectors)


def _eigenvalues(value: Any, what: str) -> list[Any]:
    """The eigenvalues of a square matrix, each as often as it is one, in increasing
    order: by real part, then by imaginary part; an error where they hold symbols."""
    values = _square(value, what).eigenvals(multiple=True)
    try:
        return sorted(values, key=lambda v: (float(sympy.re(v)), float(sympy.im(v))))
    except TypeError:
        raise EvaluationError(
            f"{what} orders the eigenvalues, but these hold symbols: {', '.join(map(str, values))}"
        ) from None


def _numeric_eigenvalues(xp: Any, matrix: Any) -> Any:
    """The eigenvalues of an array, in increasing order, as :func:`_eigenvalues`."""
    values = xp.linalg.eigvals(matrix)
    return xp.sort_complex(values) if xp.iscomplexobj(values) else xp.sort(values)


def _inverse(matrix: sympy.Matrix) -> sympy.Matrix:
    """The inverse of a square matrix; an error where it has none."""
    try:
        return matrix.inv()
    except sympy.matrices.exceptions.NonInvertibleMatrixError:
        raise EvaluationError(NO_INVERSE) from None


def _norm(matrix: sympy.Matrix, p: Any) -> Any:
    """The norm of a matrix of order ``p``: of a vector, a row or a column, its p-norm."""
    order = _order(p)
    if order == "fro" or (_vector(matrix) and order == 2):
        return sympy.sqrt(sum(sympy.Abs(entry) ** 2 for entry in matrix))
    if _vector(matrix):
        if order == float("inf"):
            return sympy.Max(*(sympy.Abs(entry) for entry in matrix))
        return sum(sympy.Abs(entry) ** order for entry in matrix) ** (1 / sympy.sympify(order))
    if order in (1, 2, float("inf")):
        return sympy.simplify(matrix.norm(sympy.oo if order == float("inf") else order))
    raise EvaluationError(
        f"the norm of a matrix is of order 1, 2, \\infty or F, not {p}",
        hint="for a vector's, write it as one column: \\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}",
    )


def _order(p: Any) -> Any:
    """The order of a norm as NumPy takes it: a number, infinity, or ``"fro"`` for the
    Frobenius norm, written F."""
    if isinstance(p, str) or (isinstance(p, sympy.Symbol) and p.name in ("F", "fro")):
        return "fro"
    try:
        number = float(p)
    except TypeError:
        raise EvaluationError(
            f"the order of a norm is a number, \\infty or F, not {p}",
            hint="for the Frobenius norm, write \\|A\\|_F",
        ) from None
    return int(number) if number.is_integer() else number


@map_function(
    "diag",
    summary="`diag(a, b, ...)`, a diagonal matrix; of a vector, the matrix with it on its "
    "diagonal",
    examples=[
        Example("diag(1, 2)", [[1, 0], [0, 2]], "standard"),
        Example("diag(a, b, c)", [[a, 0, 0], [0, b, 0], [0, 0, c]], "standard"),
        Example(r"\operatorname{diag}\begin{pmatrix} 2 \\ 3 \end{pmatrix}", [[2, 0], [0, 3]],
                "numpy.diag([2, 3])"),
    ],
    lifts=False,
)  # fmt: skip
class Diagonal:
    def exact(self, first: Any, *rest: Any) -> Any:
        if not rest and _matrix_like(first) and _vector(vector := _matrix(first)):
            return _array(sympy.diag(*vector))
        return sympy.ImmutableDenseNDimArray(sympy.diag(first, *rest).tolist())


@map_function(
    "det",
    summary="the determinant, `\\det A`",
    examples=[
        Example("det([[1, 2], [3, 4]])", -2, "standard"),
        Example("det([[a, b], [c, d]])", a * d - b * c, "standard"),
        Example(r"\det \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}", 5,
                "numpy.linalg.det([[2, 1], [1, 3]])"),
    ],
    points=_SQUARE,
    scalar=True,
)  # fmt: skip
class Determinant:
    def exact(self, matrix: Any) -> Any:
        """The determinant of a matrix; a scalar is its own determinant."""
        if _matrix_like(matrix):
            return sympy.simplify(_square(matrix, "a determinant").det())
        return matrix

    def numeric(self, xp: Any, matrix: Any) -> Any:
        return xp.linalg.det(matrix)


@map_function(
    "inverse",
    aliases=["inv"],
    summary="the inverse of a matrix, `A^{-1}`",
    examples=[
        Example("inverse([[1, 2], [3, 4]])",
                [[-2, 1], [sympy.Rational(3, 2), sympy.Rational(-1, 2)]], "standard"),
        Example("inv([[2, 0], [0, 4]])",
                [[sympy.Rational(1, 2), 0], [0, sympy.Rational(1, 4)]], "standard"),
    ],
    points=_SQUARE,
)  # fmt: skip
class Inverse:
    def exact(self, matrix: Any) -> Any:
        return _array(_inverse(_square(matrix, "an inverse")))

    def numeric(self, xp: Any, matrix: Any) -> Any:
        return xp.linalg.inv(matrix)


@map_function(
    "pinv",
    summary="the pseudoinverse of a matrix, `A^+`, Moore and Penrose's",
    examples=[
        Example(r"A := \begin{pmatrix} 1 & 2 \\ 2 & 4 \\ 3 & 6 \end{pmatrix}" "\nA^+",
                [[sympy.Rational(1, 70), sympy.Rational(1, 35), sympy.Rational(3, 70)],
                 [sympy.Rational(1, 35), sympy.Rational(2, 35), sympy.Rational(3, 35)]],
                "numpy.linalg.pinv([[1, 2], [2, 4], [3, 6]])"),
        Example("pinv([[2, 0], [0, 4]])",
                [[sympy.Rational(1, 2), 0], [0, sympy.Rational(1, 4)]],
                "Golub and Van Loan (2013), section 5.5.2: of an invertible matrix, its inverse"),
    ],
    points=[*_SQUARE, ([[1, 2], [2, 4], [3, 6]],)],
)  # fmt: skip
class PseudoInverse:
    def exact(self, matrix: Any) -> Any:
        return _simplified(_matrix(matrix, "pinv").pinv())

    def numeric(self, xp: Any, matrix: Any) -> Any:
        return xp.linalg.pinv(matrix)


@map_function(
    "transpose",
    summary="the transpose of a matrix, `A^T`",
    examples=[
        Example("transpose([[1, 2], [3, 4]])", [[1, 3], [2, 4]], "standard"),
        Example("transpose([[1, 2, 3]])", [[1], [2], [3]], "standard"),
    ],
    points=[*_SQUARE, ([[1, 2, 3]],)],
    lifts=False,
)
class Transpose:
    def exact(self, matrix: Any) -> Any:
        return _array(_matrix(matrix, "transpose").T)

    def numeric(self, xp: Any, matrix: Any) -> Any:
        return xp.swapaxes(matrix, -1, -2)


@map_function(
    "ctranspose",
    summary="the conjugate transpose of a matrix, `A^\\dagger` or `A^*`",
    examples=[
        Example(r"\begin{pmatrix} 1 & i \\ 0 & 2 \end{pmatrix}^\dagger",
                [[1, 0], [-sympy.I, 2]], "numpy.conj([[1, 1j], [0, 2]]).T"),
        Example("ctranspose([[a, b]])", [[sympy.conjugate(a)], [sympy.conjugate(b)]],
                "standard"),
    ],
    points=[*_SQUARE, ([[1, 2, 3]],)],
    lifts=False,
)  # fmt: skip
class ConjugateTranspose:
    def exact(self, matrix: Any) -> Any:
        return _array(_matrix(matrix, "ctranspose").H)

    def numeric(self, xp: Any, matrix: Any) -> Any:
        return xp.conj(xp.swapaxes(matrix, -1, -2))


@map_function(
    "trace",
    aliases=["tr", "Tr"],
    latex=[r"\operatorname{tr} A", r"\operatorname{tr}(A)", r"\operatorname{Tr} A",
           r"\operatorname{Tr}(A)"],
    summary="the trace of a matrix; of an equation between tensors, its contraction",
    examples=[
        Example("trace([[1, 2], [3, 4]])", 5, "standard"),
        Example("trace([[a, b], [c, d]])", a + d, "standard"),
        Example(r"A := \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}"
                "\n\\operatorname{tr} A", 15, "Strang (2016), section 6.1"),
    ],
    points=_SQUARE,
    scalar=True,
)  # fmt: skip
class Trace:
    def exact(self, A: Any) -> Any:
        """The trace of a matrix; a scalar is its own trace, as its own determinant."""
        if _matrix_like(A):
            return _square(A, "a trace").trace()
        return A

    def numeric(self, xp: Any, A: Any) -> Any:
        return xp.trace(A)


@map_function(
    "rank",
    latex=[r"\operatorname{rank} A", r"\operatorname{rank}(A)"],
    summary="the rank of a matrix, the dimension of the space its columns span",
    examples=[
        Example(r"\operatorname{rank} \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 "
                r"\end{pmatrix}", 2, "Strang (2016), section 3.2"),
        Example("rank([[1, 2], [2, 4]])", 1, "numpy.linalg.matrix_rank([[1, 2], [2, 4]])"),
    ],
    points=[*_SQUARE, ([[1, 2], [2, 4], [3, 6]],)],
    scalar=True,
)  # fmt: skip
class Rank:
    def exact(self, A: Any) -> Any:
        return sympy.Integer(_matrix(A, "rank").rank(simplify=True))

    def numeric(self, xp: Any, A: Any) -> Any:
        return xp.linalg.matrix_rank(A)


@map_function(
    "adjugate",
    aliases=["adj"],
    latex=[r"\operatorname{adj} A", r"\operatorname{adj}(A)"],
    summary="the adjugate of a matrix, the transpose of its cofactors, `\\operatorname{adj} A`",
    examples=[
        Example(r"\operatorname{adj} \begin{pmatrix} a & b \\ c & d \end{pmatrix}",
                [[d, -b], [-c, a]], "Strang (2016), section 5.3"),
        Example("adj([[1, 2, 3], [0, 1, 4], [5, 6, 0]])",
                [[-24, 18, 5], [20, -15, -4], [-5, 4, 1]],
                "numpy.linalg.det(A) * numpy.linalg.inv(A), A = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]"),
    ],
    points=[*_SQUARE, ([[1, 2], [2, 4]],)],
)  # fmt: skip
class Adjugate:
    def exact(self, A: Any) -> Any:
        return _array(_square(A, "an adjugate").adjugate())

    def numeric(self, xp: Any, A: Any) -> Any:
        n = A.shape[-1]
        if n == 1:
            return xp.ones((1, 1))
        others = [xp.asarray([i for i in range(n) if i != row]) for row in range(n)]
        minors = xp.stack(
            [xp.stack([A[xp.ix_(others[i], others[j])] for j in range(n)]) for i in range(n)]
        )
        signs = (-1.0) ** (xp.arange(n)[:, None] + xp.arange(n)[None, :])
        return xp.swapaxes(signs * xp.linalg.det(minors), -1, -2)


@map_function(
    "null",
    latex=[r"\operatorname{null} A", r"\operatorname{null}(A)"],
    summary="the null space of a matrix, `\\ker A`: a basis of it, as the columns of a matrix",
    examples=[
        Example(r"\ker \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}", [[-2], [1]],
                "Strang (2016), section 3.1: the solutions of x + 2y = 0"),
        Example(r"\operatorname{null}\begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{pmatrix}",
                [[-1], [-1], [1]], "Strang (2016), section 3.1: the solutions of x + z = 0 "
                "and y + z = 0"),
    ],
    lifts=False,
)  # fmt: skip
class NullSpace:
    def exact(self, A: Any) -> Any:
        matrix = _matrix(A, "null")
        basis = matrix.nullspace(simplify=True)
        return _array(sympy.Matrix.hstack(*basis) if basis else sympy.zeros(matrix.cols, 0))


@map_function(
    "dim",
    summary="the dimension of the space the columns of a matrix span, as `\\dim \\ker A`, "
    "the dimension of its null space",
    examples=[
        Example(r"\dim \ker \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}", 1,
                "Strang (2016), section 3.1: n - rank = 2 - 1"),
        Example(r"L := \begin{pmatrix} 2 & -1 & -1 & 0 & 0 \\ -1 & 2 & -1 & 0 & 0 \\ "
                r"-1 & -1 & 2 & 0 & 0 \\ 0 & 0 & 0 & 1 & -1 \\ 0 & 0 & 0 & -1 & 1 "
                r"\end{pmatrix}" "\n\\dim \\ker L", 2,
                "Chung (1997), lemma 1.7: one zero eigenvalue for each component"),
    ],
    lifts=False,
    scalar=True,
)  # fmt: skip
class Dimension:
    def exact(self, V: Any) -> Any:
        return sympy.Integer(_matrix(V, "dim").rank(simplify=True))


@map_function(
    "vec",
    latex=[r"\operatorname{vec} A", r"\operatorname{vec}(A)"],
    summary="the columns of a matrix stacked into one, `\\operatorname{vec} A`",
    examples=[
        Example(r"\operatorname{vec} \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}",
                [[1], [3], [2], [4]], "numpy.reshape([[1, 2], [3, 4]], (-1, 1), order='F')"),
        Example("vec([[a, b]])", [[a], [b]], "Magnus and Neudecker (2019), section 2.4"),
    ],
    points=[*_SQUARE, ([[1, 2, 3]],)],
    lifts=False,
)  # fmt: skip
class Vectorisation:
    def exact(self, A: Any) -> Any:
        return _array(_matrix(A, "vec").T.reshape(len(_matrix(A)), 1))

    def numeric(self, xp: Any, A: Any) -> Any:
        return xp.reshape(xp.swapaxes(A, -1, -2), (-1, 1))


@map_function(
    "norm",
    summary="the norm of a matrix or a vector, written between double bars with its order p "
    "as a subscript: 1, 2, `\\infty`, or F for Frobenius's",
    examples=[
        Example(r"\left\| \begin{pmatrix} 1 & 2 & -1 \\ 0 & 3 & -1 \\ 5 & -1 & 1 \end{pmatrix}"
                r" \right\|_\infty", 7, "Burden and Faires (2011), section 7.1"),
        Example(r"\|\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\|_2",
                5.464985704219043, "numpy.linalg.norm([[1, 2], [3, 4]], 2)"),
        Example(r"\|\begin{pmatrix} 1 \\ -2 \\ 2 \end{pmatrix}\|_1", 5,
                "numpy.linalg.norm([1, -2, 2], 1)"),
    ],
    points=[([[1, 2], [3, 4]], 2), ([[1, 2], [3, 4]], 1), ([[1, 2], [3, 4]], sympy.oo),
            ([[1], [-2], [2]], 2), ([[1, -2, 2]], 1)],
    scalar=True,
)  # fmt: skip
class Norm:
    def exact(self, x: Any, p: Any = 2) -> Any:
        return _norm(_matrix(x, "norm"), p)

    def numeric(self, xp: Any, x: Any, p: Any = 2) -> Any:
        order = _order(p)
        if 1 in x.shape[-2:]:
            return xp.linalg.norm(xp.ravel(x), 2 if order == "fro" else order)
        return xp.linalg.norm(x, order)


@map_function(
    "cond",
    latex=[r"\kappa(A)", r"\kappa_p(A)"],
    summary="the condition number of a matrix, `\\kappa(A)`, its norm times its inverse's, of "
    "order 2, or another p",
    examples=[
        Example(r"\kappa\left(\begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}\right)",
                2.618033988749895, "numpy.linalg.cond([[2, 1], [1, 3]])"),
        Example(r"\kappa_\infty\left(\begin{pmatrix} 1 & 2 \\ 1.0001 & 2 \end{pmatrix}\right)",
                60002, "Burden and Faires (2011), section 7.5", tolerance=1e-9),
    ],
    points=[([[2, 1], [1, 3]], 2), ([[4, 1], [2, 3]], 1), ([[4, 1], [2, 3]], sympy.oo)],
    scalar=True,
)  # fmt: skip
class ConditionNumber:
    def exact(self, A: Any, p: Any = 2) -> Any:
        matrix = _square(A, "a condition number")
        if _order(p) == 2:
            values = [abs(v) for v in matrix.singular_values()]
            if any(v == 0 for v in values):
                return sympy.oo
            return sympy.simplify(sympy.Max(*values) / sympy.Min(*values))
        return sympy.simplify(_norm(matrix, p) * _norm(_inverse(matrix), p))

    def numeric(self, xp: Any, A: Any, p: Any = 2) -> Any:
        return xp.linalg.cond(A, _order(p))


@map_function(
    "spectral_radius",
    latex=r"\rho(A)",
    summary="the spectral radius of a matrix, `\\rho(A)`, the largest modulus of its "
    "eigenvalues; ρ of anything else is not",
    examples=[
        Example(r"T := \begin{pmatrix} 0 & \frac{1}{4} & 0 \\ \frac{1}{4} & 0 & \frac{1}{4} \\ "
                r"0 & \frac{1}{4} & 0 \end{pmatrix}" "\n\\rho(T)", sympy.sqrt(2) / 4,
                "Burden and Faires (2011), section 7.3"),
        Example("spectral_radius([[1, 2], [3, 4]])", (5 + sympy.sqrt(33)) / 2,
                "max(abs(numpy.linalg.eigvals([[1, 2], [3, 4]])))"),
    ],
    points=_SQUARE,
    scalar=True,
)  # fmt: skip
class SpectralRadius:
    def exact(self, A: Any) -> Any:
        values = _square(A, "a spectral radius").eigenvals(multiple=True)
        return sympy.simplify(sympy.Max(*(sympy.Abs(v) for v in values)))

    def numeric(self, xp: Any, A: Any) -> Any:
        return xp.max(xp.abs(xp.linalg.eigvals(A)))


@map_function(
    "lambda_max",
    latex=[r"\lambda_{\max}(A)", r"\lambda_{\max} A"],
    summary="the largest eigenvalue of a matrix, by its real part, `\\lambda_{\\max}(A)`",
    examples=[
        Example(r"\lambda_{\max}\begin{pmatrix} 0 & 1 & 1 & 1 \\ 1 & 0 & 1 & 1 \\ 1 & 1 & 0 & 1 "
                r"\\ 1 & 1 & 1 & 0 \end{pmatrix}", 3,
                "Chung (1997), section 1.2: a k-regular graph has largest eigenvalue k"),
        Example("lambda_max([[2, 1], [1, 3]])", (5 + sympy.sqrt(5)) / 2,
                "max(numpy.linalg.eigvalsh([[2, 1], [1, 3]]))"),
    ],
    points=_SQUARE,
    scalar=True,
)  # fmt: skip
class LargestEigenvalue:
    def exact(self, A: Any) -> Any:
        return _eigenvalues(A, "lambda_max")[-1]

    def numeric(self, xp: Any, A: Any) -> Any:
        return _numeric_eigenvalues(xp, A)[-1]


@map_function(
    "lambda_min",
    latex=[r"\lambda_{\min}(A)", r"\lambda_{\min} A"],
    summary="the smallest eigenvalue of a matrix, by its real part, `\\lambda_{\\min}(A)`",
    examples=[
        Example(r"\lambda_{\min}\left(\begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}\right)",
                (5 - sympy.sqrt(5)) / 2, "min(numpy.linalg.eigvalsh([[2, 1], [1, 3]]))"),
        Example("lambda_min([[4, 1], [2, 3]])", 2, "min(numpy.linalg.eigvals([[4, 1], [2, 3]]))"),
    ],
    points=_SQUARE,
    scalar=True,
)  # fmt: skip
class SmallestEigenvalue:
    def exact(self, A: Any) -> Any:
        return _eigenvalues(A, "lambda_min")[0]

    def numeric(self, xp: Any, A: Any) -> Any:
        return _numeric_eigenvalues(xp, A)[0]


@map_function(
    "eigenvalue",
    latex=r"\lambda_k(A)",
    summary="the eigenvalues of a matrix counted from the smallest, `\\lambda_k(A)`: "
    "`\\lambda_1(A)` is the smallest, as for a graph's Laplacian",
    examples=[
        Example(r"L := \begin{pmatrix} 2 & -1 & 0 & 0 & -1 \\ -1 & 2 & -1 & 0 & 0 \\ "
                r"0 & -1 & 2 & -1 & 0 \\ 0 & 0 & -1 & 2 & -1 \\ -1 & 0 & 0 & -1 & 2 "
                r"\end{pmatrix}" "\n\\lambda_2(L)", 2 - 2 * sympy.cos(2 * sympy.pi / 5),
                "Chung (1997), example 1.5: the cycle C_5"),
        Example("eigenvalue(2, [[2, 1], [1, 3]])", (5 + sympy.sqrt(5)) / 2,
                "sorted(numpy.linalg.eigvalsh([[2, 1], [1, 3]]))[1]"),
    ],
    points=[(1, *_SQUARE[0]), (2, *_SQUARE[1]), (2, *_SQUARE[2])],
    scalar=True,
)  # fmt: skip
class Eigenvalue:
    def exact(self, k: Any, A: Any) -> Any:
        values = _eigenvalues(A, "eigenvalue")
        return values[_counted(k, len(values)) - 1]

    def numeric(self, xp: Any, k: Any, A: Any) -> Any:
        values = _numeric_eigenvalues(xp, A)
        return values[_counted(k, values.shape[-1]) - 1]


def _counted(k: Any, count: int) -> int:
    """Which eigenvalue λ_k is, counted from 1, the smallest, to ``count``, the largest; an
    error for a k that counts none, as 0, -1, 3/2 or a symbol. Data gives k as a decimal."""
    given = whole_number(k.item() if hasattr(k, "item") else k)
    if not (given.is_Integer and 1 <= given <= count):
        written = sympy.nsimplify(given) if given.is_Float else given
        raise EvaluationError(
            f"this matrix has the eigenvalues \\lambda_1 to \\lambda_{count}, counted from the "
            f"smallest, and no \\lambda_{{{written}}}"
        )
    return int(given)


@map_function(
    "spec",
    summary="the spectrum of a matrix, `\\operatorname{spec}(A)`: its eigenvalues, each as "
    "often as it is one, from the smallest",
    examples=[
        Example(r"\operatorname{spec}\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}", [1, 3],
                "numpy.linalg.eigvalsh([[2, 1], [1, 2]])"),
        Example("spec([[a, 0], [0, a]])", [a, a], "standard"),
    ],
    points=_SQUARE,
    lifts=False,
)  # fmt: skip
class Spectrum:
    def exact(self, A: Any) -> Any:
        matrix = _square(A, "a spectrum")
        try:
            values = _eigenvalues(matrix, "spec")
        except EvaluationError:
            values = matrix.eigenvals(multiple=True)
        return sympy.ImmutableDenseNDimArray(values)

    def numeric(self, xp: Any, A: Any) -> Any:
        return _numeric_eigenvalues(xp, A)


@map_function(
    "expm",
    summary="the exponential of a matrix, `e^{A}` or `\\exp(A)` of a matrix A",
    requires="scipy",
    examples=[
        Example(r"A := \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}" "\ne^{A t}",
                [[sympy.cos(t), sympy.sin(t)], [-sympy.sin(t), sympy.cos(t)]],
                "Strang (2016), section 6.3: the rotation through t"),
        Example(r"\exp\left(\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\right)",
                [[51.96895619870755, 74.73656456700692],
                 [112.10484685051041, 164.07380304921796]],
                "scipy.linalg.expm([[1, 2], [3, 4]])", tolerance=1e-9),
    ],
    points=_SQUARE,
)  # fmt: skip
class MatrixExponential:
    def exact(self, A: Any) -> Any:
        matrix = _square(A, "the exponential of a matrix")
        return _of_decimals(matrix, "expm") or _simplified(matrix.exp())

    def numeric(self, xp: Any, A: Any) -> Any:
        return _linalg(xp).expm(A)


@map_function(
    "logm",
    summary="the logarithm of a matrix, `\\log A` of a matrix A, the principal one",
    requires="scipy",
    examples=[
        Example(r"\log \begin{pmatrix} e & 0 \\ 0 & e^{2} \end{pmatrix}", [[1, 0], [0, 2]],
                "Higham (2008), section 11.1: of a diagonal matrix, the logarithms of its "
                "diagonal"),
        Example("logm([[2, 1], [1, 3]])",
                [[0.589514485735048, 0.430408940964004], [0.43040894096400406, 1.0199234266990522]],
                "scipy.linalg.logm([[2, 1], [1, 3]])", tolerance=1e-9),
    ],
    points=_SQUARE,
)  # fmt: skip
class MatrixLogarithm:
    def exact(self, A: Any) -> Any:
        matrix = _square(A, "the logarithm of a matrix")
        return _of_decimals(matrix, "logm") or _simplified(matrix.log())

    def numeric(self, xp: Any, A: Any) -> Any:
        if xp.__name__.startswith("jax"):
            return _through_eigenvalues(xp, A, xp.log)
        import scipy.linalg

        return scipy.linalg.logm(A)


@map_function(
    "sqrtm",
    summary="the square root of a matrix, `\\sqrt{A}` or `A^{1/2}` of a matrix A, the "
    "principal one",
    requires="scipy",
    examples=[
        Example(r"V := \operatorname{diag}(4, 9, 25)" "\nV^{1/2}",
                [[2, 0, 0], [0, 3, 0], [0, 0, 5]], "Higham (2008), section 6.1"),
        Example(r"\sqrt{\begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}}",
                [[1.3763819204711734, 0.3249196962329062],
                 [0.32491969623290634, 1.7013016167040795]],
                "scipy.linalg.sqrtm([[2, 1], [1, 3]])", tolerance=1e-9),
    ],
    points=_SQUARE,
)  # fmt: skip
class MatrixSquareRoot:
    def exact(self, A: Any) -> Any:
        matrix = _square(A, "the square root of a matrix")
        return _of_decimals(matrix, "sqrtm") or _simplified(matrix ** sympy.Rational(1, 2))

    def numeric(self, xp: Any, A: Any) -> Any:
        return _linalg(xp).sqrtm(A)


@map_function(
    "matrix_power",
    summary="a matrix to a power, `A^k`, for a power that is no whole number: a symbol, "
    "or a fraction",
    requires="scipy",
    examples=[
        Example(r"\Lambda := \begin{pmatrix} 1 & 0 \\ 0 & 6 \end{pmatrix}" "\n\\Lambda^k",
                [[1, 0], [0, 6**k]], "Strang (2016), section 6.2"),
        Example(r"D := \operatorname{diag}(4, 1)" "\nD^{-1/2}",
                [[sympy.Rational(1, 2), 0], [0, 1]],
                "scipy.linalg.fractional_matrix_power([[4, 0], [0, 1]], -0.5)"),
    ],
    points=[(*_SQUARE[0], 0.5), (*_SQUARE[1], 3), (*_SQUARE[2], -1.5)],
)  # fmt: skip
class MatrixPower:
    def exact(self, A: Any, p: Any) -> Any:
        matrix, power = _square(A, "a power of a matrix"), sympy.sympify(p)
        if power.is_number and (power.is_Float or matrix.has(sympy.Float)):
            decimals = _of_decimals(matrix, "fractional_matrix_power", float(power))
            if decimals is not None:
                return decimals
        return _simplified(matrix**power)

    def numeric(self, xp: Any, A: Any, p: Any) -> Any:
        number = float(p)
        if number.is_integer():
            return xp.linalg.matrix_power(A, int(number))
        if xp.__name__.startswith("jax"):
            return _through_eigenvalues(xp, A, lambda values: values**number)
        import scipy.linalg

        return scipy.linalg.fractional_matrix_power(A, number)


@map_function(
    "kron",
    summary="the Kronecker product of two matrices, `A \\otimes B`",
    examples=[
        Example(r"\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \otimes "
                r"\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}",
                [[0, 0, 0, -sympy.I], [0, 0, sympy.I, 0], [0, -sympy.I, 0, 0],
                 [sympy.I, 0, 0, 0]],
                "numpy.kron([[0, 1], [1, 0]], [[0, -1j], [1j, 0]])"),
        Example("kron([[1, 2]], [[a], [b]])", [[a, 2 * a], [b, 2 * b]],
                "Horn and Johnson (1991), section 4.2"),
    ],
    points=[(*_SQUARE[0], [[0, 1], [1, 0]]), (*_SQUARE[2], [[1, 2, 3]])],
    lifts=False,
)  # fmt: skip
class KroneckerProduct:
    def exact(self, A: Any, B: Any) -> Any:
        return _array(sympy.kronecker_product(_matrix(A, "kron"), _matrix(B, "kron")))

    def numeric(self, xp: Any, A: Any, B: Any) -> Any:
        return xp.kron(A, B)


@map_function(
    "direct_sum",
    summary="the direct sum of matrices, `A \\oplus B`, the block diagonal matrix of them",
    requires="scipy",
    examples=[
        Example(r"\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta "
                r"\end{pmatrix} \oplus \begin{pmatrix} 1 \end{pmatrix}",
                [[sympy.cos(sympy.Symbol("theta")), -sympy.sin(sympy.Symbol("theta")), 0],
                 [sympy.sin(sympy.Symbol("theta")), sympy.cos(sympy.Symbol("theta")), 0],
                 [0, 0, 1]], "Hall (2015), section 4.3: the rotations about an axis"),
        Example("direct_sum([[1, 2], [3, 4]], [[5]])", [[1, 2, 0], [3, 4, 0], [0, 0, 5]],
                "scipy.linalg.block_diag([[1, 2], [3, 4]], [[5]])"),
    ],
    points=[(*_SQUARE[0], [[5]]), (*_SQUARE[1], [[1, 2, 3]])],
    lifts=False,
)  # fmt: skip
class DirectSum:
    def exact(self, A: Any, *rest: Any) -> Any:
        blocks = [_matrix(block, "direct_sum") if _matrix_like(block) else block
                  for block in (A, *rest)]  # fmt: skip
        return _array(sympy.diag(*blocks))

    def numeric(self, xp: Any, A: Any, *rest: Any) -> Any:
        return _linalg(xp).block_diag(A, *rest)
