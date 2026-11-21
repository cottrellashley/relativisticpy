r"""Linear algebra on components, symbolic or numerical.

A tensor with some indices up and the rest down is a matrix: its upper axes are the
rows and its lower ones the columns, so :math:`A^a{}_b` is a square matrix and the
two-spin operator :math:`H^{ik}{}_{jl}` a 4 × 4 one. These functions take components
and the number of leading axes that are rows. SymPy does exact work on symbolic
components and NumPy and SciPy numerical work on numbers::

    >>> import sympy
    >>> values, vectors = eig(sympy.Array([[2, 1], [1, 2]]), rows=1)
    >>> values
    [1, 3]
"""

from __future__ import annotations

import math
from typing import Any

import sympy
from sympy import ImmutableDenseNDimArray

from relativisticpy.errors import EvaluationError
from relativisticpy.tensors import numerical


def _square(components: Any, rows: int) -> tuple[Any, tuple[int, ...], tuple[int, ...]]:
    """The components as a matrix, with the shapes of its row and column axes."""
    shape = tuple(components.shape)
    row_shape, column_shape = shape[:rows], shape[rows:]
    size = (math.prod(row_shape), math.prod(column_shape))
    if numerical.is_numerical(components):
        if numerical.is_sparse(components):
            return components, row_shape, column_shape
        return components.reshape(size), row_shape, column_shape
    return (
        sympy.Matrix(size[0], size[1], list(sympy.flatten(components.tolist()))),
        row_shape,
        column_shape,
    )


def _array(matrix: Any, shape: tuple[int, ...]) -> Any:
    if numerical.is_numerical(matrix):
        return matrix.reshape(shape)
    return ImmutableDenseNDimArray(list(sympy.flatten(matrix.tolist())), shape)


def inverse(components: Any) -> Any:
    """The inverse of a two-index tensor, as a matrix."""
    if numerical.is_numerical(components):
        return numerical.xp(components).linalg.inv(numerical.dense(components))
    matrix, _, _ = _square(components, 1)
    if matrix.det() == 0:
        raise EvaluationError("this tensor has no inverse: its determinant is zero")
    return _array(matrix.inv(), tuple(components.shape))


def solve(matrix: Any, rhs: Any, transpose: bool = False) -> Any:
    """The solution :math:`x` of :math:`A x = b`, for :math:`(A^{-1})^{ab} b_b` without
    forming the inverse: by Cholesky factorisation when :math:`A` is symmetric positive
    definite, as covariances and metrics of length are, and LU otherwise. The first axis
    of ``rhs`` meets the matrix's columns (its rows, if ``transpose``)."""
    if not (numerical.is_numerical(matrix) or numerical.is_numerical(rhs)):
        a = sympy.Matrix(matrix.tolist())
        a = a.T if transpose else a
        shape = tuple(rhs.shape)
        b = sympy.Matrix(shape[0], math.prod(shape[1:]), list(sympy.flatten(rhs.tolist())))
        return ImmutableDenseNDimArray(list(sympy.flatten(a.LUsolve(b).tolist())), shape)
    np = numerical.numpy()
    a = numerical.as_array(matrix)
    b = numerical.dense(numerical.as_array(rhs))
    a = a.T if transpose else a
    block = b.reshape(b.shape[0], -1)
    if numerical.is_jax(a) or numerical.is_jax(b):  # differentiable, and on a GPU if there is one
        solution = numerical.xp(a, b).linalg.solve(a, block)
    elif numerical.is_sparse(a):
        from scipy.sparse.linalg import spsolve

        solution = np.asarray(spsolve(a.tocsc(), block)).reshape(a.shape[0], -1)
    else:
        solution = _dense_solve(a, block)
    return solution.reshape((a.shape[1] if transpose else a.shape[0], *b.shape[1:]))


def _dense_solve(a: Any, block: Any) -> Any:
    np = numerical.numpy()
    if _hermitian(a):
        from scipy.linalg import LinAlgError, cho_factor, cho_solve

        try:
            return cho_solve(cho_factor(a), block)
        except LinAlgError:
            pass  # symmetric but not positive definite
    return np.linalg.solve(a, block)


def _hermitian(matrix: Any) -> bool:
    """Whether a numerical matrix is its own conjugate transpose, to rounding error in
    its largest component. NumPy's default tolerance would call any matrix of small
    numbers symmetric, and Cholesky factorisation and ``eigh`` read only half of it."""
    np = numerical.numpy()
    if numerical.is_sparse(matrix):
        largest = abs(matrix).max() if matrix.nnz else 0.0
        difference = abs(matrix - matrix.conj().T)
        return bool(difference.max() <= _ROUNDING * largest) if difference.nnz else True
    scale = float(np.max(np.abs(matrix), initial=0.0))
    return bool(np.allclose(matrix, matrix.conj().T, rtol=0.0, atol=_ROUNDING * scale))


#: Rounding error, relative to the largest component of a matrix.
_ROUNDING = 1e-12


def eig(
    components: Any, rows: int, metric: Any = None, count: int | None = None
) -> tuple[Any, Any]:
    """Eigenvalues and eigenvectors of a map, rows to columns, in increasing order; the
    eigenvectors' last axis is the mode. With a ``metric``, of the bilinear form relative
    to it: :math:`S_{ab} v^b = \\lambda g_{ab} v^b`. With a ``count``, only that many,
    the largest (the smallest, if negative), as for a large sparse matrix."""
    if not numerical.is_numerical(components) and not numerical.is_numerical(metric):
        return _symbolic_eig(components, rows, metric)
    matrix, row_shape, _ = _square(components, rows)
    np = numerical.numpy()
    if numerical.is_jax(matrix):  # Hermitian, as JAX differentiates eigenvalues only of those
        values, vectors = numerical.xp(matrix).linalg.eigh(matrix)
        return values, vectors.reshape((*row_shape, vectors.shape[1]))
    if count is not None or numerical.is_sparse(matrix):
        from scipy.sparse.linalg import eigsh

        k = abs(count) if count else min(6, matrix.shape[0] - 1)
        values, vectors = eigsh(matrix, k=k, M=metric, which="SA" if (count or 0) < 0 else "LA")
    elif metric is not None:
        from scipy.linalg import eigh

        values, vectors = eigh(matrix, numerical.dense(numerical.as_array(metric)))
    elif _hermitian(matrix):
        values, vectors = np.linalg.eigh(matrix)
    else:
        values, vectors = np.linalg.eig(matrix)
    order = np.argsort(values.real, kind="stable")
    values, vectors = values[order], vectors[:, order]
    return values, vectors.reshape((*row_shape, vectors.shape[1]))


def _symbolic_eig(components: Any, rows: int, metric: Any) -> tuple[Any, Any]:
    matrix, row_shape, _ = _square(components, rows)
    if metric is not None:  # S v = λ g v is g⁻¹S v = λ v
        matrix = sympy.Matrix(metric.tolist()).inv() * matrix
    pairs = []
    for value, multiplicity, vectors in matrix.eigenvects():
        for vector in vectors[:multiplicity]:
            pairs.append((sympy.simplify(value), [sympy.simplify(x) for x in vector]))
    pairs.sort(key=lambda pair: sympy.default_sort_key(pair[0]))
    if len(pairs) != matrix.rows:
        raise EvaluationError("this map has too few eigenvectors to diagonalise it")
    values = ImmutableDenseNDimArray([value for value, _ in pairs], (len(pairs),))
    columns = [[vector[i] for _, vector in pairs] for i in range(matrix.rows)]
    vectors = ImmutableDenseNDimArray(columns, (matrix.rows, len(pairs)))
    return values, vectors.reshape(*row_shape, len(pairs))


def svd(components: Any, rows: int) -> tuple[Any, Any, Any]:
    """The singular value decomposition :math:`X = U\\,\\mathrm{diag}(s)\\,V^\\dagger`,
    thin, with singular values in decreasing order: ``U`` and ``V`` have the mode as
    their last axis."""
    matrix, row_shape, column_shape = _square(components, rows)
    if not numerical.is_numerical(matrix):
        u, s, v = matrix.singular_value_decomposition()
        k = s.rows
        values = ImmutableDenseNDimArray([s[i, i] for i in range(k)], (k,))
        return _array(u, (*row_shape, k)), values, _array(v, (*column_shape, k))
    np = numerical.numpy()
    u, s, vh = np.linalg.svd(numerical.dense(matrix), full_matrices=False)
    k = len(s)
    return u.reshape((*row_shape, k)), s, vh.conj().T.reshape((*column_shape, k))


def det(components: Any, rows: int) -> Any:
    matrix, _, _ = _square(components, rows)
    if numerical.is_numerical(matrix):
        return numerical.xp(matrix).linalg.det(numerical.dense(matrix))
    return sympy.simplify(matrix.det())


def trace(components: Any, rows: int) -> Any:
    matrix, _, _ = _square(components, rows)
    if numerical.is_numerical(matrix):
        return numerical.xp(matrix).trace(numerical.dense(matrix))
    return matrix.trace()


def function(name: str, components: Any, rows: int) -> Any:
    """A function of a matrix, as of a map: ``exp`` (:math:`e^{-iHt}`), ``sqrt`` or
    ``log``; not of its components one by one."""
    matrix, row_shape, column_shape = _square(components, rows)
    shape = (*row_shape, *column_shape)
    if numerical.is_numerical(matrix):
        import scipy.linalg

        compute = {"exp": scipy.linalg.expm, "sqrt": scipy.linalg.sqrtm, "log": scipy.linalg.logm}
        return compute[name](numerical.dense(matrix)).reshape(shape)
    if name == "exp":
        return _array(sympy.simplify(matrix.exp()), shape)
    if name == "log":
        return _array(sympy.simplify(matrix.log()), shape)
    return _array(sympy.simplify(matrix.pow(sympy.Rational(1, 2))), shape)


def diag(vector: Any) -> Any:
    """The diagonal matrix of a vector's components."""
    if numerical.is_numerical(vector):
        return numerical.xp(vector).diag(numerical.dense(vector))
    n = vector.shape[0]
    return ImmutableDenseNDimArray(
        [vector[i] if i == j else 0 for i in range(n) for j in range(n)], (n, n)
    )


def hadamard(left: Any, right: Any) -> Any:
    """The elementwise product of two arrays of the same shape."""
    if numerical.involves([left, right]):
        return numerical.as_array(left) * numerical.as_array(right)
    if not isinstance(left, sympy.NDimArray):
        return left * right
    return ImmutableDenseNDimArray(
        [
            a * b
            for a, b in zip(
                sympy.flatten(left.tolist()), sympy.flatten(right.tolist()), strict=True
            )
        ],
        left.shape,
    )
