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
    """Eigenvalues and eigenvectors of a map, rows to columns, in increasing order (of
    their real parts, then imaginary); the eigenvectors' last axis is the mode. With a
    ``metric``, of the bilinear form relative to it: :math:`S_{ab} v^b = \\lambda g_{ab}
    v^b`. With a ``count``, only that many, the largest (the smallest, if negative): of a
    large sparse matrix, from an iterative solver, which finds a few without the rest."""
    if not numerical.is_numerical(components) and not numerical.is_numerical(metric):
        values, vectors = _symbolic_eig(components, rows, metric)
        return _symbolic_count(values, vectors, count) if count is not None else (values, vectors)
    matrix, row_shape, _ = _square(components, rows)
    if metric is not None and not numerical.is_sparse(metric):
        metric = numerical.as_array(metric)
    size = matrix.shape[0]
    if numerical.is_jax(matrix) or numerical.is_jax(metric):
        values, vectors = _jax_eig(matrix, metric)
    elif numerical.is_sparse(matrix) and count is not None and abs(count) < size - 1:
        values, vectors = _sparse_eig(matrix, metric, count)
    elif numerical.is_sparse(matrix) and size > _DENSE:
        raise EvaluationError(
            f"the spectrum of a sparse matrix of {size} rows is found a few eigenvalues at a time",
            hint="give how many, the largest or, if negative, the smallest: spec(L, -6)",
        )
    else:
        values, vectors = _dense_eig(matrix, metric)
    values, vectors = _ordered(values, vectors, count)
    return values, vectors.reshape((*row_shape, vectors.shape[-1]))


#: Sparse matrices up to this size have their whole spectrum computed as dense ones.
_DENSE = 4096


def _dense_eig(matrix: Any, metric: Any) -> tuple[Any, Any]:
    """Every eigenvalue: by ``eigh`` for a Hermitian map, or a Hermitian form relative to a
    positive definite metric, and the general algorithm otherwise."""
    import scipy.linalg

    np = numerical.numpy()
    a = numerical.dense(matrix)
    if metric is None:
        return np.linalg.eigh(a) if _hermitian(a) else _real(np.linalg.eig(a), a)
    g = numerical.dense(metric)
    if _hermitian(a) and _hermitian(g):
        try:
            values, vectors = scipy.linalg.eigh(a, g)
        except np.linalg.LinAlgError:
            pass  # the metric is not positive definite
        else:
            return values, vectors
    return _real(scipy.linalg.eig(a, g), a, g)


def _real(pairs: tuple[Any, Any], *matrices: Any) -> tuple[Any, Any]:
    """Eigenvalues and eigenvectors as real numbers where real matrices give only real
    ones, which the general algorithm returns as complex numbers."""
    np = numerical.numpy()
    values, vectors = pairs
    if all(np.isrealobj(matrix) for matrix in matrices) and not np.any(values.imag):
        return values.real, vectors.real
    return values, vectors


def _sparse_eig(matrix: Any, metric: Any, count: int) -> tuple[Any, Any]:
    """The ``count`` largest eigenvalues of a sparse matrix (the smallest, if negative),
    by Lanczos's method where it is Hermitian and Arnoldi's otherwise."""
    from scipy.sparse.linalg import eigs, eigsh

    largest = count > 0
    if _hermitian(matrix) and (metric is None or _hermitian(metric)):
        values, vectors = eigsh(matrix, k=abs(count), M=metric, which="LA" if largest else "SA")
        return values, vectors
    pairs = eigs(matrix, k=abs(count), M=metric, which="LR" if largest else "SR")
    return _real(pairs, matrix, *([] if metric is None else [metric]))


def _jax_eig(matrix: Any, metric: Any) -> tuple[Any, Any]:
    """Eigenvalues with JAX, which differentiates those of Hermitian maps: a matrix it
    is differentiating, whose numbers are not known yet, is taken to be one. A metric is
    met by Cholesky factorisation, :math:`g = L L^\\dagger`, which leaves
    :math:`(L^{-1} S L^{-\\dagger}) y = \\lambda y`, with :math:`v = L^{-\\dagger} y`."""
    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular

    a = jnp.asarray(numerical.dense(matrix))
    g = None if metric is None else jnp.asarray(numerical.dense(metric))
    known = [_concrete(x) for x in (a, g) if x is not None]
    hermitian = all(_hermitian(x) for x in known if x is not None)
    if g is not None and hermitian and all(x is not None for x in known):
        hermitian = _positive_definite(known[1])
    if not hermitian:  # JAX computes these on a CPU only
        values, vectors = jnp.linalg.eig(a if g is None else jnp.linalg.solve(g, a))
        return values, vectors
    if g is None:
        values, vectors = jnp.linalg.eigh(a)
        return values, vectors
    lower = jnp.linalg.cholesky(g)
    half = solve_triangular(lower, a, lower=True)
    inner = solve_triangular(lower, half.conj().T, lower=True).conj().T
    values, vectors = jnp.linalg.eigh(inner)
    return values, solve_triangular(lower.conj().T, vectors, lower=False)


def _concrete(array: Any) -> Any:
    """A JAX array's numbers, or ``None`` while JAX traces it to differentiate it."""
    try:
        return numerical.numpy().asarray(array)
    except TypeError:  # a tracer has no numbers yet
        return None


def _positive_definite(matrix: Any) -> bool:
    np = numerical.numpy()
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return False
    return True


def _ordered(values: Any, vectors: Any, count: int | None) -> tuple[Any, Any]:
    """Eigenvalues in increasing order, of real parts and then imaginary ones, with their
    eigenvectors; with a ``count``, the largest that many, or the smallest if negative."""
    xp = numerical.xp(values)
    if xp.iscomplexobj(values):
        order = xp.lexsort((values.imag, values.real))
    elif xp is numerical.numpy():
        order = xp.argsort(values, kind="stable")
    else:
        order = xp.argsort(values)
    values, vectors = values[order], vectors[:, order]
    if count is not None:
        chosen = slice(len(values) - count, None) if count > 0 else slice(None, -count)
        values, vectors = values[chosen], vectors[:, chosen]
    return values, vectors


def _symbolic_count(values: Any, vectors: Any, count: int) -> tuple[Any, Any]:
    """The largest ``count`` of exact eigenvalues, or the smallest if negative, where
    they are numbers, which can be ordered."""
    try:
        keys = [(complex(value).real, complex(value).imag) for value in values]
    except TypeError:
        raise EvaluationError(
            "symbolic eigenvalues cannot be ordered, so not the largest or smallest of them",
            hint="ask for every eigenvalue, without a count",
        ) from None
    order = sorted(range(len(keys)), key=keys.__getitem__)
    order = order[len(order) - count :] if count > 0 else order[:-count]
    rows = vectors.shape[:-1]
    flat = vectors.reshape(math.prod(rows), vectors.shape[-1]).tomatrix()
    chosen = ImmutableDenseNDimArray([values[i] for i in order], (len(order),))
    columns = flat.extract(list(range(flat.rows)), order)
    return chosen, _array(columns, (*rows, len(order)))


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
    """The singular value decomposition :math:`X = U\\,\\mathrm{diag}(s)\\,V^T`, thin, with
    singular values in decreasing order: ``U`` and ``V`` have the mode as their last axis,
    so that :math:`X_{ta} = U_{tk} \\mathrm{diag}(s)_{kl} V_{al}` in index notation, which
    conjugates nothing. Of complex data, ``V`` holds the conjugates of the right singular
    vectors."""
    matrix, row_shape, column_shape = _square(components, rows)
    if not numerical.is_numerical(matrix):
        u, s, v = matrix.singular_value_decomposition()  # X = U S V^†
        k = s.rows
        values = ImmutableDenseNDimArray([s[i, i] for i in range(k)], (k,))
        return _array(u, (*row_shape, k)), values, _array(v.conjugate(), (*column_shape, k))
    np = numerical.numpy()
    u, s, vh = np.linalg.svd(numerical.dense(matrix), full_matrices=False)
    k = len(s)
    return u.reshape((*row_shape, k)), s, vh.T.reshape((*column_shape, k))


def det(components: Any, rows: int) -> Any:
    matrix, _, _ = _square(components, rows)
    if numerical.is_numerical(matrix):
        return numerical.xp(matrix).linalg.det(numerical.dense(matrix))
    return sympy.simplify(matrix.det())


def trace(components: Any, rows: int) -> Any:
    matrix, _, _ = _square(components, rows)
    if numerical.is_sparse(matrix):
        return matrix.diagonal().sum()
    if numerical.is_numerical(matrix):
        return numerical.xp(matrix).trace(matrix)
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
    """The elementwise product of two arrays of the same shape. A sparse matrix's is
    sparse; its ``*`` is the matrix product, which is not this."""
    if numerical.involves([left, right]):
        left, right = numerical.as_array(left), numerical.as_array(right)
        if numerical.is_jax(left) or numerical.is_jax(right):
            return numerical.xp(left, right).multiply(numerical.dense(left), numerical.dense(right))
        if numerical.is_sparse(left):
            return left.multiply(right).tocsr()
        if numerical.is_sparse(right):
            return right.multiply(left).tocsr()
        return left * right
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
