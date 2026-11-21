r"""Numerical components: NumPy arrays, and SciPy for what NumPy lacks.

Components are numerical when they are NumPy arrays, SciPy sparse matrices or JAX
arrays: data bound from Python, and whatever is computed from it. The kernels of
:mod:`relativisticpy.tensors.arrays` hand them here, so the compiler and its IR are
the same for 16 symbolic components of a metric and for a million returns.

Contractions are planned: ``numpy.einsum`` finds the cheapest order in which to
multiply the operands, and the plan is kept for the next contraction of the same
shapes. A contraction whose operands are too large to hold at once is done in slices
along an index it sums over, and the slices added, which keeps memory bounded::

    >>> import numpy as np
    >>> x = np.arange(6.0).reshape(3, 2)
    >>> contract([(x, (0, 1)), (x, (0, 2))], (1, 2), (3, 2, 2))   # x_{ta} x_{tb}
    array([[20., 26.],
           [26., 35.]])

NumPy is an optional dependency: ``pip install "relativisticpy[numeric]"``.
"""

from __future__ import annotations

import contextlib
import functools
import math
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from typing import Any

import sympy

from relativisticpy.errors import EvaluationError

#: Contractions whose largest operand has more elements than this are done in slices.
CHUNK = 2**25


def numpy() -> Any:
    """NumPy, or an error that says how to install it."""
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            'numerical components need NumPy: pip install "relativisticpy[numeric]"'
        ) from error
    return np


def is_numerical(value: Any) -> bool:
    """Whether ``value`` is a numerical array or number: NumPy, SciPy sparse or JAX."""
    module = type(value).__module__ or ""
    return module.split(".")[0] in ("numpy", "jax", "jaxlib") or module.startswith("scipy.sparse")


def involves(values: Sequence[Any]) -> bool:
    return any(is_numerical(value) for value in values)


def is_sparse(value: Any) -> bool:
    return (type(value).__module__ or "").startswith("scipy.sparse")


def is_jax(value: Any) -> bool:
    """Whether ``value`` is a JAX array, or a tracer JAX differentiates through."""
    return (type(value).__module__ or "").split(".")[0] in ("jax", "jaxlib")


def _xp(*values: Any) -> Any:
    """The array module of the values: JAX's for JAX arrays, NumPy's otherwise."""
    if any(is_jax(value) for value in values):
        import jax.numpy as jnp

        return jnp
    return numpy()


def number(value: Any) -> Any:
    """A SymPy number as a Python one, to combine with numerical arrays."""
    if is_numerical(value) or isinstance(value, int | float | complex):
        return value
    value = sympy.sympify(value)
    if value.is_number:
        complex_value = complex(value)
        return complex_value.real if complex_value.imag == 0 else complex_value
    raise EvaluationError(
        f"{value} is a symbol, so it cannot be combined with numerical components",
        hint=f"give {value} a value, as in {value} := 0.5",
    )


def as_array(value: Any) -> Any:
    """Components as a numerical array: SymPy arrays of numbers are converted, and so is
    a NumPy matrix, which a sparse matrix and a dense array add up to, and whose ``*``
    and indexing are a matrix's, not an array's."""
    if is_numerical(value):
        np = numpy()
        return widened(np.asarray(value) if isinstance(value, np.matrix) else value)
    np = numpy()
    if isinstance(value, sympy.NDimArray):
        flat = [number(entry) for entry in sympy.flatten(value.tolist())] if value.shape else []
        kind = complex if any(isinstance(entry, complex) for entry in flat) else float
        return np.array(flat, dtype=kind).reshape(value.shape)
    return np.asarray(number(value))


def widened(array: Any) -> Any:
    """An array of booleans or small integers as 64-bit integers. NumPy's arithmetic keeps
    the type of its operands, so the Gram matrix of 8-bit pixels would wrap around at 256
    and a product of masks would be a logical one."""
    dtype = getattr(array, "dtype", None)
    if dtype is None or is_jax(array) or dtype.kind not in "biu" or dtype.itemsize >= 8:
        return array
    return array.astype(numpy().int64)


def xp(*values: Any) -> Any:
    """The array module to compute with: JAX's if any value is a JAX array."""
    return _xp(*values)


def dense(value: Any) -> Any:
    return value.toarray() if is_sparse(value) else value


# -- contraction -----------------------------------------------------------------------


def contract(
    operands: Sequence[tuple[Any, Sequence[Hashable]]],
    output: Sequence[Hashable],
    sizes: int | Sequence[int] | Mapping[Hashable, int] = 0,
) -> Any:
    """Einstein summation over labelled operands, as
    :func:`relativisticpy.tensors.arrays.contract`, with ``numpy.einsum``."""
    arrays = [as_array(components) for components, _ in operands]
    numbers: dict[Hashable, int] = {}
    labels = [[numbers.setdefault(label, len(numbers)) for label in group] for _, group in operands]
    out = [numbers[label] for label in output]
    if any(is_sparse(array) for array in arrays):
        return _sparse_contract(arrays, labels, out)
    xp = _xp(*arrays)
    largest = max((math.prod(array.shape) for array in arrays), default=0)
    if largest > CHUNK and xp is numpy():
        sliced = _chunked(arrays, labels, out)
        if sliced is not None:
            return sliced
    return _einsum(xp, arrays, labels, out)


def _einsum(
    xp: Any, arrays: Sequence[Any], labels: Sequence[Sequence[int]], out: Sequence[int]
) -> Any:
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if max((n for group in labels for n in group), default=0) >= len(letters):
        raise EvaluationError("a contraction of more than 52 distinct indices")
    spec = ",".join("".join(letters[n] for n in group) for group in labels)
    spec += "->" + "".join(letters[n] for n in out)
    if xp is not numpy():
        return xp.einsum(spec, *arrays)
    path = _path(spec, tuple(tuple(array.shape) for array in arrays))
    result = xp.einsum(spec, *arrays, optimize=path)
    return result if result.ndim else result[()]


@functools.lru_cache(maxsize=512)
def _path(spec: str, shapes: tuple[tuple[int, ...], ...]) -> list[Any]:
    """The cheapest order of contraction for these shapes, found once."""
    np = numpy()
    dummies = [np.empty(shape) for shape in shapes]
    strategy = "optimal" if len(shapes) <= 4 else "greedy"
    return list(np.einsum_path(spec, *dummies, optimize=strategy)[0])


def _chunked(arrays: Sequence[Any], labels: Sequence[Sequence[int]], out: Sequence[int]) -> Any:
    """The contraction in slices along a summed index of the largest operand, each slice
    of bounded size, added together; ``None`` if no index can be sliced."""
    np = numpy()
    biggest = max(range(len(arrays)), key=lambda i: math.prod(arrays[i].shape))
    candidates = [
        label
        for label in labels[biggest]
        if label not in out and all(group.count(label) <= 1 for group in labels)
    ]
    if not candidates:
        return None
    axis_of = {label: labels[biggest].index(label) for label in candidates}
    label = max(candidates, key=lambda n: arrays[biggest].shape[axis_of[n]])
    length = arrays[biggest].shape[axis_of[label]]
    step = max(1, CHUNK * length // math.prod(arrays[biggest].shape))
    total = None
    for start in range(0, length, step):
        window = slice(start, min(start + step, length))
        parts = [
            array[(slice(None),) * group.index(label) + (window,)] if label in group else array
            for array, group in zip(arrays, labels, strict=True)
        ]
        part = _einsum(np, [np.asarray(p) for p in parts], labels, out)
        total = part if total is None else total + part
    return total


def _sparse_contract(
    arrays: Sequence[Any], labels: Sequence[Sequence[int]], out: Sequence[int]
) -> Any:
    """Contractions with sparse matrices, a pair at a time. A sparse matrix meets an
    operand it shares one index with, summed nowhere else, as a matrix product: a vector
    or a dense array, or another sparse matrix, whose product stays sparse. Dense operands
    that share an index are contracted together first, keeping what the rest needs, so
    that L_{ij} x^j y^i is two products of the matrix with a vector. What cannot be done
    so is densified, if small. Operands JAX traces meet a sparse matrix as JAX's own."""
    operands = [(array, tuple(group)) for array, group in zip(arrays, labels, strict=True)]
    while len(operands) > 1:
        pair = _pair(operands, out)
        if pair is None:
            break
        i, j = pair
        others = [operand for k, operand in enumerate(operands) if k not in (i, j)]
        needed = {label for _, group in others for label in group} | set(out)
        operands = [*others, _product(operands[i], operands[j], needed)]
    if len(operands) == 1 and (alone := _arranged(*operands[0], out)) is not None:
        return alone
    if sum(math.prod(array.shape) for array, _ in operands) > 10**7:
        raise EvaluationError(
            "this product of a sparse matrix is too large to compute densely",
            hint="split the product, as in x^i := L^i{}_j v^j, then y_k := x^i w_{ik}",
        )
    arrays = [dense(array) for array, _ in operands]
    return _einsum(_xp(*arrays), arrays, [group for _, group in operands], out)


def _pair(operands: Sequence[tuple[Any, tuple[int, ...]]], out: Sequence[int]) -> Any:
    """The next two operands of a sparse contraction: a sparse matrix and a dense operand
    it is a matrix product with, then two dense operands with an index in common, then two
    sparse matrices; ``None`` if no two can be multiplied so."""
    found: dict[str, tuple[int, int]] = {}
    for i, (a, la) in enumerate(operands):
        for j, (b, lb) in enumerate(operands):
            if i == j or not set(la) & set(lb):
                continue
            if is_sparse(a) and _matrix_product(operands, i, j, out) is not None:
                found.setdefault("sparse" if is_sparse(b) else "dense", (i, j))
            elif not (is_sparse(a) or is_sparse(b)):
                found.setdefault("both dense", (i, j))
    return next((found[kind] for kind in ("dense", "both dense", "sparse") if kind in found), None)


def _matrix_product(
    operands: Sequence[tuple[Any, tuple[int, ...]]], i: int, j: int, out: Sequence[int]
) -> int | None:
    """The index a sparse matrix, operand ``i``, sums with operand ``j`` as a matrix
    product: their one index in common, written nowhere else; ``None`` if there is none."""
    (_, la), (b, lb) = operands[i], operands[j]
    shared = set(la) & set(lb)
    if len(la) != 2 or la[0] == la[1] or len(shared) != 1:
        return None
    (label,) = shared
    elsewhere = any(label in group for k, (_, group) in enumerate(operands) if k not in (i, j))
    if elsewhere or label in out or lb.count(label) != 1:
        return None
    if is_sparse(b) and (len(lb) != 2 or lb[0] == lb[1]):
        return None
    return label


def _product(
    left: tuple[Any, tuple[int, ...]], right: tuple[Any, tuple[int, ...]], needed: set[int]
) -> tuple[Any, tuple[int, ...]]:
    """Two operands of a sparse contraction multiplied, with their labels: as a matrix
    product if one is a sparse matrix, summing the index they share, and otherwise with
    ``einsum``, summing what nothing else ``needed`` has."""
    (a, la), (b, lb) = left, right
    if not is_sparse(a):
        (a, la), (b, lb) = (b, lb), (a, la)
    if not is_sparse(a):
        keep = tuple(label for label in dict.fromkeys(la + lb) if label in needed)
        xp = _xp(a, b)
        return _einsum(xp, [a, b], [la, lb], keep), keep
    (label,) = set(la) & set(lb)
    matrix = a if la[1] == label else a.T  # its columns are summed
    row = la[0] if la[1] == label else la[1]
    if is_sparse(b):
        other = b if lb[0] == label else b.T
        return matrix @ other, (row, lb[1] if lb[0] == label else lb[0])
    xp = _xp(b)
    moved = xp.moveaxis(b, lb.index(label), 0)
    if is_jax(b):
        from jax.experimental import sparse as jax_sparse

        matrix = jax_sparse.BCOO.from_scipy_sparse(matrix)
    product = matrix @ moved.reshape(moved.shape[0], -1)
    shape = (matrix.shape[0], *moved.shape[1:])
    return xp.asarray(product).reshape(shape), (row, *(m for m in lb if m != label))


def _arranged(array: Any, group: tuple[int, ...], out: Sequence[int]) -> Any:
    """The last operand of a contraction, its axes in the order of ``out``, and summed
    over any index not there; ``None`` for a sparse matrix that would need densifying."""
    if not is_sparse(array):
        return _einsum(_xp(array), [array], [group], out)
    if len(set(group)) == 2 and sorted(group) == sorted(out):
        return array if list(group) == list(out) else array.T
    np = numpy()
    if group[0] == group[1]:  # the trace, or the diagonal
        return array.diagonal().sum() if not out else array.diagonal()
    if not out:
        return array.sum()
    if len(out) == 1 and out[0] in group:  # summed over the other index
        return np.asarray(array.sum(axis=1 - group.index(out[0]))).ravel()
    return None


# -- components ---------------------------------------------------------------------


def permute(array: Any, axes: Sequence[int]) -> Any:
    if is_sparse(array):
        return array.T if list(axes) == [1, 0] else array
    return _xp(array).transpose(array, list(axes))


def select(array: Any, fixed: Mapping[int, int]) -> Any:
    if is_sparse(array):  # a row or a column, without the rest of the matrix
        if len(fixed) == 2:
            return array.tocsr()[fixed[0], fixed[1]]
        part = array.tocsr()[fixed[0], :] if 0 in fixed else array.tocsc()[:, fixed[1]]
        return numpy().asarray(part.toarray()).ravel()
    index = tuple(fixed.get(axis, slice(None)) for axis in range(len(array.shape)))
    result = array[index]
    return result[()] if getattr(result, "ndim", 1) == 0 else result


def symmetrize(array: Any, axes: Sequence[int], *, antisymmetric: bool) -> Any:
    from itertools import permutations

    from sympy.combinatorics import Permutation

    np = numpy()
    array = dense(array)
    total = np.zeros_like(array, dtype=np.result_type(array, 1.0))
    orderings = list(permutations(range(len(axes))))
    for ordering in orderings:
        full = list(range(array.ndim))
        for target, source in zip(axes, (axes[k] for k in ordering), strict=True):
            full[target] = source
        sign = Permutation(list(ordering)).signature() if antisymmetric else 1
        total = total + sign * np.transpose(array, full)
    return total / len(orderings)


#: Elementwise functions by the name of RelativisticPy's builtin.
_UFUNCS = {
    "sin": "sin", "cos": "cos", "tan": "tan", "asin": "arcsin", "acos": "arccos",
    "atan": "arctan", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "exp": "exp",
    "log": "log", "ln": "log", "sqrt": "sqrt", "abs": "abs", "Abs": "abs",
    "sign": "sign", "floor": "floor", "ceil": "ceil", "conjugate": "conj",
    "re": "real", "im": "imag",
}  # fmt: skip


def elementwise(name: str) -> bool:
    """Whether NumPy (and JAX) have their own version of the builtin ``name``."""
    return name in _UFUNCS


def apply(array: Any, func: Callable[..., Any], name: str, extra: Sequence[Any] = ()) -> Any:
    """``func`` applied to every component: NumPy's own version where there is one, and
    otherwise its translation into NumPy and SciPy, made once from what it does to a
    symbol, as ``erf`` becomes ``scipy.special.erf``; one component at a time, through
    SymPy, only where there is none."""
    xp = _xp(array)
    array = dense(array)
    if not extra and (ufunc := _UFUNCS.get(name)) is not None:
        return getattr(xp, ufunc)(array)
    numbers = [number(e) for e in extra]
    np = numpy()
    try:
        translated = _translated(func, tuple(numbers)) if xp is np else None
    except TypeError:  # what cannot be remembered, as a JAX array cannot
        translated = None
    if translated is not None:
        try:
            result = np.asarray(translated(array))
        except (NameError, TypeError, ValueError, AttributeError):
            pass  # SymPy wrote what NumPy cannot compute
        else:
            if result.dtype != object:  # a constant is one for every component
                return np.broadcast_to(result, array.shape).copy() if result.ndim == 0 else result
    vectorised = np.vectorize(lambda x: number(func(sympy.sympify(x), *numbers)))
    return vectorised(array)


@functools.lru_cache(maxsize=256)
def _translated(func: Callable[..., Any], extra: tuple[Any, ...]) -> Callable[[Any], Any] | None:
    """``func`` as a function of NumPy arrays, through what it does to a symbol; ``None``
    if it does not give an expression of the symbol."""
    x = sympy.Dummy("x")
    try:
        expression = func(x, *extra)
    except (TypeError, ValueError, AttributeError, sympy.SympifyError):
        return None
    if not isinstance(expression, sympy.Expr):
        return None
    modules = ["scipy", "numpy"] if _has_scipy() else ["numpy"]
    translated: Callable[[Any], Any] = sympy.lambdify(x, expression, modules=modules)
    return translated


def _has_scipy() -> bool:
    try:
        import scipy.special  # noqa: F401
    except ImportError:  # pragma: no cover - depends on the environment
        return False
    return True


def close(lhs: Any, rhs: Any) -> bool:
    """Whether two numerical values agree to rounding error: to a billionth of each other,
    or to a trillionth of the numbers they were computed from, as an equation's sides were
    from the terms of a sum, or else of their own largest components. There is no absolute
    tolerance: the mass of an electron is not twice itself, nor zero."""
    np = numpy()
    lhs, rhs = np.asarray(dense(as_array(lhs))), np.asarray(dense(as_array(rhs)))
    if lhs.shape != rhs.shape and lhs.size != 1 and rhs.size != 1:
        return False
    scale = max(_OPERANDS.get(), magnitude(lhs), magnitude(rhs))
    return bool(np.allclose(lhs, rhs, rtol=RELATIVE, atol=ROUNDING * scale))


#: How far numbers may differ and agree: relative to each other, and to the magnitude of
#: the numbers they were computed from, which rounding error is relative to.
RELATIVE, ROUNDING = 1e-9, 1e-12

#: The magnitude of the numbers the sides of an equation being decided were computed
#: from: the terms of a side that is a sum, such as the residual E[PP] - m m - C.
_OPERANDS: ContextVar[float] = ContextVar("operands", default=0.0)


@contextlib.contextmanager
def operands(scale: float) -> Iterator[None]:
    """Decide equations, within the block, to rounding error in numbers of magnitude
    ``scale``, which their sides were computed from."""
    token = _OPERANDS.set(scale)
    try:
        yield
    finally:
        _OPERANDS.reset(token)


def operand_scale() -> float:
    """The magnitude :func:`operands` gives, or 0 outside it."""
    return _OPERANDS.get()


def magnitude(value: Any) -> float:
    """The largest absolute value of a number or of the components of a numerical array;
    0 for anything else, such as a symbol."""
    if isinstance(value, int | float | complex | sympy.NDimArray) or getattr(
        value, "is_number", False
    ):
        try:
            value = as_array(value)
        except EvaluationError:  # symbols
            return 0.0
    if not is_numerical(value):
        return 0.0
    if is_sparse(value):
        return float(abs(value).max()) if value.nnz else 0.0
    np = numpy()
    return float(np.max(np.abs(np.asarray(value)), initial=0.0))
