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

import functools
import math
from collections.abc import Callable, Hashable, Mapping, Sequence
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
    """Components as a numerical array: SymPy arrays of numbers are converted."""
    if is_numerical(value):
        return value
    np = numpy()
    if isinstance(value, sympy.NDimArray):
        flat = [number(entry) for entry in sympy.flatten(value.tolist())] if value.shape else []
        kind = complex if any(isinstance(entry, complex) for entry in flat) else float
        return np.array(flat, dtype=kind).reshape(value.shape)
    return np.asarray(number(value))


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
    """Contractions with a sparse matrix: as matrix products where they are one, a
    sparse matrix times a vector or a dense matrix; otherwise densified if small."""
    if len(arrays) == 2:
        (a, b), (la, lb) = arrays, labels
        if is_sparse(b) and not is_sparse(a):
            (a, b), (la, lb) = (b, a), (lb, la)
        shared = [n for n in la if n in lb]
        if is_sparse(a) and len(shared) == 1 and len(la) == 2 and len(set(la)) == 2:
            (n,) = shared
            matrix = a if la[1] == n else a.T  # summing over the matrix's columns
            other = numpy().moveaxis(dense(b), list(lb).index(n), 0)
            result = matrix @ other.reshape(other.shape[0], -1)
            free_a = la[0] if la[1] == n else la[1]
            rest = [m for m in lb if m != n]
            result = (
                numpy()
                .asarray(result)
                .reshape((a.shape[0] if la[1] == n else a.shape[1], *other.shape[1:]))
            )
            have = [free_a, *rest]
            if sorted(have) == sorted(out):
                return numpy().transpose(result, [have.index(m) for m in out]) if out else result
    if sum(math.prod(array.shape) for array in arrays) > 10**7:
        raise EvaluationError(
            "a sparse tensor can be contracted with one dense operand at a time",
            hint="split the product, as in x^i := L^i{}_j v^j, then y_k := x^i w_{ik}",
        )
    return _einsum(numpy(), [dense(array) for array in arrays], labels, out)


# -- components ---------------------------------------------------------------------


def permute(array: Any, axes: Sequence[int]) -> Any:
    if is_sparse(array):
        return array.T if list(axes) == [1, 0] else array
    return _xp(array).transpose(array, list(axes))


def select(array: Any, fixed: Mapping[int, int]) -> Any:
    index = tuple(fixed.get(axis, slice(None)) for axis in range(len(array.shape)))
    result = dense(array)[index]
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
    """``func`` applied to every component: NumPy's own version where there is one."""
    xp = _xp(array)
    array = dense(array)
    if not extra and (ufunc := _UFUNCS.get(name)) is not None:
        return getattr(xp, ufunc)(array)
    numbers = [number(e) for e in extra]
    vectorised = numpy().vectorize(lambda x: number(func(sympy.sympify(x), *numbers)))
    return vectorised(array)


def close(lhs: Any, rhs: Any) -> bool:
    """Whether two numerical values agree to rounding error."""
    np = numpy()
    lhs, rhs = np.asarray(dense(as_array(lhs))), np.asarray(dense(as_array(rhs)))
    if lhs.shape != rhs.shape and lhs.size != 1 and rhs.size != 1:
        return False
    scale = max(
        1.0, float(np.max(np.abs(lhs), initial=0.0)), float(np.max(np.abs(rhs), initial=0.0))
    )
    return bool(np.allclose(lhs, rhs, rtol=1e-9, atol=1e-12 * scale))
