"""Kernels on the component arrays of tensors.

Components are SymPy arrays (``ImmutableDenseNDimArray``) for tensors of rank
one and above, and plain SymPy expressions for scalars; or numerical arrays, which
are handed to :mod:`relativisticpy.tensors.numerical`. These functions know
nothing about index names or positions: the compiler and :class:`Tensor` work
those out and call these with axes and labels.

The workhorse is :func:`contract`, a sparse Einstein summation. Curvature
tensors of the metrics people study are mostly zeros, so it multiplies only
nonzero components and chooses the order in which to contract the operands.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from itertools import permutations, product
from typing import Any, TypeAlias

import sympy
from sympy import Expr, ImmutableDenseNDimArray, derive_by_array, permutedims
from sympy.combinatorics import Permutation

from relativisticpy.tensors import numerical

Array: TypeAlias = ImmutableDenseNDimArray
#: A tensor's components: an array, or an expression for a scalar.
Components: TypeAlias = Any
Label: TypeAlias = Hashable
_Sparse: TypeAlias = dict[tuple[int, ...], Expr]


def to_array(value: Any) -> Array:
    """Convert nested lists, a SymPy matrix or array to an immutable SymPy array;
    numerical arrays stay as they are."""
    if isinstance(value, ImmutableDenseNDimArray) or numerical.is_numerical(value):
        return value
    if isinstance(value, sympy.MatrixBase):
        return ImmutableDenseNDimArray(value.tolist())
    return ImmutableDenseNDimArray(value)


def rank(components: Components) -> int:
    if numerical.is_numerical(components):
        return len(getattr(components, "shape", ()))
    return components.rank() if isinstance(components, sympy.NDimArray) else 0


def flatten(components: Components) -> list[Expr]:
    """The components in row-major order."""
    if numerical.is_numerical(components):
        return list(numerical.numpy().ravel(numerical.dense(components)))
    if not isinstance(components, sympy.NDimArray):
        return [components]
    return list(components.reshape(len(components))) if components.rank() else [components[()]]


def zeros(dim: int, rank: int) -> Components:
    if rank == 0:
        return sympy.S.Zero
    return ImmutableDenseNDimArray.zeros(*([dim] * rank))


def from_flat(values: Sequence[Expr], dim: int, rank: int) -> Components:
    if rank == 0:
        return sympy.sympify(values[0])
    return ImmutableDenseNDimArray(list(values), (dim,) * rank)


def apply(components: Components, func: Callable[[Expr], Expr]) -> Components:
    """Apply ``func`` to every component."""
    if numerical.is_numerical(components):
        return numerical.apply(components, func, getattr(func, "__name__", ""))
    if isinstance(components, sympy.NDimArray):
        return components.applyfunc(func)
    return func(components)


def permute(components: Components, axes: Sequence[int]) -> Components:
    """Reorder the axes: the result's axis ``i`` is the input's axis ``axes[i]``."""
    if list(axes) == sorted(axes):
        return components
    if numerical.is_numerical(components):
        return numerical.permute(components, axes)
    return permutedims(components, list(axes))


def select(components: Array, fixed: Mapping[int, int]) -> Components:
    """Fix some axes to a component: ``select(T, {0: 1})`` is ``T[1, :, :]``."""
    if numerical.is_numerical(components):
        return numerical.select(components, fixed)
    key = tuple(fixed.get(axis, slice(None)) for axis in range(components.rank()))
    result = components[key]
    return result if isinstance(result, sympy.NDimArray) else sympy.sympify(result)


def partial(components: Components, coordinates: Sequence[sympy.Symbol]) -> Array:
    """The array of partial derivatives, with the new axis first: ``∂_a T_{bc}``."""
    return derive_by_array(components, list(coordinates))


def symmetrize(components: Array, axes: Sequence[int], *, antisymmetric: bool) -> Array:
    """Average over every permutation of ``axes``, with signs if ``antisymmetric``.

    ``T_{(ab)} = (T_{ab} + T_{ba}) / 2`` and ``T_{[ab]} = (T_{ab} - T_{ba}) / 2``.
    """
    if numerical.is_numerical(components):
        return numerical.symmetrize(components, axes, antisymmetric=antisymmetric)
    axes = list(axes)
    terms = []
    for order in permutations(range(len(axes))):
        perm = list(range(components.rank()))
        for position, source in zip(axes, order, strict=True):
            perm[position] = axes[source]
        term = permutedims(components, perm)
        if antisymmetric and Permutation(list(order)).signature() < 0:
            term = -term
        terms.append(term)
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total / math.factorial(len(axes))


# ---------------------------------------------------------------------------
# Einstein summation
# ---------------------------------------------------------------------------


def contract(
    operands: Sequence[tuple[Components, Sequence[Label]]],
    output: Sequence[Label],
    dim: int | Sequence[int],
) -> Components:
    """Einstein summation over labelled operands.

    Each operand pairs components with one label per axis. Labels shared
    between operands, or repeated within one, are summed over unless they are
    in ``output``, which gives the labels of the result's axes in order::

        contract([(g, "ab"), (ginv, "bc")], "ac", 4)   # g_{ab} g^{bc}
        contract([(riemann, "abac")], "bc", 4)          # R^{a}{}_{bac}

    ``dim`` is the size of every axis, or, with axes of different sizes, a sequence
    giving the size of the axes of each integer label.

    Only nonzero components are multiplied, and operands are contracted
    pairwise in a greedy order that keeps intermediate results small.
    """
    if numerical.involves([components for components, _ in operands]):
        return numerical.contract(operands, output, dim)
    output = tuple(output)
    terms = [(_sparse(components), tuple(labels)) for components, labels in operands]
    for position, (_, labels) in enumerate(terms):
        if len(labels) != rank(operands[position][0]):
            raise ValueError(f"operand {position} has rank {rank(operands[position][0])}")
    terms = [
        _reduce(entries, labels, _needed(terms, output, skip=i))
        for i, (entries, labels) in enumerate(terms)
    ]
    while len(terms) > 1:
        i, j = _cheapest_pair(terms, output)
        merged = _multiply(terms[i], terms[j], _needed(terms, output, skip=(i, j)))
        terms = [term for k, term in enumerate(terms) if k not in (i, j)] + [merged]
    entries, labels = _reduce(*terms[0], set(output))
    if missing := set(output) - set(labels):
        raise ValueError(f"output labels {sorted(map(str, missing))} are not in any operand")
    positions = [labels.index(label) for label in output]
    shape = tuple(dim if isinstance(dim, int) else dim[int(label)] for label in output)  # type: ignore[call-overload]
    if not shape:
        return entries.get((), sympy.S.Zero)
    flat = [sympy.S.Zero] * math.prod(shape)
    for index, value in entries.items():
        flat[_flat_position([index[p] for p in positions], shape)] = value
    return ImmutableDenseNDimArray(flat, shape)


def _sparse(components: Components) -> _Sparse:
    if not isinstance(components, sympy.NDimArray) or components.rank() == 0:
        value = components[()] if isinstance(components, sympy.NDimArray) else components
        value = sympy.sympify(value)
        return {(): value} if value != 0 else {}
    indices = product(*(range(n) for n in components.shape))
    entries = zip(indices, flatten(components), strict=True)
    return {index: value for index, value in entries if value != 0}


def _needed(
    terms: Sequence[tuple[_Sparse, tuple[Label, ...]]],
    output: tuple[Label, ...],
    skip: int | tuple[int, ...],
) -> set[Label]:
    skipped = {skip} if isinstance(skip, int) else set(skip)
    needed = set(output)
    for k, (_, labels) in enumerate(terms):
        if k not in skipped:
            needed.update(labels)
    return needed


def _reduce(
    entries: _Sparse, labels: tuple[Label, ...], needed: set[Label]
) -> tuple[_Sparse, tuple[Label, ...]]:
    """Take traces over labels repeated within one operand and sum out labels
    nothing else needs."""
    kept = tuple(label for label in dict.fromkeys(labels) if label in needed)
    if kept == labels:
        return entries, labels
    accumulated: defaultdict[tuple[int, ...], list[Expr]] = defaultdict(list)
    for index, value in entries.items():
        assignment: dict[Label, int] = {}
        if all(
            assignment.setdefault(label, i) == i for label, i in zip(labels, index, strict=True)
        ):
            accumulated[tuple(assignment[label] for label in kept)].append(value)
    return _total(accumulated), kept


def _multiply(
    left: tuple[_Sparse, tuple[Label, ...]],
    right: tuple[_Sparse, tuple[Label, ...]],
    needed: set[Label],
) -> tuple[_Sparse, tuple[Label, ...]]:
    (a, a_labels), (b, b_labels) = left, right
    shared = [label for label in dict.fromkeys(a_labels) if label in b_labels]
    result = tuple(
        label
        for label in dict.fromkeys(a_labels + b_labels)
        if label not in shared or label in needed
    )
    a_at = {label: a_labels.index(label) for label in a_labels}
    b_at = {label: b_labels.index(label) for label in b_labels}
    by_key: defaultdict[tuple[int, ...], list[tuple[tuple[int, ...], Expr]]] = defaultdict(list)
    for index, value in b.items():
        by_key[tuple(index[b_at[label]] for label in shared)].append((index, value))
    sources = [(0, a_at[label]) if label in a_at else (1, b_at[label]) for label in result]
    accumulated: defaultdict[tuple[int, ...], list[Expr]] = defaultdict(list)
    for a_index, a_value in a.items():
        for b_index, b_value in by_key.get(tuple(a_index[a_at[label]] for label in shared), ()):
            pair = (a_index, b_index)
            accumulated[tuple(pair[side][axis] for side, axis in sources)].append(a_value * b_value)
    return _total(accumulated), result


def _cheapest_pair(
    terms: Sequence[tuple[_Sparse, tuple[Label, ...]]], output: tuple[Label, ...]
) -> tuple[int, int]:
    """The pair whose product has the fewest axes, then the fewest multiplications."""
    best: tuple[tuple[int, int], tuple[int, int]] | None = None
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            needed = _needed(terms, output, skip=(i, j))
            labels = set(terms[i][1]) | set(terms[j][1])
            shared = set(terms[i][1]) & set(terms[j][1])
            axes = len({label for label in labels if label not in shared or label in needed})
            cost = (axes, len(terms[i][0]) * len(terms[j][0]))
            if best is None or cost < best[0]:
                best = (cost, (i, j))
    assert best is not None
    return best[1]


def _total(accumulated: Mapping[tuple[int, ...], list[Expr]]) -> _Sparse:
    totals = {index: sympy.Add(*values) for index, values in accumulated.items()}
    return {index: value for index, value in totals.items() if value != 0}


def _flat_position(index: Sequence[int], shape: Sequence[int]) -> int:
    position = 0
    for i, size in zip(index, shape, strict=True):
        position = position * size + i
    return position
