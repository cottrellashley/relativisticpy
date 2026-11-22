r"""Binding data from Python, and the spaces its axes range over.

``wb["R_{ta}"] = returns`` binds a tensor whose first index ranges over the rows of
``returns`` and its second over its columns, and ``wb["J"] = 1.5`` a number: the rules by
which each axis ranges over a space are those of :meth:`Workbook.__setitem__
<relativisticpy.workbook.Workbook.__setitem__>`. A definition binds the data of a Python
object by the same rules, as ``P_{ta} := \texttt{prices}`` does, and a statement reads
data so where it writes ``\texttt{signal}_t``. :func:`declare_space` declares a space for
indices to range over, as :meth:`Workbook.space <relativisticpy.workbook.Workbook.space>`
does::

    >>> import numpy as np
    >>> from relativisticpy.compiler import Session
    >>> session = Session()
    >>> name, entries = written_tensor("R_{ta}", 2)
    >>> bind_data(session, "R_{ta}", name, entries, data_of(np.ones((3, 2)), "R_{ta}"), "numpy")
    >>> [str(space) for space in session.globals.lookup("R").node.type.spaces]
    ['the space of t', 'the space of a']
"""

from __future__ import annotations

import numbers
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import sympy

from relativisticpy.compiler.scope import Scope, TensorBinding
from relativisticpy.compiler.session import Session
from relativisticpy.errors import CompileError, DataError, RelativisticPyError, UsageError
from relativisticpy.ir import SCALAR, Node, TensorType
from relativisticpy.language import Source, Span, parse
from relativisticpy.language import syntax as ast
from relativisticpy.language.parser import split_index_labels
from relativisticpy.language.vocabulary import GREEK, GREEK_VARIANTS
from relativisticpy.tensors import Index, Indices, Space, arrays, numerical
from relativisticpy.tensors.numerical import double_precision

__all__ = [
    "bind_data",
    "data_of",
    "data_tensor",
    "declare_space",
    "in_basis_order",
    "labelled_spaces",
    "rank_of",
    "value_node",
    "written_name",
    "written_tensor",
]


def written_name(written: str) -> tuple[str, list[tuple[str, bool]] | None]:
    """What ``wb[written]`` names: a name, without index labels, or a tensor with its index
    labels, each with whether it is upper: ``("x", None)`` for ``x``, and
    ``("R", [("t", False), ("a", False)])`` for ``R_{ta}``. What is neither is refused."""
    try:
        program = parse(Source(written, "<binding>"))
    except RelativisticPyError:
        program = None
    statements = program.statements if program is not None else ()
    expr = (
        statements[0].expr
        if len(statements) == 1 and isinstance(statements[0], ast.Evaluate)
        else None
    )
    if isinstance(expr, ast.Name):
        return expr.id, None
    if isinstance(expr, ast.Scripted) and isinstance(expr.base, ast.Name):
        name = expr.base.id
        entries: list[tuple[str, bool]] = []
        head, underscore, tail = name.partition("_")
        if underscore and tail.isalpha():  # B_a^b reads as the name B_a; it is B_{a}{}^{b}
            name = head
            entries.extend((label, False) for label in split_index_labels(tail))
        count = len(entries)
        for script in expr.scripts:
            for item in script.indices.items if script.indices is not None else ():
                count += 1
                if item.label is None or item.value is not None or item.derivative is not None:
                    continue
                # X^i_j reads as X^{i_j}; as a binding it can only mean X^{i}{}_{j}
                first, *rest = str(item.label).split("_")
                entries.append((first, script.kind == "^"))
                entries.extend((label, False) for label in rest if label)
                count += sum(1 for label in rest if label)
        if entries and len(entries) == count == len({label for label, _ in entries}):
            return name, entries
    raise DataError(
        f"cannot bind {written!r}: write a name, as in x, or a tensor with distinct index "
        "labels, as in R_{ta}"
    )


def written_tensor(written: str, rank: int) -> tuple[str, list[tuple[str, bool]]] | None:
    """The tensor ``wb[written] = data`` binds data of ``rank`` axes to, with its index
    labels: ``R_{ta}``, or ``y_t`` for data with axes; ``None`` for a name bound to a value."""
    name, entries = written_name(written)
    if entries is None and "_" in name and rank:
        base, _, tail = name.partition("_")  # y_t: the tensor y, with index t
        return base, [(label, False) for label in split_index_labels(tail)]
    return None if entries is None else (name, entries)


def rank_of(value: Any) -> int:
    """How many axes a value bound from Python has: none for a number."""
    shape = getattr(value, "shape", None)
    if shape is None and isinstance(value, list | tuple):
        return 1
    return len(shape) if shape is not None else 0


def value_node(session: Session, written: str, value: Any) -> Node:
    """The node of a number or a SymPy expression bound to a name from Python, as it is.
    Text, which the language reads, is compiled by the workbook that binds it."""
    if numerical.is_numerical(value) and getattr(value, "shape", ()) != ():
        raise DataError(
            f"an array is bound to a tensor written with its indices, as in {written}^{{a}}"
        )
    if numerical.is_numerical(value):
        scalar = value[()] if isinstance(value, numerical.numpy().ndarray) else value
        return session.graph.const(scalar, SCALAR)  # a view of nothing
    if isinstance(value, sympy.Basic | numbers.Number):  # never text, which SymPy would run
        return session.graph.const(sympy.sympify(value, strict=True), SCALAR)
    raise DataError(
        f"{written} can be bound to a number, a SymPy expression or text in the "
        f"language, not {_kind(value)}"
    )


def data_of(value: Any, written: str) -> tuple[Any, list[tuple[str, ...] | None], list[str | None]]:
    """Components, the names of each axis's elements and each axis's name, from a pandas
    object; just the components from anything else. The components are a copy, so that
    data changed in place afterwards changes nothing bound. Text is refused, never read
    as SymPy would read it, as Python."""
    module = type(value).__module__ or ""
    if module.startswith("pandas"):
        axes = [value.index] if value.ndim == 1 else [value.index, value.columns]
        bases = [None if _is_range(axis) else tuple(str(x) for x in axis) for axis in axes]
        names = [str(axis.name) if axis.name is not None else None for axis in axes]
        return _copy(_numbers(value.to_numpy(), written)), bases, names
    if numerical.is_numerical(value):
        rank = len(getattr(value, "shape", ()))
        return _copy(_numbers(value, written)), [None] * rank, [None] * rank
    if isinstance(value, sympy.NDimArray | sympy.MatrixBase):
        array = arrays.to_array(value)
        return array, [None] * array.rank(), [None] * array.rank()
    if not isinstance(value, list | tuple):
        raise DataError(
            f"the data for {written} is {_kind(value)}: bind an array, a pandas object, a "
            "SciPy sparse matrix, a SymPy array or nested lists"
        )
    _refuse_text(value, written)
    try:
        array = numerical.numpy().array(value)
    except ValueError:
        where = _ragged(value) or "its rows are not all of one length"
        raise DataError(f"the data for {written} is ragged: {where}") from None
    if array.dtype == object:  # SymPy expressions among the numbers
        array = arrays.to_array(value)
        return array, [None] * array.rank(), [None] * array.rank()
    return array, [None] * array.ndim, [None] * array.ndim


def bind_data(
    session: Session,
    written: str,
    name: str,
    entries: Sequence[tuple[str, bool]],
    data: tuple[Any, Sequence[tuple[str, ...] | None], Sequence[str | None]],
    backend: str,
    span: Span | None = None,
) -> None:
    r"""Bind data to the tensor ``name`` with the index labels ``entries``, as
    ``wb[written] = data`` binds it, and as a definition binds the data of a Python object,
    as in ``P_{ta} := \texttt{prices}``: ``data`` is the components, the names of each
    axis's elements and each axis's own name, as :func:`data_of` gives them, and each axis
    ranges over a space by the rules of :meth:`Workbook.__setitem__
    <relativisticpy.workbook.Workbook.__setitem__>`. ``span`` is where a definition writes
    it."""
    node, indices = data_tensor(session, written, name, entries, data, backend, span)
    session.globals.bind(name, TensorBinding(node, indices))


def data_tensor(
    session: Session,
    written: str,
    name: str,
    entries: Sequence[tuple[str, bool]],
    data: tuple[Any, Sequence[tuple[str, ...] | None], Sequence[str | None]],
    backend: str,
    span: Span | None = None,
    labels: Mapping[str, Space] | None = None,
) -> tuple[Node, Indices]:
    r"""The node of data written as the tensor ``name`` with the index labels ``entries``,
    and its indices, each axis over a space, as :func:`bind_data` binds it and as
    ``\texttt{signal}_t`` reads it: there, a label that already ranges over a space in the
    statement, by ``labels``, ranges over it on the data too, as a declared one does."""
    known = labels or {}
    components, bases, axis_names = data
    shape = tuple(getattr(components, "shape", ()))
    if len(shape) != len(entries):
        raise DataError(
            f"{written} has {_count(len(entries), 'index', 'indices')}, but the data "
            f"has {_count(len(shape), 'axis', 'axes')}",
            hint="write one index for each axis of the data, as in R_{ta} for a table",
        )
    complex_data = numerical.is_numerical(components) and numerical.numpy().iscomplexobj(components)
    field: Literal["real", "complex"] = "complex" if complex_data else "real"
    spaces: list[Space] = []
    for axis, ((label, _), size, basis, axis_name) in enumerate(
        zip(entries, shape, bases, axis_names, strict=True)
    ):
        earlier = [(entries[k][0], space) for k, space in enumerate(spaces)]
        space = known.get(label) or _axis_space(
            session, name, label, size, basis, axis_name, field, earlier
        )
        if space.dim != size:
            raise DataError(
                f"the index {label} ranges over {space}, of dimension {space.dim}, "
                f"but this axis of the data has {size} components"
            )
        if space.name == axis_name:  # a space a pandas axis names, for later data
            session.spaces.setdefault(axis_name, space)
        if basis is not None and space.basis is not None and basis != space.basis:
            what = f"axis {axis + 1} of {written}"
            components = in_basis_order(components, axis, basis, space, what)
        spaces.append(space)
    if backend == "jax" and numerical.is_numerical(components):
        import jax
        import jax.numpy as jnp

        double_precision(jax)
        if not numerical.is_sparse(components):  # a sparse matrix stays SciPy's
            components = jnp.asarray(components)
    indices = Indices(Index(label, upper) for label, upper in entries)
    return session.graph.const(components, TensorType(indices, tuple(spaces)), span), indices


def labelled_spaces(
    session: Session, name: str, label: str, earlier: Sequence[tuple[str, Space]]
) -> list[Space]:
    """The spaces with a basis that data bound now may meet by label: those of the earlier
    axes of the same data, those of the same label on the tensors already bound, then those
    of every other axis already bound, and those named."""
    candidates = [space for _, space in earlier]
    candidates += _spaces_of_label(session.globals, name, label)
    candidates += [
        space
        for _, binding in session.globals.bindings()
        if isinstance(binding, TensorBinding) and isinstance(binding.node.type, TensorType)
        for space in binding.node.type.spaces
        if not space.spacetime
    ]
    candidates += session.spaces.values()
    return [space for space in candidates if space.basis is not None]


def in_basis_order(
    components: Any, axis: int, basis: tuple[str, ...], space: Space, what: str
) -> Any:
    """Data labelled by pandas along ``axis``, put in the order of its space's basis; an
    error if its labels are not the space's elements."""
    assert space.basis is not None
    if sorted(basis) != sorted(space.basis):
        foreign = [element for element in basis if element not in space.basis]
        missing = [element for element in space.basis if element not in basis]
        details = []
        if foreign:
            details.append(f"{_listed(foreign)} {'is' if len(foreign) == 1 else 'are'} not one")
        if missing:
            details.append(f"{_listed(missing)} {'is' if len(missing) == 1 else 'are'} missing")
        raise CompileError(
            f"the labels of {what} are not the elements of {space}: "
            + ("; ".join(details) if details else "some appear more than once")
        )
    position = {element: number for number, element in enumerate(basis)}
    order = [position[element] for element in space.basis]
    return numerical.numpy().take(components, order, axis=axis)


def declare_space(
    session: Session,
    name: str,
    dim: int | None = None,
    *,
    basis: Iterable[Any] | None = None,
    labels: Iterable[str] = (),
    field: str = "real",
    measure: Any = None,
) -> Space:
    r"""Declare a space in ``session`` for indices to range over, as ``\mathcal{A} :=
    \mathbb{R}^{500}`` does: its ``dim``, or the names of its ``basis``; index ``labels``
    that range over it; and a ``measure``, weights for its expectations (uniform
    otherwise), taken by label from a pandas Series. :meth:`Workbook.space
    <relativisticpy.workbook.Workbook.space>` declares one so."""
    names = tuple(str(b) for b in basis) if basis is not None else None
    size = len(names) if names is not None else dim
    if size is None:
        raise UsageError("give the dimension of the space, or the names of its basis")
    if field not in ("real", "complex"):
        raise UsageError(f"a space's field is real or complex, not {field!r}")
    kind: Literal["real", "complex"] = "complex" if field == "complex" else "real"
    try:
        space = Space(name, size, basis=names, field=kind)
    except ValueError as error:
        raise UsageError(str(error)) from None
    if measure is not None:
        weights, labelled, _ = data_of(measure, f"the measure on {name}")
        weights = numerical.numpy().asarray(numerical.dense(weights), dtype=float)
        if weights.shape != (size,):
            given = len(weights) if weights.ndim == 1 else None
            raise DataError(
                f"a measure on {name} needs {size} weights, one for each element, but "
                + (
                    f"{given} {'is' if given == 1 else 'are'} given"
                    if given is not None
                    else f"the weights have {_count(weights.ndim, 'axis', 'axes')}"
                )
            )
        if labelled[0] is not None and names is not None:
            what = f"the measure on {name}"
            weights = in_basis_order(weights, 0, labelled[0], space, what)
        session.measures[space] = session.graph.const(weights / weights.sum(), SCALAR)
    session.spaces[name] = space
    for label in labels:
        session.labels[label] = space
    return space


def _numbers(array: Any, written: str) -> Any:
    """A numerical array, refused if it holds text."""
    dtype = getattr(array, "dtype", None)
    if dtype is not None and dtype.kind in "USa":
        raise DataError(f"the data for {written} is text: bind numbers")
    if dtype is not None and dtype.kind == "O":
        _refuse_text(array.tolist(), written)
    return array


def _refuse_text(value: Any, written: str) -> None:
    """Refuse text anywhere in nested lists: SymPy would read it as Python, and run it."""
    if isinstance(value, str | bytes):
        raise DataError(
            f"the data for {written} holds text, {value[:40]!r}: bind numbers or SymPy expressions",
            hint="to compute with an expression written as text, write it in a program",
        )
    if isinstance(value, list | tuple):
        for item in value:
            _refuse_text(item, written)


def _ragged(value: Sequence[Any]) -> str:
    """Where nested lists are not rectangular, as a message says it; empty if they are."""
    nested = [isinstance(item, list | tuple) for item in value]
    if not any(nested):
        return ""
    if not all(nested):
        odd = (
            nested.index(True) if nested.count(True) < nested.count(False) else nested.index(False)
        )
        what = "a list" if nested[odd] else "a number"
        others = "numbers" if nested[odd] else "lists"
        return f"entry {odd + 1} is {what}, the others are {others}"
    lengths = [len(item) for item in value]
    for position, length in enumerate(lengths):
        if length != lengths[0]:
            return f"entry {position + 1} has {length} entries, but entry 1 has {lengths[0]}"
    for position, item in enumerate(value):
        inner = _ragged(item)
        if inner:
            return f"in entry {position + 1}, {inner}"
    return ""


def _copy(array: Any) -> Any:
    """A numerical array of one's own: a copy, except of a JAX array, which cannot change,
    with small integers widened so that arithmetic cannot overflow them."""
    if numerical.is_jax(array):
        return array
    if numerical.is_sparse(array):
        return numerical.widened(array.copy())
    return numerical.widened(numerical.numpy().array(array))


def _axis_space(
    session: Session,
    name: str,
    label: str,
    size: int,
    basis: tuple[str, ...] | None,
    axis_name: str | None,
    field: Literal["real", "complex"],
    earlier: Sequence[tuple[str, Space]],
) -> Space:
    """The space an axis of the data bound to the tensor ``name`` ranges over, given its
    index ``label`` and the name of its pandas axis, after the ``earlier`` axes' labels
    and spaces: the rules of :meth:`Workbook.__setitem__
    <relativisticpy.workbook.Workbook.__setitem__>`."""
    declared = session.labels.get(label)
    if declared is not None:
        return declared
    if basis is not None:  # labelled data meets labelled data by label, never by position
        for space in labelled_spaces(session, name, label, earlier):
            if space.dim == size and sorted(space.basis or ()) == sorted(basis):
                return space
    if axis_name is not None:
        named = session.spaces.get(axis_name)
        return named if named is not None else Space(axis_name, size, basis=basis, field=field)
    candidates = [space for other, space in earlier if _alike(other, label)]
    candidates += _spaces_of_label(session.globals, name, label)
    for space in candidates:
        if space.dim == size and _same_elements(space.basis, basis):
            return space
    return Space(f"the space of {label}", size, basis=basis, field=field)


def _alike(one: str, other: str) -> bool:
    """Whether two index labels are of a kind: from the same run of the alphabet, a to h
    as assets a and b are, i to n as nodes i and j, o to z as times s and t; or Greek."""
    return _run(one) == _run(other)


def _run(label: str) -> str:
    if label in GREEK or label in GREEK_VARIANTS:
        return "Greek"
    first = label[:1].lower()
    for start, end in (("a", "h"), ("i", "n"), ("o", "z")):
        if start <= first <= end:
            return start
    return label  # alike only to itself


def _spaces_of_label(scope: Scope, name: str, label: str) -> list[Space]:
    """The spaces the index ``label`` ranges over in tensors already bound, other than
    spacetime: in the tensor ``name`` first, then in the others, the latest first."""
    bindings = scope.bindings()
    ordered = [b for n, b in bindings if n == name] + [b for n, b in bindings[::-1] if n != name]
    return [
        space
        for binding in ordered
        if isinstance(binding, TensorBinding) and isinstance(binding.node.type, TensorType)
        for index, space in zip(binding.indices, binding.node.type.spaces, strict=True)
        if index.name == label and not space.spacetime
    ]


def _same_elements(one: tuple[str, ...] | None, other: tuple[str, ...] | None) -> bool:
    """Whether two bases may name the same elements: they do, or one has no names."""
    return one is None or other is None or sorted(one) == sorted(other)


def _is_range(axis: Any) -> bool:
    """Whether a pandas axis is the default numbering, 0, 1, 2, ..., not names."""
    return type(axis).__name__ == "RangeIndex" and axis.start == 0 and axis.step == 1


def _kind(value: Any) -> str:
    """What kind of Python value ``value`` is, as a message says it: ``int``, ``None``."""
    return "None" if value is None else type(value).__name__


def _count(number: int, one: str, many: str) -> str:
    """``1 index``, ``2 indices``."""
    return f"{number} {one if number == 1 else many}"


def _listed(names: Sequence[str]) -> str:
    shown = ", ".join(names[:3])
    return shown + (f" and {len(names) - 3} more" if len(names) > 3 else "")
