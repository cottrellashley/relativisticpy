r"""The :class:`Tensor` value: components, named indices and the coordinates they refer to."""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import product
from typing import Any

import sympy
from sympy import Expr, Symbol

from relativisticpy.errors import DataError, EvaluationError, IndexNotationError, UsageError
from relativisticpy.tensors import arrays, numerical
from relativisticpy.tensors.indices import Index, Indices
from relativisticpy.tensors.spaces import Space

#: The arithmetic of NumPy's functions, which tensors do themselves.
_OPERATORS: dict[str, Callable[[Any, Any], Any]] = {
    "multiply": operator.mul,
    "add": operator.add,
    "subtract": operator.sub,
    "divide": operator.truediv,
    "true_divide": operator.truediv,
}

#: A component of a tensor, or the value of a contraction with no free index: a SymPy
#: expression, or a number for numerical components.
Scalar = Any

#: How a component is named in a key: by position, or by the name of a coordinate or of
#: an element of a basis, as in ``G['r', 't', 't']`` and ``w['MSFT']``.
Key = Any


class Tensor:
    r"""A tensor in a coordinate basis.

    ``components`` is a SymPy array whose axes follow ``indices``; ``coordinates``
    are the symbols the components are expressed in. Arithmetic follows index
    notation: ``+`` aligns operands by index name and ``*`` sums over every
    index that appears once up and once down::

        >>> from sympy import symbols
        >>> t, x = symbols("t x")
        >>> eta = Tensor([[-1, 0], [0, 1]], "_{ab}", (t, x), name="eta")
        >>> v = Tensor([2, 3], "^{b}", (t, x), name="v")
        >>> eta * v          # η_{ab} v^b
        η_{ab} v^b in (t, x): 2 components non-zero
          η_{tb} v^b = −2
          η_{xb} v^b = 3

    A tensor is shown by its name and indices, then one line for each non-zero
    component, labelled by coordinate: see :mod:`relativisticpy.display`. Components
    are taken by position or by coordinate: ``eta[0, 0]``, ``eta['t', 't']``.

    Components may also be numerical, a NumPy array, with each axis ranging over one of
    ``spaces``: assets, days, a qubit. :meth:`to_numpy` and :meth:`to_pandas` give them
    back, labelled by the bases of their spaces.

    ``source`` is the expression the tensor is the value of, as written, such as
    ``\Gamma^{r}_{tb}`` or ``2 R_{\mu\nu}``: shown in place of the name, it keeps the
    indices a slice fixes, and names a result that has no name of its own.
    """

    __slots__ = ("components", "coordinates", "indices", "name", "source", "spaces")

    def __init__(
        self,
        components: Any,
        indices: Indices | Sequence[Index] | str,
        coordinates: Sequence[Symbol] = (),
        *,
        name: str | None = None,
        spaces: Sequence[Space] | None = None,
        source: str | None = None,
    ) -> None:
        if isinstance(components, str) and components.lstrip()[:1] in ("_", "^"):
            raise TypeError(
                f"Tensor(components, indices, ...) takes the components first; "
                f"{components!r} looks like indices. "
                f"Did you mean Tensor({indices!r}, {components!r})?"
            )
        self.indices = indices if isinstance(indices, Indices) else Indices(indices)
        self.coordinates = tuple(coordinates)
        self.name = name
        self.spaces = tuple(spaces) if spaces is not None else None
        self.source = source
        repeated = sorted({i.name for i in self.indices if self.indices.names.count(i.name) > 1})
        if repeated:
            raise IndexNotationError(
                f"index {repeated[0]} appears twice on one tensor: a tensor's indices are "
                "distinct, and a repeated label is a trace, written as a contraction"
            )
        if numerical.is_numerical(components):
            self.components = components
            shape = tuple(getattr(components, "shape", ()))
            if len(shape) != len(self.indices):
                raise DataError(_misfit(shape, len(self.indices)))
            return
        self.components = arrays.to_array(components) if self.indices else sympy.sympify(components)
        if not self.indices:
            return
        shape = tuple(self.components.shape)
        if len(shape) != len(self.indices):
            raise DataError(_misfit(shape, len(self.indices)))
        if self.spaces is not None:
            if shape != tuple(space.dim for space in self.spaces):
                sizes = " × ".join(str(space.dim) for space in self.spaces)
                raise DataError(f"components of shape {_times(shape)} do not fit spaces of {sizes}")
        elif len(set(shape)) != 1:
            raise DataError(
                f"symbolic components must be square, one axis for each coordinate, "
                f"unless their spaces are given: these are {_times(shape)}; pass "
                f"spaces=, one Space for each axis"
            )
        elif self.coordinates and shape[0] != len(self.coordinates):
            raise DataError(
                f"components have dimension {shape[0]} but there are "
                f"{len(self.coordinates)} coordinates"
            )

    # -- shape -----------------------------------------------------------------

    @property
    def rank(self) -> int:
        """The number of indices: 2 for :math:`g_{\\mu\\nu}`."""
        return len(self.indices)

    ndim = rank

    @property
    def dim(self) -> int:
        """The number of components along each axis: 4 in four dimensions. For a tensor
        whose axes differ in size, such as returns over days and assets, it is a
        ``ValueError``: see :attr:`shape`."""
        if not self.indices:
            return len(self.coordinates)
        sizes = set(self.shape)
        if len(sizes) != 1:
            raise UsageError(
                f"the axes of {self._written()} have different sizes, {_times(self.shape)}: "
                "use shape"
            )
        return int(self.shape[0])

    @property
    def shape(self) -> tuple[int, ...]:
        """The number of components along each axis: ``(4, 4, 4)`` for Christoffel's."""
        return tuple(int(n) for n in self.components.shape) if self.indices else ()

    @property
    def size(self) -> int:
        """The number of components: 64 for :math:`\\Gamma^a{}_{bc}` in four dimensions."""
        total = 1
        for n in self.shape:
            total *= n
        return total

    @property
    def numerical(self) -> bool:
        """Whether the components are numbers in an array, rather than SymPy's."""
        return numerical.is_numerical(self.components)

    @property
    def dtype(self) -> Any:
        """The type of numerical components, as NumPy names it; ``object`` for SymPy's."""
        if self.numerical:
            return self.components.dtype
        return numerical.numpy().dtype(object)

    @property
    def free_symbols(self) -> set[Symbol]:
        """The symbols the components are expressed in."""
        if self.numerical:
            return set()
        found: set[Symbol] = set()
        for component in arrays.flatten(self.components):
            found |= getattr(component, "free_symbols", set())
        return found

    def to_numpy(self, values: dict[Any, Any] | None = None, **named: Any) -> Any:
        """The components as a NumPy array. Symbolic components need a value for each
        symbol, given by name, as ``g.to_numpy(M=1, r=3, theta=0.5)``."""
        given = {str(key): value for key, value in {**(values or {}), **named}.items()}
        components = self.components
        if given and not self.numerical:
            components = arrays.apply(components, lambda c: _substituted(c, given))
        try:
            array = numerical.as_array(components)
        except EvaluationError:
            raise self._symbolic(components) from None
        return numerical.numpy().asarray(numerical.dense(array))

    def _symbolic(self, components: Any) -> EvaluationError:
        """The error for symbolic components where numbers are wanted."""
        found: set[str] = set()
        for component in arrays.flatten(components):
            found |= {str(s) for s in getattr(component, "free_symbols", ())}
        names = sorted(found, key=lambda n: (n not in {str(c) for c in self.coordinates}, n))
        values = ", ".join(f"{n}=..." for n in names[:4])
        return EvaluationError(
            f"{self._written()} has symbolic components (in {', '.join(names)}): "
            "give them values first",
            hint=f"tensor.to_numpy({values}), or relativisticpy.numeric.evaluate(tensor, {values})",
        )

    def to_pandas(self) -> Any:
        """The components as a pandas Series (one index) or DataFrame (two), labelled by
        the bases of their spaces, or by the coordinates."""
        import pandas as pd

        data = self.to_numpy()
        names = [self.labels(axis) for axis in range(self.rank)]
        if self.rank == 1:
            return pd.Series(data, index=names[0], name=self.name)
        if self.rank == 2:
            return pd.DataFrame(data, index=names[0], columns=names[1])
        raise UsageError("only a tensor with one or two indices is a Series or a DataFrame")

    def labels(self, axis: int) -> list[Any]:
        """The names of the components along an axis: the basis of its space, the
        coordinates of spacetime, or else their positions, ``0, 1, 2, ...``."""
        space = self.spaces[axis] if self.spaces is not None else None
        if self.coordinates and (space is None or space.spacetime):
            return [str(c) for c in self.coordinates]
        if space is not None and space.basis is not None:
            return list(space.basis)
        return list(range(self.shape[axis]))

    _basis = labels  # the name it had

    # -- components by position or by name ----------------------------------------

    def __getitem__(self, key: Key) -> Scalar | Tensor:
        """A component, by position or by name: ``G[1, 0, 0]``, ``G['r', 't', 't']``,
        ``G['θ', 'φ', 'φ']`` or ``w['MSFT']``. A key with a slice, such as
        ``G['r', 't', :]``, gives the tensor of the components left free."""
        keys = key if isinstance(key, tuple) else (key,)
        if not all(_is_key(k) for k in keys):  # an array of positions, a mask: as NumPy
            return self.components[key]
        if not self.indices:
            raise TypeError(f"{self._written()} is a scalar: it has no components to take")
        if len(keys) > self.rank:
            raise IndexError(
                f"{self._written()} has {self.rank} indices, but {len(keys)} are given"
            )
        positions = [self._position(axis, k) for axis, k in enumerate(keys)]
        positions += [slice(None)] * (self.rank - len(positions))
        if all(isinstance(p, int) for p in positions):
            return self.components[tuple(positions)]
        fixed = {axis: p for axis, p in enumerate(positions) if isinstance(p, int)}
        if any(isinstance(p, slice) and p != slice(None) for p in positions):
            return self.components[tuple(positions)]  # a range of components: an array
        kept = [axis for axis in range(self.rank) if axis not in fixed]
        components = arrays.select(self.components, fixed) if fixed else self.components
        return Tensor(
            components,
            Indices(self.indices[axis] for axis in kept),
            self.coordinates,
            name=self.name,
            spaces=[self.spaces[axis] for axis in kept] if self.spaces is not None else None,
            source=self._sliced_source(fixed),
        )

    def _position(self, axis: int, key: Key) -> int | slice:
        if isinstance(key, slice):
            return key
        if isinstance(key, int) or (hasattr(key, "__index__") and not isinstance(key, str)):
            return int(key)
        names = self.labels(axis)
        wanted = _label_name(key)
        for position, name in enumerate(names):
            if _label_name(name) == wanted:
                return position
        shown = ", ".join(str(n) for n in names[:8]) + (", ..." if len(names) > 8 else "")
        what = (
            "a coordinate; the coordinates are"
            if self.coordinates and self._spacetime(axis)
            else f"an element of the basis of axis {axis}, which is"
        )
        raise KeyError(f"{key!r} is not {what} {shown}")

    def _spacetime(self, axis: int) -> bool:
        space = self.spaces[axis] if self.spaces is not None else None
        return space is None or space.spacetime

    def _sliced_source(self, fixed: dict[int, int]) -> str | None:
        """The written form of a slice: the name with the indices it fixes, such as
        ``\\Gamma^{r}_{tb}``. A result without a name, such as ``2 R_{\\mu\\nu}``, has
        none to write."""
        if self.name is None:
            return None
        from relativisticpy.display import component_label, text_name
        from relativisticpy.tensors.indices import text_label

        labels = [
            component_label(self, axis, fixed[axis], "text")
            if axis in fixed
            else text_label(self.indices[axis].name)
            for axis in range(self.rank)
        ]
        return text_name(self.name) + self.indices.text(labels)

    def __iter__(self) -> Iterator[Any]:
        """The components along the first axis, as NumPy iterates over an array."""
        if not self.indices:
            raise TypeError(f"{self._written()} is a scalar: it has no axes to iterate over")
        if self.numerical:
            return iter(numerical.dense(self.components))
        if self.rank == 1:
            return (self.components[i] for i in range(self.shape[0]))
        return (arrays.select(self.components, {0: i}) for i in range(self.shape[0]))

    def __len__(self) -> int:
        """The number of components along the first axis, as for a NumPy array (1 for a
        scalar); see :attr:`size` for the number of components."""
        if not self.indices:
            return 1
        return int(self.shape[0])

    # -- comparison and display ------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tensor):
            return NotImplemented
        if sorted(self.indices, key=str) != sorted(other.indices, key=str):
            return False
        try:
            theirs = _aligned(other, self.indices)
        except IndexNotationError:
            return False
        if tuple(getattr(theirs, "shape", ())) != tuple(getattr(self.components, "shape", ())):
            return False
        if self.numerical or other.numerical:  # to rounding error
            try:
                return numerical.close(self.components, theirs)
            except EvaluationError:  # symbols, which no numbers equal
                return False
        return bool(self.components == theirs)

    def __hash__(self) -> int:
        # Equal tensors have the same indices and shape. Numerical ones are equal to
        # rounding error, which no hash of their components could respect.
        return hash((frozenset(self.indices), tuple(sorted(self.shape))))

    # The forms of relativisticpy.display, by which every surface shows a tensor.

    def _display_unicode(self) -> str:
        from relativisticpy import display

        return display.tensor_form(self, "unicode")

    def _display_text(self) -> str:
        from relativisticpy import display

        return display.tensor_form(self, "text")

    def _display_latex(self) -> str:
        from relativisticpy import display

        return display.tensor_form(self, "latex")

    def _display_data(self) -> dict[str, Any]:
        from relativisticpy import display

        return display.tensor_data(self)

    def __str__(self) -> str:
        from relativisticpy import display

        return display.unicode(self)

    __repr__ = __str__

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text(str(self) if not cycle else "Tensor(...)")

    def _repr_latex_(self) -> str:
        from relativisticpy import display

        return display.jupyter_latex(self)

    def _repr_html_(self) -> str | None:
        """An HTML table for numerical data, in notebooks; symbolic tensors are typeset."""
        from relativisticpy import display

        return display.jupyter_html(self)

    def _display_(self) -> Any:
        """What marimo shows: the table or the typeset tensor of :mod:`~relativisticpy.display`."""
        from relativisticpy import display

        return display.marimo(self)

    def _latex(self, printer: Any = None) -> str:
        """The LaTeX SymPy's printer uses, as in ``sympy.latex(T)``."""
        return self.latex()

    def _sympy_(self) -> Any:
        """The components, for SymPy: ``sympy.sympify(T)`` is the array of components."""
        if self.numerical:
            return sympy.ImmutableDenseNDimArray(self.to_numpy().tolist())
        return self.components

    def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
        np = numerical.numpy()
        if self.numerical:
            array = np.asarray(numerical.dense(self.components))
        else:
            flat = arrays.flatten(self.components)
            array = np.empty(len(flat), dtype=object)
            array[:] = flat
            array = array.reshape(self.shape)
        return array.astype(dtype) if dtype is not None else array

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        """NumPy's functions keep the indices of numerical components: ``np.exp(A)`` is a
        tensor with the indices of ``A``."""
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented
        np = numerical.numpy()
        inputs = tuple(v.item() if isinstance(v, np.generic) else v for v in inputs)
        operation = _OPERATORS.get(ufunc.__name__)
        if operation is not None and len(inputs) == 2 and not kwargs:
            return operation(*inputs)  # a number and a tensor: as Python combines them
        if not self.numerical:
            return NotImplemented
        arguments = []
        for value in inputs:
            if isinstance(value, Tensor):
                if value is not self and value.indices != self.indices:
                    return NotImplemented
                arguments.append(np.asarray(value))
            else:
                arguments.append(value)
        result = ufunc(*arguments, **kwargs)
        if getattr(result, "shape", None) != self.shape:
            return result
        return Tensor(result, self.indices, self.coordinates, spaces=self.spaces)

    def latex(self) -> str:
        """Typeset the tensor: its non-zero components, labelled by coordinate."""
        from relativisticpy import display

        return display.latex(self)

    def nonzero(self) -> dict[tuple[Any, ...], Scalar]:
        """The nonzero components, keyed by the names of their coordinates, or of the
        elements of their spaces' bases, or else by position: ``{('t', 't', 'r'): ...}``."""
        names = [self.labels(axis) for axis in range(self.rank)]
        if self.numerical:
            np = numerical.numpy()
            entries: Iterable[tuple[tuple[Any, ...], Any]]
            if numerical.is_sparse(self.components):  # without the zeros
                found = self.components.tocoo()
                entries = zip(zip(found.row, found.col, strict=True), found.data, strict=True)
            else:
                data = self.to_numpy()
                entries = ((tuple(index), data[tuple(index)]) for index in np.argwhere(data != 0))
            return {
                tuple(names[axis][int(i)] for axis, i in enumerate(index)): value
                for index, value in entries
                if value != 0
            }
        if not self.indices:
            return {(): self.components} if self.components != 0 else {}
        result = {}
        for index in product(*(range(n) for n in self.shape)):
            value = self.components[index]
            if value != 0:
                result[tuple(names[axis][i] for axis, i in enumerate(index))] = value
        return result

    # -- components ------------------------------------------------------------

    def applyfunc(self, func: Callable[[Expr], Expr]) -> Tensor:
        """Apply ``func`` to every component."""
        return self._like(arrays.apply(self.components, func))

    def simplify(self, **kwargs: Any) -> Tensor:
        """Simplify every component with SymPy's ``simplify``."""
        return self.applyfunc(lambda c: sympy.simplify(c, **kwargs))

    def subs(self, *args: Any, **kwargs: Any) -> Tensor:
        """Substitute into every component, as SymPy's ``subs`` does: ``g.subs(M, 1)``."""
        return self.applyfunc(lambda c: c.subs(*args, **kwargs))

    def diff(self, *symbols: Any) -> Tensor:
        """Differentiate every component: ``g.diff(r)``."""
        return self.applyfunc(lambda c: sympy.diff(c, *symbols))

    def relabel(self, indices: Indices | Sequence[Index] | str) -> Tensor:
        """The same components under new index labels (positions must match)."""
        new = indices if isinstance(indices, Indices) else Indices(indices)
        if [i.upper for i in new] != [i.upper for i in self.indices]:
            raise IndexNotationError(f"cannot relabel {self.indices} as {new}: positions differ")
        return Tensor(self.components, new, self.coordinates, name=self.name, spaces=self.spaces)

    @property
    def T(self) -> Tensor:  # NumPy's name for the transpose
        """The transpose: the axes, and their indices, in the other order."""
        axes = list(range(self.rank))[::-1]
        return Tensor(
            arrays.permute(self.components, axes) if self.rank > 1 else self.components,
            Indices(self.indices[axis] for axis in axes),
            self.coordinates,
            name=self.name,
            spaces=[self.spaces[axis] for axis in axes] if self.spaces is not None else None,
            source=self.source,
        )

    def _like(self, components: Any, indices: Indices | None = None) -> Tensor:
        return Tensor(
            components,
            self.indices if indices is None else indices,
            self.coordinates,
            name=self.name,
            spaces=self.spaces,
            source=self.source,
        )

    def _derived(self, components: Any, source: str | None) -> Tensor:
        """A tensor computed from this one, named by what it is the value of."""
        return Tensor(components, self.indices, self.coordinates, spaces=self.spaces, source=source)

    def _written(self) -> str:
        """The tensor as written, for messages: ``g_{\\mu\\nu}``."""
        from relativisticpy import display

        if not self.indices:
            return "a scalar"
        return display.tensor_head(self, "text") or f"the tensor {self.indices}"

    # -- arithmetic --------------------------------------------------------------

    def _scalar(self, value: object) -> Any:
        """A scalar to combine with the components: a plain number with numerical ones,
        which would otherwise become an array of SymPy objects."""
        return numerical.number(value) if self.numerical else sympy.sympify(value)

    def __neg__(self) -> Tensor:
        return self._derived(-self.components, _expression("-", self))

    def __add__(self, other: object) -> Tensor:
        if not isinstance(other, Tensor):
            if self.rank == 0:
                return self._like(self.components + self._scalar(other))
            return NotImplemented
        result = self.components + _aligned(other, self.indices)
        return self._derived(result, _expression("+", self, other))

    def __radd__(self, other: object) -> Tensor:
        return self.__add__(other)

    def __sub__(self, other: object) -> Tensor:
        if not isinstance(other, Tensor):
            if self.rank == 0:
                return self._like(self.components - self._scalar(other))
            return NotImplemented
        result = self.components - _aligned(other, self.indices)
        return self._derived(result, _expression("-", self, other))

    def __rsub__(self, other: object) -> Tensor:
        return (-self).__add__(other)

    def __mul__(self, other: object) -> Tensor | Scalar:
        if not isinstance(other, Tensor):
            scaled = self.components * self._scalar(other)
            return self._derived(scaled, _expression("*", other, self))
        return contract(self, other)

    def __rmul__(self, other: object) -> Tensor:
        scaled = self._scalar(other) * self.components
        return self._derived(scaled, _expression("*", other, self))

    def __truediv__(self, other: object) -> Tensor:
        if isinstance(other, Tensor):
            return NotImplemented
        divided = self.components / self._scalar(other)
        return self._derived(divided, _expression("/", self, other))


def contract(*tensors: Tensor) -> Tensor | Scalar:
    """Multiply tensors, summing over each label that appears once up and once down.
    Each index has as many components as the axes it labels, and ranges over their space,
    as the result's free indices do: returns from 500 assets over 2,520 days times
    weights on the assets are a tensor over the days.

    Returns a :class:`Tensor`, or a SymPy expression when no index is left free.
    """
    occurrences: dict[str, list[Index]] = {}
    sizes: dict[str, int] = {}
    spaces: dict[str, Space] = {}
    for tensor in tensors:
        for axis, index in enumerate(tensor.indices):
            occurrences.setdefault(index.name, []).append(index)
            size = int(tensor.shape[axis])
            if sizes.setdefault(index.name, size) != size:
                raise IndexNotationError(
                    f"index {index.name!r} has {sizes[index.name]} components in one place "
                    f"and {size} in another"
                )
            if tensor.spaces is not None:
                space = tensor.spaces[axis]
                if spaces.setdefault(index.name, space) != space:
                    raise IndexNotationError(
                        f"index {index.name!r} ranges over {spaces[index.name]} in one place "
                        f"and {space} in another"
                    )
    free = []
    for name, found in occurrences.items():
        if len(found) > 2:
            raise IndexNotationError(f"index {name!r} appears {len(found)} times")
        if len(found) == 2 and found[0].upper == found[1].upper:
            position = "an upper" if found[0].upper else "a lower"
            raise IndexNotationError(
                f"index {name!r} appears twice as {position} index; "
                "a summed index must appear once up and once down"
            )
        if len(found) == 1:
            free.append(found[0])
    numbers = {name: number for number, name in enumerate(occurrences)}
    components = arrays.contract(
        [(t.components, [numbers[name] for name in t.indices.names]) for t in tensors],
        [numbers[i.name] for i in free],
        [sizes[name] for name in occurrences],
    )
    if not free:
        return components
    coordinates = next((t.coordinates for t in tensors if t.coordinates), ())
    known = all(i.name in spaces for i in free)
    return Tensor(
        components,
        Indices(free),
        coordinates,
        spaces=[spaces[i.name] for i in free] if known else None,
        source=_expression("*", *tensors),
    )


def _aligned(tensor: Tensor, indices: Indices) -> Any:
    """``tensor``'s components with axes reordered to match ``indices``."""
    if sorted(tensor.indices, key=str) != sorted(indices, key=str):
        raise IndexNotationError(
            f"the free indices of a sum must agree: {indices} and {tensor.indices}"
        )
    return arrays.permute(tensor.components, [tensor.indices.index(i) for i in indices])


def _indices(text: str | Iterable[Index]) -> Indices:
    """Indices from text, as a tensor's are written, or from :class:`Index` values."""
    return Indices(text)


def _expression(operation: str, *operands: object) -> str | None:
    """The text of an operation on tensors, ``\\eta_{ab} v^{b}``, so that its result is
    shown as what it is the value of; ``None`` where an operand has no name to show."""
    from relativisticpy import display

    shown = []
    for operand in operands:
        if isinstance(operand, Tensor):
            head = display.tensor_head(operand, "text")
            if head is None:
                return None
            if operand.source is not None and operation in ("*", "-") and " " in head:
                head = f"({head})"
            shown.append(head)
        else:
            shown.append(display.text(operand))
    if operation == "-" and len(shown) == 1:
        return f"-{shown[0]}"
    if operation == "*":
        if len(shown) == 2 and not isinstance(operands[0], Tensor):
            scalar = shown[0] if shown[0].replace(".", "").isalnum() else f"({shown[0]})"
            return f"{scalar} {shown[1]}"
        return " ".join(shown)
    if operation == "/":
        return f"{shown[0]}/{shown[1]}"
    return f" {operation} ".join(shown)


def _is_key(key: Any) -> bool:
    """Whether a key names one position: a number, a slice, or a name."""
    if isinstance(key, int | str | slice | Symbol):
        return True
    return hasattr(key, "__index__") and getattr(key, "shape", ()) == ()


def _label_name(label: Any) -> str:
    """A label as a plain name, so that ``'θ'``, ``'\\theta'`` and ``'theta'`` match."""
    from relativisticpy.language.vocabulary import GREEK_VARIANTS, UNICODE_LETTERS

    text = str(label).strip()
    if text.startswith("\\"):
        text = text[1:]
    text = GREEK_VARIANTS.get(text, text)
    return "".join(UNICODE_LETTERS.get(character, character) for character in text)


def _substituted(component: Any, values: dict[str, Any]) -> Any:
    """A component with the values given for its symbols, matched by name."""
    component = sympy.sympify(component)
    matches = {s: values[s.name] for s in component.free_symbols if s.name in values}
    return component.subs(matches) if matches else component


def _times(shape: Sequence[int]) -> str:
    return " × ".join(str(n) for n in shape)


def _misfit(shape: tuple[int, ...], rank: int) -> str:
    axes = "axis" if len(shape) == 1 else "axes"
    count = "index" if rank == 1 else "indices"
    verb = "is" if rank == 1 else "are"
    shown = _times(shape) or "()"
    return f"components of shape {shown} have {len(shape)} {axes}, but there {verb} {rank} {count}"
