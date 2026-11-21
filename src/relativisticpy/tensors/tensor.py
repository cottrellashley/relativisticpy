"""The :class:`Tensor` value: components, named indices and the coordinates they refer to."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from itertools import product
from typing import Any

import sympy
from sympy import Expr, Symbol

from relativisticpy.errors import IndexNotationError
from relativisticpy.tensors import arrays, numerical
from relativisticpy.tensors.indices import Index, Indices, latex_label
from relativisticpy.tensors.spaces import Space


class Tensor:
    """A tensor in a coordinate basis.

    ``components`` is a SymPy array whose axes follow ``indices``; ``coordinates``
    are the symbols the components are expressed in. Arithmetic follows index
    notation: ``+`` aligns operands by index name and ``*`` sums over every
    index that appears once up and once down::

        >>> from sympy import symbols
        >>> t, x = symbols("t x")
        >>> eta = Tensor([[-1, 0], [0, 1]], "_{a b}", (t, x), name="eta")
        >>> v = Tensor([2, 3], "^{b}", (t, x), name="v")
        >>> eta * v          # η_{ab} v^b
        Tensor('_{a}', [-2, 3])

    ``str`` gives the components, as RelativisticPy has always printed them.

    Components may also be numerical, a NumPy array, with each axis ranging over one of
    ``spaces``: assets, days, a qubit. :meth:`to_numpy` and :meth:`to_pandas` give them
    back, labelled by the bases of their spaces.
    """

    __slots__ = ("components", "coordinates", "indices", "name", "spaces")

    def __init__(
        self,
        components: Any,
        indices: Indices | Sequence[Index] | str,
        coordinates: Sequence[Symbol] = (),
        *,
        name: str | None = None,
        spaces: Sequence[Space] | None = None,
    ) -> None:
        self.indices = indices if isinstance(indices, Indices) else _indices(indices)
        self.coordinates = tuple(coordinates)
        self.name = name
        self.spaces = tuple(spaces) if spaces is not None else None
        if numerical.is_numerical(components):
            self.components = components
            if len(getattr(components, "shape", ())) != len(self.indices):
                raise ValueError(
                    f"components of shape {components.shape} do not fit {len(self.indices)} indices"
                )
            return
        self.components = arrays.to_array(components) if self.indices else sympy.sympify(components)
        if self.indices and self.spaces is not None:
            if tuple(self.components.shape) != tuple(space.dim for space in self.spaces):
                raise ValueError(f"components of shape {self.components.shape} do not fit")
        elif self.indices:
            shape = self.components.shape
            if len(shape) != len(self.indices) or len(set(shape)) != 1:
                raise ValueError(
                    f"components of shape {shape} do not fit {len(self.indices)} indices"
                )
            if self.coordinates and shape[0] != len(self.coordinates):
                raise ValueError(
                    f"components have dimension {shape[0]} but there are "
                    f"{len(self.coordinates)} coordinates"
                )

    # -- shape -----------------------------------------------------------------

    @property
    def rank(self) -> int:
        return len(self.indices)

    @property
    def dim(self) -> int:
        if self.indices:
            return int(self.components.shape[0])
        return len(self.coordinates)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.components.shape) if self.indices else ()

    @property
    def numerical(self) -> bool:
        """Whether the components are numbers in an array, rather than SymPy's."""
        return numerical.is_numerical(self.components)

    def to_numpy(self) -> Any:
        """The components as a NumPy array."""
        return numerical.numpy().asarray(numerical.dense(numerical.as_array(self.components)))

    def to_pandas(self) -> Any:
        """The components as a pandas Series (one index) or DataFrame (two), labelled by
        the bases of their spaces."""
        import pandas as pd

        data = self.to_numpy()
        names = [self._basis(axis) for axis in range(self.rank)]
        if self.rank == 1:
            return pd.Series(data, index=names[0], name=self.name)
        if self.rank == 2:
            return pd.DataFrame(data, index=names[0], columns=names[1])
        raise ValueError("only a tensor with one or two indices is a Series or a DataFrame")

    def _basis(self, axis: int) -> list[str]:
        """The names of the components along an axis."""
        space = self.spaces[axis] if self.spaces is not None else None
        if space is not None and space.basis is not None:
            return list(space.basis)
        if self.coordinates and (space is None or space.spacetime):
            return [str(c) for c in self.coordinates]
        return [str(i) for i in range(self.shape[axis])]

    def __getitem__(self, key: Any) -> Any:
        return self.components[key]

    def __iter__(self) -> Any:
        return iter(self.components)

    def __len__(self) -> int:
        return len(self.components) if self.indices else 1

    # -- comparison and display ------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tensor):
            return NotImplemented
        if self.numerical or other.numerical:
            return self.indices == other.indices and numerical.close(
                self.components, other.components
            )
        return self.indices == other.indices and self.components == other.components

    def __hash__(self) -> int:
        if self.numerical:
            return hash((self.indices, id(self.components)))
        return hash((self.indices, self.components))

    def __str__(self) -> str:
        return str(self.components)

    def __repr__(self) -> str:
        return f"Tensor({str(self.indices)!r}, {self.components})"

    def _repr_latex_(self) -> str:
        return f"$\\displaystyle {self.latex()}$"

    def latex(self) -> str:
        """Typeset the tensor: a matrix for rank one and two, its nonzero components above."""
        head = latex_label(self.name) if self.name else "T"
        if self.rank == 0:
            return str(sympy.latex(self.components))
        if self.numerical:
            if max(self.shape) > 8 or self.rank > 2:
                sizes = " \\times ".join(str(n) for n in self.shape)
                field = (
                    "\\mathbb{C}"
                    if numerical.numpy().iscomplexobj(self.components)
                    else "\\mathbb{R}"
                )
                return f"{head}{self.indices.latex()} \\in {field}^{{{sizes}}}"
            data = self.to_numpy().tolist()
            matrix = sympy.Matrix(data if self.rank == 2 else [data]).applyfunc(
                lambda value: sympy.N(sympy.sympify(value), 6)  # six significant figures
            )
            return f"{head}{self.indices.latex()} = {sympy.latex(matrix)}"
        if self.rank <= 2:
            matrix = sympy.Matrix(self.components.tolist())
            if self.rank == 1:
                matrix = matrix.T
            return f"{head}{self.indices.latex()} = {sympy.latex(matrix)}"
        nonzero = self.nonzero()
        if not nonzero:
            return f"{head}{self.indices.latex()} = 0"
        lines = [
            f"{head}{self.indices.latex([latex_label(n) for n in labels])} &= {sympy.latex(value)}"
            for labels, value in nonzero.items()
        ]
        return "\\begin{aligned}" + " \\\\ ".join(lines) + "\\end{aligned}"

    def nonzero(self) -> dict[tuple[str, ...], Expr]:
        """The nonzero components, keyed by coordinate names: ``{('t', 't', 'r'): ...}``."""
        if self.numerical:
            np = numerical.numpy()
            names = [self._basis(axis) for axis in range(self.rank)]
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
        labels = [str(c) for c in self.coordinates] or [str(i) for i in range(self.dim)]
        result = {}
        for index in product(range(self.dim), repeat=self.rank):
            value = self.components[index]
            if value != 0:
                result[tuple(labels[i] for i in index)] = value
        return result

    # -- components ------------------------------------------------------------

    def applyfunc(self, func: Callable[[Expr], Expr]) -> Tensor:
        """Apply ``func`` to every component."""
        return self._like(arrays.apply(self.components, func))

    def simplify(self, **kwargs: Any) -> Tensor:
        return self.applyfunc(lambda c: sympy.simplify(c, **kwargs))

    def subs(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.applyfunc(lambda c: c.subs(*args, **kwargs))

    def relabel(self, indices: Indices | Sequence[Index] | str) -> Tensor:
        """The same components under new index labels (positions must match)."""
        new = indices if isinstance(indices, Indices) else _indices(indices)
        if [i.upper for i in new] != [i.upper for i in self.indices]:
            raise IndexNotationError(f"cannot relabel {self.indices} as {new}: positions differ")
        return Tensor(self.components, new, self.coordinates, name=self.name, spaces=self.spaces)

    def _like(self, components: Any, indices: Indices | None = None) -> Tensor:
        return Tensor(
            components,
            self.indices if indices is None else indices,
            self.coordinates,
            name=self.name,
            spaces=self.spaces,
        )

    # -- arithmetic --------------------------------------------------------------

    def __neg__(self) -> Tensor:
        return self._like(-self.components)

    def __add__(self, other: object) -> Tensor:
        if not isinstance(other, Tensor):
            if self.rank == 0:
                return self._like(self.components + sympy.sympify(other))
            return NotImplemented
        return self._like(self.components + _aligned(other, self.indices))

    __radd__ = __add__

    def __sub__(self, other: object) -> Tensor:
        if not isinstance(other, Tensor):
            if self.rank == 0:
                return self._like(self.components - sympy.sympify(other))
            return NotImplemented
        return self._like(self.components - _aligned(other, self.indices))

    def __rsub__(self, other: object) -> Tensor:
        return (-self).__add__(other)

    def __mul__(self, other: object) -> Any:
        if not isinstance(other, Tensor):
            return self._like(self.components * sympy.sympify(other))
        return contract(self, other)

    def __rmul__(self, other: object) -> Tensor:
        return self._like(sympy.sympify(other) * self.components)

    def __truediv__(self, other: object) -> Tensor:
        if isinstance(other, Tensor):
            return NotImplemented
        return self._like(self.components / sympy.sympify(other))


def contract(*tensors: Tensor) -> Any:
    """Multiply tensors, summing over each label that appears once up and once down.

    Returns a :class:`Tensor`, or a SymPy expression when no index is left free.
    """
    occurrences: dict[str, list[Index]] = {}
    for tensor in tensors:
        for index in tensor.indices:
            occurrences.setdefault(index.name, []).append(index)
    free = []
    for name, found in occurrences.items():
        if len(found) > 2:
            raise IndexNotationError(f"index {name!r} appears {len(found)} times")
        if len(found) == 2 and found[0].upper == found[1].upper:
            position = "upper" if found[0].upper else "lower"
            raise IndexNotationError(
                f"index {name!r} appears twice as an {position} index; "
                "a summed index must appear once up and once down"
            )
        if len(found) == 1:
            free.append(found[0])
    dims = {t.dim for t in tensors if t.rank}
    if len(dims) > 1:
        raise ValueError(f"cannot contract tensors of different dimensions {sorted(dims)}")
    dim = dims.pop() if dims else 0
    components = arrays.contract(
        [(t.components, t.indices.names) for t in tensors], [i.name for i in free], dim
    )
    if not free:
        return components
    coordinates = next((t.coordinates for t in tensors if t.coordinates), ())
    return Tensor(components, Indices(free), coordinates)


def _aligned(tensor: Tensor, indices: Indices) -> Any:
    """``tensor``'s components with axes reordered to match ``indices``."""
    if sorted(tensor.indices, key=str) != sorted(indices, key=str):
        raise IndexNotationError(
            f"the free indices of a sum must agree: {indices} and {tensor.indices}"
        )
    return arrays.permute(tensor.components, [tensor.indices.index(i) for i in indices])


def _indices(text: str | Iterable[Index]) -> Indices:
    if not isinstance(text, str):
        return Indices(text)
    from relativisticpy.errors import ParseError
    from relativisticpy.language.parser import parse_expression
    from relativisticpy.language.syntax import Scripted

    try:
        node = parse_expression("T " + text)  # the space keeps "_a" from joining the name
    except ParseError as error:
        raise ValueError(f"cannot read indices from {text!r}") from error
    if text and not isinstance(node, Scripted):
        raise ValueError(f"cannot read indices from {text!r}")
    result = []
    for script in node.scripts if isinstance(node, Scripted) else ():
        if script.indices is None:
            raise ValueError(f"cannot read indices from {text!r}")
        for item in script.indices.items:
            if item.label is None or item.value is not None:
                raise ValueError(f"index labels only, please: {text!r}")
            result.append(Index(item.label, script.kind == "^"))
    return Indices(result)
