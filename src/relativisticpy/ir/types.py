"""The static types of IR values.

The compiler knows the type of every value before anything is computed, which
is what lets it check index notation up front: the free indices of a sum must
agree, a summed index must appear once up and once down, and so on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from relativisticpy.tensors.indices import Indices
from relativisticpy.tensors.spaces import Space


class Type:
    """Base class of IR types."""


@dataclass(frozen=True)
class ScalarType(Type):
    """A symbolic scalar: a number or SymPy expression."""

    def __str__(self) -> str:
        return "Scalar"


@dataclass(frozen=True)
class TensorType(Type):
    """Components whose axes follow ``indices``, each ranging over its space in
    ``spaces``: spacetime for General Relativity, or assets, days, a qubit."""

    indices: Indices
    spaces: tuple[Space, ...]

    def __post_init__(self) -> None:
        if len(self.spaces) != len(self.indices):
            raise ValueError(f"{len(self.indices)} indices but {len(self.spaces)} spaces")

    @property
    def rank(self) -> int:
        return len(self.indices)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(space.dim for space in self.spaces)

    def space(self, name: str) -> Space:
        """The space the index called ``name`` ranges over."""
        return self.spaces[self.indices.names.index(name)]

    def with_indices(self, indices: Indices) -> TensorType:
        """The same spaces, with the indices renamed or moved."""
        return TensorType(indices, self.spaces)

    def __str__(self) -> str:
        return f"Tensor{self.indices}"


@dataclass(frozen=True)
class EquationType(Type):
    """An equation between two values of type ``of``."""

    of: Type

    def __str__(self) -> str:
        return f"Equation[{self.of}]"


@dataclass(frozen=True)
class ValueType(Type):
    """Any other value: a list, a string, the result of ``solve``."""

    def __str__(self) -> str:
        return "Value"


@dataclass(frozen=True)
class NotationAlgebra:
    """The tensors abstract index notation is written with: ``dim`` dimensions, the
    metric called ``metric``, and ``number``, which tells apart the algebras a session
    makes as its coordinates change, so that notation kept with one is read in the next."""

    dim: int
    metric: str
    number: int


@dataclass(frozen=True)
class NotationType(ValueType):
    """A value of abstract index notation, kept as notation rather than components, as
    :math:`G_{\\mu\\nu} = 8\\pi G T_{\\mu\\nu}` is before T has components: an
    ``expression``, an ``equation``, or a ``term`` that one is built from. ``sides`` are the
    free indices of the expression or term, or of each side of the equation, as notation's
    rules give them; ``algebra`` is the one it is written in."""

    role: Literal["term", "expression", "equation"]
    sides: tuple[Indices, ...]
    algebra: NotationAlgebra

    @property
    def free(self) -> Indices:
        """The free indices of an expression or a term, or of an equation's left side."""
        return self.sides[0]

    @property
    def equation(self) -> bool:
        return self.role == "equation"

    def __str__(self) -> str:
        shown = list(dict.fromkeys(f"Notation{side}" for side in self.sides))
        return f"Equation[{', '.join(shown)}]" if self.equation else shown[0]


SCALAR = ScalarType()
VALUE = ValueType()
