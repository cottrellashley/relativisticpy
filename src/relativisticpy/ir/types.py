"""The static types of IR values.

The compiler knows the type of every value before anything is computed, which
is what lets it check index notation up front: the free indices of a sum must
agree, a summed index must appear once up and once down, and so on.
"""

from __future__ import annotations

from dataclasses import dataclass

from relativisticpy.tensors.indices import Indices


class Type:
    """Base class of IR types."""


@dataclass(frozen=True)
class ScalarType(Type):
    """A symbolic scalar: a number or SymPy expression."""

    def __str__(self) -> str:
        return "Scalar"


@dataclass(frozen=True)
class TensorType(Type):
    """Components of shape ``(dim,) * len(indices)`` whose axes follow ``indices``."""

    indices: Indices
    dim: int

    @property
    def rank(self) -> int:
        return len(self.indices)

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


SCALAR = ScalarType()
VALUE = ValueType()
