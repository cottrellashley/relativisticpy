"""IR operations.

An operation is pure data: what to compute, not how. Operations are frozen
and hashable, so the graph can recognise two requests for the same
computation. The executor's kernels (:mod:`relativisticpy.runtime.kernels`)
give each one its meaning.

The tensor operations form a small core. Raising and lowering indices,
covariant derivatives and traces all lower to :class:`Contract`,
:class:`Partial`, :class:`Permute` and :class:`Add`, so a backend needs to
implement only these.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sympy import Symbol


class Op:
    """Base class of operations."""

    def label(self) -> str:
        """How the operation prints in the IR listing."""
        return type(self).__name__.lower()


# -- leaves ------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Const(Op):
    """A value known when compiling. Constants are compared by identity."""

    value: Any

    def label(self) -> str:
        text = str(self.value)
        return f"const {text if len(text) <= 48 else text[:45] + '...'}"


# -- scalars and values -------------------------------------------------------


@dataclass(frozen=True)
class Call(Op):
    """Apply a Python function, usually a SymPy one, to the argument values."""

    func: Callable[..., Any] = field(compare=True)
    name: str = ""

    def label(self) -> str:
        return f"call {self.name or getattr(self.func, '__name__', '?')}"


@dataclass(frozen=True)
class Add(Op):
    """Sum the arguments: scalars, or tensors whose axes are already aligned."""

    signs: tuple[int, ...]

    def label(self) -> str:
        return "add " + " ".join("+" if sign > 0 else "-" for sign in self.signs)


@dataclass(frozen=True)
class Mul(Op):
    """Multiply values: scalars, and matrices as matrices."""


@dataclass(frozen=True)
class Scale(Op):
    """Multiply a tensor, the last argument, by scalars."""


@dataclass(frozen=True)
class Pow(Op):
    """Raise the first argument to the power of the second."""


# -- tensors ------------------------------------------------------------------


@dataclass(frozen=True)
class Contract(Op):
    """Einstein summation. ``labels`` holds one tuple of axis labels per argument and
    ``output`` the labels of the result's axes."""

    labels: tuple[tuple[int, ...], ...]
    output: tuple[int, ...]
    dim: int

    def label(self) -> str:
        letters = "abcdefghijklmnopqrstuvwxyz"
        operands = ",".join("".join(letters[i % 26] for i in group) for group in self.labels)
        return f"contract {operands}->{''.join(letters[i % 26] for i in self.output)}"


@dataclass(frozen=True)
class Permute(Op):
    """Reorder axes: axis ``i`` of the result is axis ``axes[i]`` of the argument."""

    axes: tuple[int, ...]

    def label(self) -> str:
        return f"permute {self.axes}"


@dataclass(frozen=True)
class Select(Op):
    """Fix axes to components: ``fixed`` pairs an axis with a component number."""

    fixed: tuple[tuple[int, int], ...]

    def label(self) -> str:
        return "select " + ", ".join(f"{axis}={value}" for axis, value in self.fixed)


@dataclass(frozen=True)
class Partial(Op):
    """Partial derivatives along each coordinate, as a new first axis."""

    coordinates: tuple[Symbol, ...]


@dataclass(frozen=True)
class Symmetrize(Op):
    """(Anti)symmetrise over ``axes``."""

    axes: tuple[int, ...]
    antisymmetric: bool

    def label(self) -> str:
        kind = "antisymmetrize" if self.antisymmetric else "symmetrize"
        return f"{kind} {self.axes}"


@dataclass(frozen=True)
class Map(Op):
    """Apply a function to every component of the first argument, passing the
    remaining argument values after the component. Run in parallel when the
    executor has workers."""

    func: Callable[..., Any]
    name: str = ""

    def label(self) -> str:
        return f"map {self.name or getattr(self.func, '__name__', '?')}"


@dataclass(frozen=True)
class MakeArray(Op):
    """Assemble an array of the given shape from scalar arguments, in row-major order."""

    shape: tuple[int, ...]

    def label(self) -> str:
        return f"array {self.shape}"


@dataclass(frozen=True)
class ToArray(Op):
    """Convert a value (a list, a matrix) to components of the given shape."""

    shape: tuple[int, ...]

    def label(self) -> str:
        return f"to-array {self.shape}"


# -- geometry -----------------------------------------------------------------


@dataclass(frozen=True)
class Geometry(Op):
    """A geometric quantity of the metric, computed by a dedicated kernel:
    ``metric``, ``inverse``, ``determinant``, ``christoffel``, ``riemann``,
    ``ricci``, ``ricci_scalar``, ``einstein``, ``weyl``, ``kretschmann`` or
    ``kronecker``."""

    quantity: str
    coordinates: tuple[Symbol, ...]

    def label(self) -> str:
        return self.quantity


@dataclass(frozen=True)
class LineElement(Op):
    """Read metric components off a line element written in the differentials."""

    coordinates: tuple[Symbol, ...]
    differentials: tuple[Symbol, ...]

    def label(self) -> str:
        return "line-element"


# -- equations ----------------------------------------------------------------


@dataclass(frozen=True)
class Equation(Op):
    """An equation between the two arguments: SymPy's ``Eq`` for scalars, a
    :class:`~relativisticpy.tensors.TensorEquation` for tensors."""
