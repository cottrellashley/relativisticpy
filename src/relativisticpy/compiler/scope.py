"""Compile-time state: scopes, bindings, and the operators that act on what follows them."""

from __future__ import annotations

from dataclasses import dataclass, field

from relativisticpy.ir import Node
from relativisticpy.language import syntax as ast
from relativisticpy.language.source import Span
from relativisticpy.tensors import Indices


@dataclass(frozen=True)
class VariableBinding:
    """A name for a value: ``x := 2``."""

    node: Node


@dataclass(frozen=True)
class TensorBinding:
    """A tensor defined in the program, ``T^{a}_{b} := ...``, with the index
    positions it was defined with."""

    node: Node
    indices: Indices


@dataclass(frozen=True)
class FunctionBinding:
    """A function defined in the program. Calls are compiled inline, in a child of
    the scope the function was defined in (lexical scoping)."""

    definition: ast.FunctionDefinition
    closure: Scope


Binding = VariableBinding | TensorBinding | FunctionBinding


class Scope:
    """Names bound at one level of nesting: the program, or a function body."""

    def __init__(self, parent: Scope | None = None) -> None:
        self.parent = parent
        self._bindings: dict[str, Binding] = {}

    def lookup(self, name: str) -> Binding | None:
        scope: Scope | None = self
        while scope is not None:
            if name in scope._bindings:
                return scope._bindings[name]
            scope = scope.parent
        return None

    def bind(self, name: str, binding: Binding) -> None:
        self._bindings[name] = binding

    def child(self) -> Scope:
        return Scope(self)


@dataclass(frozen=True)
class Context:
    """What the compiler knows about the statement being compiled."""

    latex: bool = False
    line_element: bool = False
    calls: tuple[str, ...] = ()  # user functions being inlined, to catch recursion


@dataclass(frozen=True)
class Output:
    """A value the program shows: the node to compute, where it was written and,
    for a tensor such as ``Ric_{ab}``, the name it was written with."""

    node: Node
    span: Span
    name: str | None = None


# -- operators: compile-time values that act on the factor after them ------------


@dataclass(frozen=True)
class Operator:
    """Something that acts on the factor written after it, as ``∂_μ`` does in
    ``∂_μ φ`` and ``sin`` in ``sin x``."""

    span: Span

    def describe(self) -> str:
        return self.span.text


@dataclass(frozen=True)
class IndexDerivative(Operator):
    """``∂_μ`` (``d_{μ}``) or ``∇_μ`` (``D_{μ}``), with one index item per derivative."""

    covariant: bool
    items: tuple[tuple[ast.IndexItem, bool], ...] = field(default=())  # (item, upper)


@dataclass(frozen=True)
class LeibnizDerivative(Operator):
    """``\\frac{d}{dx}`` or ``\\frac{\\partial^2}{\\partial x \\partial y}``."""

    variables: tuple[tuple[Node, int], ...] = ()


@dataclass(frozen=True)
class FunctionReference(Operator):
    """A function named without parentheses, as in plain-text ``sin x``."""

    name: str = ""


@dataclass(frozen=True)
class Dalembertian(Operator):
    """``□``, the wave operator :math:`\\nabla^\\mu \\nabla_\\mu`."""


@dataclass(frozen=True)
class LieDerivative(Operator):
    """``£_V``, the Lie derivative along the vector field ``V``."""

    vector: str = ""


@dataclass(frozen=True)
class DifferentialMarker(Operator):
    """The ``d`` of ``d\\theta`` in a line element."""
