"""Compile-time state: scopes, bindings, and the operators that act on what follows them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

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

    def bindings(self) -> list[tuple[str, Binding]]:
        """The names bound at this level, with what they are bound to, in the order they
        were first bound."""
        return list(self._bindings.items())

    def saved(self) -> dict[str, Binding]:
        """The bindings at this level, to put back with :meth:`restore`."""
        return dict(self._bindings)

    def restore(self, saved: dict[str, Binding]) -> None:
        """Bind at this level what :meth:`saved` gave, and nothing else."""
        self._bindings = dict(saved)

    def child(self) -> Scope:
        return Scope(self)

    def copy(self) -> Scope:
        """A scope with the same bindings and parent, whose bindings change apart."""
        scope = Scope(self.parent)
        scope._bindings = dict(self._bindings)
        return scope


@dataclass(frozen=True)
class Context:
    """What the compiler knows about the statement being compiled: whether it is LaTeX or a
    line element, the user functions being inlined, to catch recursion, and the definition
    whose ``body`` is being compiled, the innermost of them, which is read as it was where
    it is written."""

    latex: bool = False
    line_element: bool = False
    calls: tuple[str, ...] = ()
    body: ast.FunctionDefinition | None = None


@dataclass(frozen=True)
class Output:
    """A value the program shows: the node to compute, where it was written and,
    for a tensor such as ``Ric_{ab}``, the name it was written with. A ``note`` says
    how the statement was read where that could surprise, as when ``K = ...`` compares
    a name that already has a value rather than giving it a new one."""

    node: Node
    span: Span
    name: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class Check:
    """What a definition needs of a value it makes, as a metric must be one, computed as
    the statement runs rather than while it is compiled, so that it fails where the
    definition is written: ``node``, which must be computed. Notation is checked so too,
    each operation of it in the order it is written. Where it cannot be,
    ``diagnosis``, if there is one, gives the error that says why in the terms of what
    was written, or ``None``, leaving the error of ``node``. To an evaluation error,
    ``context`` adds notes and ``hint`` a hint; ``notes`` say how the statement was read,
    up to the definition."""

    node: Node
    span: Span | None
    diagnosis: Node | None = None
    notes: tuple[str, ...] = ()
    context: tuple[str, ...] = ()
    hint: str | None = None


# -- operators: compile-time values that act on what is written after them --------


@dataclass(frozen=True)
class Operator:
    """Something that acts on what is written after it, as far as its ``reach``: the
    factor after it, as ``∂_μ`` does in ``∂_μ φ ∂^μ φ`` and ``sin`` in ``sin x``, or the
    rest of the term, as ``\\frac{d}{dx}`` does in ``\\frac{d}{dx} x^2 y^3``."""

    span: Span

    #: What it acts on: the ``factor`` after it, or the rest of the ``term``.
    reach: ClassVar[Literal["factor", "term"]] = "factor"

    def describe(self) -> str:
        return self.span.text


@dataclass(frozen=True)
class IndexDerivative(Operator):
    """``∂_μ`` (``d_{μ}``) or ``∇_μ`` (``D_{μ}``), with one index item per derivative."""

    covariant: bool
    items: tuple[tuple[ast.IndexItem, bool], ...] = field(default=())  # (item, upper)


@dataclass(frozen=True)
class LeibnizDerivative(Operator):
    """``\\frac{d}{dx}`` or ``\\frac{\\partial^2}{\\partial x \\partial y}``, which acts on the
    rest of the term; ``partial`` where it is written with ∂, and so holds the other names
    constant."""

    variables: tuple[tuple[Node, int], ...] = ()
    partial: bool = False
    reach: ClassVar[Literal["factor", "term"]] = "term"


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
