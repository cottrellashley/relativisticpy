"""The state a workbook keeps between calls: coordinates, the metric, declared
symbols and the names of the built-in tensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sympy
from sympy import Symbol

from relativisticpy.abstract import Algebra
from relativisticpy.compiler.scope import Scope
from relativisticpy.errors import CompileError
from relativisticpy.ir import SCALAR, Graph, Node, TensorType, ops
from relativisticpy.language.source import Span
from relativisticpy.runtime import Executor
from relativisticpy.tensors import Index, Indices
from relativisticpy.tensors.spaces import Space, spacetime

#: The names of the built-in tensors and derivatives, by role, and the
#: statement that renames each: ``MetricSymbol := G`` makes the metric ``G``.
DEFAULT_NAMES = {
    "metric": "g",
    "connection": "C",
    "riemann": "R",
    "ricci": "Ric",
    "einstein": "G",
    "weyl": "W",
    "partial": "d",
    "covariant": "D",
}
RENAMING_STATEMENTS = {
    "MetricSymbol": "metric",
    "ConnectionSymbol": "connection",
    "RiemannSymbol": "riemann",
    "RicciSymbol": "ricci",
    "EinsteinSymbol": "einstein",
    "EinsteinTensorSymbol": "einstein",
    "WeylSymbol": "weyl",
    "DerivativeSymbol": "partial",
    "CovariantDerivativeSymbol": "covariant",
}
#: Statements that declare symbols with assumptions: ``Reals := [x, y]``.
ASSUMPTION_STATEMENTS = {
    "Constants": {"constant": True},
    "Reals": {"real": True},
    "Integers": {"integer": True},
    "Complex": {"complex": True},
    "Positive": {"positive": True},
}

_GEOMETRY_TYPES = {
    "inverse": (True, True),
    "christoffel": (True, False, False),
    "riemann": (True, False, False, False),
    "ricci": (False, False),
    "einstein": (False, False),
    "riemann_lower": (False, False, False, False),
    "weyl": (False, False, False, False),
    "kronecker": (True, False),
}


@dataclass
class MetricState:
    """The metric in force: its node, and the names its definition mentions
    (a metric written in terms of ``R`` keeps ``R`` an ordinary symbol)."""

    node: Node
    names: frozenset[str]


@dataclass
class Session:
    """Everything the compiler remembers from one statement to the next."""

    executor: Executor = field(default_factory=Executor)
    graph: Graph = field(default_factory=Graph)
    globals: Scope = field(default_factory=Scope)
    coordinates: tuple[Symbol, ...] | None = None
    metric: MetricState | None = None
    assumptions: dict[str, Symbol] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_NAMES))
    # Functions of the coordinates, written once with their arguments as in \alpha(t, r):
    # as in physics texts, a bare \alpha then means \alpha(t, r).
    functions: dict[str, tuple[Symbol, ...]] = field(default_factory=dict)
    # Whether the coordinates were read from a line element's differentials rather than
    # declared; only then may a later line element bring new ones.
    coordinates_inferred: bool = False
    # Spaces other than spacetime, by name, and the index labels declared to range over
    # them, as by a, b \in \mathcal{A}. A space may be given a metric, by the node of
    # the tensor that is its metric, and a measure, by the node of its weights.
    spaces: dict[str, Space] = field(default_factory=dict)
    labels: dict[str, Space] = field(default_factory=dict)
    space_metrics: dict[Space, Node] = field(default_factory=dict)
    measures: dict[Space, Node] = field(default_factory=dict)
    _algebra: Algebra | None = field(default=None, repr=False)

    def algebra(self) -> Algebra:
        """The tensors of abstract index notation, in as many dimensions as there are
        coordinates (four before there are any), with the metric's name in force.
        Notation kept with an earlier algebra is read in this one (see
        :func:`~relativisticpy.abstract.rebased`)."""
        dim = len(self.coordinates) if self.coordinates else 4
        metric = self.names["metric"]
        algebra = self._algebra
        if algebra is None or algebra.dim != dim or algebra.metric.name != metric:
            algebra = self._algebra = Algebra(dim, metric=metric)
        return algebra

    def spacetime(self, span: Span, why: str = "index notation") -> Space:
        """The space of the coordinates in force."""
        return spacetime(self.require_coordinates(span, why))

    def set_coordinates(self, coordinates: tuple[Symbol, ...], *, inferred: bool = False) -> None:
        """Change coordinates, forgetting what belonged to the old ones."""
        if coordinates != self.coordinates:
            self.metric = None
            self.functions.clear()
        self.coordinates = coordinates
        self.coordinates_inferred = inferred

    def symbol(self, name: str) -> Symbol:
        """The symbol called ``name``, with any assumptions declared for it."""
        return self.assumptions.get(name) or sympy.Symbol(name)

    def require_coordinates(self, span: Span, why: str = "index notation") -> tuple[Symbol, ...]:
        if self.coordinates is None:
            raise CompileError(
                f"{why} needs coordinates",
                span=span,
                hint="declare them first, for example  Coordinates := [t, r, \\theta, \\phi]",
            )
        return self.coordinates

    def require_metric(self, span: Span, why: str) -> MetricState:
        self.require_coordinates(span, why)
        if self.metric is None:
            raise CompileError(
                "there is no metric yet" if why == "the metric" else f"{why} needs a metric",
                span=span,
                hint="define one first, for example  g_{\\mu\\nu} := diag(-1, 1, 1, 1)",
            )
        return self.metric

    def geometry(self, quantity: str, span: Span) -> Node:
        """The node computing a geometric quantity of the metric in force.

        Nodes are interned, so every request for, say, the Christoffel symbols
        of one metric is the same node, computed once.
        """
        if quantity == "kronecker":  # needs coordinates, not a metric
            coordinates = self.require_coordinates(span, _describe(quantity))
            indices = Indices([Index("a", True), Index("b")])
            kronecker = ops.Geometry("kronecker", coordinates)
            kind = TensorType(indices, (spacetime(coordinates),) * 2)
            return self.graph.add(kronecker, (), kind, span)
        metric = self.require_metric(span, _describe(quantity)).node
        assert self.coordinates is not None
        patch = self.coordinates
        space = spacetime(patch)

        def build(name: str, *args: Node) -> Node:
            if name in _GEOMETRY_TYPES:
                positions = _GEOMETRY_TYPES[name]
                labels = "abcd"
                indices = Indices(Index(labels[i], up) for i, up in enumerate(positions))
                kind: Any = TensorType(indices, (space,) * len(indices))
            else:
                kind = SCALAR
            return self.graph.add(ops.Geometry(name, patch), args, kind, metric.span)

        def node(name: str) -> Node:
            if name == "metric":
                return metric
            dependencies = {
                "inverse": ("metric",),
                "determinant": ("metric",),
                "christoffel": ("metric", "inverse"),
                "riemann": ("christoffel",),
                "ricci": ("christoffel",),
                "ricci_scalar": ("ricci", "inverse"),
                "einstein": ("ricci", "ricci_scalar", "metric"),
                "riemann_lower": ("riemann", "metric"),
                "weyl": ("riemann_lower", "ricci", "ricci_scalar", "metric"),
                "kretschmann": ("riemann", "metric", "inverse"),
            }[name]
            return build(name, *(node(dependency) for dependency in dependencies))

        return node(quantity)

    def evaluate(self, node: Node) -> Any:
        """Compute a value now, while compiling (used for spacetimes and constants)."""
        return self.executor.evaluate(node)


def _describe(quantity: str) -> str:
    return {
        "metric": "the metric",
        "inverse": "raising an index",
        "determinant": "the metric determinant",
        "christoffel": "the Christoffel connection",
        "riemann": "the Riemann tensor",
        "ricci": "the Ricci tensor",
        "ricci_scalar": "the Ricci scalar",
        "einstein": "the Einstein tensor",
        "riemann_lower": "the Riemann tensor",
        "weyl": "the Weyl tensor",
        "kretschmann": "the Kretschmann scalar",
        "kronecker": "the Kronecker delta",
    }[quantity]
