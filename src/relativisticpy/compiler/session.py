"""The state a workbook keeps between calls: coordinates, the metric, declared
symbols and the names of the built-in tensors."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import sympy
from sympy import Symbol

from relativisticpy.compiler.scope import Binding, Scope
from relativisticpy.errors import CompileError
from relativisticpy.ir import SCALAR, Graph, Node, NotationAlgebra, TensorType, ops
from relativisticpy.language.source import Span
from relativisticpy.tensors import Index, Indices
from relativisticpy.tensors.spaces import Space, spacetime

if TYPE_CHECKING:
    from relativisticpy.integrations._objects import ObjectReader

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

#: The positions of the indices of each geometric tensor, upper or not, in order.
GEOMETRY_POSITIONS: dict[str, tuple[bool, ...]] = {
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


@dataclass(frozen=True)
class Saved:
    """What a statement may change in a session, as it was before the statement, so that
    the statement can be undone where a check of what it defines fails: the names bound,
    the coordinates and the metric, the functions of the coordinates, and the names
    written as factors."""

    bindings: dict[str, Binding]
    coordinates: tuple[Symbol, ...] | None
    coordinates_inferred: bool
    metric: MetricState | None
    functions: dict[str, tuple[Symbol, ...]]
    dropped: Span | None
    factors: frozenset[str]


@dataclass
class Session:
    """Everything the compiler remembers from one statement to the next. It computes
    nothing: values are computed by an executor, from the graph, as a program runs."""

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
    # Where the coordinates last changed while a metric was in force, which dropped it.
    dropped: Span | None = None
    # The names written as factors so far, as \rho is in p = w\rho: a symbol, so that
    # \rho(1 + 3w) later is a product, not a function.
    factors: set[str] = field(default_factory=set)
    # The equations stated so far between names nothing defines, as x^3 + y^3 = 6xy, by
    # the names each relates and where it is written: they tie y to x, so that a later
    # dy/dx is not the derivative of a constant.
    relations: list[tuple[frozenset[str], Span]] = field(default_factory=list)
    # Spaces other than spacetime, by name, and the index labels declared to range over
    # them, as by a, b \in \mathcal{A}. A space may be given a metric, by the node of
    # the tensor that is its metric, and a measure, by the node of its weights.
    spaces: dict[str, Space] = field(default_factory=dict)
    labels: dict[str, Space] = field(default_factory=dict)
    space_metrics: dict[Space, Node] = field(default_factory=dict)
    measures: dict[Space, Node] = field(default_factory=dict)
    # The tensors of one's own that notation in each algebra is written with, by name and
    # number of indices: what its components may need.
    notation_tensors: dict[NotationAlgebra, set[tuple[str, int]]] = field(default_factory=dict)
    # The conventions a program states, by what they are of: a Fourier transform defined
    # by its integral, \mathcal{F}\{f\}(\omega) := \int ... e^{-i\omega t} dt.
    conventions: dict[str, Any] = field(default_factory=dict)
    # What reads the Python objects a program names, as \texttt{prices}: the scope the
    # workbook was given, or none (see relativisticpy.integrations._objects).
    objects: ObjectReader | None = field(default=None, repr=False)
    _algebra: NotationAlgebra | None = field(default=None, repr=False)
    _algebras: int = field(default=0, repr=False)

    def algebra(self) -> NotationAlgebra:
        """The tensors of abstract index notation, in as many dimensions as there are
        coordinates (four before there are any), with the metric's name in force: a new
        algebra whenever either changes, and in a fork. Notation kept with an earlier
        algebra is read in this one (see :func:`~relativisticpy.abstract.rebased`)."""
        dim = len(self.coordinates) if self.coordinates else 4
        metric = self.names["metric"]
        algebra = self._algebra
        if algebra is None or algebra.dim != dim or algebra.metric != metric:
            self._algebras += 1
            algebra = self._algebra = NotationAlgebra(dim, metric, self._algebras)
        return algebra

    def fork(self) -> Session:
        """A session in the same state whose changes leave this one as it is: to compile a
        program in order to look at it, without running it. It shares the graph, which only
        ever gains nodes."""
        return replace(
            self,
            globals=self.globals.copy(),
            assumptions=dict(self.assumptions),
            names=dict(self.names),
            functions=dict(self.functions),
            factors=set(self.factors),
            relations=list(self.relations),
            spaces=dict(self.spaces),
            labels=dict(self.labels),
            space_metrics=dict(self.space_metrics),
            measures=dict(self.measures),
            notation_tensors={key: set(value) for key, value in self.notation_tensors.items()},
            conventions=dict(self.conventions),
            _algebra=None,
        )

    def save(self) -> Saved:
        """What a statement may change, as it is now, to put back with :meth:`restore`."""
        return Saved(
            self.globals.saved(),
            self.coordinates,
            self.coordinates_inferred,
            self.metric,
            dict(self.functions),
            self.dropped,
            frozenset(self.factors),
        )

    def restore(self, saved: Saved) -> None:
        """Undo a statement: put back what :meth:`save` gave before it."""
        self.globals.restore(saved.bindings)
        self.coordinates = saved.coordinates
        self.coordinates_inferred = saved.coordinates_inferred
        self.metric = saved.metric
        self.functions.clear()
        self.functions.update(saved.functions)
        self.dropped = saved.dropped
        self.factors.clear()
        self.factors.update(saved.factors)

    def spacetime(self, span: Span, why: str = "index notation") -> Space:
        """The space of the coordinates in force."""
        return spacetime(self.require_coordinates(span, why))

    def set_coordinates(
        self, coordinates: tuple[Symbol, ...], *, inferred: bool = False, span: Span | None = None
    ) -> None:
        """Change coordinates, forgetting what belonged to the old ones; ``span`` is where
        they changed, to say so of the metric it drops."""
        if coordinates != self.coordinates:
            if self.metric is not None:
                self.dropped = span
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
                hint="declare them first, for example Coordinates := [t, r, \\theta, \\phi]",
            )
        return self.coordinates

    def require_metric(self, span: Span, why: str) -> MetricState:
        self.require_coordinates(span, why)
        if self.metric is None:
            raise CompileError(
                "there is no metric yet" if why == "the metric" else f"{why} needs a metric",
                span=span,
                hint="define one first, for example g_{\\mu\\nu} := diag(-1, 1, 1, 1)",
                notes=self.dropped_note(),
            )
        return self.metric

    def dropped_note(self) -> list[str]:
        """The note that the metric was dropped when the coordinates changed, if it was."""
        if self.dropped is None:
            return []
        return [f"the metric was dropped when the coordinates changed on line {self.dropped.line}"]

    def geometry(self, quantity: str, span: Span) -> Node:
        """The node computing a geometric quantity of the metric in force.

        Nodes are interned, so every request for, say, the Christoffel symbols
        of one metric is the same node, computed once.
        """
        if quantity == "kronecker":  # needs coordinates, not a metric
            coordinates = self.require_coordinates(span, describe(quantity))
            indices = Indices([Index("a", True), Index("b")])
            kronecker = ops.Geometry("kronecker", coordinates)
            kind = TensorType(indices, (spacetime(coordinates),) * 2)
            return self.graph.add(kronecker, (), kind, span)
        metric = self.require_metric(span, describe(quantity)).node
        assert self.coordinates is not None
        patch = self.coordinates
        if quantity == "weyl" and len(patch) < 3:  # said where it is asked for
            where = f"defined on line {metric.span.line}" if metric.span is not None else "in force"
            raise CompileError(
                "the Weyl tensor is defined in three dimensions or more",
                span=span,
                notes=[f"the metric {where} is {len(patch)}-dimensional"],
            )
        space = spacetime(patch)

        def build(name: str, *args: Node) -> Node:
            if name in GEOMETRY_POSITIONS:
                positions = GEOMETRY_POSITIONS[name]
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


def describe(quantity: str) -> str:
    """What a geometric quantity is called: ``ricci`` is the Ricci tensor."""
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
    }.get(quantity, quantity)
