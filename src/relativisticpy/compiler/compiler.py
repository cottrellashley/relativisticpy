r"""The compiler: from syntax tree to IR.

This is where notation acquires meaning. The parser keeps every reading of
ambiguous notation; the compiler, knowing what each name refers to, chooses:

* ``x^{a}`` is a component of ``x`` if ``x`` is a tensor and a power otherwise;
  ``r_{s}`` is the symbol ``r_s`` unless ``r`` is a tensor.
* ``R_{\mu\nu}`` is the Ricci tensor and ``R^{a}{}_{bcd}`` the Riemann tensor:
  built-in tensors are recognised by name *and* number of indices.
* ``y = 2`` assigns ``y``; ``y(x) = 2`` is an equation; ``ds^2 = ...`` defines
  the metric.
* In LaTeX, an unknown run of letters is a product: ``\frac{2GM}{r}`` means
  :math:`2GM/r`. In plain text, ``GM`` is one name.
* ``G_{\mu\nu} = 8\pi G T_{\mu\nu}``, before ``T`` has components, is kept as
  index notation (:mod:`relativisticpy.abstract`), to be solved as notation and
  turned into components where it meets them.

It also checks index notation before anything is computed: the free indices of
a sum must agree, an index may appear at most twice, and a summed index must
appear once up and once down. Tensor expressions are lowered to a small set of
operations; raising an index, a covariant derivative or a trace become
contractions with the metric or the connection.
"""

from __future__ import annotations

import contextlib
import itertools
import math
import operator
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import reduce
from typing import Any, Literal

import sympy
from sympy import Symbol
from sympy.tensor import tensor as symbolic

from relativisticpy import abstract
from relativisticpy.abstract import IndexEquation, IndexExpression
from relativisticpy.compiler.builtins import BUILTINS, CONSTANTS, Builtin
from relativisticpy.compiler.scope import (
    Context,
    Dalembertian,
    DifferentialMarker,
    FunctionBinding,
    FunctionReference,
    IndexDerivative,
    LeibnizDerivative,
    LieDerivative,
    Operator,
    Output,
    Scope,
    TensorBinding,
    VariableBinding,
)
from relativisticpy.compiler.session import (
    ASSUMPTION_STATEMENTS,
    RENAMING_STATEMENTS,
    MetricState,
    Session,
)
from relativisticpy.diffgeom import Metric, differential, geodesic_equations
from relativisticpy.errors import CompileError, IndexNotationError
from relativisticpy.ir import (
    SCALAR,
    VALUE,
    EquationType,
    Node,
    TensorType,
    Type,
    dependencies,
    ops,
)
from relativisticpy.language import syntax as ast
from relativisticpy.language.parser import split_index_labels
from relativisticpy.language.source import Span
from relativisticpy.language.vocabulary import GREEK
from relativisticpy.tensors import Index, Indices, Tensor, linalg
from relativisticpy.tensors.equation import TensorEquation
from relativisticpy.tensors.indices import latex_label
from relativisticpy.tensors.spaces import Space, spacetime

Term = Node | Operator


@dataclass(frozen=True)
class TensorSource:
    """Something indices can be put on: a defined tensor or a built-in one."""

    node: Node
    indices: Indices
    name: str


#: Built-in tensors by name and number of indices.
_BUILTIN_TENSORS: dict[str, dict[int, str]] = {
    "Gamma": {3: "christoffel"},
    "R": {4: "riemann", 2: "ricci"},
    "delta": {2: "kronecker"},
}
_ROLE_TENSORS = {
    "connection": {3: "christoffel"},
    "riemann": {4: "riemann", 2: "ricci"},
    "ricci": {2: "ricci"},
    "einstein": {2: "einstein"},
    "weyl": {4: "weyl"},
}
_COMPARE = {
    "<": sympy.Lt,
    ">": sympy.Gt,
    "<=": sympy.Le,
    ">=": sympy.Ge,
    "==": sympy.Eq,
    "!=": sympy.Ne,
}
_INTEGER = re.compile(r"\d+")


class Compiler:
    """Compiles programs against a :class:`Session`, which it updates as it goes."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.graph = session.graph
        # Every component selection made, in order, so that a covariant derivative
        # can tell which components were chosen inside its operand.
        self._selections: list[Node] = []
        self._fresh = itertools.count()  # for index labels no one wrote, as in □ = ∇^a ∇_a
        # The spaces the index labels of the statement being compiled range over, and
        # the labels an expectation binds while its operand is compiled.
        self._labels: dict[str, Space] = {}
        self._bound: frozenset[str] = frozenset()

    # ======================================================================
    # Statements
    # ======================================================================

    def compile(self, program: ast.Program) -> list[Output]:
        """Compile a program; return the values it shows, in order."""
        outputs: list[Output] = []
        for statement in program.statements:
            outputs.extend(self.statement(statement, self.session.globals))
        return outputs

    def statement(
        self, statement: ast.Statement, scope: Scope, outer: Context | None = None
    ) -> list[Output]:
        saved = self._labels
        self._labels = self._infer_labels(statement, scope)
        try:
            return self._statement(statement, scope, outer)
        finally:
            self._labels = saved

    def _statement(
        self, statement: ast.Statement, scope: Scope, outer: Context | None
    ) -> list[Output]:
        ctx = Context(latex=statement.latex, calls=outer.calls if outer else ())
        if isinstance(statement, ast.FunctionDefinition):
            scope.bind(statement.name, FunctionBinding(statement, scope))
            return []
        if isinstance(statement, ast.Define) and self._states_identity(statement, scope):
            target = statement.targets[0]
            equation = ast.Equation(target, statement.value, span=statement.span)
            statement = ast.Evaluate(equation, span=statement.span, latex=statement.latex)
        if isinstance(statement, ast.Define):
            if len(statement.targets) > 1:
                self._destructure(statement.targets, statement.value, scope, ctx)
            else:
                self._define(statement.targets[0], statement.value, scope, ctx)
            return []
        assert isinstance(statement, ast.Evaluate)
        expr = statement.expr
        if isinstance(expr, ast.Relation) and self._declares(expr):
            return []
        if isinstance(expr, ast.Chain):  # a = b = c: each link is a claim of its own
            return [
                Output(self._value_or_notation(link, scope, ctx), link.span)
                for link in expr.links()
            ]
        if isinstance(expr, ast.Equation) and self._defines(expr.lhs, scope):
            if self._is_notation(expr.rhs, scope, ctx, equation=True):  # nothing to define it
                return [Output(self._notation(expr, scope, ctx), expr.span)]
            self._define(expr.lhs, expr.rhs, scope, ctx)
            return []
        if isinstance(expr, ast.Equation) and self._is_notation(expr, scope, ctx, equation=True):
            return [Output(self._notation(expr, scope, ctx), expr.span)]
        name = (
            expr.base.id
            if isinstance(expr, ast.Scripted) and isinstance(expr.base, ast.Name)
            else None
        )
        return [Output(self.value(expr, scope, ctx), expr.span, name)]

    def _states_identity(self, statement: ast.Define, scope: Scope) -> bool:
        """Whether a definition written with ``\\equiv`` says what a built-in already is,
        as ``R_{\\mu\\nu} \\equiv R^\\alpha{}_{\\mu\\alpha\\nu}`` does: then it is a claim to
        check, which leaves the built-in as it is, rather than a new meaning for its name.
        The metric is defined this way, as any new tensor is."""
        if len(statement.targets) != 1:
            return False
        target, value = statement.targets[0], statement.value
        written = Span(statement.span.source, target.span.end, value.span.start).text
        if "\\equiv" not in written and "≡" not in written:
            return False
        if isinstance(target, ast.Name):
            return scope.lookup(target.id) is None and self._builtin_scalar(target.id) is not None
        pattern = _tensor_pattern(target) if isinstance(target, ast.Scripted) else None
        if pattern is None or scope.lookup(pattern[0]) is not None:
            return False
        return self._builtin_tensor(pattern[0], len(pattern[1])) not in (None, "metric")

    def _defines(self, target: ast.Expr, scope: Scope) -> bool:
        """Whether ``target = value`` is a definition rather than an equation. What already
        has a meaning is compared, not overwritten: a tensor that exists, written with its
        indices or, as data is, without braces (``w_a``), a component, as in ``g_{tt}`` or
        ``T_{tr}``, whose indices name coordinates, and, with a metric in force, a built-in
        scalar, as in ``R = 0``."""
        if _is_line_element(target):
            return True
        if isinstance(target, ast.Name):
            if scope.lookup(target.id) is not None:
                return True  # a name defined again, as x = 5 and then x = 6
            return _indexed_name(target, scope) is None and self._builtin_scalar(target.id) is None
        if not isinstance(target, ast.Scripted) or not isinstance(target.base, ast.Name):
            return False
        if any(s.indices is None for s in target.scripts):
            return False
        items = [item for s in target.scripts for item in s.indices.items]  # type: ignore[union-attr]
        if not all(_plain_label(item) for item in items):
            return False
        labels = [str(item.label) for item in items]
        if len(set(labels)) != len(labels) or any(map(self._names_component, labels)):
            return False  # a component, or a trace: an equation about what is there
        name = target.base.id
        if name == self.session.names["metric"]:
            return True
        # A tensor that already exists is compared, not overwritten: use := to redefine it.
        if scope.lookup(name) is not None:
            return False
        return self._builtin_tensor(name, len(items)) is None

    def _names_component(self, label: str) -> bool:
        """Whether an index label names a component: a coordinate's name, as ``t`` in
        ``g_{tt}``, unless the label ranges over a space other than spacetime."""
        space = self._labels.get(label) or self.session.labels.get(label)
        if space is not None and not space.spacetime:
            return False
        return any(c.name == label for c in self.session.coordinates or ())

    def _define(self, target: ast.Expr, value: ast.Expr, scope: Scope, ctx: Context) -> None:
        if isinstance(target, ast.Name):
            name = target.id
            if name == "Coordinates":
                self._define_coordinates(value)
            elif name in ASSUMPTION_STATEMENTS:
                self._declare(value, ASSUMPTION_STATEMENTS[name])
            elif name in RENAMING_STATEMENTS:
                self._rename(RENAMING_STATEMENTS[name], value)
            elif (space := self._space_written(value, name, scope, ctx)) is not None:
                self.session.spaces[name] = space
            else:
                value_node = self._value_or_notation(value, scope, ctx, definition=True)
                if (indexed := _indexed_target(name, value_node)) is not None:
                    base, indices = indexed  # v_a := ..., a tensor with free index a: v_{a}
                    node = self._align(value_node, indices, value.span, what="a definition")
                    scope.bind(base, TensorBinding(node, indices))
                else:
                    scope.bind(name, VariableBinding(value_node))
            return
        if _is_line_element(target):
            self._define_line_element(value, scope, ctx, target.span.to(value.span))
            return
        if not isinstance(target, ast.Scripted) or not isinstance(target.base, ast.Name):
            raise CompileError("cannot define this", span=target.span)
        entries = self._entries(target, "a definition")
        if target.base.id == self.session.names["metric"]:
            self._define_metric(target, entries, value, scope, ctx)
        else:
            self._define_tensor(target, entries, value, scope, ctx)

    def _destructure(
        self, targets: Sequence[ast.Expr], value: ast.Expr, scope: Scope, ctx: Context
    ) -> None:
        parts = self._decomposition(value, scope, ctx)
        if parts is not None:
            if len(parts) != len(targets):
                raise CompileError(
                    f"this gives {len(parts)} values, but {len(targets)} are named",
                    span=value.span,
                )
            for target, part in zip(targets, parts, strict=True):
                self._bind_part(target, part, scope)
            return
        values = self.value(value, scope, ctx)
        for position, target in enumerate(targets):
            if not isinstance(target, ast.Name):
                raise CompileError("only names can be defined together", span=target.span)
            index = self.graph.const(position, VALUE)
            item = self.graph.add(
                ops.Call(operator.getitem, "item"), (values, index), VALUE, target.span
            )
            scope.bind(target.id, VariableBinding(item))

    def _bind_part(self, target: ast.Expr, part: Node, scope: Scope) -> None:
        """Name one part of a decomposition: ``\\lambda_k``, ``v^a{}_k`` or a name."""
        if isinstance(target, ast.Name):
            base, underscore, tail = target.id.partition("_")
            labels = split_index_labels(tail) if underscore and tail.isalpha() else []
            if isinstance(part.type, TensorType) and len(labels) == part.type.rank:
                # s_k: the tensor s, its index k where the part has it
                indices = Indices(
                    Index(label, index.upper)
                    for label, index in zip(labels, part.type.indices, strict=True)
                )
                scope.bind(
                    base, TensorBinding(self._name_axes(part, indices, target.span), indices)
                )
                return
            scope.bind(target.id, VariableBinding(part))
            return
        if not isinstance(target, ast.Scripted) or not isinstance(target.base, ast.Name):
            raise CompileError("only names and tensors can be defined together", span=target.span)
        entries = self._entries(target, "a definition")
        indices = Indices(Index(str(item.label), upper) for item, upper in entries)
        if not isinstance(part.type, TensorType) or part.type.rank != len(indices):
            raise IndexNotationError(
                f"{target.base.id} is given {len(indices)} indices for a value with "
                f"{part.type.rank if isinstance(part.type, TensorType) else 0}",
                span=target.span,
            )
        named = self._name_axes(part, indices, target.span)
        node = self._align(named, indices, target.span, what="a definition")
        scope.bind(target.base.id, TensorBinding(node, indices))

    def _define_coordinates(self, value: ast.Expr) -> None:
        names = _names(value, "Coordinates")
        if len(set(names)) != len(names):
            raise CompileError("coordinates must be distinct", span=value.span)
        self.session.set_coordinates(tuple(self.session.symbol(name) for name in names))

    def _declare(self, value: ast.Expr, assumptions: dict[str, bool]) -> None:
        coordinates = {c.name for c in self.session.coordinates or ()}
        for name in _names(value, "a declaration"):
            if name in coordinates:
                raise CompileError(
                    f"{name} is already a coordinate",
                    span=value.span,
                    hint="declare assumptions before the coordinates",
                )
            self.session.assumptions[name] = sympy.Symbol(name, **assumptions)

    def _rename(self, role: str, value: ast.Expr) -> None:
        if not isinstance(value, ast.Name):
            raise CompileError("expected a name", span=value.span)
        self.session.names[role] = value.id

    def _define_metric(
        self,
        target: ast.Scripted,
        entries: list[tuple[ast.IndexItem, bool]],
        value: ast.Expr,
        scope: Scope,
        ctx: Context,
    ) -> None:
        if len(entries) != 2 or not all(_plain_label(item) for item, _ in entries):
            raise IndexNotationError(
                "the metric has two indices", span=target.span, hint="write g_{\\mu\\nu} := ..."
            )
        first = str(entries[0][0].label)
        space = self._labels.get(first) or self.session.labels.get(first)
        if space is not None and not space.spacetime:
            self._define_space_metric(space, target, entries, value, scope, ctx)
            return
        positions = {upper for _, upper in entries}
        if len(positions) != 1:
            raise IndexNotationError(
                "give the metric with both indices down, g_{\\mu\\nu}, or both up, g^{\\mu\\nu}",
                span=target.span,
            )
        components = self._spacetime(value, scope, ctx)
        coordinates = self.session.require_coordinates(target.span, "a metric")
        if components is None:
            components = self._components(value, (len(coordinates),) * 2, scope, ctx)
        if isinstance(components.type, TensorType):  # g_{ab} := h_{ab}, as written
            written = Indices(Index(str(item.label), upper) for item, upper in entries)
            components = self._align(components, written, value.span, what="a definition")
        if positions == {True}:
            components = self.graph.add(
                ops.Geometry("inverse", coordinates), (components,), VALUE, value.span
            )
        indices = Indices([Index("a"), Index("b")])
        metric = self.graph.add(
            ops.Geometry("metric", coordinates),
            (components,),
            TensorType(indices, (spacetime(coordinates),) * 2),
            target.span.to(value.span),
        )
        self.session.metric = MetricState(metric, _mentioned(value, latex=ctx.latex))

    def _define_space_metric(
        self,
        space: Space,
        target: ast.Scripted,
        entries: list[tuple[ast.IndexItem, bool]],
        value: ast.Expr,
        scope: Scope,
        ctx: Context,
    ) -> None:
        """``g_{ab} := \\Sigma_{ab}``: the metric of the space a and b range over, which
        raises and lowers their indices from then on. Given with both indices up, it is the
        inverse metric."""
        indices = Indices(Index(str(item.label), upper) for item, upper in entries)
        if indices[0].upper != indices[1].upper:
            raise IndexNotationError(
                "give a metric with both indices down, g_{ab}, or both up, g^{ab}",
                span=target.span,
            )
        node = self.value(value, scope, ctx)
        if not isinstance(node.type, TensorType):
            node = self.graph.add(
                ops.ToArray((space.dim, space.dim)),
                (node,),
                TensorType(indices, (space, space)),
                value.span,
            )
        node = self._align(self._name_axes(node, indices, value.span), indices, value.span)
        assert isinstance(node.type, TensorType)
        if node.type.spaces != (space, space):
            raise IndexNotationError(
                f"the metric of {space} has both indices over {space}", span=target.span
            )
        if indices[0].upper:
            node = self._inverse(node, value.span)
        assert isinstance(node.type, TensorType)
        lower = Indices([Index("a"), Index("b")])
        self.session.space_metrics[space] = self.graph.retype(node, node.type.with_indices(lower))

    def _space_written(
        self, value: ast.Expr, name: str, scope: Scope, ctx: Context
    ) -> Space | None:
        """The space ``\\mathbb{R}^{n}`` or ``\\mathbb{C}^{n}`` describes, named ``name``;
        ``None`` for any other value."""
        if not (
            isinstance(value, ast.Scripted)
            and isinstance(value.base, ast.Name)
            and value.base.id in ("ℝ", "ℂ")
            and len(value.scripts) == 1
            and value.scripts[0].kind == "^"
            and value.scripts[0].body is not None
        ):
            return None
        size = self.session.evaluate(self.value(value.scripts[0].body, scope, ctx))
        if not (sympy.sympify(size).is_integer and size >= 0):
            raise CompileError(
                f"the dimension of a space is a whole number, not {size}", span=value.span
            )
        return Space(name, int(size), field="complex" if value.base.id == "ℂ" else "real")

    def _declares(self, relation: ast.Relation) -> bool:
        """``a, b \\in \\mathcal{A}``: declare that the labels range over a space, and say
        so; ``False`` for any other relation, which is read but not computed."""
        names = relation.lhs.items if isinstance(relation.lhs, ast.Tuple) else (relation.lhs,)
        if relation.op != "∈" or not all(isinstance(n, ast.Name) for n in names):
            return False
        rhs = relation.rhs
        space = self.session.spaces.get(rhs.id) if isinstance(rhs, ast.Name) else None
        if space is None and isinstance(rhs, ast.Scripted):
            written = rhs.span.text.strip()
            scope = self.session.globals
            space = self._space_written(rhs, written, scope, Context(latex=True))
            if space is not None:
                space = self.session.spaces.setdefault(written, space)
        if space is None:
            return False
        for label in names:
            assert isinstance(label, ast.Name)
            self.session.labels[label.id] = space
        return True

    def _spacetime(self, value: ast.Expr, scope: Scope, ctx: Context) -> Node | None:
        """Components of a spacetime such as ``Schwarzschild(M)``, which also brings its
        coordinates; ``None`` for any other value."""
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and scope.lookup(value.func.id) is None
            and value.func.id[:1].isupper()
            and value.func.id in BUILTINS
        ):
            return None
        metric = self.session.evaluate(self.value(value, scope, ctx))
        if not isinstance(metric, Metric):  # pragma: no cover - every capitalised builtin is one
            return None
        # Its coordinates are plain symbols: one declared with assumptions, as by
        # Positive := [r], takes the place of its namesake.
        declared = {c: self.session.symbol(c.name) for c in metric.coordinates}
        self.session.set_coordinates(tuple(declared.values()))
        return self.graph.const(metric.components.xreplace(declared), VALUE, value.span)

    def _define_line_element(self, value: ast.Expr, scope: Scope, ctx: Context, span: Span) -> None:
        # Without declared coordinates, the differentials name them: dt, dr and d\Omega
        # are t, r, \theta and \phi.
        session = self.session
        named = _line_element_coordinates(value)
        current = {c.name for c in session.coordinates or ()}
        replaceable = session.coordinates is None or session.coordinates_inferred
        if named and replaceable and not set(named) <= current:
            coordinates = tuple(session.symbol(name) for name in named)
            session.set_coordinates(coordinates, inferred=True)
        coordinates = self.session.require_coordinates(span, "a line element")
        differentials = tuple(differential(c) for c in coordinates)
        inner = scope.child()
        for coordinate, d in zip(coordinates, differentials, strict=True):
            inner.bind("d" + coordinate.name, VariableBinding(self.graph.const(d, SCALAR)))
        expr = self.value(value, inner, replace(ctx, line_element=True))
        components = self.graph.add(
            ops.LineElement(coordinates, differentials), (expr,), VALUE, span
        )
        indices = Indices([Index("a"), Index("b")])
        metric = self.graph.add(
            ops.Geometry("metric", coordinates),
            (components,),
            TensorType(indices, (spacetime(coordinates),) * 2),
            span,
        )
        self.session.metric = MetricState(metric, _mentioned(value, latex=ctx.latex))

    def _define_tensor(
        self,
        target: ast.Scripted,
        entries: list[tuple[ast.IndexItem, bool]],
        value: ast.Expr,
        scope: Scope,
        ctx: Context,
    ) -> None:
        assert isinstance(target.base, ast.Name)
        name = target.base.id
        for item, _ in entries:
            if not _plain_label(item):
                raise IndexNotationError(
                    "a tensor is defined with index labels only", span=item.span
                )
        labels = [item.label for item, _ in entries]
        if len(set(labels)) != len(labels):
            raise IndexNotationError("the indices of a definition must differ", span=target.span)
        indices = Indices(Index(str(item.label), upper) for item, upper in entries)
        if _is_literal(value):
            spaces = self._spaces_of(indices.names, target.span, "a tensor")
            kind = TensorType(indices, spaces)
            node = self.graph.retype(self._components(value, kind.shape, scope, ctx), kind)
            scope.bind(name, TensorBinding(node, indices))
            return
        result = self._components_of(self.value(value, scope, ctx), scope, ctx, value.span)
        if isinstance(result.type, TensorType):
            result = self._name_axes(result, indices, value.span)
            node = self._align(result, indices, value.span, what="a definition")
        elif result.type == SCALAR and _subscripted_symbol(target):
            scope.bind(_subscripted_name(target), VariableBinding(result))
            return
        elif result.type == SCALAR and not _is_zero(result):
            raise CompileError(
                f"the components of {name} must be a list, a matrix or 0",
                span=value.span,
                hint=f"for example {name}{_written(target.scripts)} := [...]",
            )
        else:
            kind = TensorType(indices, self._spaces_of(indices.names, target.span, "a tensor"))
            node = self.graph.add(ops.ToArray(kind.shape), (result,), kind, value.span)
        scope.bind(name, TensorBinding(node, indices))

    def _components(
        self, value: ast.Expr, shape: tuple[int, ...], scope: Scope, ctx: Context
    ) -> Node:
        """Components of ``shape`` from a list or matrix literal, checked as written, or
        from any value, checked when computed."""
        rank = len(shape)
        if _is_literal(value):
            cells = _literal_cells(value, rank)
            if cells is None:
                raise CompileError(f"expected {_shape_text(shape)} components", span=value.span)
            found, entries = cells
            if found != shape:
                coordinates = self.session.coordinates
                where = (
                    f"for {shape[0]} coordinates"
                    if coordinates and set(shape) == {len(coordinates)}
                    else "for the spaces of its indices"
                )
                raise CompileError(
                    f"expected {_shape_text(shape)} components {where}, "
                    f"but found {_shape_text(found)}",
                    span=value.span,
                )
            nodes = [self.value(entry, scope, ctx) for entry in entries]
            for node, entry in zip(nodes, entries, strict=True):
                if isinstance(node.type, TensorType | EquationType):
                    raise CompileError("a component must be a scalar", span=entry.span)
            return self.graph.add(ops.MakeArray(shape), nodes, VALUE, value.span)
        result = self.value(value, scope, ctx)
        if isinstance(result.type, TensorType):
            if result.type.rank != rank:
                raise IndexNotationError(
                    f"expected a tensor with {rank} indices, not {result.type.rank}",
                    span=value.span,
                )
            return result
        return self.graph.add(ops.ToArray(shape), (result,), VALUE, value.span)

    # ======================================================================
    # Expressions
    # ======================================================================

    def value(self, node: ast.Expr, scope: Scope, ctx: Context) -> Node:
        """Compile an expression that must produce a value. A builtin's name with nothing
        to act on, as N is in N^2 (the lapse) or \\frac{N}{2}, is a symbol."""
        result = self.term(node, scope, ctx)
        if isinstance(result, FunctionReference) and scope.lookup(result.name) is None:
            return self.graph.const(self.session.symbol(result.name), SCALAR, result.span)
        if isinstance(result, Operator):
            raise CompileError(
                f"{result.describe()} needs something to act on, as in {result.describe()} f",
                span=result.span,
            )
        return result

    def term(self, node: ast.Expr, scope: Scope, ctx: Context) -> Term:
        """Compile an expression, which may be an operator such as ``∂_μ``."""
        handler: Callable[[Any, Scope, Context], Term] = getattr(self, f"_{type(node).__name__}")
        return handler(node, scope, ctx)

    # -- atoms ---------------------------------------------------------------

    def _Number(self, node: ast.Number, scope: Scope, ctx: Context) -> Node:
        text = node.text
        number = (
            sympy.Integer(int(text)) if _INTEGER.fullmatch(text.lstrip("-")) else sympy.Float(text)
        )
        return self.graph.const(number, SCALAR, node.span)

    def _String(self, node: ast.String, scope: Scope, ctx: Context) -> Node:
        return self.graph.const(node.value, VALUE, node.span)

    def _Name(self, node: ast.Name, scope: Scope, ctx: Context) -> Term:
        name = node.id
        binding = scope.lookup(name)
        if isinstance(binding, VariableBinding | TensorBinding):
            return binding.node
        if binding is None and (indexed := _indexed_name(node, scope)) is not None:
            return self.term(indexed, scope, ctx)  # y_t, where y is a tensor: y_{t}
        if isinstance(binding, FunctionBinding):
            return FunctionReference(node.span, name)
        if ctx.line_element and name == "d":
            return DifferentialMarker(node.span)
        if name == "d" and node.verbatim:  # {\rm d}t = ... outside a line element
            raise CompileError(
                "a relation between differentials such as {\\rm d}t is read but not computed",
                span=node.span,
            )
        session = self.session
        if name == "Coordinates" and session.coordinates is not None:
            return self.graph.const(list(session.coordinates), VALUE, node.span)
        if (quantity := self._builtin_scalar(name)) is not None:
            return self._geometry(quantity, node.span)
        if name in session.assumptions:
            return self.graph.const(session.assumptions[name], SCALAR, node.span)
        if name in CONSTANTS:
            return self.graph.const(CONSTANTS[name], SCALAR, node.span)
        if (arguments := session.functions.get(name)) is not None:  # \alpha is \alpha(t, r)
            args = [self.graph.const(argument, SCALAR) for argument in arguments]
            return self._call(sympy.Function(name), name, args, SCALAR, node.span)
        # In LaTeX a run of letters is a product, as typeset maths reads it: re^{x} is
        # r e^{x}. A builtin keeps its name when called, as in diff(f, x).
        if ctx.latex and _splittable(node):
            return self._latex_letters(node, scope, ctx)
        # A builtin applies to what follows it, as sin does in sin x; in LaTeX only if it
        # is written as a command, \operatorname{sign} x, so that N dt^2 is a product.
        if _names_builtin(node) and (node.verbatim or not ctx.latex):
            return FunctionReference(node.span, name)
        return self.graph.const(session.symbol(name), SCALAR, node.span)

    def _builtin_scalar(self, name: str) -> str | None:
        """The geometric quantity that ``name``, bound to nothing in the program, denotes
        with a metric in force: the metric determinant ``g``, the Ricci scalar ``Ric`` or
        ``R``, or the Kretschmann scalar. A name the metric itself is written with, such as
        the radius R of a sphere, or a coordinate's name, stays a symbol; ``Ric`` is always
        the Ricci scalar."""
        session = self.session
        if session.metric is None:
            return None
        own = name in session.metric.names or any(c.name == name for c in session.coordinates or ())
        if name == session.names["metric"] and not own:
            return "determinant"
        if name == session.names["ricci"] or (name == "R" and not own):
            return "ricci_scalar"
        if name == "Kretschmann":
            return "kretschmann"
        return None

    def _latex_letters(self, node: ast.Name, scope: Scope, ctx: Context) -> Node:
        r"""In LaTeX, ``GM`` is :math:`G \cdot M`: one factor per letter (or Greek name)."""
        factors = []
        offset = node.span.start if node.span.text == node.id else None
        for position, label in enumerate(split_index_labels(node.id)):
            span = node.span
            if offset is not None:
                span = Span(node.span.source, offset, offset + len(label))
                offset += len(label)
            factors.append(self.value(_letter(label, span, first=position == 0), scope, ctx))
        return self._multiply(factors, node.span)

    def _Dots(self, node: ast.Dots, scope: Scope, ctx: Context) -> Node:
        raise CompileError(
            "⋯ stands for terms that are not written, so it cannot be computed", span=node.span
        )

    def _Relation(self, node: ast.Relation, scope: Scope, ctx: Context) -> Node:
        if node.op == "≡":  # A \equiv B, where A cannot be defined: an identity
            return self._Equation(ast.Equation(node.lhs, node.rhs, span=node.span), scope, ctx)
        raise CompileError(
            f"'{node.op}' is read but not computed: RelativisticPy decides equations and "
            "inequalities",
            span=node.span,
        )

    def _Chain(self, node: ast.Chain, scope: Scope, ctx: Context) -> Node:
        raise CompileError(
            "a chain of relations, a = b = c, is a statement of its own", span=node.span
        )

    def _Operator(self, node: ast.Operator, scope: Scope, ctx: Context) -> Term:
        if node.kind == "box":
            return Dalembertian(node.span)
        if node.kind == "lie":
            raise CompileError(
                "\\pounds needs the vector field it is along",
                span=node.span,
                hint="for example \\pounds_V T^{\\mu}",
            )
        symbol = "\\partial" if node.kind == "partial" else "\\nabla"
        raise CompileError(
            f"{symbol} needs an index or a variable",
            span=node.span,
            hint=f"for example {symbol}_\\mu",
        )

    # -- arithmetic ----------------------------------------------------------

    def _Unary(self, node: ast.Unary, scope: Scope, ctx: Context) -> Node:
        if node.op == "*":
            raise CompileError("the Hodge dual is not supported yet", span=node.span)
        if node.op in ("±", "∓"):  # ±x is both x and -x
            negative = ast.Unary("-", node.operand, span=node.span)
            pair = (node.operand, negative) if node.op == "±" else (negative, node.operand)
            return self.value(ast.List(pair, span=node.span), scope, ctx)
        operand = self.value(_operand(node.operand), scope, ctx)
        if node.op == "+":
            return operand
        if node.op == "-" and isinstance(operand.op, ops.Const) and operand.type == SCALAR:
            return self.graph.const(-operand.op.value, SCALAR, node.span)
        if node.op == "not":
            return self._call(sympy.Not, "not", [operand], SCALAR, node.span)
        minus_one = self.graph.const(sympy.S.NegativeOne, SCALAR)
        return self._multiply([minus_one, operand], node.span)

    def _Binary(self, node: ast.Binary, scope: Scope, ctx: Context) -> Term:
        if node.op in ("+", "-"):
            return self._sum(node, scope, ctx)
        if node.op in ("±", "∓"):  # a ± b is both a + b and a - b
            plus = ast.Binary("+", node.left, node.right, span=node.span)
            minus = ast.Binary("-", node.left, node.right, span=node.span)
            pair = (plus, minus) if node.op == "±" else (minus, plus)
            return self.value(ast.List(pair, span=node.span), scope, ctx)
        if node.op == "∧":
            return self._wedge(node, scope, ctx)
        if node.op == "∘":
            raise CompileError("the composition of maps is not supported", span=node.span)
        if node.op == "⊙":
            return self._hadamard(node, scope, ctx)
        if node.op == "*":
            return self._product(node, scope, ctx)
        if node.op in ("and", "or"):
            func = sympy.And if node.op == "and" else sympy.Or
            args = [self.value(node.left, scope, ctx), self.value(node.right, scope, ctx)]
            return self._call(func, node.op, args, SCALAR, node.span)
        left = self.value(_operand(node.left), scope, ctx)
        right = self.value(_operand(node.right), scope, ctx)
        if node.op == "/":
            return self._divide(left, right, node.span)
        return self._power(left, right, node.span)

    def _hadamard(self, node: ast.Binary, scope: Scope, ctx: Context) -> Node:
        """``A_{ta} \\odot B_{ta}``: the elementwise product, of tensors with the same free
        indices, summing nothing. Index notation has no other way to say it."""
        left = self.value(_operand(node.left), scope, ctx)
        right = self.value(_operand(node.right), scope, ctx)
        if not (isinstance(left.type, TensorType) and isinstance(right.type, TensorType)):
            return self._multiply([left, right], node.span)
        right = self._align(right, left.type.indices, node.span, what="an elementwise product")
        return self.graph.add(
            ops.Call(linalg.hadamard, "hadamard"), (left, right), left.type, node.span
        )

    def _wedge(self, node: ast.Binary, scope: Scope, ctx: Context) -> Node:
        """The wedge product of forms written with indices:
        :math:`(A \\wedge B)_{\\mu\\nu} = 2 A_{[\\mu} B_{\\nu]}`."""
        left = self.value(_operand(node.left), scope, ctx)
        right = self.value(_operand(node.right), scope, ctx)
        ranks = [
            len(value.type.indices) if isinstance(value.type, TensorType) else 0
            for value in (left, right)
        ]
        if not any(ranks):
            raise CompileError(
                "a wedge product needs forms written with their indices",
                span=node.span,
                hint="for example A_{\\mu} \\wedge B_{\\nu}",
            )
        product = self._multiply([left, right], node.span)
        assert isinstance(product.type, TensorType)
        indices = product.type.indices
        if len(indices) != sum(ranks) or any(index.upper for index in indices):
            raise IndexNotationError(
                "a wedge product is of forms: their indices are all down and all different",
                span=node.span,
            )
        if 0 in ranks:
            return product
        result = self._symmetrize(product, "[", list(indices.names), node.span)
        p, q = ranks
        factor = sympy.factorial(p + q) / (sympy.factorial(p) * sympy.factorial(q))
        return self._multiply([self.graph.const(factor, SCALAR), result], node.span)

    def _sum(self, node: ast.Binary, scope: Scope, ctx: Context) -> Node:
        # The terms in order, without recursion: a sum may have thousands of them.
        terms: list[tuple[int, ast.Expr]] = []
        pending: list[tuple[int, ast.Expr]] = [(1, node)]
        while pending:
            sign, expr = pending.pop()
            if isinstance(expr, ast.Binary) and expr.op in ("+", "-"):
                pending.append((sign if expr.op == "+" else -sign, expr.right))
                pending.append((sign, expr.left))
            else:
                terms.append((sign, expr))
        values = [
            (
                sign,
                self._components_of(self.value(_operand(term), scope, ctx), scope, ctx, term.span),
                term,
            )
            for sign, term in terms
        ]
        tensors = [value for _, value, _ in values if isinstance(value.type, TensorType)]
        if not tensors:
            kind = VALUE if any(v.type == VALUE for _, v, _ in values) else SCALAR
            signs = tuple(sign for sign, _, _ in values)
            return self.graph.add(ops.Add(signs), [v for _, v, _ in values], kind, node.span)
        if self._bound:
            spread = self._broadcast([value for _, value, _ in values], node.span)
            values = [
                (sign, new, term) for (sign, _, term), new in zip(values, spread, strict=True)
            ]
            tensors = [value for _, value, _ in values if isinstance(value.type, TensorType)]
        reference = tensors[0].type
        assert isinstance(reference, TensorType)
        kept: list[tuple[int, Node]] = []
        for sign, value, term in values:
            if isinstance(value.type, TensorType):
                kept.append((sign, self._align(value, reference.indices, term.span)))
            elif not _is_zero(value):
                raise IndexNotationError(
                    f"cannot add a scalar to a tensor with free indices {reference.indices}",
                    span=term.span,
                )
        signs = tuple(sign for sign, _ in kept)
        return self.graph.add(ops.Add(signs), [value for _, value in kept], reference, node.span)

    def _broadcast(self, values: list[Node], span: Span) -> list[Node]:
        """Inside an expectation over t, a term without t is constant in t, as \\mu_a is
        in R_{ta} - \\mu_a: it is spread along t, so that the terms can be added."""
        widest: dict[str, tuple[Index, Space]] = {}
        for value in values:
            if isinstance(value.type, TensorType):
                for index, space in zip(value.type.indices, value.type.spaces, strict=True):
                    widest.setdefault(index.name, (index, space))
        spread: list[Node] = []
        for value in values:
            have = value.type.indices.names if isinstance(value.type, TensorType) else ()
            missing = [name for name in widest if name not in have]
            if missing and all(name in self._bound for name in missing):
                factors = [(value, _indices(value))] if have else []
                for name in missing:
                    index, space = widest[name]
                    ones = self._ones((name, index.upper), span, space)
                    factors.append((ones, Indices([index])))
                if have:
                    value = self._contract(factors, span)
                else:
                    product = self._contract(factors, span) if len(factors) > 1 else factors[0][0]
                    value = self._multiply([value, product], span)
            spread.append(value)
        return spread

    def _statistic_factors(
        self, factors: list[ast.Expr], scope: Scope, ctx: Context
    ) -> list[ast.Expr | Node]:
        """Each statistic, 𝔼_t, Var_t or Cov_t, with the factor after it it acts on,
        compiled to one node; the other factors as they are."""
        grouped: list[ast.Expr | Node] = []
        position = 0
        while position < len(factors):
            factor = factors[position]
            statistic = _statistic(factor)
            if statistic is None:
                grouped.append(factor)
                position += 1
                continue
            if position + 1 == len(factors):
                raise CompileError(
                    f"{factor.span.text} needs something to act on, as in "
                    f"{factor.span.text}[X_{{{statistic[1]}}}]",
                    span=factor.span,
                )
            operand = factors[position + 1]
            span = factor.span.to(operand.span)
            grouped.append(self._statistic(*statistic, operand, span, scope, ctx))
            position += 2
        return grouped

    def _statistic(
        self, kind: str, label: str, operand: ast.Expr, span: Span, scope: Scope, ctx: Context
    ) -> Node:
        r"""``\mathbb{E}_t[X]``, ``\operatorname{Var}_t(X)``, ``\operatorname{Cov}_t(X, Y)``
        and the Ledoit–Wolf covariance, ``\operatorname{LedoitWolf}_t(R_{ta})``. The
        statistic binds ``t`` as a sum does: inside it, every factor shares ``t``."""
        operand = _operand(operand)
        items = operand.items if isinstance(operand, ast.Tuple) else (operand,)
        expected = 2 if kind == "covariance" else 1
        if len(items) != expected:
            raise CompileError(
                f"{span.text.split('(')[0].split('[')[0]} takes {expected} "
                f"argument{'s' if expected > 1 else ''}",
                span=span,
            )
        with self._binding(label):
            nodes = [self.value(_operand(item), scope, ctx) for item in items]
        for node in nodes:
            if not (isinstance(node.type, TensorType) and label in node.type.indices.names):
                raise IndexNotationError(
                    f"a statistic over {latex_label(label)} needs the index "
                    f"{latex_label(label)} in what it acts on",
                    span=span,
                )
        if kind == "expectation":
            return self._expectation(nodes[0], label, span)
        if kind == "ledoit-wolf":
            return self._ledoit_wolf(nodes[0], label, span)
        centered = [self._centered(node, label, span) for node in nodes]
        if kind == "variance":
            square = self.graph.add(
                ops.Call(linalg.hadamard, "hadamard"),
                (centered[0], centered[0]),
                centered[0].type,
                span,
            )
            return self._expectation(square, label, span)
        with self._binding(label):
            product = self._multiply(centered, span)
        return self._expectation(product, label, span)

    def _centered(self, node: Node, label: str, span: Span) -> Node:
        """``X - \\mathbb{E}_t[X]``, the mean spread along ``t`` again."""
        mean = self._expectation(node, label, span)
        with self._binding(label):
            kept, subtracted = self._broadcast([node, mean], span)
            assert isinstance(kept.type, TensorType)
            subtracted = self._align(subtracted, kept.type.indices, span)
            return self.graph.add(ops.Add((1, -1)), (kept, subtracted), kept.type, span)

    def _expectation(self, node: Node, label: str, span: Span) -> Node:
        """The average over ``label`` with its space's measure: a contraction with the
        measure, fused with the contraction that made ``node``, so that the product of
        everything averaged is never formed."""
        assert isinstance(node.type, TensorType)
        names = node.type.indices.names
        if label not in names:
            return node
        index = node.type.indices[names.index(label)]
        space = node.type.space(label)
        measure = self._measure(space, Index(label, not index.upper), span)
        remaining = Indices(i for i in node.type.indices if i.name != label)
        spaces = tuple(
            s for i, s in zip(node.type.indices, node.type.spaces, strict=True) if i.name != label
        )
        kind: Type = TensorType(remaining, spaces) if remaining else SCALAR
        if isinstance(node.op, ops.Contract):
            number = node.op.output[names.index(label)]
            op = ops.Contract(
                (*node.op.labels, (number,)),
                tuple(n for n in node.op.output if n != number),
                node.op.sizes,
            )
            return self.graph.add(op, (*node.args, measure), kind, span)
        assert isinstance(measure.type, TensorType)
        with self._unbinding(label):
            return self._contract(
                [(node, node.type.indices), (measure, measure.type.indices)], span
            )

    def _measure(self, space: Space, index: Index, span: Span) -> Node:
        """The weights an expectation over ``space`` averages with: those declared for it,
        or the same for every element."""
        kind = TensorType(Indices([index]), (space,))
        declared = self.session.measures.get(space)
        if declared is not None:
            return self.graph.retype(declared, kind)
        return self.graph.add(
            ops.Call(_uniform, "uniform"), (self.graph.const(space.dim, VALUE),), kind, span
        )

    def _ledoit_wolf(self, node: Node, label: str, span: Span) -> Node:
        """The Ledoit–Wolf shrinkage estimate of the covariance of ``node`` over
        ``label``, from scikit-learn: better conditioned than the sample covariance when
        there are few samples for the number of variables."""
        assert isinstance(node.type, TensorType)
        if node.type.rank != 2:
            raise IndexNotationError(
                "the Ledoit–Wolf covariance is of samples of a vector, X_{ta}", span=span
            )
        names = node.type.indices.names
        other = 1 - names.index(label)
        if other == 0:
            node = self._permute(node, (1, 0), Indices(reversed(node.type.indices)))
        assert isinstance(node.type, TensorType)
        space = node.type.spaces[1]
        kind = TensorType(Indices([Index("#a"), Index("#b")]), (space, space))
        return self.graph.add(ops.Call(_ledoit_wolf, "ledoit-wolf"), (node,), kind, span)

    @contextlib.contextmanager
    def _binding(self, label: str) -> Iterator[None]:
        saved = self._bound
        self._bound = saved | {label}
        try:
            yield
        finally:
            self._bound = saved

    @contextlib.contextmanager
    def _unbinding(self, label: str) -> Iterator[None]:
        saved = self._bound
        self._bound = saved - {label}
        try:
            yield
        finally:
            self._bound = saved

    def _product(self, node: ast.Binary, scope: Scope, ctx: Context) -> Term:
        factors: list[ast.Expr] = []
        left: ast.Expr = node
        while isinstance(left, ast.Binary) and left.op == "*":
            factors.insert(0, left.right)
            left = left.left
        factors.insert(0, left)
        factors = [part for factor in factors for part in _unsigned(factor)]
        grouped = self._statistic_factors(factors, scope, ctx)
        if len(grouped) == 1 and isinstance(grouped[0], Node):
            return grouped[0]
        terms: list[Term] = []
        chosen: list[list[Node]] = []  # the components each factor selects
        for factor in grouped:
            mark = len(self._selections)
            if isinstance(factor, Node):  # a statistic, already compiled
                terms.append(factor)
                chosen.append([])
                continue
            inner = scope
            if terms and isinstance(terms[-1], DifferentialMarker) and _names_omega(factor):
                # in a line element d\Omega is the solid angle, whatever \Omega means
                inner = scope.child()
                inner.bind("Omega", VariableBinding(self.graph.const(Symbol("Omega"), SCALAR)))
            term = self.term(_operand(factor), inner, ctx)
            if isinstance(term, Node):
                term = self._components_of(term, inner, ctx, factor.span)
            terms.append(term)
            chosen.append(self._selections[mark:])
        # Operators act on what follows them, innermost (rightmost) first. As in
        # physics texts, ∂_μ acts on the next factor (∂_μφ ∂^μφ is (∂_μφ)(∂^μφ)) while
        # d/dx acts on the rest of the term (d/dx x² y³ is d/dx (x² y³)).
        stack: list[Node] = []
        selected: list[list[Node]] = []
        for term, selections in zip(reversed(terms), reversed(chosen), strict=True):
            if isinstance(term, Operator):
                if not stack:
                    raise CompileError(
                        f"{term.describe()} needs something to act on", span=term.span
                    )
                if isinstance(term, LeibnizDerivative):
                    rest = self._multiply(stack[::-1], node.span)
                    stack = [self._apply(term, rest, node.span, scope, ctx)]
                    selected = [[s for group in selected for s in group]]
                else:
                    whole = isinstance(term, Dalembertian | LieDerivative)
                    if whole or (isinstance(term, IndexDerivative) and term.covariant):
                        self._require_whole(stack[-1], selected[-1], term.span)
                    stack[-1] = self._apply(term, stack[-1], node.span, scope, ctx)
            else:
                stack.append(term)
                selected.append(selections)
        result = self._multiply(stack[::-1], node.span)
        return self._spanning_brackets(factors, result, node.span)

    def _multiply(self, operands: Sequence[Node], span: Span) -> Node:
        if len(operands) == 1:
            return operands[0]
        scalars = [o for o in operands if not isinstance(o.type, TensorType)]
        tensors = [o for o in operands if isinstance(o.type, TensorType)]
        if not tensors:
            kind = VALUE if any(o.type == VALUE for o in scalars) else SCALAR
            return self.graph.add(ops.Mul(), scalars, kind, span)
        tensors = self._solves(tensors, span)
        product = tensors[0]
        if len(tensors) > 1:
            product = self._contract([(t, _indices(t)) for t in tensors], span)
        if not scalars:
            return product
        return self.graph.add(ops.Scale(), [*scalars, product], product.type, span)

    def _solves(self, tensors: list[Node], span: Span) -> list[Node]:
        """Replace an inverse and what it multiplies, :math:`(A^{-1})^{ab} b_b`, by the
        solution of :math:`A x = b`: solving is faster and more accurate than inverting,
        and with a symmetric :math:`A` it is a Cholesky solve."""
        tensors = list(tensors)
        while True:
            inverse = next(
                (
                    t
                    for t in tensors
                    if isinstance(t.op, ops.Inverse)
                    and any(o is not t and _shares(o, t) for o in tensors)
                ),
                None,
            )
            if inverse is None or self._bound:
                return tensors
            assert isinstance(inverse.type, TensorType)
            first, second = inverse.type.indices
            others = [t for t in tensors if t is not inverse]
            along, transpose = (second, False)
            if not any(along.name in _indices(t).names for t in others):
                along, transpose = (first, True)
            connected = [t for t in others if along.name in _indices(t).names]
            rhs = connected[0]
            if len(connected) > 1:
                rhs = self._contract([(t, _indices(t)) for t in connected], span)
            assert isinstance(rhs.type, TensorType)
            names = rhs.type.indices.names
            axis = names.index(along.name)
            order = [axis, *(i for i in range(len(names)) if i != axis)]
            rhs = self._permute(rhs, order, Indices(rhs.type.indices[i] for i in order))
            assert isinstance(rhs.type, TensorType)
            kept = first if not transpose else second
            space = inverse.type.space(kept.name)
            indices = Indices([kept, *rhs.type.indices[1:]])
            kind = TensorType(indices, (space, *rhs.type.spaces[1:]))
            solution = self.graph.add(
                ops.Call(linalg.solve, "solve"),
                (inverse.args[0], rhs, self.graph.const(transpose, VALUE)),
                kind,
                span,
            )
            tensors = [t for t in others if all(t is not c for c in connected)] + [solution]

    def _linear_algebra(self, name: str, node: ast.Call, scope: Scope, ctx: Context) -> Node | None:
        """Linear algebra on a tensor: ``eig`` and ``spec`` (its spectrum), ``svd``,
        ``det``, ``tr``, ``inv``, ``diag`` of a vector, and ``exp``, ``sqrt`` and ``log``
        of a map named without indices, :math:`e^{-iHt}`. ``None`` leaves the call to the
        builtins, as for a matrix written as a list."""
        argument = node.args[0]
        if name in ("exp", "sqrt", "log") and _mentions_tensor(argument):
            return None  # written with its indices: one component at a time
        if name not in ("eig", "spec", "svd", "tr") and not self._names_tensor(argument, scope):
            return None  # a builtin of plain matrices and numbers, as diag(1, x^2)
        tensor = self.value(argument, scope, ctx)
        if not isinstance(tensor.type, TensorType):
            return None
        span = node.span
        if name == "diag":
            if tensor.type.rank != 1:
                raise IndexNotationError("diag makes a matrix of a vector", span=span)
            index = tensor.type.indices[0]
            indices = Indices([Index("#r", index.upper), Index("#c", index.upper)])
            space = tensor.type.spaces[0]
            kind = TensorType(indices, (space, space))
            return self.graph.add(ops.Call(linalg.diag, "diag"), (tensor,), kind, span)
        if name == "inv":
            if tensor.type.rank != 2:
                raise IndexNotationError("only a tensor with two indices has an inverse", span=span)
            return self._inverse(tensor, span)
        if name in ("eig", "svd"):
            parts = self._decomposition(node, scope, ctx)
            assert parts is not None
            return self.graph.add(ops.Call(_tuple, name), parts, VALUE, span)
        operator, rows, metric = self._as_map(tensor, span, form=name == "spec")
        if name in ("det", "tr", "trace"):
            func = linalg.det if name == "det" else linalg.trace
            return self.graph.add(
                ops.Call(_matrix_call(func), name),
                (operator, self.graph.const(rows, VALUE)),
                SCALAR,
                span,
            )
        if name in ("exp", "sqrt", "log"):
            return self.graph.add(
                ops.Call(_matrix_function, name),
                (self.graph.const(name, VALUE), operator, self.graph.const(rows, VALUE)),
                operator.type,
                span,
            )
        parts = self._eig(operator, rows, metric, node, scope, ctx, values_only=True)
        return parts[0]

    def _as_map(
        self, tensor: Node, span: Span, *, form: bool = True
    ) -> tuple[Node, int, Node | None]:
        """A tensor as a map, its upper indices first: the tensor, the number of its row
        axes and, for a bilinear form with both indices down, the metric its eigenvalues
        are taken relative to; without ``form``, such a tensor is its matrix of
        components, as for the determinant of g_{ab}."""
        assert isinstance(tensor.type, TensorType)
        indices, spaces = tensor.type.indices, tensor.type.spaces
        upper = [i for i, index in enumerate(indices) if index.upper]
        lower = [i for i, index in enumerate(indices) if not index.upper]
        if len(indices) == 2 and len(upper) != 1 and not form:
            return tensor, 1, None
        if not upper and len(lower) == 2:  # a bilinear form
            space = spaces[0]
            if spaces[1] != space:
                raise IndexNotationError(
                    "a bilinear form takes two vectors of one space", span=span
                )
            if space.spacetime:
                metric: Node | None = self._geometry("metric", span)
            else:
                metric = self.session.space_metrics.get(space)
            return tensor, 1, metric
        if [spaces[i] for i in upper] != [spaces[i] for i in lower]:
            raise IndexNotationError(
                "a map takes its upper indices to its lower ones over the same spaces, "
                "as A^{a}{}_{b} does",
                span=span,
                hint="raise or lower an index, as in eig(\\Sigma^{a}{}_{b})",
            )
        order = upper + lower
        permuted = self._permute(tensor, order, Indices(indices[i] for i in order))
        return permuted, len(upper), None

    def _names_tensor(self, node: ast.Expr, scope: Scope) -> bool:
        """Whether an expression mentions a tensor: written with indices, or by a name."""
        for current in _walk(node):
            if isinstance(current, ast.Scripted) and _tensor_pattern(current):
                return True
            if isinstance(current, ast.Name):
                binding = scope.lookup(current.id)
                if isinstance(binding, TensorBinding) or (
                    isinstance(binding, VariableBinding)
                    and isinstance(binding.node.type, TensorType)
                ):
                    return True
        return False

    def _decomposition(self, value: ast.Expr, scope: Scope, ctx: Context) -> list[Node] | None:
        """The parts of ``eig`` (eigenvalues and eigenvectors) or ``svd`` (U, s and V),
        as typed tensors; ``None`` for any other value."""
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in ("eig", "svd")
            and scope.lookup(value.func.id) is None
            and value.args
        ):
            return None
        tensor = self.value(value.args[0], scope, ctx)
        if not isinstance(tensor.type, TensorType):
            raise CompileError(
                f"{value.func.id} takes a tensor written with its indices", span=value.span
            )
        if value.func.id == "svd":
            return self._svd(tensor, value)
        operator, rows, metric = self._as_map(tensor, value.span)
        return self._eig(operator, rows, metric, value, scope, ctx, values_only=False)

    def _eig(
        self,
        operator: Node,
        rows: int,
        metric: Node | None,
        call: ast.Call,
        scope: Scope,
        ctx: Context,
        *,
        values_only: bool,
    ) -> list[Node]:
        assert isinstance(operator.type, TensorType)
        count = None
        if len(call.args) > 1:
            count = self.session.evaluate(self.value(call.args[1], scope, ctx))
            if not sympy.sympify(count).is_integer:
                raise CompileError("the number of eigenvalues is a whole number", span=call.span)
            count = int(count)
        size = math.prod(space.dim for space in operator.type.spaces[:rows])
        modes = Space(f"eigenvectors of {call.args[0].span.text}", abs(count) if count else size)
        args = [operator, self.graph.const(rows, VALUE), metric or self.graph.const(None, VALUE)]
        args.append(self.graph.const(count, VALUE))
        both = self.graph.add(ops.Call(linalg.eig, "eig"), args, VALUE, call.span)
        values = self._part(both, 0, TensorType(Indices([Index("#k")]), (modes,)), call.span)
        if values_only:
            return [values]
        row_indices = Indices(Index(f"#r{i}", True) for i in range(rows))
        kind = TensorType(
            Indices([*row_indices, Index("#k")]), (*operator.type.spaces[:rows], modes)
        )
        return [values, self._part(both, 1, kind, call.span)]

    def _svd(self, tensor: Node, call: ast.Call) -> list[Node]:
        """``U_{tk}, s_k, V_{ak} := \\operatorname{svd}(R_{ta})``: the first index's
        space, the singular values and the second's."""
        assert isinstance(tensor.type, TensorType)
        if tensor.type.rank != 2:
            raise IndexNotationError("svd takes a tensor with two indices", span=call.span)
        first, second = tensor.type.indices
        rows, columns = tensor.type.spaces
        modes = Space(f"singular vectors of {call.args[0].span.text}", min(rows.dim, columns.dim))
        both = self.graph.add(
            ops.Call(linalg.svd, "svd"), (tensor, self.graph.const(1, VALUE)), VALUE, call.span
        )
        mode = Index("#k")
        return [
            self._part(
                both,
                0,
                TensorType(Indices([Index("#r", first.upper), mode]), (rows, modes)),
                call.span,
            ),
            self._part(both, 1, TensorType(Indices([mode]), (modes,)), call.span),
            self._part(
                both,
                2,
                TensorType(Indices([Index("#c", second.upper), mode]), (columns, modes)),
                call.span,
            ),
        ]

    def _part(self, whole: Node, position: int, kind: Type, span: Span) -> Node:
        index = self.graph.const(position, VALUE)
        return self.graph.add(ops.Call(operator.getitem, "item"), (whole, index), kind, span)

    def _divide(self, numerator: Node, denominator: Node, span: Span) -> Node:
        if isinstance(denominator.type, TensorType):
            raise IndexNotationError("cannot divide by a tensor", span=span)
        minus_one = self.graph.const(sympy.S.NegativeOne, SCALAR)
        reciprocal = self.graph.add(ops.Pow(), (denominator, minus_one), denominator.type, span)
        return self._multiply([numerator, reciprocal], span)

    def _power(self, base: Node, exponent: Node, span: Span) -> Node:
        for value in (base, exponent):
            if isinstance(value.type, TensorType):
                raise IndexNotationError(
                    "cannot raise a tensor to a power",
                    span=span,
                    hint="contract it with itself instead, as in V_{a} V^{a}",
                )
        kind = VALUE if VALUE in (base.type, exponent.type) else SCALAR
        return self.graph.add(ops.Pow(), (base, exponent), kind, span)

    # -- relations -----------------------------------------------------------

    def _Equation(self, node: ast.Equation, scope: Scope, ctx: Context) -> Node:
        lhs = self._components_of(self.value(node.lhs, scope, ctx), scope, ctx, node.lhs.span)
        rhs = self._components_of(self.value(node.rhs, scope, ctx), scope, ctx, node.rhs.span)
        tensor = next((v.type for v in (lhs, rhs) if isinstance(v.type, TensorType)), None)
        if tensor is not None:
            assert isinstance(tensor, TensorType)
            lhs, rhs = (
                self._align(v, tensor.indices, side.span)
                if isinstance(v.type, TensorType)
                else self._zero_side(v, tensor, side.span)
                for v, side in ((lhs, node.lhs), (rhs, node.rhs))
            )
            return self.graph.add(ops.Equation(), (lhs, rhs), EquationType(tensor), node.span)
        return self.graph.add(ops.Equation(), (lhs, rhs), EquationType(SCALAR), node.span)

    def _zero_side(self, value: Node, tensor: TensorType, span: Span) -> Node:
        """The side of a tensor equation that is not a tensor: zero, or components."""
        if value.type == VALUE:
            return self.graph.add(ops.ToArray(tensor.shape), (value,), tensor, span)
        if not _is_zero(value):
            raise IndexNotationError(
                f"one side of this equation is a tensor with free indices {tensor.indices} "
                "and the other is not",
                span=span,
            )
        return value

    def _Comparison(self, node: ast.Comparison, scope: Scope, ctx: Context) -> Node:
        operands = [self.value(o, scope, ctx) for o in node.operands]
        relations = [
            self._call(_COMPARE[op], op, [a, b], SCALAR, node.span)
            for op, a, b in zip(node.ops, operands, operands[1:], strict=False)
        ]
        if len(relations) == 1:
            return relations[0]
        return self._call(sympy.And, "and", relations, SCALAR, node.span)

    # -- functions -----------------------------------------------------------

    def _Call(self, node: ast.Call, scope: Scope, ctx: Context) -> Node:
        func = node.func
        if isinstance(func, ast.Prime):
            return self._prime(func, node.args, node.span, scope, ctx)
        assert isinstance(func, ast.Name)
        if func.id in _NOTATION_OPERATIONS and self._is_notation(node, scope, ctx, operation=True):
            return self._notation(node, scope, ctx)
        name = func.id
        binding = scope.lookup(name)
        if isinstance(binding, FunctionBinding):
            args = [self.value(a, scope, ctx) for a in node.args]
            return self._inline(binding, args, node.span, ctx)
        trace = binding is None and name == "trace" and len(node.args) == 1
        if trace and (traced := self._metric_trace(node.args[0], scope, ctx)) is not None:
            return traced
        if binding is None and name in _LINEAR_ALGEBRA and node.args:
            done = self._linear_algebra(name, node, scope, ctx)
            if done is not None:
                return done
        names = {c.name for c in self.session.coordinates or ()}
        if isinstance(binding, VariableBinding) and all(
            isinstance(a, ast.Name) and a.id in names for a in node.args
        ):  # \alpha(r), once \alpha is defined: it says only what \alpha depends on
            return binding.node
        one_name = len(node.args) == 1 and isinstance(node.args[0], ast.Name)
        product = (name in names and not one_name) or (
            ctx.latex and (isinstance(binding, VariableBinding) or name in CONSTANTS)
        )
        if product:  # r(\partial_1 \beta - 1), with r a coordinate: as typeset, a product
            group = node.args[0] if len(node.args) == 1 else ast.Tuple(node.args, span=node.span)
            times = ast.Binary("*", func, group, implicit=True, span=node.span)
            return self.value(times, scope, ctx)
        if binding is not None:
            raise CompileError(f"{name} is not a function", span=func.span)
        if name == "geodesic_equations":
            return self._geodesic_equations(node, scope, ctx)
        builtin = _names_builtin(func)  # \gamma(r) is an unknown function, not Euler's
        if ctx.latex and binding is None and not builtin and _splittable(func):
            # 2Gm(r) in LaTeX is 2 G m(r): only the last letter is the function
            labels = split_index_labels(name)
            last = _letter(labels[-1], func.span, first=False)
            leading = [
                _letter(label, func.span, first=n == 0) for n, label in enumerate(labels[:-1])
            ]
            factors = [self.value(n, scope, ctx) for n in leading]
            factors.append(self.value(replace(node, func=last), scope, ctx))
            return self._multiply(factors, node.span)
        if builtin:
            args = [self.value(a, scope, ctx) for a in node.args]
            return self._builtin(BUILTINS[name], args, node.span)
        args = [self.value(a, scope, ctx) for a in node.args]
        for arg, source in zip(args, node.args, strict=True):
            if isinstance(arg.type, TensorType):
                raise CompileError(f"cannot pass a tensor to {name}()", span=source.span)
        values = [arg.op.value if isinstance(arg.op, ops.Const) else None for arg in args]
        coordinates = self.session.coordinates or ()
        # Later, a bare name means this; but a coordinate, as r in r(t), stays a coordinate.
        if values and all(value in coordinates for value in values) and name not in names:
            self.session.functions[name] = tuple(values)
        function = sympy.Function(name)
        return self._call(function, name, args, SCALAR, node.span)

    def _metric_trace(self, argument: ast.Expr, scope: Scope, ctx: Context) -> Node | None:
        """``trace(R_{\\mu\\nu})``: the two indices of a tensor on spacetime contracted with
        the metric, as ``trace`` does for notation, rather than the sum of the diagonal
        components, which depends on the coordinates. ``None`` for any other argument."""
        if not self._names_tensor(argument, scope):
            return None
        tensor = self.value(argument, scope, ctx)
        kind = tensor.type
        if not (isinstance(kind, TensorType) and kind.rank == 2):
            return None
        if not all(space.spacetime for space in kind.spaces):
            return None
        indices = Indices(Index(f"#t{k}", index.upper) for k, index in enumerate(kind.indices))
        node = self.graph.retype(tensor, kind.with_indices(indices))
        if indices[0].upper == indices[1].upper:  # T^{a}{}_{a}, raised with the metric
            node, indices = self._move_index(node, indices, 1, argument.span)
        assert isinstance(node.type, TensorType)
        paired = Indices([indices[0], Index(indices[0].name, indices[1].upper)])
        return self._trace(self.graph.retype(node, node.type.with_indices(paired)), argument.span)

    def _builtin(self, builtin: Builtin, args: list[Node], span: Span) -> Node:
        if message := builtin.check_arity(len(args)):
            raise CompileError(message, span=span)
        first = args[0] if args else None
        if builtin.componentwise and first is not None and isinstance(first.type, TensorType):
            op = ops.Map(builtin.func, builtin.name)
            return self.graph.add(op, args, first.type, span)
        args = [self._as_object(arg) for arg in args]
        kind = SCALAR if builtin.scalar and all(a.type == SCALAR for a in args) else VALUE
        return self._call(builtin.func, builtin.name, args, kind, span)

    def _as_object(self, node: Node) -> Node:
        """Wrap tensor components in a :class:`Tensor` for a function that wants one."""
        if isinstance(node.type, TensorType):
            make = _TensorFactory(node.type.indices, self.session.coordinates or ())
            return self.graph.add(ops.Call(make, "tensor"), (node,), VALUE, node.span)
        if isinstance(node.type, EquationType) and isinstance(node.type.of, TensorType):
            make_eq = _EquationFactory(node.type.of.indices, self.session.coordinates or ())
            return self.graph.add(ops.Call(make_eq, "equation"), (node,), VALUE, node.span)
        return node

    def _inline(self, binding: FunctionBinding, args: list[Node], span: Span, ctx: Context) -> Node:
        definition = binding.definition
        if definition.name in ctx.calls:
            raise CompileError(
                f"{definition.name} calls itself; recursive functions are not supported",
                span=span,
            )
        if len(args) != len(definition.params):
            count = len(definition.params)
            raise CompileError(
                f"{definition.name}() takes {count} argument{'s' * (count != 1)}, not {len(args)}",
                span=span,
            )
        scope = binding.closure.child()
        for param, arg in zip(definition.params, args, strict=True):
            scope.bind(param, VariableBinding(arg))
        inner = Context(latex=definition.latex, calls=(*ctx.calls, definition.name))
        if isinstance(definition.body, ast.Block):
            result = None
            for statement in definition.body.statements:
                outputs = self.statement(statement, scope, inner)
                if outputs:
                    result = outputs[-1].node
            if result is None:
                raise CompileError(
                    f"{definition.name} does not return a value",
                    span=definition.span,
                    hint="end the function body with an expression",
                )
            return result
        return self.value(definition.body, scope, inner)

    def _prime(
        self, prime: ast.Prime, args: tuple[ast.Expr, ...], span: Span, scope: Scope, ctx: Context
    ) -> Node:
        """``f'(x)``: differentiate ``f`` with respect to its argument, then apply it."""
        name = prime.func.id
        if len(args) != 1:
            raise CompileError(f"{name}{chr(39) * prime.order}(x) takes one argument", span=span)
        point = self.value(args[0], scope, ctx)
        dummy = self.graph.const(sympy.Dummy(name), SCALAR)
        binding = scope.lookup(name)
        if isinstance(binding, FunctionBinding):
            body = self._inline(binding, [dummy], span, ctx)
        elif binding is not None:
            raise CompileError(f"{name} is not a function", span=prime.func.span)
        elif _names_builtin(prime.func):
            body = self._builtin(BUILTINS[name], [dummy], span)
        else:
            body = self._call(sympy.Function(name), name, [dummy], SCALAR, span)
        order = self.graph.const(prime.order, VALUE)
        return self._call(_derivative_at, "derivative", [body, dummy, order, point], SCALAR, span)

    def _geodesic_equations(self, node: ast.Call, scope: Scope, ctx: Context) -> Node:
        if len(node.args) > 1:
            raise CompileError("geodesic_equations() takes at most one argument", span=node.span)
        parameter = (
            self.value(node.args[0], scope, ctx)
            if node.args
            else self.graph.const(sympy.Symbol("tau"), SCALAR)
        )
        christoffel = self._geometry("christoffel", node.span)
        coordinates = self.graph.const(self.session.coordinates, VALUE)
        return self._call(
            geodesic_equations,
            "geodesic_equations",
            [christoffel, coordinates, parameter],
            VALUE,
            node.span,
        )

    def _require_whole(self, operand: Node, selections: Sequence[Node], span: Span) -> None:
        """A covariant derivative acts on whole tensors. Components chosen as the operand
        is accessed, as in ∇_μ T^{μt}, are taken after the derivative (see
        :meth:`_index_derivative`); components chosen inside it, as in ∇_μ (A^μ B^t),
        are an error rather than a guess at which was meant."""
        inner = operand.args[0] if isinstance(operand.op, ops.Select) else operand
        inside = {node.id for node in dependencies([inner])}
        if any(selection.id in inside for selection in selections):
            raise IndexNotationError(
                "a covariant derivative acts on whole tensors, not on components chosen "
                "inside its operand",
                span=span,
                hint="define the operand as a tensor first, as in X^{\\mu\\nu} := "
                "A^{\\mu} B^{\\nu}, then write \\nabla_\\mu X^{\\mu t}; or name a "
                "component to use it as a function, as in f := g_{tt}",
            )

    def _apply(
        self, operator_: Operator, operand: Node, span: Span, scope: Scope, ctx: Context
    ) -> Node:
        """Apply an operator to the value after it; ``span`` is the expression written."""
        if isinstance(operator_, FunctionReference):
            binding = scope.lookup(operator_.name)
            if isinstance(binding, FunctionBinding):
                return self._inline(binding, [operand], span, ctx)
            return self._builtin(BUILTINS[operator_.name], [operand], span)
        if isinstance(operator_, LeibnizDerivative):
            return self._differentiate(operand, operator_.variables, span)
        if isinstance(operator_, DifferentialMarker):
            return self._differential(operand, span)
        if isinstance(operator_, Dalembertian):
            return self._dalembertian(operand, span)
        if isinstance(operator_, LieDerivative):
            return self._lie_derivative(operator_.vector, operand, span, scope)
        assert isinstance(operator_, IndexDerivative)
        result = operand
        for item, upper in reversed(operator_.items):
            result = self._index_derivative(result, item, upper, operator_.covariant, span)
        return result

    # -- calculus ------------------------------------------------------------

    def _Derivative(self, node: ast.Derivative, scope: Scope, ctx: Context) -> Term:
        variables = tuple((self.value(v, scope, ctx), order) for v, order in node.variables)
        if node.expr is None:
            return LeibnizDerivative(node.span, variables)
        return self._differentiate(self.value(node.expr, scope, ctx), variables, node.span)

    def _differentiate(self, expr: Node, variables: Sequence[tuple[Node, int]], span: Span) -> Node:
        vectors = [v for v, _ in variables if isinstance(v.type, TensorType)]
        if vectors:
            vector = vectors[0]
            assert isinstance(vector.type, TensorType)
            if len(variables) != 1 or variables[0][1] != 1 or vector.type.rank != 1:
                raise IndexNotationError(
                    "a derivative with respect to a vector takes one first derivative",
                    span=span,
                )
            if isinstance(expr.type, TensorType):
                raise IndexNotationError(
                    "only a scalar can be differentiated with respect to a vector", span=span
                )
            index = vector.type.indices[0].flip()
            kind = TensorType(Indices([index]), vector.type.spaces)
            return self.graph.add(ops.Call(_gradient, "gradient"), (expr, vector), kind, span)
        orders = self.graph.const(tuple(order for _, order in variables), VALUE)
        args = [expr, orders, *(v for v, _ in variables)]
        if isinstance(expr.type, TensorType):
            return self.graph.add(ops.Map(_differentiate, "diff"), args, expr.type, span)
        return self._call(_differentiate, "diff", args, expr.type, span)

    def _Differential(self, node: ast.Differential, scope: Scope, ctx: Context) -> Node:
        raise CompileError(
            "a differential d{...} needs a denominator, as in d{f}/d{x}", span=node.span
        )

    def _Integral(self, node: ast.Integral, scope: Scope, ctx: Context) -> Node:
        if node.domain is not None or any(bound.dimension for bound in node.bounds):
            raise CompileError(
                "integrals over regions of spacetime, as in d^4x, are not supported yet",
                span=node.span,
            )
        inner = scope.child()
        limits = []
        for bound in node.bounds:
            symbol = self._bound_variable(bound.variable, inner)
            if bound.lower is None and bound.upper is None:
                limits.append(symbol)
                continue
            if bound.lower is None or bound.upper is None:
                raise CompileError("give both limits of a definite integral", span=bound.span)
            lower = self.value(bound.lower, scope, ctx)
            upper = self.value(bound.upper, scope, ctx)
            limits.append(self._call(_limits, "limits", [symbol, lower, upper], VALUE, bound.span))
        integrand = self.value(node.integrand, inner, ctx)
        if isinstance(integrand.type, TensorType):
            return self.graph.add(
                ops.Map(sympy.integrate, "integrate"),
                [integrand, *limits],
                integrand.type,
                node.span,
            )
        return self._call(
            sympy.integrate, "integrate", [integrand, *limits], integrand.type, node.span
        )

    def _BigOperator(self, node: ast.BigOperator, scope: Scope, ctx: Context) -> Node:
        if node.lower is None or node.upper is None:
            raise CompileError(
                f"\\{node.op} needs its limits",
                span=node.span,
                hint=f"for example \\{node.op}_{{n=0}}^{{3}}",
            )
        inner = scope.child()
        variable = self._bound_variable(node.variable, inner)
        lower = self.value(node.lower, scope, ctx)
        upper = self.value(node.upper, scope, ctx)
        body = self.value(node.body, inner, ctx)
        func = _summation if node.op == "sum" else _product
        args = [body, variable, lower, upper]
        if isinstance(body.type, TensorType):  # component by component, as an integral is
            return self.graph.add(ops.Map(func, node.op), args, body.type, node.span)
        return self._call(func, node.op, args, body.type, node.span)

    def _Limit(self, node: ast.Limit, scope: Scope, ctx: Context) -> Node:
        inner = scope.child()
        variable = self._bound_variable(node.variable, inner)
        point = self.value(node.point, scope, ctx)
        body = self.value(node.body, inner, ctx)
        direction = self.graph.const(node.direction or "+-", VALUE)
        if isinstance(body.type, TensorType):
            return self.graph.add(
                ops.Map(_limit, "limit"), [body, variable, point, direction], body.type, node.span
            )
        return self._call(_limit, "limit", [body, variable, point, direction], SCALAR, node.span)

    def _bound_variable(self, variable: ast.Expr, scope: Scope) -> Node:
        """Bind a summation, integration or limit variable as a plain symbol."""
        if not isinstance(variable, ast.Name):
            raise CompileError("expected a variable", span=variable.span)
        symbol = self.graph.const(self.session.symbol(variable.id), SCALAR, variable.span)
        scope.bind(variable.id, VariableBinding(symbol))
        return symbol

    def _Root(self, node: ast.Root, scope: Scope, ctx: Context) -> Node:
        radicand = self.value(node.radicand, scope, ctx)
        if node.degree is None:
            return self._builtin(BUILTINS["sqrt"], [radicand], node.span)
        degree = self.value(node.degree, scope, ctx)
        return self._builtin(BUILTINS["root"], [radicand, degree], node.span)

    def _Factorial(self, node: ast.Factorial, scope: Scope, ctx: Context) -> Node:
        return self._builtin(
            BUILTINS["factorial"], [self.value(node.operand, scope, ctx)], node.span
        )

    def _Abs(self, node: ast.Abs, scope: Scope, ctx: Context) -> Node:
        operand = self.value(node.operand, scope, ctx)
        if isinstance(operand.type, TensorType) and operand.type.rank == 1:
            return self._norm(operand, node.span)
        return self._builtin(BUILTINS["Abs"], [operand], node.span)

    def _norm(self, vector: Node, span: Span) -> Node:
        """``\\|v\\|``, the length of a vector with the metric of its space:
        :math:`\\sqrt{g_{ab} v^a v^b}`."""
        assert isinstance(vector.type, TensorType)
        index = vector.type.indices[0]
        lowered, _ = self._move_index(vector, vector.type.indices, 0, span)
        twin = Indices([Index(index.name, not index.upper)])
        square = self._contract([(vector, vector.type.indices), (lowered, twin)], span)
        half = self.graph.const(sympy.Rational(1, 2), SCALAR)
        return self.graph.add(ops.Pow(), (square, half), SCALAR, span)

    def _Substitution(self, node: ast.Substitution, scope: Scope, ctx: Context) -> Node:
        expr = self.value(node.expr, scope, ctx)
        pairs = []
        for target, value in node.bindings:
            pairs += [self.value(target, scope, ctx), self.value(value, scope, ctx)]
        return self._substituted(expr, pairs, node.span)

    def _substituted(self, expr: Node, pairs: list[Node], span: Span) -> Node:
        """``expr`` with every substitution made at once: in each component of a tensor,
        and in each side of an equation."""
        if isinstance(expr.op, ops.Equation):
            sides = [self._substituted(side, pairs, span) for side in expr.args]
            return self.graph.add(ops.Equation(), sides, expr.type, span)
        if isinstance(expr.type, TensorType):
            return self.graph.add(ops.Map(_substitute, "subs"), [expr, *pairs], expr.type, span)
        return self._call(_substitute, "subs", [expr, *pairs], expr.type, span)

    def _Accent(self, node: ast.Accent, scope: Scope, ctx: Context) -> Node:
        if node.accent in ("dot", "ddot"):
            return self._time_derivative(self.value(node.base, scope, ctx), node.accent, node.span)
        if not isinstance(node.base, ast.Name):
            raise CompileError(f"\\{node.accent} needs a single symbol", span=node.base.span)
        return self.value(
            ast.Name(node.base.id + node.accent, verbatim=True, span=node.span), scope, ctx
        )

    def _time_derivative(self, value: Node, accent: str, span: Span) -> Node:
        order = self.graph.const(1 if accent == "dot" else 2, VALUE)
        if isinstance(value.type, TensorType):
            return self.graph.add(
                ops.Map(_time_derivative, accent), (value, order), value.type, span
            )
        return self._call(_time_derivative, accent, [value, order], value.type, span)

    # -- collections -----------------------------------------------------------

    def _List(self, node: ast.List, scope: Scope, ctx: Context) -> Node:
        items = [self.value(item, scope, ctx) for item in node.items]
        return self._call(_list, "list", items, VALUE, node.span)

    def _Tuple(self, node: ast.Tuple, scope: Scope, ctx: Context) -> Node:
        items = [self.value(item, scope, ctx) for item in node.items]
        return self._call(_tuple, "tuple", items, VALUE, node.span)

    def _Matrix(self, node: ast.Matrix, scope: Scope, ctx: Context) -> Node:
        cells = [self.value(cell, scope, ctx) for row in node.rows for cell in row]
        shape = (len(node.rows), len(node.rows[0]))
        array = self.graph.add(ops.MakeArray(shape), cells, VALUE, node.span)
        if node.environment == "vmatrix":
            return self._builtin(BUILTINS["det"], [array], node.span)
        return array

    def _Piecewise(self, node: ast.Piecewise, scope: Scope, ctx: Context) -> Node:
        args = []
        for value, condition in node.pieces:
            args.append(self.value(value, scope, ctx))
            args.append(
                self.value(condition, scope, ctx)
                if condition is not None
                else self.graph.const(sympy.true, SCALAR)
            )
        return self._call(_piecewise, "piecewise", args, SCALAR, node.span)

    def _Prime(self, node: ast.Prime, scope: Scope, ctx: Context) -> Node:
        name = node.func.id + "'" * node.order
        raise CompileError(f"{name} needs an argument, as in {name}(x)", span=node.span)

    # ======================================================================
    # Scripts and tensors
    # ======================================================================

    def _Scripted(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Term:
        base = node.base
        if isinstance(base, ast.Operator) and base.kind == "lie":
            return self._lie_operator(node)
        if isinstance(base, ast.Operator) and base.kind == "box":
            raise CompileError("\\Box takes no indices", span=node.span)
        if isinstance(base, ast.Operator):
            return self._derivative_operator(node, covariant=base.kind == "nabla")
        if isinstance(base, ast.Accent) and base.accent in ("dot", "ddot"):
            inner = ast.Scripted(base.base, node.scripts, span=node.span)
            return self._time_derivative(self.value(inner, scope, ctx), base.accent, node.span)
        if (inverse := self._inverse_access(node, scope, ctx)) is not None:
            return inverse
        if (result := self._indexed_result(node, scope, ctx)) is not None:
            return result
        if isinstance(base, ast.Name):
            name = base.id
            binding = scope.lookup(name)
            names = self.session.names
            if (
                binding is None
                and name in (names["partial"], names["covariant"])
                and all(s.indices is not None and s.indices.items for s in node.scripts)
            ):
                return self._derivative_operator(node, covariant=name == names["covariant"])
            rank = _rank(node)
            labels_only = rank is not None and all(
                _plain_label(item)
                for s in node.scripts
                for item in s.indices.items  # type: ignore[union-attr]
            )
            source = None
            if binding is None and labels_only and (pattern := _tensor_pattern(node)):
                source = self._space_tensor(name, pattern[1], node.span)
            if source is None:
                source = self._tensor_source(name, binding, rank, node.span, strict=labels_only)
            differentiated = _has_derivative(node)
            if source is None and rank == 0 and differentiated:  # \phi_{;\mu}
                source = self._scalar_source(base, binding, scope, ctx)
            if source is not None:
                return self._access(source, node, scope, ctx)
            if differentiated:  # never a symbol named phi_mu
                raise self._underivable(node, base, binding, rank)
        if (ones := _tensor_pattern_of_one(node)) is not None:  # 𝟙_a, or 1_a
            return self._ones(ones, node.span)
        return self._script_math(node, scope, ctx)

    def _scalar_source(
        self, base: ast.Name, binding: object, scope: Scope, ctx: Context
    ) -> TensorSource | None:
        """A scalar written with derivative indices only, as ``\\phi_{;\\mu}``: one defined
        in the program, a built-in one such as ``R``, or a function of the coordinates
        written once with its arguments. It has no indices of its own."""
        if isinstance(binding, VariableBinding) and binding.node.type == SCALAR:
            return TensorSource(binding.node, Indices(), base.id)
        written = self._builtin_scalar(base.id) is not None or base.id in self.session.functions
        if binding is None and written:
            return TensorSource(self.value(base, scope, ctx), Indices(), base.id)
        return None

    def _underivable(
        self, node: ast.Scripted, base: ast.Name, binding: object, rank: int | None
    ) -> CompileError:
        """The error for a comma or semicolon derivative of what has none to take."""
        name, written = base.id, base.span.text
        if binding is not None:
            return CompileError(
                f"{name} is not a scalar or a tensor, so it has no derivative", span=node.span
            )
        if not rank:
            return CompileError(
                f"{name} is not defined, so it has no derivative",
                span=node.span,
                hint=f"define it first, for example {written} := ...",
            )
        labels = "".join(_EXAMPLE_LABELS[:rank])
        return CompileError(
            f"{name} is not a tensor",
            span=node.span,
            hint=f"define it first, for example {written}_{{{labels}}} := [...]",
        )

    def _inverse_access(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Node | None:
        """``(\\Sigma^{-1})^{ab}``: the inverse of a two-index tensor, written with indices
        of its own, in the positions opposite to the tensor's."""
        base = node.base
        if not (
            isinstance(base, ast.Scripted)
            and len(base.scripts) == 1
            and base.scripts[0].kind == "^"
            and _is_minus_one(base.scripts[0].body)
            and isinstance(base.base, ast.Name)
        ):
            return None
        matrix = self.value(base.base, scope, ctx)
        if not isinstance(matrix.type, TensorType) or matrix.type.rank != 2:
            return None
        inverse = self._inverse(matrix, base.span)
        assert isinstance(inverse.type, TensorType)
        source = TensorSource(inverse, inverse.type.indices, f"{base.base.id}^{{-1}}")
        return self._access(source, node, scope, ctx)

    def _indexed_result(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Node | None:
        """``\\operatorname{diag}(k)_{ij}``: the tensor a call gives, with indices written
        on it, by position. ``f(x)^2`` stays a power: its script is not index labels."""
        base = node.base
        plain = all(
            script.indices is not None and all(_plain_label(item) for item in script.indices.items)
            for script in node.scripts
        )
        if not (plain and isinstance(base, ast.Call) and self._names_tensor(base, scope)):
            return None
        value = self.value(base, scope, ctx)
        if not isinstance(value.type, TensorType):
            return None
        return self._access(
            TensorSource(value, value.type.indices, base.span.text), node, scope, ctx
        )

    def _space_tensor(
        self, name: str, entries: Sequence[tuple[str, bool]], span: Span
    ) -> TensorSource | None:
        """The Kronecker delta and the metric of a space other than spacetime: the
        identity, or the metric declared for the space, as in g_{ab} := \\Sigma_{ab}."""
        quantity = self._builtin_tensor(name, len(entries))
        if quantity not in ("kronecker", "metric"):
            return None
        space = self._labels.get(entries[0][0]) or self.session.labels.get(entries[0][0])
        if space is None or space.spacetime:
            return None
        declared = self.session.space_metrics.get(space)
        if quantity == "metric" and declared is not None:
            assert isinstance(declared.type, TensorType)
            return TensorSource(declared, declared.type.indices, name)
        upper = quantity == "kronecker"
        indices = Indices([Index("a", upper), Index("b")])
        node = self.graph.add(
            ops.Call(_identity, "identity"),
            (self.graph.const(space.dim, VALUE),),
            TensorType(indices, (space, space)),
            span,
        )
        return TensorSource(node, indices, name)

    def _ones(self, label: tuple[str, bool], span: Span, space: Space | None = None) -> Node:
        """𝟙_a: the covector whose components are all one, which sums what it meets."""
        name, upper = label
        space = space or self._space_of(name, span)
        kind = TensorType(Indices([Index(name, upper)]), (space,))
        return self.graph.add(
            ops.Call(_ones, "ones"), (self.graph.const(space.dim, VALUE),), kind, span
        )

    def _builtin_tensor(self, name: str, rank: int | None) -> str | None:
        """The geometric quantity a built-in tensor name denotes with ``rank`` indices."""
        names = self.session.names
        if name == names["metric"]:
            return "metric" if rank == 2 else None
        candidates: dict[int, str] = {}
        for role, by_rank in _ROLE_TENSORS.items():
            if names[role] == name:
                candidates.update(by_rank)
        for count, quantity in _BUILTIN_TENSORS.get(name, {}).items():
            candidates.setdefault(count, quantity)
        if name == "C":  # most texts write the Weyl tensor C_{abcd}
            candidates.setdefault(4, "weyl")
        return candidates.get(rank) if rank is not None else None

    def _tensor_source(
        self, name: str, binding: object, rank: int | None, span: Span, *, strict: bool = False
    ) -> TensorSource | None:
        """What ``name`` with ``rank`` indices refers to, if it is a tensor. With
        ``strict`` (the scripts are plain index labels), a built-in tensor given the
        wrong number of indices is an error rather than an ordinary symbol."""
        if isinstance(binding, TensorBinding):
            return TensorSource(binding.node, binding.indices, name)
        if isinstance(binding, VariableBinding) and isinstance(binding.node.type, TensorType):
            return TensorSource(binding.node, binding.node.type.indices, name)
        if binding is not None or rank is None:
            return None
        quantity = self._builtin_tensor(name, rank)
        if quantity is None:
            ranks = [n for n in range(1, 5) if self._builtin_tensor(name, n)]
            if ranks and strict:
                expected = " or ".join(str(n) for n in ranks)
                raise IndexNotationError(
                    f"{name} has {expected} indices, but {rank} "
                    f"{'is' if rank == 1 else 'are'} given",
                    span=span,
                )
            return None
        node = self._geometry(quantity, span)
        assert isinstance(node.type, TensorType)
        return TensorSource(node, node.type.indices, name)

    def _geometry(self, quantity: str, span: Span) -> Node:
        return self.session.geometry(quantity, span)

    def _entries(self, node: ast.Scripted, purpose: str) -> list[tuple[ast.IndexItem, bool]]:
        """The index items of every script, each with whether it is upper."""
        entries: list[tuple[ast.IndexItem, bool]] = []
        for script in node.scripts:
            if script.indices is None:
                if script.index_error is not None:
                    raise script.index_error
                raise IndexNotationError(f"expected indices in {purpose}", span=script.span)
            for item in script.indices.items:
                if item.label == "⋯":
                    raise CompileError(
                        "⋯ stands for indices that are not written, so this cannot be computed",
                        span=item.span,
                    )
                entries.append((item, script.kind == "^"))
        return entries

    def _access(self, source: TensorSource, node: ast.Scripted, scope: Scope, ctx: Context) -> Node:
        """Put indices on a tensor: raise or lower them, take components, derivatives
        (``T_{a,b}``, ``T_{a;b}``), traces and (anti)symmetrisations."""
        entries = self._entries(node, f"the indices of {source.name}")
        main = [(item, upper) for item, upper in entries if item.derivative is None]
        derivatives = [(item, upper) for item, upper in entries if item.derivative is not None]
        if any(item.derivative is None for item, _ in entries[len(main) :]):
            raise IndexNotationError(
                "derivative indices come after the tensor's own: T_{ab;c}", span=node.span
            )
        rank = len(source.indices)
        if len(main) != rank:
            example = source.name + "".join(str(i) for i in source.indices)
            raise IndexNotationError(
                f"{source.name} has {rank} ind{'ex' if rank == 1 else 'ices'}, but "
                f"{len(main)} {'is' if len(main) == 1 else 'are'} given",
                span=node.span,
                hint=f"for example {example}",
            )
        spaces = _spaces(source.node)
        if spaces is None or any(space.spacetime for space in spaces):
            self.session.require_coordinates(node.span)
        if spaces is None:
            spaces = (self.session.spacetime(node.span),) * rank
        for (item, _), space in zip(main, spaces, strict=True):
            if _plain_label(item):
                self._check_label(str(item.label), space, node.span)
        current = Indices(Index(f"#{k}", index.upper) for k, index in enumerate(source.indices))
        result = source.node  # a scalar, \phi in \phi_{;\mu}, stays one
        if rank:
            result = self.graph.retype(source.node, TensorType(current, spaces))
        for position, (_, upper) in enumerate(main):
            if current[position].upper != upper:
                result, current = self._move_index(result, current, position, node.span)
        for number, (item, upper) in enumerate(derivatives):
            covariant = item.derivative == ";"
            label = f"#d{number}"
            result = self._derivative_of(result, current, label, covariant, node.span)
            moved = (*range(1, len(current) + 1), 0)
            current = Indices([*current, Index(label)])
            result = self._permute(result, moved, current)
            if upper:
                result, current = self._move_index(result, current, len(current) - 1, node.span)
        return self._label(result, [*main, *derivatives], node.span)

    def _label(
        self,
        result: Node,
        entries: Sequence[tuple[ast.IndexItem, bool]],
        span: Span,
        fixed: Mapping[int, int] | None = None,
    ) -> Node:
        """Name the axes after the items written, take traces and (anti)symmetrisations,
        then fix components: those the items name, and any already ``fixed`` by position.
        Components are fixed last so that a covariant derivative of the result can still
        reach the whole tensor (see :meth:`_index_derivative`)."""
        assert isinstance(result.type, TensorType)
        spaces = result.type.spaces
        labels: list[Index] = []
        values: dict[str, int] = {}
        for position, (item, upper) in enumerate(entries):
            if fixed and position in fixed:
                value: int | None = fixed[position]
            else:
                value = self._index_value(item, spaces[position])
            if value is not None:
                values[f"#v{position}"] = value
                labels.append(Index(f"#v{position}", upper))
            else:
                labels.append(Index(str(item.label), upper))
        result = self.graph.retype(result, TensorType(Indices(labels), spaces))
        result = self._trace(result, span)
        result = self._brackets(result, entries, span)
        return self._select(result, values, span)

    def _select(self, node: Node, values: dict[str, int], span: Span) -> Node:
        """Fix the axes named in ``values`` to those components."""
        if not values or not isinstance(node.type, TensorType):
            return node
        indices = node.type.indices
        fixed = tuple(
            (axis, values[index.name]) for axis, index in enumerate(indices) if index.name in values
        )
        remaining = Indices(index for index in indices if index.name not in values)
        spaces = tuple(s for s, i in zip(node.type.spaces, indices, strict=True) if i in remaining)
        kind: Type = TensorType(remaining, spaces) if remaining else SCALAR
        selection = self.graph.add(ops.Select(fixed), (node,), kind, span)
        self._selections.append(selection)
        return selection

    def _index_value(self, item: ast.IndexItem, space: Space) -> int | None:
        """The component an index item fixes: ``a=0``, ``a:t``, or the name of an element
        of the space's basis, a coordinate's name in spacetime."""
        names = list(space.basis or ())
        what = "coordinates" if space.spacetime else f"elements of {space.name}"
        value = item.value
        if value is None:
            return names.index(item.label) if item.label in names else None
        if isinstance(value, ast.Number):
            number = int(value.text)
            if number >= space.dim:
                raise IndexNotationError(
                    f"index value {number} is out of range for {space.dim} {what}",
                    span=value.span,
                )
            return number
        assert isinstance(value, ast.Name)
        if value.id not in names:
            noun = "a coordinate" if space.spacetime else f"an element of {space.name}"
            raise IndexNotationError(
                f"{value.id} is not {noun}",
                span=value.span,
                hint=f"the {what} are {', '.join(names)}" if names else None,
            )
        return names.index(value.id)

    def _move_index(
        self, node: Node, indices: Indices, position: int, span: Span
    ) -> tuple[Node, Indices]:
        """Raise or lower one index with the metric of its space: spacetime's metric, one
        declared for the space, or the identity, which leaves the components as they are."""
        index = indices[position]
        assert isinstance(node.type, TensorType)
        space = node.type.spaces[position]
        if space.spacetime:
            metric = self._geometry("metric" if index.upper else "inverse", span)
        elif (declared := self.session.space_metrics.get(space)) is not None:
            metric = declared if index.upper else self._inverse(declared, span)
        else:
            renamed = Indices([index.flip() if p == position else i for p, i in enumerate(indices)])
            return self.graph.retype(node, node.type.with_indices(renamed)), renamed
        placeholder = Index(f"#m{position}", not index.upper)
        metric_indices = Indices([placeholder, index.flip()])
        output = Indices([placeholder if p == position else i for p, i in enumerate(indices)])
        moved = self._contract([(node, indices), (metric, metric_indices)], span, output=output)
        renamed = Indices([index.flip() if p == position else i for p, i in enumerate(indices)])
        assert isinstance(moved.type, TensorType)
        return self.graph.retype(moved, moved.type.with_indices(renamed)), renamed

    def _permute(self, node: Node, axes: Sequence[int], indices: Indices) -> Node:
        assert isinstance(node.type, TensorType)
        kind = TensorType(indices, tuple(node.type.spaces[axis] for axis in axes))
        if list(axes) == list(range(len(axes))):
            return self.graph.retype(node, kind)
        return self.graph.add(ops.Permute(tuple(axes)), (node,), kind, node.span)

    def _align(self, value: Node, indices: Indices, span: Span, what: str = "a sum") -> Node:
        """``value``'s components with axes in the order of ``indices``."""
        assert isinstance(value.type, TensorType)
        have = value.type.indices
        if sorted(have, key=str) != sorted(indices, key=str):
            raise IndexNotationError(
                f"the free indices of {what} must agree: {indices} and {have}", span=span
            )
        return self._permute(value, [have.index(i) for i in indices], indices)

    def _contract(
        self,
        factors: Sequence[tuple[Node, Indices]],
        span: Span,
        output: Indices | None = None,
    ) -> Node:
        """Multiply tensors, summing each index that appears once up and once down, or,
        in a space whose metric is the identity, twice in any position. An index ranges
        over one space. A label an expectation binds is shared by every factor and kept."""
        seen: dict[str, list[tuple[Index, Space]]] = {}
        for node, indices in factors:
            spaces = _spaces(node)
            assert spaces is not None
            assert len(spaces) == len(indices)
            for index, space in zip(indices, spaces, strict=True):
                seen.setdefault(index.name, []).append((index, space))
        for name, found in seen.items():
            label = _display(name)
            over = list(dict.fromkeys(space for _, space in found))
            if len(over) > 1:
                raise IndexNotationError(
                    f"index {label} ranges over {over[0]} in one place and {over[1]} in "
                    "another; an index ranges over one space",
                    span=span,
                )
            if name in self._bound:
                continue
            if len(found) > 2:
                raise IndexNotationError(
                    f"index {label} appears {len(found)} times; an index may appear at most twice",
                    span=span,
                )
            positions = {index.upper for index, _ in found}
            if len(found) == 2 and len(positions) == 1 and not self._euclidean(over[0]):
                position = "an upper" if found[0][0].upper else "a lower"
                raise IndexNotationError(
                    f"index {label} appears twice as {position} index; "
                    "a summed index must appear once up and once down",
                    span=span,
                )
        if output is None:
            kept = {
                name: found[0][0]
                for name, found in seen.items()
                if len(found) == 1 or name in self._bound
            }
            output = Indices(kept.values())
        numbers = {name: number for number, name in enumerate(seen)}
        labels = tuple(tuple(numbers[i.name] for i in indices) for _, indices in factors)
        sizes = tuple(found[0][1].dim for found in seen.values())
        spaces = tuple(seen[index.name][0][1] for index in output)
        kind: Type = TensorType(output, spaces) if output else SCALAR
        op = ops.Contract(labels, tuple(numbers[i.name] for i in output), sizes)
        return self.graph.add(op, [node for node, _ in factors], kind, span)

    def _euclidean(self, space: Space) -> bool:
        """Whether the space's metric is the identity, so that an index may be summed in
        any position: neither spacetime nor a space given a metric."""
        return not space.spacetime and space not in self.session.space_metrics

    def _inverse(self, node: Node, span: Span) -> Node:
        """The inverse of a two-index tensor: its indices change position."""
        assert isinstance(node.type, TensorType)
        assert node.type.rank == 2
        flipped = Indices(index.flip() for index in node.type.indices)
        return self.graph.add(ops.Inverse(), (node,), node.type.with_indices(flipped), span)

    # -- spaces ----------------------------------------------------------------------
    #
    # An index label ranges over a space: one it was declared in (a, b \in \mathcal{A}),
    # or, within a statement, the space of the first axis it is written on. Labels that
    # neither declare nor infer range over spacetime.

    def _space_of(self, name: str, span: Span, why: str = "index notation") -> Space:
        space = self._labels.get(name) or self.session.labels.get(name)
        return space if space is not None else self.session.spacetime(span, why)

    def _spaces_of(self, names: Sequence[str], span: Span, why: str) -> tuple[Space, ...]:
        return tuple(self._space_of(name, span, why) for name in names)

    def _check_label(self, name: str, space: Space, span: Span) -> None:
        """That an index written on an axis over ``space`` may range over it; the first
        time in a statement, it then does, for the rest of the statement."""
        expected = self._labels.get(name) or self.session.labels.get(name)
        if expected is None:
            self._labels[name] = space
        elif expected != space:
            raise IndexNotationError(
                f"the index {latex_label(name)} ranges over {expected}, but is written "
                f"here on an axis over {space}",
                span=span,
            )

    def _infer_labels(self, statement: ast.Statement, scope: Scope) -> dict[str, Space]:
        """The spaces of the labels written on tensors with spaces of their own in a
        statement, found before it is compiled, so that 𝟙_c in 𝟙_c (\\Sigma^{-1})^{cd}
        knows its space from what comes after it."""
        labels: dict[str, Space] = {}
        for current in _walk(statement):
            if not isinstance(current, ast.Scripted):
                continue
            base = current.base
            if isinstance(base, ast.Scripted) and len(base.scripts) == 1:  # (\Sigma^{-1})^{cd}
                if not _is_minus_one(base.scripts[0].body):
                    continue
                base = base.base
            if not isinstance(base, ast.Name):
                continue
            binding = scope.lookup(base.id)
            node = binding.node if isinstance(binding, VariableBinding | TensorBinding) else None
            spaces = _spaces(node) if node is not None else None
            pattern = _tensor_pattern(replace(current, base=base))
            if not spaces or pattern is None or len(pattern[1]) != len(spaces):
                continue
            for (label, _), space in zip(pattern[1], spaces, strict=True):
                if space.spacetime:
                    continue
                expected = labels.get(label) or self.session.labels.get(label)
                if expected is not None and expected != space:
                    raise IndexNotationError(
                        f"the index {latex_label(label)} ranges over {expected}, but is "
                        f"written here on an axis over {space}",
                        span=current.span,
                    )
                labels[label] = space
        return labels

    def _name_axes(self, value: Node, indices: Indices, span: Span) -> Node:
        """Name the axes of a value whose indices no one wrote, such as the eigenvectors
        eig returns, by position after the indices of a definition."""
        assert isinstance(value.type, TensorType)
        current = value.type.indices
        if len(current) != len(indices) or not all(i.name.startswith("#") for i in current):
            return value
        for position, wanted in enumerate(indices):
            if current[position].upper != wanted.upper:
                value, current = self._move_index(value, current, position, span)
        assert isinstance(value.type, TensorType)
        for wanted, space in zip(indices, value.type.spaces, strict=True):
            self._check_label(wanted.name, space, span)
        return self.graph.retype(value, value.type.with_indices(indices))

    def _trace(self, node: Node, span: Span) -> Node:
        """Sum over indices repeated within one tensor: ``R^{a}{}_{bac}``."""
        if not isinstance(node.type, TensorType):
            return node
        names = node.type.indices.names
        if len(set(names)) == len(names):
            return node
        return self._contract([(node, node.type.indices)], span)

    def _brackets(
        self, node: Node, entries: Sequence[tuple[ast.IndexItem, bool]], span: Span
    ) -> Node:
        """(Anti)symmetrise over bracketed indices that open and close within ``entries``."""
        for kind, labels, _ in _bracket_groups([entries], span, closed=False):
            node = self._symmetrize(node, kind, labels, span)
        return node

    def _spanning_brackets(self, factors: Sequence[ast.Expr], node: Node, span: Span) -> Node:
        """Brackets opened in one factor and closed in another, as in the Bianchi identity
        ``\\nabla_{[\\lambda} R_{\\rho\\sigma]\\mu\\nu}``."""
        groups = []
        entries = []
        for factor in factors:
            if isinstance(factor, ast.Scripted):
                own = [
                    (i, s.kind == "^") for s in factor.scripts if s.indices for i in s.indices.items
                ]
                entries.append(own)
        flat = [item for own in entries for item, _ in own]
        if not any(item.opens or item.closes for item in flat):
            return node
        for kind, labels, spans_factors in _bracket_groups(entries, span, closed=True):
            if spans_factors:
                groups.append((kind, labels))
        for kind, labels in groups:
            node = self._symmetrize(node, kind, labels, span)
        return node

    def _symmetrize(self, node: Node, kind: str, labels: Sequence[str], span: Span) -> Node:
        if not isinstance(node.type, TensorType):
            raise IndexNotationError("only free indices can be (anti)symmetrised", span=span)
        names = list(node.type.indices.names)
        missing = [label for label in labels if label not in names]
        if missing:
            raise IndexNotationError(
                f"only free indices can be (anti)symmetrised, not {', '.join(missing)}", span=span
            )
        axes = tuple(names.index(label) for label in labels)
        op = ops.Symmetrize(axes, antisymmetric=kind == "[")
        return self.graph.add(op, (node,), node.type, span)

    # -- derivatives with indices ------------------------------------------------

    def _lie_operator(self, node: ast.Scripted) -> LieDerivative:
        """``\\pounds_V``: the subscript names the vector field."""
        script = node.scripts[0]
        name = script.body.id if isinstance(script.body, ast.Name) else None
        if len(node.scripts) != 1 or script.kind != "_" or name is None:
            raise CompileError(
                "\\pounds needs one subscript, the vector field it is along",
                span=node.span,
                hint="for example \\pounds_V T^{\\mu}",
            )
        return LieDerivative(node.span, name)

    def _dalembertian(self, operand: Node, span: Span) -> Node:
        """``□X = ∇^a ∇_a X``."""
        self.session.require_coordinates(span, "\\Box")
        item = ast.IndexItem(f"#box{next(self._fresh)}", span=span)
        inner = self._index_derivative(operand, item, False, True, span)
        return self._index_derivative(inner, item, True, True, span)

    def _lie_derivative(self, vector: str, operand: Node, span: Span, scope: Scope) -> Node:
        """The Lie derivative along ``V``:
        :math:`£_V T^{a}{}_{b} = V^c \\partial_c T^{a}{}_{b} - (\\partial_c V^a) T^{c}{}_{b}
        + (\\partial_b V^c) T^{a}{}_{c}`, with one such term for every index."""
        space = self.session.spacetime(span, "a Lie derivative")
        binding = scope.lookup(vector)
        if not isinstance(binding, TensorBinding) or len(binding.indices) != 1:
            raise CompileError(
                f"\\pounds_{vector} needs {vector} to be a vector field",
                span=span,
                hint=f"define it first, for example {vector}^{{\\mu}} := [1, 0, 0, 0]",
            )
        if not binding.indices[0].upper:
            raise IndexNotationError(
                f"\\pounds_{vector} is along a vector, with its index up", span=span
            )
        field_ = self.graph.retype(binding.node, TensorType(Indices([Index("#v", True)]), (space,)))
        selection: tuple[tuple[int, int], ...] = ()
        if isinstance(operand.op, ops.Select):  # as for ∇: differentiate, then select
            selection, operand = operand.op.fixed, operand.args[0]
        names = operand.type.indices if isinstance(operand.type, TensorType) else Indices()
        current = Indices(Index(f"#{k}", i.upper) for k, i in enumerate(names))
        if names:
            assert isinstance(operand.type, TensorType)
            operand = self.graph.retype(operand, operand.type.with_indices(current))
        kind = TensorType(current, (space,) * len(current)) if current else SCALAR
        derivative = self._derivative_of(operand, current, "#c", False, span)
        terms = [
            (
                1,
                self._contract(
                    [
                        (field_, Indices([Index("#c", True)])),
                        (derivative, Indices([Index("#c"), *current])),
                    ],
                    span,
                    output=current,
                ),
            )
        ]
        field_derivative = self._derivative_of(
            field_, Indices([Index("#v", True)]), "#c", False, span
        )
        for position, index in enumerate(current):
            swapped = Indices(
                Index("#e", index.upper) if p == position else i for p, i in enumerate(current)
            )
            if index.upper:  # - (∂_e V^a) T^{..e..}
                field_indices, sign = Indices([Index("#e"), Index(index.name, True)]), -1
            else:  # + (∂_b V^e) T_{..e..}
                field_indices, sign = Indices([Index(index.name), Index("#e", True)]), 1
            terms.append(
                (
                    sign,
                    self._contract(
                        [(field_derivative, field_indices), (operand, swapped)],
                        span,
                        output=current,
                    ),
                )
            )
        result = self.graph.add(
            ops.Add(tuple(s for s, _ in terms)), [t for _, t in terms], kind, span
        )
        if not names:
            return result
        entries = [(ast.IndexItem(name.name, span=span), name.upper) for name in names]
        return self._label(result, entries, span, dict(selection))

    def _derivative_operator(self, node: ast.Scripted, *, covariant: bool) -> IndexDerivative:
        """``\\partial_\\mu`` or ``\\nabla^\\mu``: one derivative for each index. A number
        written as a superscript is an order, never an index: ``\\partial_0^2`` and
        ``\\partial^2_0`` are \\partial_0 \\partial_0, and with no subscript
        ``\\partial^2`` is \\partial^\\mu \\partial_\\mu and ``\\nabla^2`` is
        \\nabla^\\mu \\nabla_\\mu, which is \\Box."""
        scripts = node.scripts
        orders = [s for s in scripts if s.kind == "^" and isinstance(s.body, ast.Number)]
        if len(orders) == 1 and len(scripts) <= 2:
            order = orders[0]
            assert isinstance(order.body, ast.Number)
            count = int(order.body.text) if order.body.text.isdigit() else 0
            rest = tuple(s for s in scripts if s is not order)
            symbol = "\\nabla" if covariant else "\\partial"
            if not rest:  # \nabla^2: the wave operator, with an index no one wrote
                if count != 2:
                    raise CompileError(
                        f"{symbol}^{order.body.text} has no meaning: a number on {symbol} is "
                        f"the order of a derivative, and {symbol}^2 is {symbol}^\\mu {symbol}_\\mu",
                        span=node.span,
                        hint=f"for a component, name it, as in {symbol}^t or {symbol}^{{\\mu=0}}",
                    )
                self.session.require_coordinates(node.span, f"{symbol}^2")
                item = ast.IndexItem(f"#box{next(self._fresh)}", span=node.span)
                return IndexDerivative(node.span, covariant, ((item, True), (item, False)))
            once = self._entries(replace(node, scripts=rest), "a derivative")
            if count < 1 or len(once) not in (1, count):
                raise IndexNotationError(
                    f"a derivative of order {order.body.text} takes one index, or {count}",
                    span=node.span,
                    hint=f"for example {symbol}_t^2 or {symbol}^2_{{tx}}",
                )
            # \partial_0^2 is \partial_0 \partial_0; \partial^2_{tx} is \partial_t \partial_x
            entries = once * count if len(once) == 1 else once
        else:
            entries = self._entries(node, "a derivative")
        for item, _ in entries:
            if item.derivative is not None:
                raise IndexNotationError("unexpected derivative index", span=item.span)
        return IndexDerivative(node.span, covariant, tuple(entries))

    def _index_derivative(
        self, operand: Node, item: ast.IndexItem, upper: bool, covariant: bool, span: Span
    ) -> Node:
        """``∂_μ X`` or ``∇_μ X``: the derivative gets the first axis."""
        if (
            self.session.coordinates is None
            and item.label
            and not isinstance(operand.type, TensorType)
        ):
            variable = self.graph.const(self.session.symbol(item.label), SCALAR)
            return self._differentiate(operand, [(variable, 1)], span)
        self.session.require_coordinates(span, "a derivative with an index")
        selection: tuple[tuple[int, int], ...] = ()
        if covariant and isinstance(operand.op, ops.Select):
            # ∇ acts on the whole tensor and the components asked for are taken after:
            # ∇_μ T^{μt} is the t component of ∇_μ T^{μν}, which needs every component.
            selection, operand = operand.op.fixed, operand.args[0]
        if isinstance(operand.type, TensorType):
            current = Indices(Index(f"#{k}", i.upper) for k, i in enumerate(operand.type.indices))
            names = operand.type.indices
            operand = self.graph.retype(operand, operand.type.with_indices(current))
        else:
            current, names = Indices(), Indices()
        result = self._derivative_of(operand, current, "#d", covariant, span)
        indices = Indices([Index("#d"), *current])
        if upper:
            result, indices = self._move_index(result, indices, 0, span)
        entries = [(item, upper)] + [
            (ast.IndexItem(name.name, span=span), name.upper) for name in names
        ]
        fixed = {axis + 1: value for axis, value in selection}
        return self._label(result, entries, span, fixed)

    def _derivative_of(
        self, operand: Node, indices: Indices, label: str, covariant: bool, span: Span
    ) -> Node:
        """The derivative of ``operand`` (with axes ``indices``) as a new first axis
        labelled ``label``; covariant derivatives add one connection term per index."""
        coordinates = self.session.require_coordinates(span, "a derivative")
        new = Index(label)
        own = _spaces(operand) or ()
        kind = TensorType(Indices([new, *indices]), (spacetime(coordinates), *own))
        result = self.graph.add(ops.Partial(coordinates), (operand,), kind, span)
        if not covariant or not indices:
            return result
        gamma = self._geometry("christoffel", span)
        terms: list[tuple[int, Node]] = [(1, result)]
        for position, index in enumerate(indices):
            dummy = Index("#e", index.upper)
            swapped = Indices(dummy if p == position else i for p, i in enumerate(indices))
            if index.upper:  # + Γ^{a}_{c e} T^{..e..}
                gamma_indices = Indices([index, new, dummy.flip()])
                sign = 1
            else:  # - Γ^{e}_{c a} T_{..e..}
                gamma_indices = Indices([dummy.flip(), new, index])
                sign = -1
            term = self._contract(
                [(gamma, gamma_indices), (operand, swapped)], span, output=kind.indices
            )
            terms.append((sign, term))
        signs = tuple(sign for sign, _ in terms)
        return self.graph.add(ops.Add(signs), [t for _, t in terms], kind, span)

    def _differential(self, operand: Node, span: Span) -> Node:
        """``d`` applied to a coordinate in a line element: ``d\\theta^2`` is ``(dθ)²``."""
        coordinates = self.session.coordinates or ()
        base, power = operand, None
        if isinstance(operand.op, ops.Pow):
            base, power = operand.args
        symbol = base.op.value if isinstance(base.op, ops.Const) else None
        if symbol in coordinates:
            result = self.graph.const(differential(symbol), SCALAR, span)
        elif isinstance(symbol, Symbol) and symbol.name == "Omega":
            result = self._solid_angle(coordinates, span)
        else:
            raise CompileError("in a line element, d must be followed by a coordinate", span=span)
        if power is not None:
            result = self.graph.add(ops.Pow(), (result, power), SCALAR, span)
        return result

    def _solid_angle(self, coordinates: Sequence[Symbol], span: Span) -> Node:
        r"""The solid angle element :math:`d\Omega = \sqrt{d\theta^2 + \sin^2\theta\, d\phi^2}`,
        for spherical coordinates."""
        names = {c.name: c for c in coordinates}
        if "theta" not in names or "phi" not in names:
            raise CompileError("d\\Omega needs coordinates named \\theta and \\phi", span=span)
        theta, phi = names["theta"], names["phi"]
        element = sympy.sqrt(
            differential(theta) ** 2 + sympy.sin(theta) ** 2 * differential(phi) ** 2
        )
        return self.graph.const(element, SCALAR, span)

    def _script_math(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Node:
        """Scripts on something that is not a tensor: subscripts name a symbol
        (``r_{s}`` is ``r_s``) and superscripts are powers."""
        base, scripts = node.base, list(node.scripts)
        if ctx.latex and isinstance(base, ast.Name) and _splittable(base):
            whole = self.term(base, scope, ctx)
            labels = split_index_labels(base.id)
            if (
                isinstance(whole, Node)
                and isinstance(whole.op, ops.Mul)
                and len(whole.args) == len(labels)
            ):
                # re^{x} is r e^{x}: in LaTeX a script binds to the last letter only, which
                # may be a tensor, as in Rg_{\mu\nu}
                last = _letter(labels[-1], base.span, first=False)
                rest = self.value(replace(node, base=last), scope, ctx)
                return self._multiply([*whole.args[:-1], rest], node.span)
        decorations = [s for s in scripts if s.kind == "^" and _decoration(s)]
        if decorations and isinstance(base, ast.Name):  # r^* is a name, not a power
            marks = "".join(s.body.id for s in decorations if isinstance(s.body, ast.Name))
            base = ast.Name(f"{base.id}^{marks}", verbatim=True, span=base.span)
            scripts = [s for s in scripts if s not in decorations]
        subscripts = [s for s in scripts if s.kind == "_"]
        powers = [s for s in scripts if s.kind == "^"]
        if subscripts:
            if not isinstance(base, ast.Name):
                raise CompileError("only a name can have a subscript", span=node.span)
            items = [i for s in subscripts if s.indices for i in s.indices.items]
            if len(items) > 1 and all(_plain_label(i) for i in items):
                written = _written(subscripts)
                if any(self._names_component(str(i.label)) for i in items):  # T_{tt}
                    written = "_{" + "".join(_EXAMPLE_LABELS[: len(items)]) + "}"
                raise CompileError(
                    f"{base.id} is not a tensor",
                    span=node.span,
                    hint=f"define it first, for example {base.id}{written} := [...]",
                )
            symbol_name = _subscripted_name(ast.Scripted(base, tuple(subscripts), span=node.span))
            value = self.value(ast.Name(symbol_name, verbatim=True, span=node.span), scope, ctx)
        else:
            value = self.value(base, scope, ctx)
        exponents = []
        for script in powers:
            if script.body is None or isinstance(script.body, ast.Equation):
                name = base.id if isinstance(base, ast.Name) else "this"
                raise CompileError(f"{name} is not a tensor", span=script.span)
            exponents.append(self.value(script.body, scope, ctx))
        if not exponents:
            return value
        exponent = exponents[-1]
        for previous in reversed(exponents[:-1]):  # x^a^b is x^(a^b)
            exponent = self._power(previous, exponent, node.span)
        return self._power(value, exponent, node.span)

    # -- abstract index notation -------------------------------------------------------
    #
    # An equation with a tensor that has no components, such as T_{\mu\nu} in Einstein's
    # equations before T is defined, is kept as notation (relativisticpy.abstract), and
    # solve, trace, simplify and subs work on it as notation. Where components are needed,
    # in a definition or a component equation, notation is turned into components with
    # the metric in force.

    def _value_or_notation(
        self, node: ast.Expr, scope: Scope, ctx: Context, *, definition: bool = False
    ) -> Node:
        """The value of an expression, or of a link of a chain, or what a name is defined
        as: notation if it can only be read as notation. Without a metric, an equation or
        a definition of tensors can only be."""
        equation = definition or isinstance(node, ast.Equation)
        if self._is_notation(node, scope, ctx, equation=equation):
            return self._notation(node, scope, ctx)
        return self.value(node, scope, ctx)

    def _is_notation(
        self,
        node: ast.Expr,
        scope: Scope,
        ctx: Context,
        *,
        equation: bool = False,
        operation: bool = False,
    ) -> bool:
        """Whether an expression is to be read as notation: it mentions a new tensor (a
        name with two or more indices and no components), or, without a metric, something
        kept as notation.

        In an ``equation`` or a definition, or the argument of an ``operation`` on
        notation such as ``simplify``, built-in tensors count too when there is no
        metric, and so do new tensors with one index when a contraction, as in
        u^\\alpha u_\\alpha, shows that index notation is in use; but only where the
        indices follow the rules of the notation and none is a coordinate, since
        \\Gamma^\\theta_{r\\theta} = 1/r is a component. A tensor written with a coordinate
        as an index never counts: ``G_{tt} = 8\\pi T_{tt}`` is about components, which T
        has not yet; nor does the tensor ``solve`` is asked to solve for. In an operation,
        something kept as notation always counts."""
        builtin = single = False
        labels: set[str] = set()
        for current in _used(node):
            if isinstance(current, ast.Name):
                binding = scope.lookup(current.id)
                notation = isinstance(binding, VariableBinding) and _notation_of(binding.node)
                # With a metric, notation is computed where it meets components.
                if notation and (operation or self.session.metric is None):
                    return True
            elif isinstance(current, ast.Scripted) and (pattern := _tensor_pattern(current)):
                written = self._letter_run(current, scope, ctx)
                if isinstance(written, ast.Binary):  # G T_{\mu\nu}: the tensor is the last
                    assert isinstance(written.right, ast.Scripted)
                    pattern = _tensor_pattern(written.right) or pattern
                name, entries = pattern
                if any(self._names_component(label) for label, _ in entries):
                    return False
                labels.update(label for label, _ in entries)
                if scope.lookup(name) is None and name not in BUILTINS and name not in CONSTANTS:
                    if self._builtin_tensor(name, len(entries)) is not None:
                        builtin = True
                    elif any(self._builtin_tensor(name, rank) for rank in range(5)):
                        pass  # a built-in with the wrong number of indices: an error
                    elif len(entries) >= 2:
                        return True
                    else:
                        single = True
        builtin = builtin and self.session.metric is None
        if not (equation or operation) or not (builtin or single):
            return False
        if any(c.name in labels for c in self.session.coordinates or ()):
            return False
        try:
            _, summed = self._index_structure(node, scope)
        except IndexNotationError:
            return False
        return builtin or summed

    def _index_structure(
        self, node: ast.Expr, scope: Scope
    ) -> tuple[frozenset[tuple[str, bool]] | None, bool]:
        """The free indices of notation, with their positions, read from its syntax, and
        whether any index is summed. The free indices are ``None`` where the syntax cannot
        tell them, as for the result of ``solve``.

        Raises :class:`IndexNotationError` where the indices break the rules: an index
        three times in one term, a summed index that is not once up and once down, a
        division by a tensor, or terms or sides whose free indices differ."""
        summed = False

        def contract(entries: list[tuple[str, bool]], span: Span) -> frozenset[tuple[str, bool]]:
            nonlocal summed
            free: set[tuple[str, bool]] = set()
            for label in dict.fromkeys(label for label, _ in entries):
                positions = [up for other, up in entries if other == label]
                text = latex_label(label)
                if len(positions) > 2:
                    raise IndexNotationError(
                        f"the index {text} appears {len(positions)} times in one term",
                        span=span,
                        hint="an index appears once, or twice to be summed",
                    )
                if len(positions) == 2 and positions[0] == positions[1]:
                    raise IndexNotationError(
                        f"the index {text} appears twice in one term, both times "
                        + ("up" if positions[0] else "down"),
                        span=span,
                        hint="a summed index is written once up and once down",
                    )
                if len(positions) == 2:
                    summed = True
                else:
                    free.add((label, positions[0]))
            return frozenset(free)

        def side(node: ast.Expr) -> frozenset[tuple[str, bool]] | None:
            if _zero_literal(node):  # zero has whatever indices the other side has
                return None
            found = uses(node)
            return None if found is None else contract(found, node.span)

        def uses(node: ast.Expr) -> list[tuple[str, bool]] | None:
            """Every index written in a term, summed or free; ``None`` if unknown."""
            node = _operand(node)
            if isinstance(node, ast.Equation):
                lhs, rhs = side(node.lhs), side(node.rhs)
                if lhs is not None and rhs is not None and lhs != rhs:
                    raise IndexNotationError(
                        "the two sides have different free indices: "
                        f"{_free_latex(lhs)} on the left, {_free_latex(rhs)} on the right",
                        span=node.span,
                    )
                either = lhs if lhs is not None else rhs
                return None if either is None else sorted(either)
            if isinstance(node, ast.Binary) and node.op in ("+", "-"):
                common: frozenset[tuple[str, bool]] | None = None
                for _, term in _signed_terms(node):
                    free = side(term)
                    if free is None:
                        continue
                    if common is not None and free != common:
                        raise IndexNotationError(
                            "the terms of a sum have different free indices: "
                            f"{_free_latex(common)} and {_free_latex(free)}",
                            span=node.span,
                        )
                    common = free
                return None if common is None else sorted(common)
            if isinstance(node, ast.Binary) and node.op == "*":
                entries: list[tuple[str, bool]] = []
                known = True
                for factor in _product_factors(node):
                    written = uses(factor)
                    if written is None:
                        known = False
                    else:
                        entries.extend(written)
                if not known:
                    contract(entries, node.span)  # what the known factors break, still
                    return None
                return entries
            if isinstance(node, ast.Binary) and node.op == "/":
                if side(node.right):
                    raise IndexNotationError("cannot divide by a tensor", span=node.right.span)
                return uses(node.left)
            if isinstance(node, ast.Unary) and node.op in ("-", "+", "±", "∓"):
                return uses(node.operand)
            if isinstance(node, ast.Scripted):
                if pattern := _tensor_pattern(node):
                    return pattern[1]
                items = [
                    (item, script.kind == "^")
                    for script in node.scripts
                    if script.indices is not None
                    for item in script.indices.items
                ]
                if any(item.derivative or item.opens or item.closes for item, _ in items):
                    return None  # derivatives and symmetrisations
                plain = [(str(item.label), up) for item, up in items if _plain_label(item)]
                if isinstance(node.base, ast.Operator):  # \partial_\mu, a factor of a product
                    return plain
                if isinstance(node.base, ast.Name) or not plain:  # a component, or a power
                    return None if side(node.base) else []
                return None
            if isinstance(node, ast.Name):
                binding = scope.lookup(node.id)
                if isinstance(binding, VariableBinding) and (kept := _notation_of(binding.node)):
                    if isinstance(kept, IndexEquation):
                        return None
                    return [(index.name, index.is_up) for index in abstract.free(kept.value)]
                return []
            if isinstance(node, ast.Call):
                # An operator applied, as in \nabla_\mu(...), is a derivative.
                if not isinstance(node.func, ast.Name):
                    return None
                arguments = [side(argument) for argument in node.args]
                if node.func.id in _NOTATION_OPERATIONS or any(a is None or a for a in arguments):
                    return None  # a function of tensors, or of what cannot be told
                return []
            if isinstance(node, ast.Number | ast.Accent | ast.Root | ast.Abs | ast.Factorial):
                return []
            return None

        return side(node), summed

    def _notation(self, node: ast.Expr, scope: Scope, ctx: Context) -> Node:
        return self.graph.const(self._abstract(node, scope, ctx), VALUE, node.span)

    def _abstract(
        self, node: ast.Expr, scope: Scope, ctx: Context
    ) -> IndexEquation | IndexExpression:
        node = _operand(node)
        space = self.session.algebra()
        if isinstance(node, ast.Name):
            binding = scope.lookup(node.id)
            if isinstance(binding, VariableBinding) and (found := _notation_of(binding.node)):
                return found
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _NOTATION_OPERATIONS
        ):
            return self._notation_operation(node, scope, ctx)
        if isinstance(node, ast.Chain):
            raise CompileError("give the links of a chain one at a time", span=node.span)
        self._index_structure(node, scope)  # what breaks the rules, before SymPy sees it
        if isinstance(node, ast.Equation):
            lhs = self._tensorial(node.lhs, scope, ctx)
            return IndexEquation(lhs, self._tensorial(node.rhs, scope, ctx), space)
        return IndexExpression(self._tensorial(node, scope, ctx), space)

    def _notation_operation(
        self, node: ast.Call, scope: Scope, ctx: Context
    ) -> IndexEquation | IndexExpression:
        assert isinstance(node.func, ast.Name)
        name, args, space = node.func.id, node.args, self.session.algebra()
        if not args:
            raise CompileError(f"{name} needs an equation", span=node.span)
        try:
            subject = self._abstract(args[0], scope, ctx)
            if name == "solve":
                if len(args) != 2 or not isinstance(subject, IndexEquation):
                    raise CompileError(
                        "solve takes an equation and the tensor to solve it for",
                        span=node.span,
                        hint="for example solve(E, T_{\\mu\\nu})",
                    )
                self._index_structure(args[1], scope)
                target = self._tensorial(args[1], scope, ctx)
                if not isinstance(target, symbolic.Tensor):
                    raise CompileError(
                        "give the tensor to solve for with its indices, as in T_{\\mu\\nu}",
                        span=args[1].span,
                    )
                return IndexEquation(
                    target, abstract.solve(subject.lhs, subject.rhs, target, space), space
                )
            if name in ("LHS", "RHS"):
                if not isinstance(subject, IndexEquation):
                    raise CompileError(f"{name} takes an equation", span=node.span)
                return IndexExpression(subject.lhs if name == "LHS" else subject.rhs, space)
            transform: Callable[[Any], Any]
            if name == "subs":
                rule = self._abstract(args[1], scope, ctx) if len(args) == 2 else None
                if not isinstance(rule, IndexEquation):
                    raise CompileError(
                        "subs takes an expression and a rule",
                        span=node.span,
                        hint="for example subs(E, u^\\alpha u_\\alpha = -1)",
                    )

                def transform(value: Any) -> Any:
                    assert isinstance(rule, IndexEquation)
                    return abstract.substitute(value, rule.lhs, rule.rhs, space)

            elif name == "trace":

                def transform(value: Any) -> Any:
                    return abstract.trace(value, space)

            else:

                def transform(value: Any) -> Any:
                    return abstract.canonical(value, space)

            if isinstance(subject, IndexEquation):
                return IndexEquation(transform(subject.lhs), transform(subject.rhs), space)
            return IndexExpression(transform(subject.value), space)
        except CompileError as error:
            if error.span is None:
                error.span = node.span
            raise

    def _tensorial(self, node: ast.Expr, scope: Scope, ctx: Context) -> Any:
        """An expression read as abstract index notation: a SymPy tensor expression."""
        node = _operand(node)
        space = self.session.algebra()
        if isinstance(node, ast.Binary) and node.op in ("+", "-"):
            total: Any = sympy.S.Zero
            for sign, term in _signed_terms(node):
                value = self._tensorial(term, scope, ctx)
                value = value if sign > 0 else -value
                try:
                    total = value if total == 0 else total + value
                except ValueError as error:  # SymPy's own check of free indices
                    raise IndexNotationError(
                        f"the free indices of a sum must agree: {error}", span=node.span
                    ) from None
            return total
        if isinstance(node, ast.Binary) and node.op == "*":
            product: Any = sympy.S.One
            for factor in _product_factors(node):
                try:
                    product = product * self._tensorial(factor, scope, ctx)
                except ValueError as error:
                    raise IndexNotationError(str(error), span=node.span) from None
            return product
        if isinstance(node, ast.Binary) and node.op == "/":
            denominator = self._tensorial(node.right, scope, ctx)
            if isinstance(denominator, symbolic.TensExpr):
                raise IndexNotationError("cannot divide by a tensor", span=node.right.span)
            return self._tensorial(node.left, scope, ctx) / denominator
        if isinstance(node, ast.Unary) and node.op in ("-", "+"):
            value = self._tensorial(node.operand, scope, ctx)
            return -value if node.op == "-" else value
        if (isinstance(node, ast.Scripted) and isinstance(node.base, ast.Operator)) or (
            isinstance(node, ast.Derivative | ast.Differential) and _mentions_tensor(node)
        ):
            raise CompileError(
                "derivatives are not yet part of tensor notation; they work on components",
                span=node.span,
            )
        if isinstance(node, ast.Scripted) and (pattern := _tensor_pattern(node)):
            written = self._letter_run(node, scope, ctx)
            if written is not node:
                return self._tensorial(written, scope, ctx)
            return self._abstract_tensor(pattern, scope, node.span)
        if isinstance(node, ast.Name):
            binding = scope.lookup(node.id)
            if isinstance(binding, VariableBinding) and (found := _notation_of(binding.node)):
                if isinstance(found, IndexEquation):
                    raise CompileError(f"{node.id} is an equation", span=node.span)
                return found.value
            if binding is None and node.id == "R":
                return space.ricci_scalar
            if isinstance(binding, TensorBinding):  # a tensor without indices: its trace
                return Symbol(node.id)
        if isinstance(node, ast.Equation):
            raise CompileError("an equation cannot be part of an expression", span=node.span)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in _NOTATION_OPERATIONS:
                found = self._notation_operation(node, scope, ctx)
                if isinstance(found, IndexEquation):
                    raise CompileError(
                        "an equation cannot be part of an expression", span=node.span
                    )
                return found.value
            function = isinstance(scope.lookup(name), FunctionBinding) or _names_builtin(node.func)
            if not function and len(node.args) == 1 and _mentions_tensor(node):
                # \kappa(T_{\mu\nu} - ...), as typeset: a product
                times = ast.Binary("*", node.func, node.args[0], implicit=True, span=node.span)
                return self._tensorial(times, scope, ctx)
        if _mentions_tensor(node):  # not a scalar, though nothing else here reads it
            raise IndexNotationError(
                "tensor notation reads sums and products of tensors, and not yet this",
                span=node.span,
            )
        value = self.value(node, scope, ctx)
        if isinstance(value.type, TensorType):
            raise IndexNotationError(
                "in tensor notation, write a tensor with its indices", span=node.span
            )
        scalar = sympy.sympify(self.session.evaluate(value))
        if not isinstance(scalar, sympy.Expr):
            raise IndexNotationError(
                "in tensor notation, each factor is a scalar or a tensor with indices",
                span=node.span,
            )
        return scalar

    def _letter_run(self, node: ast.Scripted, scope: Scope, ctx: Context) -> ast.Expr:
        """In LaTeX, ``8\\pi GT_{\\mu\\nu}`` is G times T_{\\mu\\nu}: the indices on a run of
        letters that names nothing belong to its last letter, as a script does."""
        base = node.base
        if not (ctx.latex and isinstance(base, ast.Name) and _splittable(base)):
            return node
        name = base.id
        known = name in BUILTINS or name in CONSTANTS or scope.lookup(name) is not None
        if known or any(self._builtin_tensor(name, rank) for rank in range(5)):
            return node
        letters = split_index_labels(name)
        if len(letters) < 2:
            return node
        factors: list[ast.Expr] = [
            _letter(letter, base.span, first=n == 0) for n, letter in enumerate(letters[:-1])
        ]
        factors.append(replace(node, base=_letter(letters[-1], base.span, first=False)))
        return reduce(lambda a, b: ast.Binary("*", a, b, implicit=True, span=node.span), factors)

    def _abstract_tensor(
        self, pattern: tuple[str, list[tuple[str, bool]]], scope: Scope, span: Span
    ) -> Any:
        name, entries = pattern
        space = self.session.algebra()
        binding = scope.lookup(name)
        if isinstance(binding, VariableBinding) and _notation_of(binding.node):
            raise IndexNotationError(
                f"{name} stands for notation, whose indices are its own",
                span=span,
                hint=f"write it without indices, as in trace({name})",
            )
        quantity = None
        if not isinstance(binding, TensorBinding):
            quantity = self._builtin_tensor(name, len(entries))
        head = space.head(_NOTATION_NAMES.get(quantity or "", name), len(entries), quantity)
        try:
            return head(*(space.index(label, upper) for label, upper in entries))
        except ValueError as error:
            raise IndexNotationError(str(error), span=span) from None

    def _components_of(self, node: Node, scope: Scope, ctx: Context, span: Span) -> Node:
        """A value kept as notation, turned into components with the metric in force."""
        found = _notation_of(node)
        if found is None:
            return node
        if isinstance(found, IndexEquation):
            raise CompileError(
                "an equation in index notation has no components",
                span=span,
                hint="take one side of it, as in RHS(E)",
            )
        return self._lower(found.value, scope, ctx, span)

    def _lower(self, value: Any, scope: Scope, ctx: Context, span: Span) -> Node:
        """Turn notation into components: write each term in the language, with its
        scalar coefficient bound to a name, and compile it."""
        space = self.session.algebra()
        traces = space.traces()
        inner = scope.child()
        fresh = itertools.count()
        summands: list[ast.Expr] = []
        for coefficient, tensors in abstract.terms(value):
            present = [symbol for symbol in traces if coefficient.has(symbol)]
            try:
                monomials = (
                    sympy.Poly(coefficient, *present).terms() if present else [((), coefficient)]
                )
            except sympy.PolynomialError:
                raise CompileError(
                    "cannot turn this into components: a trace appears other than as a factor",
                    span=span,
                ) from None
            for powers, scalar in monomials:
                name = f"#c{next(fresh)}"
                inner.bind(name, VariableBinding(self.graph.const(sympy.sympify(scalar), SCALAR)))
                factors: list[ast.Expr] = [ast.Name(name, verbatim=True, span=span)]
                for symbol, power in zip(present, powers, strict=True):
                    for _ in range(power):
                        factors.append(_trace_syntax(symbol, traces[symbol], fresh, span))
                labels: dict[str, str] = {}
                for tensor in tensors:
                    factors.append(_tensor_syntax(tensor, space, labels, fresh, span))
                summands.append(
                    reduce(lambda a, b: ast.Binary("*", a, b, implicit=True, span=span), factors)
                )
        if not summands:
            return self.graph.const(sympy.S.Zero, SCALAR, span)
        total = reduce(lambda a, b: ast.Binary("+", a, b, span=span), summands)
        return self.value(total, inner, replace(ctx, latex=False))

    # -- helpers -------------------------------------------------------------

    def _call(
        self, func: Callable[..., Any], name: str, args: Sequence[Node], kind: Type, span: Span
    ) -> Node:
        return self.graph.add(ops.Call(func, name), list(args), kind, span)


# ---------------------------------------------------------------------------
# Kernels used by Call and Map nodes (module-level, so worker processes can use them)
# ---------------------------------------------------------------------------


def _list(*items: Any) -> list[Any]:
    return list(items)


def _tuple(*items: Any) -> tuple[Any, ...]:
    return tuple(items)


def _limits(variable: Any, lower: Any, upper: Any) -> tuple[Any, Any, Any]:
    return (variable, lower, upper)


def _summation(body: Any, variable: Any, lower: Any, upper: Any) -> Any:
    return sympy.Sum(body, (variable, lower, upper))


def _product(body: Any, variable: Any, lower: Any, upper: Any) -> Any:
    return sympy.Product(body, (variable, lower, upper))


def _limit(body: Any, variable: Any, point: Any, direction: str) -> Any:
    if direction == "+-" and point in (sympy.oo, -sympy.oo):
        direction = "-" if point == sympy.oo else "+"
    return sympy.limit(body, variable, point, direction)


def _differentiate(expr: Any, orders: tuple[int, ...], *variables: Any) -> Any:
    return sympy.diff(expr, *((v, n) for v, n in zip(variables, orders, strict=True)))


def _derivative_at(body: Any, dummy: Any, order: int, point: Any) -> Any:
    return sympy.diff(body, dummy, order).subs(dummy, point)


def _gradient(expr: Any, vector: Any) -> Any:
    return sympy.ImmutableDenseNDimArray([sympy.diff(expr, component) for component in vector])


def _substitute(expr: Any, *pairs: Any) -> Any:
    """An evaluation bar, ``x - y |_{x=y, y=x}``: every substitution at once, as a
    change of variables is, rather than one after another."""
    rules = list(zip(pairs[::2], pairs[1::2], strict=True))
    return sympy.sympify(expr).subs(rules, simultaneous=True)


def _piecewise(*args: Any) -> Any:
    return sympy.Piecewise(*zip(args[::2], args[1::2], strict=True))


def _time_derivative(expr: Any, order: int) -> Any:
    r"""``\dot{x}``: the derivative with respect to the single variable that the
    functions in ``x`` depend on; a plain symbol ``a`` gives the symbol ``adot``."""
    expr = sympy.sympify(expr)
    applied = expr.atoms(sympy.core.function.AppliedUndef)
    parameters = {argument for function in applied for argument in function.args}
    if len(parameters) == 1:
        return sympy.diff(expr, parameters.pop(), order)
    if isinstance(expr, Symbol):
        return sympy.Symbol(expr.name + ("dot" if order == 1 else "ddot"))
    raise ValueError(
        f"cannot tell what {expr} changes with: write it as a function of one variable, e.g. a(t)"
    )


@dataclass(frozen=True)
class _TensorFactory:
    """Wraps components in a Tensor (picklable, for worker processes)."""

    indices: Indices
    coordinates: tuple[Symbol, ...]

    def __call__(self, components: Any) -> Tensor:
        return Tensor(components, _public(self.indices), self.coordinates)


@dataclass(frozen=True)
class _EquationFactory:
    indices: Indices
    coordinates: tuple[Symbol, ...]

    def __call__(self, equation: TensorEquation) -> TensorEquation:
        return TensorEquation(equation.lhs, equation.rhs, _public(self.indices), self.coordinates)


# ---------------------------------------------------------------------------
# Syntax helpers
# ---------------------------------------------------------------------------


def _public(indices: Indices) -> Indices:
    return Indices(Index(_display(i.name), i.upper) for i in indices)


def _display(name: str) -> str:
    return name.lstrip("#")


def _indices(node: Node) -> Indices:
    assert isinstance(node.type, TensorType)
    return node.type.indices


def _is_zero(node: Node) -> bool:
    return isinstance(node.op, ops.Const) and node.op.value == 0


def _unsigned(factor: ast.Expr) -> list[ast.Expr]:
    """Take the sign off a factor, so that ``-\\partial_\\mu \\phi`` and ``-d t^2`` let
    the operator act on what follows: ``-x`` becomes ``-1``, ``x``."""
    if isinstance(factor, ast.Unary) and factor.op == "-":
        return [ast.Number("-1", span=factor.span), *_unsigned(factor.operand)]
    return [factor]


def _operand(node: ast.Expr) -> ast.Expr:
    """``[x + 1]`` used as an operand is a grouping, as LaTeX's brackets are."""
    if isinstance(node, ast.List) and len(node.items) == 1:
        return node.items[0]
    return node


def _plain_label(item: ast.IndexItem) -> bool:
    return (
        item.label is not None
        and item.value is None
        and item.derivative is None
        and item.opens is None
        and item.closes is None
    )


def _rank(node: ast.Scripted) -> int | None:
    """The number of index items, or ``None`` if a script cannot be read as indices."""
    count = 0
    for script in node.scripts:
        if script.indices is None:
            return None
        count += sum(1 for item in script.indices.items if item.derivative is None)
    return count


def _has_derivative(node: ast.Scripted) -> bool:
    """Whether an index is written after a comma or a semicolon, as in ``\\phi_{;\\mu}``, in
    a script that is nothing but indices: ``e^{2\\alpha(t,r)}`` is a power."""
    return any(
        item.derivative is not None
        for script in node.scripts
        if script.indices is not None and script.body is None
        for item in script.indices.items
    )


def _is_line_element(target: ast.Expr) -> bool:
    """``ds^2`` (also ``ds^{2}`` and ``d s^2``)."""

    def squared(node: ast.Expr, name: str) -> bool:
        return (
            isinstance(node, ast.Scripted)
            and isinstance(node.base, ast.Name)
            and node.base.id == name
            and len(node.scripts) == 1
            and node.scripts[0].kind == "^"
            and isinstance(node.scripts[0].body, ast.Number)
            and node.scripts[0].body.text == "2"
        )

    if squared(target, "ds"):
        return True
    return (
        isinstance(target, ast.Binary)
        and target.op == "*"
        and isinstance(target.left, ast.Name)
        and target.left.id == "d"
        and squared(target.right, "s")
    )


def _splittable(node: ast.Name) -> bool:
    name = node.id
    return not node.verbatim and len(name) > 1 and name.isalpha() and name not in GREEK


def _letter(label: str, span: Span, *, first: bool) -> ast.Name:
    """One letter, or Greek name, of a run of letters read as a product, as G is of GM.
    No letter is split again. A d is the letter d, as in ad - bc, except first in its run,
    where it is the d of a differential, as in ds or dx, as {\\rm d} is."""
    return ast.Name(label, verbatim=first and label == "d", span=span)


def _written_as_letter(node: ast.Name) -> bool:
    """Whether a name is a Greek letter written as one, ``\\gamma`` or ``γ``, rather than
    spelled out as a function's name is, as in ``gamma(x)``."""
    if node.id not in GREEK:
        return False
    written = node.span.text
    return written == "\\" + node.id or len(written) == 1  # the command, or the character


def _names_builtin(node: ast.Name) -> bool:
    """Whether a name, bound to nothing in the program, is a builtin function: never a
    Greek letter written as one, so that \\gamma(r) in a metric is an unknown function."""
    return node.id in BUILTINS and not _written_as_letter(node)


def _names(value: ast.Expr, what: str) -> list[str]:
    """The names listed in ``[t, r, \\theta]``, a one-row matrix, or a single name."""
    if isinstance(value, ast.Name):
        return [value.id]
    if isinstance(value, ast.List | ast.Tuple):
        items: Sequence[ast.Expr] = value.items
    elif isinstance(value, ast.Matrix) and (len(value.rows) == 1 or len(value.rows[0]) == 1):
        items = [cell for row in value.rows for cell in row]
    else:
        raise CompileError(f"{what} lists names, as in [t, r, \\theta, \\phi]", span=value.span)
    names = []
    for item in items:
        if not isinstance(item, ast.Name):
            raise CompileError("expected a name", span=item.span)
        names.append(item.id)
    return names


def _indexed_target(name: str, value: Node) -> tuple[str, Indices] | None:
    """``v_a := ...`` with a value whose free indices are exactly ``a``: the tensor
    ``v_{a}``, with its indices where the value has them."""
    base, underscore, tail = name.partition("_")
    if not (underscore and tail.isalpha() and isinstance(value.type, TensorType)):
        return None
    labels = split_index_labels(tail)
    have = {index.name: index for index in value.type.indices}
    if sorted(labels) != sorted(have):
        return None
    return base, Indices(have[label] for label in labels)


def _indexed_name(node: ast.Name, scope: Scope) -> ast.Scripted | None:
    """``y_t`` as ``y_{t}`` when ``y`` is a tensor and ``y_t`` names nothing: data
    notation writes the indices of a tensor without braces."""
    base, underscore, tail = node.id.partition("_")
    binding = scope.lookup(base) if underscore and tail.isalpha() else None
    if not (
        isinstance(binding, TensorBinding)
        or (isinstance(binding, VariableBinding) and isinstance(binding.node.type, TensorType))
    ):
        return None
    span = node.span
    items = tuple(ast.IndexItem(label, span=span) for label in split_index_labels(tail))
    script = ast.Script("_", None, ast.IndexList(items, span=span), span=span)
    return ast.Scripted(ast.Name(base, span=span), (script,), span=span)


#: Functions that are linear algebra when given a tensor.
_LINEAR_ALGEBRA = frozenset(
    {"eig", "spec", "svd", "det", "tr", "trace", "inv", "diag", "exp", "sqrt", "log"}
)


@dataclass(frozen=True)
class _matrix_call:
    """``func(components, rows)``: picklable, for worker processes."""

    func: Callable[..., Any]

    def __call__(self, components: Any, rows: int) -> Any:
        return self.func(components, rows)


def _matrix_function(name: str, components: Any, rows: int) -> Any:
    return linalg.function(name, components, rows)


def _is_minus_one(node: ast.Expr | None) -> bool:
    return (
        isinstance(node, ast.Unary)
        and node.op == "-"
        and isinstance(node.operand, ast.Number)
        and node.operand.text == "1"
    )


def _shares(one: Node, other: Node) -> bool:
    """Whether two tensors have an index label in common."""
    return bool(set(_indices(one).names) & set(_indices(other).names))


def _identity(dim: int) -> Any:
    """The identity matrix: SymPy's for a few dimensions, NumPy's for many."""
    if dim <= _SMALL:
        return sympy.ImmutableDenseNDimArray(sympy.eye(dim).tolist())
    from relativisticpy.tensors import numerical

    return numerical.numpy().eye(dim)


def _ones(dim: int) -> Any:
    if dim <= _SMALL:
        return sympy.ImmutableDenseNDimArray([1] * dim)
    from relativisticpy.tensors import numerical

    return numerical.numpy().ones(dim)


def _uniform(dim: int) -> Any:
    """The uniform measure: weight 1/n on each of n elements."""
    if dim <= _SMALL:
        return sympy.ImmutableDenseNDimArray([sympy.Rational(1, dim)] * dim)
    from relativisticpy.tensors import numerical

    return numerical.numpy().full(dim, 1.0 / dim)


def _ledoit_wolf(samples: Any) -> Any:
    """The Ledoit–Wolf covariance of samples in rows, from scikit-learn."""
    from relativisticpy.tensors import numerical

    try:
        from sklearn.covariance import LedoitWolf
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            'the Ledoit–Wolf covariance needs scikit-learn: pip install "relativisticpy[data]"'
        ) from error
    data = numerical.dense(numerical.as_array(samples))
    return LedoitWolf(assume_centered=False).fit(data).covariance_


#: The statistics written with the index they are over, as \mathbb{E}_t.
_STATISTICS = {
    "𝔼": "expectation",
    "Var": "variance",
    "var": "variance",
    "Cov": "covariance",
    "cov": "covariance",
    "LedoitWolf": "ledoit-wolf",
}


def _statistic(node: ast.Expr) -> tuple[str, str] | None:
    """``\\mathbb{E}_t``, ``\\operatorname{Var}_t``, ...: the statistic and its index."""
    if not (
        isinstance(node, ast.Scripted)
        and isinstance(node.base, ast.Name)
        and node.base.id in _STATISTICS
        and len(node.scripts) == 1
        and node.scripts[0].kind == "_"
        and node.scripts[0].indices is not None
        and len(node.scripts[0].indices.items) == 1
        and _plain_label(node.scripts[0].indices.items[0])
    ):
        return None
    return _STATISTICS[node.base.id], str(node.scripts[0].indices.items[0].label)


# Spaces up to this size get SymPy's identity and ones, which stay exact with symbols.
_SMALL = 16


def _tensor_pattern_of_one(node: ast.Scripted) -> tuple[str, bool] | None:
    """``𝟙_a`` or ``1_a``: the ones covector, with its single index label."""
    base = node.base
    ones = (isinstance(base, ast.Number) and base.text == "1") or (
        isinstance(base, ast.Name) and base.id == "𝟙"
    )
    if not ones or len(node.scripts) != 1:
        return None
    script = node.scripts[0]
    items = script.indices.items if script.indices is not None else ()
    if len(items) != 1 or not _plain_label(items[0]):
        return None
    return str(items[0].label), script.kind == "^"


def _spaces(node: Node) -> tuple[Space, ...] | None:
    """The spaces of a tensor's axes; ``None`` for anything else."""
    return node.type.spaces if isinstance(node.type, TensorType) else None


def _walk(node: ast.Node) -> Iterator[ast.Node]:
    """Every node of a syntax tree, without recursion, for expressions of any length."""
    pending: list[object] = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, tuple):
            pending.extend(current)
        elif isinstance(current, ast.Node):
            yield current
            pending.extend(vars(current).values())


def _used(node: ast.Node) -> Iterator[ast.Node]:
    """Every node of an expression as :func:`_walk` gives them, except the tensor that
    ``solve(E, T_{\\mu\\nu})`` is asked to solve for: that names the unknown, and is not a
    tensor the expression uses."""
    pending: list[object] = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, tuple):
            pending.extend(current)
        elif isinstance(current, ast.Node):
            yield current
            solving = (
                isinstance(current, ast.Call)
                and isinstance(current.func, ast.Name)
                and current.func.id == "solve"
                and len(current.args) == 2
            )
            if isinstance(current, ast.Call) and solving:
                pending.extend((current.func, current.args[0]))
            else:
                pending.extend(vars(current).values())


def _mentioned(value: ast.Expr, *, latex: bool) -> frozenset[str]:
    """Every name written in an expression and, in LaTeX, every letter of a run of
    letters read as a product, as GM is in \\frac{2GM}{r}. The name of a builtin, such as
    FLRW in FLRW(k) or diag, is one name and never its letters."""
    found: set[str] = set()
    for node in _walk(value):
        if isinstance(node, ast.Name):
            found.add(node.id)
            builtin = node.id in BUILTINS or node.id in CONSTANTS
            if latex and _splittable(node) and not builtin:
                found.update(split_index_labels(node.id))
    return frozenset(found)


def _is_literal(value: ast.Expr) -> bool:
    return isinstance(value, ast.List | ast.Matrix)


def _literal_cells(value: ast.Expr, rank: int) -> tuple[tuple[int, ...], list[ast.Expr]] | None:
    """The shape and entries of a nested list or matrix literal."""
    if isinstance(value, ast.Matrix):
        rows = value.rows
        if rank == 1 and (len(rows) == 1 or all(len(row) == 1 for row in rows)):
            cells = [cell for row in rows for cell in row]
            return (len(cells),), cells
        if rank != 2:
            return None
        return (len(rows), len(rows[0])), [cell for row in rows for cell in row]

    def shape_of(node: ast.Expr, depth: int) -> tuple[tuple[int, ...], list[ast.Expr]] | None:
        if depth == 0:
            return ((), [node]) if not isinstance(node, ast.List) else None
        if not isinstance(node, ast.List) or not node.items:
            return None
        parts = [shape_of(item, depth - 1) for item in node.items]
        if any(part is None for part in parts):
            return None
        shapes = {part[0] for part in parts if part is not None}
        if len(shapes) != 1:
            return None
        cells = [cell for part in parts if part is not None for cell in part[1]]
        return (len(node.items), *shapes.pop()), cells

    return shape_of(value, rank)


def _shape_text(shape: tuple[int, ...]) -> str:
    return "×".join(str(n) for n in shape) if shape else "scalar"


def _subscripted_symbol(target: ast.Scripted) -> bool:
    """``r_{s}``: a single subscript, which may name a symbol."""
    return (
        len(target.scripts) == 1
        and target.scripts[0].kind == "_"
        and target.scripts[0].indices is not None
        and len(target.scripts[0].indices.items) == 1
    )


def _subscripted_name(node: ast.Scripted) -> str:
    """The symbol a subscript names: ``r_{s}`` is ``r_s`` and ``x_{0}`` is ``x_0``."""
    assert isinstance(node.base, ast.Name)
    parts = []
    for script in node.scripts:
        if script.indices is not None:
            parts.append(
                "".join(
                    item.label if item.label is not None else to_text(item.value)
                    for item in script.indices.items
                )
            )
        else:
            parts.append(script.span.text.lstrip("_").strip("{}"))
    return f"{node.base.id}_{''.join(parts)}"


def to_text(node: ast.Expr | None) -> str:
    return (
        node.text
        if isinstance(node, ast.Number)
        else (node.id if isinstance(node, ast.Name) else "")
    )


def _written(scripts: Sequence[ast.Script]) -> str:
    return "".join(s.span.text for s in scripts)


# Index labels for examples in messages, where the labels written name components.
_EXAMPLE_LABELS = ("\\mu", "\\nu", "\\rho", "\\sigma", "\\lambda", "\\kappa")


def _bracket_groups(
    factors: Sequence[Sequence[tuple[ast.IndexItem, bool]]], span: Span, *, closed: bool
) -> list[tuple[str, list[str], bool]]:
    """Bracketed groups of index labels across the index items of ``factors``, each with
    whether it spans more than one factor; with ``closed``, every bracket must close. A
    bracket takes the indices at the level it opens at, so ``T_{(a}{}^{b}{}_{c)}``
    symmetrises a and c alone, and one that closes at the other level means nothing."""
    groups: list[tuple[str, list[str], bool]] = []
    current: tuple[str, list[str], int, bool] | None = None
    for number, entries in enumerate(factors):
        for item, upper in entries:
            if item.opens is not None:
                current = (item.opens, [], number, upper)
            if current is not None and item.label is not None and upper == current[3]:
                current[1].append(item.label)
            if item.closes is not None and current is not None:
                if upper != current[3]:
                    opened, closes = ("upper", "lower") if current[3] else ("lower", "upper")
                    raise IndexNotationError(
                        f"this (anti)symmetrisation opens on {opened} indices and closes on "
                        f"{closes} ones; it is over indices at one level",
                        span=span,
                    )
                groups.append((current[0], current[1], current[2] != number))
                current = None
    if closed and current is not None:
        raise IndexNotationError("an (anti)symmetrisation bracket is never closed", span=span)
    return groups


def _decoration(script: ast.Script) -> bool:
    """A script that marks a name rather than raising it to a power: ``r^*``, ``i^+``."""
    body = script.body
    return isinstance(body, ast.Name) and body.verbatim and body.id in ("*", "+", "-", "±", "∓")


_DIFFERENTIAL_NAME = re.compile(
    r"d(" + "|".join(sorted(GREEK, key=len, reverse=True)) + r"|[A-Za-z][0-9]*)"
)


def _line_element_coordinates(value: ast.Expr) -> list[str]:
    r"""The coordinates a line element's differentials name, in the order they appear:
    ``-dt^2 + dr^2 + r^2 d\Omega^2`` names t, r, θ and φ. Empty if a differential is
    abstract, as in ``g_{\mu\nu} dx^\mu dx^\nu``, or if there are none."""
    atoms: list[ast.Expr] = []
    pending: list[ast.Expr] = [value]
    while pending:  # left to right, without recursion
        node = pending.pop()
        if isinstance(node, ast.Binary):
            pending.append(node.right)
            pending.append(node.left)
        elif isinstance(node, ast.Unary):
            pending.append(node.operand)
        else:
            atoms.append(node)
    names: list[str] = []
    for atom, following in zip(atoms, [*atoms[1:], None], strict=True):
        base = atom.base if isinstance(atom, ast.Scripted) else atom
        if isinstance(base, ast.Name) and base.id == "d" and following is not None:
            target = following
        elif isinstance(base, ast.Name) and (match := _DIFFERENTIAL_NAME.fullmatch(base.id)):
            target = ast.Name(match.group(1), span=base.span)
        else:
            continue
        if isinstance(target, ast.Scripted):
            if not all(isinstance(s.body, ast.Number) for s in target.scripts):
                return []  # dx^\mu: an abstract differential
            target = target.base
        if not isinstance(target, ast.Name):
            continue
        found = ["theta", "phi"] if target.id == "Omega" else [target.id]
        names.extend(name for name in found if name not in names)
    return names


def _names_omega(node: ast.Expr) -> bool:
    """``\\Omega`` or ``\\Omega^2``, as after the ``d`` of a solid angle."""
    base = node.base if isinstance(node, ast.Scripted) else node
    return isinstance(base, ast.Name) and base.id == "Omega"


# -- abstract index notation ---------------------------------------------------------

_NOTATION_OPERATIONS = frozenset({"solve", "trace", "simplify", "subs", "LHS", "RHS"})
# The names built-in tensors have in notation, whatever they were written as.
_NOTATION_NAMES = {
    "riemann": "R", "ricci": "R", "einstein": "G", "weyl": "C", "christoffel": "Gamma",
}  # fmt: skip


def _mentions_tensor(node: ast.Expr) -> bool:
    """Whether a tensor with indices is written anywhere in an expression."""
    return any(
        isinstance(current, ast.Scripted) and _tensor_pattern(current) for current in _walk(node)
    )


def _zero_literal(node: ast.Expr) -> bool:
    """A written 0, which has whatever free indices the other side of an equation has."""
    node = _operand(node)
    return isinstance(node, ast.Number) and float(node.text) == 0


def _free_latex(free: frozenset[tuple[str, bool]]) -> str:
    """Free indices as scripts, ``^{\\mu}_{\\nu}``, or ``none``."""
    ordered = sorted(free)
    upper = abstract.latex_labels([label for label, up in ordered if up])
    lower = abstract.latex_labels([label for label, up in ordered if not up])
    written = (f"^{{{upper}}}" if upper else "") + (f"_{{{lower}}}" if lower else "")
    return written or "none"


def _notation_of(node: Node) -> IndexEquation | IndexExpression | None:
    """The notation a node holds, if it is a value kept as notation."""
    if isinstance(node.op, ops.Const) and isinstance(
        node.op.value, IndexEquation | IndexExpression
    ):
        return node.op.value
    return None


def _tensor_pattern(node: ast.Scripted) -> tuple[str, list[tuple[str, bool]]] | None:
    """A name with index labels, as a tensor is written: ``T^{\\mu}{}_{\\nu}``."""
    if not isinstance(node.base, ast.Name):
        return None
    entries = []
    for script in node.scripts:
        if script.indices is None or not script.indices.items:
            return None
        for item in script.indices.items:
            if not _plain_label(item) or item.label == "⋯":
                return None
            entries.append((str(item.label), script.kind == "^"))
    return node.base.id, entries


def _signed_terms(node: ast.Expr) -> list[tuple[int, ast.Expr]]:
    """The terms of a sum, with their signs, in order and without recursion."""
    found: list[tuple[int, ast.Expr]] = []
    pending: list[tuple[int, ast.Expr]] = [(1, node)]
    while pending:
        sign, expr = pending.pop()
        if isinstance(expr, ast.Binary) and expr.op in ("+", "-"):
            pending.append((sign if expr.op == "+" else -sign, expr.right))
            pending.append((sign, expr.left))
        else:
            found.append((sign, expr))
    return found


def _product_factors(node: ast.Expr) -> list[ast.Expr]:
    """The factors of a product, in order and without recursion."""
    found: list[ast.Expr] = []
    pending = [node]
    while pending:
        expr = pending.pop()
        if isinstance(expr, ast.Binary) and expr.op == "*":
            pending.append(expr.right)
            pending.append(expr.left)
        else:
            found.append(expr)
    return found


def _tensor_syntax(
    tensor: Any, space: Any, labels: dict[str, str], fresh: Iterator[int], span: Span
) -> ast.Scripted:
    """A tensor of notation, written as the language writes it, dummies renamed."""
    name = tensor.head.name
    if tensor.head == space.delta:
        name = "delta"
    scripts: list[ast.Script] = []
    run: list[ast.IndexItem] = []
    position: Literal["^", "_"] | None = None
    for index in tensor.indices:
        label = index.name
        if label.startswith("L_"):
            label = labels.setdefault(label, f"#l{next(fresh)}")
        kind: Literal["^", "_"] = "^" if index.is_up else "_"
        if position is not None and kind != position:
            scripts.append(
                ast.Script(position, None, ast.IndexList(tuple(run), span=span), span=span)
            )
            run = []
        position = kind
        run.append(ast.IndexItem(label, span=span))
    if run and position is not None:
        scripts.append(ast.Script(position, None, ast.IndexList(tuple(run), span=span), span=span))
    return ast.Scripted(ast.Name(name, verbatim=True, span=span), tuple(scripts), span=span)


def _trace_syntax(symbol: Symbol, head: Any, fresh: Iterator[int], span: Span) -> ast.Expr:
    """The trace a scalar of notation stands for: R, or T^{a}{}_{a}."""
    if head is None:
        return ast.Name(symbol.name, verbatim=True, span=span)
    label = f"#t{next(fresh)}"
    up = ast.Script(
        "^", None, ast.IndexList((ast.IndexItem(label, span=span),), span=span), span=span
    )
    down = ast.Script(
        "_", None, ast.IndexList((ast.IndexItem(label, span=span),), span=span), span=span
    )
    return ast.Scripted(ast.Name(head.name, verbatim=True, span=span), (up, down), span=span)
