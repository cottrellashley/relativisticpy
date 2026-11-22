"""The lowering of definitions: names, functions, coordinates, declarations, metrics, line
elements, spaces and tensors, each bound in the scope and the session. A definition that
needs something of the value it makes, as a metric must be one, makes a check, which the
workbook computes as the statement runs.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import replace

import sympy
from sympy import Symbol

from relativisticpy import spacetimes
from relativisticpy.compiler.builtins import _SPACETIMES, SPACETIMES
from relativisticpy.compiler.scope import (
    Check,
    Context,
    FunctionBinding,
    Scope,
    TensorBinding,
    VariableBinding,
)
from relativisticpy.compiler.session import ASSUMPTION_STATEMENTS, RENAMING_STATEMENTS, MetricState
from relativisticpy.diffgeom import differential
from relativisticpy.errors import CompileError, EvaluationError, IndexNotationError
from relativisticpy.ir import SCALAR, VALUE, EquationType, Node, TensorType, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.parser import split_index_labels
from relativisticpy.language.source import Span
from relativisticpy.lowering.expressions import Expressions
from relativisticpy.reading.names import (
    BOLD_ONE,
    _bold_one,
    _indexed_target,
    _is_line_element,
    _late_note,
    _line_element_coordinates,
    _mentioned,
    _named,
    _names,
    _shown,
    _subscripted_name,
    _subscripted_symbol,
    _unprimed,
)
from relativisticpy.tensors import Index, Indices
from relativisticpy.tensors.indices import latex_label
from relativisticpy.tensors.spaces import Space, spacetime
from relativisticpy.typing.syntax import (
    _is_literal,
    _literal_cells,
    _plain_label,
    _shape_text,
    _walk,
    _written,
)
from relativisticpy.typing.typer import _KNOWN_FIRST, _is_zero, _listed

#: What := cannot define, which the reader says how it read: a product, as \psi_n(x) or
#: \Delta\phi, a time derivative, a call of what is not a name, brackets and numbers.
_UNDEFINABLE = (ast.Binary, ast.Accent, ast.Call, ast.Tuple, ast.List, ast.Number)


class Definitions(Expressions):
    """The lowering of definitions."""

    def _define(self, target: ast.Expr, value: ast.Expr, scope: Scope, ctx: Context) -> None:
        with self._atomically():
            self._define_now(target, value, scope, ctx)

    def _define_now(self, target: ast.Expr, value: ast.Expr, scope: Scope, ctx: Context) -> None:
        target = _named(target)
        if isinstance(target, ast.Prime):  # t' = ..., a name of its own
            primed = target.func.id + "'" * target.order
            target = ast.Name(primed, verbatim=True, span=target.span)
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
                late = self._written_in_metric(name)
                if late:  # its value's names join the metric's: R in it is no Ricci scalar
                    self._metric_mentions(_mentioned(value, latex=ctx.latex))
                value_node = self._value_or_notation(value, scope, ctx, definition=True)
                if (indexed := _indexed_target(name, value_node)) is not None:
                    base, indices = indexed  # v_a := ..., a tensor with free index a: v_{a}
                    node = self._align(value_node, indices, value.span, what="a definition")
                    scope.bind(base, TensorBinding(node, indices))
                    return
                self._check_free_indices(target, value_node, value.span)  # L_i = I_{ij} w_i
                if late and value_node.type == SCALAR:
                    where = ops.Call("core.where")
                    self._rewrite_metric(where, (self.graph.const(name, VALUE), value_node))
                    self._note(_late_note(target.span.text, self.session.metric))
                scope.bind(name, VariableBinding(value_node))
            return
        if _is_line_element(target):
            self._define_line_element(value, scope, ctx, target.span.to(value.span))
            return
        target = _unprimed(target)
        if _bold_one(target):  # \mathbf{1} := ..., the program's own, which wins (decision 23)
            ones = self._value_or_notation(value, scope, ctx, definition=True)
            scope.bind(BOLD_ONE, VariableBinding(ones))
            return
        if isinstance(target, _UNDEFINABLE):  # \psi_n(x), \Delta\phi, \langle p, q \rangle
            raise self._cannot_define(target, value)
        if not isinstance(target, ast.Scripted) or not isinstance(target.base, ast.Name):
            raise CompileError("cannot define this", span=target.span)
        entries = self._entries(target, "a definition")
        if target.base.id == self.session.names["metric"]:
            self._define_metric(target, entries, value, scope, ctx)
        else:
            self._define_tensor(target, entries, value, scope, ctx)

    @contextlib.contextmanager
    def _atomically(self) -> Iterator[None]:
        """Make a definition whole or not at all: if it fails, the coordinates and the
        metric are as they were before it, whatever it changed on the way."""
        session = self.session
        saved = (
            session.coordinates,
            session.coordinates_inferred,
            session.metric,
            dict(session.functions),
            session.dropped,
        )
        try:
            yield
        except BaseException:
            session.coordinates, session.coordinates_inferred, session.metric = saved[:3]
            session.functions.clear()
            session.functions.update(saved[3])
            session.dropped = saved[4]
            raise

    def _define_function(
        self, definition: ast.FunctionDefinition, scope: Scope, ctx: Context
    ) -> None:
        """``f(x) := x^2``. A function that calls itself is refused here, where it is
        written, rather than when it is called. One the metric in force is written with,
        as f is in ``ds^2 = -f(r) dt^2 + ...``, takes this definition there from now on."""
        name, shown = definition.name, _shown(definition.name, definition.latex)
        if isinstance(definition.body, ast.Expr) and any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
            for node in _walk(definition.body)
        ):
            raise CompileError(
                f"{shown} calls itself, and recursive functions are not supported",
                span=definition.span,
            )
        binding = FunctionBinding(definition, scope)
        with self._atomically():
            if self._written_in_metric(name):
                if isinstance(definition.body, ast.Expr):
                    self._metric_mentions(_mentioned(definition.body, latex=definition.latex))
                symbols = tuple(self.session.symbol(param) for param in definition.params)
                parameters = [self.graph.const(symbol, SCALAR) for symbol in symbols]
                body = self._inline(binding, parameters, definition.span, ctx)
                if body.type == SCALAR:
                    where = ops.Call("core.where_function")
                    name_node = self.graph.const((name, symbols), VALUE)
                    self._rewrite_metric(where, (name_node, body))
                    params = ", ".join(_shown(p, definition.latex) for p in definition.params)
                    self._note(_late_note(f"{shown}({params})", self.session.metric))
            scope.bind(name, binding)

    def _metric_mentions(self, names: frozenset[str]) -> None:
        """Count ``names`` among those the metric in force is written with."""
        metric = self.session.metric
        assert metric is not None
        self.session.metric = replace(metric, names=metric.names | names)

    def _rewrite_metric(self, rewrite: ops.Call, args: tuple[Node, ...]) -> None:
        """The metric in force, with what a definition made after it says put into its
        components, checked as any metric is where it is defined."""
        metric = self.session.metric
        assert metric is not None
        assert self.session.coordinates is not None
        components = self.graph.add(rewrite, (metric.node, *args), VALUE, metric.node.span)
        node = self.graph.add(
            ops.Geometry("metric", self.session.coordinates),
            (components,),
            metric.node.type,
            metric.node.span,
        )
        self._check_metric(node, metric.node.span or args[-1].span)
        self.session.metric = replace(metric, node=node)

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
            item = self.graph.add(ops.Call("core.item"), (values, index), VALUE, target.span)
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
        coordinates = tuple(self.session.symbol(name) for name in names)
        self.session.set_coordinates(coordinates, span=value.span)

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
        self._check_spacetime_name(value, scope, ctx)
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
        self._check_metric(metric, target.span.to(value.span))
        self.session.metric = MetricState(metric, _mentioned(value, latex=ctx.latex))
        self.session.dropped = None

    def _check_metric(
        self,
        metric: Node,
        span: Span | None,
        *,
        context: tuple[str, ...] = (),
        hint: str | None = None,
    ) -> None:
        """Have a metric computed as its definition runs, so that one that cannot be is
        refused there, with the reason, rather than wherever it is first used: components of
        the wrong size, components that are not symmetric, or a determinant of zero. Nothing
        is computed now: the definition makes a :class:`Check`, which the workbook runs, with
        ``core.metric_diagnosis`` to say why the components are no metric."""
        given = metric.args[0]
        if isinstance(given.op, ops.ToArray):  # diag(-1, 1, 1), checked for its shape
            given = given.args[0]
        coordinates = self.session.coordinates or ()
        why = ops.Call("core.metric_diagnosis", (("coordinates", coordinates),))
        diagnosis = self.graph.add(why, (given,), VALUE, span)
        check = Check(metric, span, diagnosis, tuple(self._notes), context, hint)
        self._checks.append(check)

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
        complex_space = self.graph.const(space.field == "complex", VALUE)
        self.session.space_metrics[space] = self.graph.add(
            ops.Call("core.space_metric"),
            (node, complex_space),
            node.type.with_indices(lower),
            target.span.to(value.span),
        )

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
        dimension = self.value(value.scripts[0].body, scope, ctx)
        size = sympy.sympify(self._constant(dimension))
        # a symbol, even an integer one, is no size: n \geq 0 cannot be decided
        if not (getattr(size, "is_integer", False) and getattr(size, "is_nonnegative", False)):
            written = value.scripts[0].body.span.text  # 3.5 as written, not 3.50000000000000
            raise CompileError(
                f"the dimension of a space is a whole number, not {written}",
                span=value.span,
                hint=None if isinstance(dimension.op, ops.Const) else _KNOWN_FIRST,
            )
        return Space(name, int(size), field="complex" if value.base.id == "ℂ" else "real")

    def _spacetime(self, value: ast.Expr, scope: Scope, ctx: Context) -> Node | None:
        """Components of a spacetime such as ``Schwarzschild(M)``, which also brings its
        coordinates, as the spacetime declares them, without computing it; ``None`` for
        any other value."""
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and scope.lookup(value.func.id) is None
            and value.func.id in SPACETIMES
        ):
            return None
        if value.func.id == "Minkowski" and value.args and isinstance(value.args[0], ast.Name):
            # Minkowski(spherical): the name of the coordinates, never a product of letters
            kind = value.args[0]
            if kind.id not in ("cartesian", "spherical") and scope.lookup(kind.id) is None:
                raise CompileError(
                    f"Minkowski takes cartesian or spherical coordinates, not {kind.span.text}",
                    span=kind.span,
                    hint="for example Minkowski(spherical)",
                )
            named = ast.Name(kind.id, verbatim=True, span=kind.span)
            value = replace(value, args=(named, *value.args[1:]))
        call = self.value(value, scope, ctx)
        # Its coordinates are plain symbols: one declared with assumptions, as by
        # Positive := [r], takes the place of its namesake.
        declared = self._spacetime_coordinates(value, call)
        coordinates = tuple(self.session.symbol(c.name) for c in declared)
        self.session.set_coordinates(coordinates, span=value.span)
        components = ops.Call("core.components", (("coordinates", coordinates),))
        return self.graph.add(components, (call,), VALUE, value.span)

    def _spacetime_coordinates(self, value: ast.Call, call: Node) -> tuple[Symbol, ...]:
        """The coordinates a spacetime declares: of Minkowski, those its first argument
        names, which is known when compiling, as the name of the coordinates or a value
        given one."""
        assert isinstance(value.func, ast.Name)
        name, factory = value.func.id, _SPACETIMES[value.func.id]
        names = spacetimes.choices(factory)
        if not names:
            return spacetimes.coordinates_of(factory)
        choice = None
        if call.args:
            first = call.args[0].op
            if not isinstance(first, ops.Const):
                written = value.args[0].span.text
                raise CompileError(
                    f"{name} takes {_listed(list(names), 'or')} coordinates, not {written}",
                    span=value.args[0].span,
                    hint=f"for example {name}({names[-1]})",
                )
            choice = first.value
        try:
            return spacetimes.coordinates_of(factory, choice)
        except ValueError as error:  # the factory's own words, as when it is called
            raise EvaluationError(str(error), span=value.span) from None

    def _define_line_element(self, value: ast.Expr, scope: Scope, ctx: Context, span: Span) -> None:
        # Without declared coordinates, the differentials name them: dt, dr and d\Omega
        # are t, r, \theta and \phi.
        session = self.session
        named = _line_element_coordinates(value)
        current = {c.name for c in session.coordinates or ()}
        replaceable = session.coordinates is None or session.coordinates_inferred
        if named and replaceable and not set(named) <= current:
            coordinates = tuple(session.symbol(name) for name in named)
            session.set_coordinates(coordinates, inferred=True, span=span)
        # d\Omega without \theta and \phi is said where d\Omega is read
        written = _line_element_coordinates(value, solid_angle=False)
        strays = [name for name in written if name not in current and name != "Omega"]
        if strays and not replaceable:
            declared = [latex_label(c.name) for c in session.coordinates or ()]
            shown = _listed([f"d{latex_label(name)}" for name in strays])
            what = "is not the differential" if len(strays) == 1 else "are not differentials"
            raise CompileError(
                f"{shown} {what} of a coordinate",
                span=span,
                notes=[f"the coordinates are {_listed(declared)}, as declared"],
                hint="write the line element in those coordinates, or declare the ones it "
                f"uses, as in Coordinates := [{', '.join(latex_label(n) for n in named)}]",
            )
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
        context: tuple[str, ...] = ()
        hint = None
        if replaceable and named and set(named) < {c.name for c in coordinates}:
            # the 2-sphere after a line element in three dimensions, said if it is no metric
            alone = _listed([latex_label(name) for name in named])
            in_force = _listed([latex_label(c.name) for c in coordinates])
            context = (
                f"the coordinates in force are {in_force}, but this line element is "
                f"written in {alone} alone",
            )
            hint = (
                f"for a metric in {alone}, declare them first: Coordinates := "
                f"[{', '.join(latex_label(name) for name in named)}]"
            )
        self._check_metric(metric, span, context=context, hint=hint)
        self.session.metric = MetricState(metric, _mentioned(value, latex=ctx.latex))
        self.session.dropped = None

    def _line_element_of(self, span: Span) -> Node:
        """``ds^2`` written alone, with a metric in force: its line element,
        :math:`g_{\\mu\\nu} dx^\\mu dx^\\nu`, in the differentials of the coordinates."""
        metric = self.session.require_metric(span, "the line element").node
        coordinates = self.graph.const(self.session.coordinates, VALUE)
        return self._call("core.line_element", [metric, coordinates], SCALAR, span)

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
