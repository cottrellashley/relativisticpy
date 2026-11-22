r"""The compiler: from syntax tree to IR.

This is where notation acquires meaning. The parser keeps every reading of
ambiguous notation; the compiler, knowing what each name refers to, chooses:

* ``x^{a}`` is a component of ``x`` if ``x`` is a tensor and a power otherwise;
  ``r_{s}`` is the symbol ``r_s`` unless ``r`` is a tensor.
* ``R_{\mu\nu}`` is the Ricci tensor and ``R^{a}{}_{bcd}`` the Riemann tensor:
  built-in tensors are recognised by name *and* number of indices.
* ``=`` defines what has no meaning yet, ``y = 2`` or ``y(x) = x^2``, and a line
  element, ``ds^2 = ...``, defines the metric; once ``y`` has a value, ``y = 3`` is a
  claim, checked, and says so in a note (``:=`` always defines). A name the metric
  is written with, defined after it, takes its value in the metric.
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
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import reduce
from typing import Any, Literal

import sympy
from sympy import Symbol

from relativisticpy import abstract, spacetimes
from relativisticpy.abstract import IndexExpression
from relativisticpy.compiler.builtins import _SPACETIMES, BUILTINS, SPACETIMES, Builtin
from relativisticpy.compiler.passes import Term
from relativisticpy.compiler.scope import (
    Check,
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
    Saved,
    Session,
)
from relativisticpy.diffgeom import differential
from relativisticpy.errors import (
    CompileError,
    EvaluationError,
    IndexNotationError,
    RelativisticPyError,
)
from relativisticpy.ir import (
    SCALAR,
    VALUE,
    EquationType,
    Graph,
    Node,
    NotationType,
    TensorType,
    Type,
    ValueType,
    dependencies,
    ops,
)
from relativisticpy.language import syntax as ast
from relativisticpy.language.parser import split_index_labels
from relativisticpy.language.source import Span
from relativisticpy.language.vocabulary import GREEK

# The tables of the language, which its reference reads from here.
from relativisticpy.reading.names import _BUILTIN_TENSORS as _BUILTIN_TENSORS
from relativisticpy.reading.names import _LINEAR_ALGEBRA as _LINEAR_ALGEBRA
from relativisticpy.reading.names import _STATISTICS as _STATISTICS
from relativisticpy.reading.names import (
    _an_index,
    _bracketed,
    _bracketed_order,
    _factor_names,
    _indexed_target,
    _is_line_element,
    _late_note,
    _letter,
    _line_element_coordinates,
    _mentioned,
    _mentions_symbol,
    _named,
    _names,
    _names_builtin,
    _names_omega,
    _script_labels,
    _shown,
    _slashed,
    _splittable,
    _statistic,
    _subscripted_name,
    _subscripted_symbol,
    _unbracketed,
    _unprimed,
)
from relativisticpy.reading.reader import Reader
from relativisticpy.tensors import Index, Indices
from relativisticpy.tensors.indices import latex_label
from relativisticpy.tensors.spaces import Space, spacetime
from relativisticpy.typing import notation as rules
from relativisticpy.typing.syntax import _NOTATION_OPERATIONS as _NOTATION_OPERATIONS
from relativisticpy.typing.syntax import (
    _bracket_groups,
    _has_derivative,
    _is_literal,
    _is_minus_one,
    _literal_cells,
    _mentions_tensor,
    _operand,
    _plain_label,
    _product_factors,
    _rank,
    _shape_text,
    _signed_terms,
    _tensor_pattern,
    _tensor_pattern_of_one,
    _walk,
    _written,
    _zero_literal,
)
from relativisticpy.typing.typer import (
    TensorSource,
    _complex,
    _indices,
    _is_zero,
    _listed,
    _public,
    _shares,
    _spaces,
)

_COMPARE = {
    "<": "core.lt",
    ">": "core.gt",
    "<=": "core.le",
    ">=": "core.ge",
    "==": "core.eq",
    "!=": "core.ne",
}
_INTEGER = re.compile(r"\d+")
#: The hint for a size computed where it must be known before anything is computed.
_KNOWN_FIRST = "it is read before anything is computed: write it as a number, or a name given one"


class Compiler(Reader):
    """Compiles programs against a :class:`Session`, which it updates as it goes."""

    # ======================================================================
    # Statements
    # ======================================================================

    def compile(self, program: ast.Program) -> list[Output]:
        """Compile a program; return the values it shows, in order."""
        outputs: list[Output] = []
        for statement in program.statements:
            outputs.extend(self.statement(statement, self.session.globals))
        return outputs

    def take_checks(self) -> list[Check]:
        """What the definitions compiled since this was last called need of the values they
        make, and the notation they write, in order (see
        :class:`~relativisticpy.compiler.scope.Check`), to compute as their statements run;
        it forgets them."""
        checks, self._checks = self._checks, []
        return checks

    def save(self) -> tuple[Saved, int]:
        """What a statement may change, as it is before it, to put back with
        :meth:`restore`: the session's state, and the notes no output carries."""
        return self.session.save(), len(self._unshown)

    def restore(self, saved: tuple[Saved, int]) -> None:
        """Undo a statement whose check failed, as if it had not compiled."""
        state, unshown = saved
        self.session.restore(state)
        del self._unshown[unshown:]

    def take_notes(self) -> list[str]:
        """The notes on how statements were read that no output carries, because the
        statements showed nothing, as a definition does (``\\Delta`` taking its value in
        the metric written with it): those of the statements compiled since this was
        last called, which it forgets."""
        notes, self._unshown = self._unshown, []
        return notes

    def statement(
        self, statement: ast.Statement, scope: Scope, outer: Context | None = None
    ) -> list[Output]:
        saved = self._labels, self._scripts, self._factors, self._text
        top = self._depth == 0
        if top:
            self._notes = []
        self._depth += 1
        try:
            self._text = statement.span.text
            self._labels = self._infer_labels(statement, scope)
            self._scripts = _script_labels(statement)
            self._factors = _factor_names(statement)
            outputs = self._statement(statement, scope, outer)
            if top:
                self.session.factors.update(self._factors)
        except RelativisticPyError as error:
            if top:  # how something was read may say why it fails
                error.notes.extend(note for note in self._notes if note not in error.notes)
            raise
        finally:
            self._labels, self._scripts, self._factors, self._text = saved
            self._depth -= 1
        if top and self._notes:
            if outputs:
                first = outputs[0]
                notes = [*([first.note] if first.note else []), *self._notes]
                outputs[0] = replace(first, note="\n".join(dict.fromkeys(notes)))
            else:
                self._unshown.extend(self._notes)
        return outputs

    def _statement(
        self, statement: ast.Statement, scope: Scope, outer: Context | None
    ) -> list[Output]:
        ctx = Context(latex=statement.latex, calls=outer.calls if outer else ())
        if isinstance(statement, ast.FunctionDefinition):
            self._define_function(statement, scope, ctx)
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
        if isinstance(expr, ast.Equation):
            return self._equals(expr, scope, ctx)
        if _is_line_element(expr) and self.session.metric is not None:
            return [Output(self._line_element_of(expr.span), expr.span)]
        name = (
            expr.base.id
            if isinstance(expr, ast.Scripted) and isinstance(expr.base, ast.Name)
            else None
        )
        return [Output(self.value(expr, scope, ctx), expr.span, name)]

    def _equals(self, equation: ast.Equation, scope: Scope, ctx: Context) -> list[Output]:
        """``lhs = rhs``: a definition of what has no meaning yet, and a claim, checked,
        about everything else (see :meth:`_reading`)."""
        lhs, rhs = equation.lhs, equation.rhs
        if _is_line_element(lhs) and _zero_literal(rhs):
            raise CompileError(
                f"{equation.span.text} says that a path is null: a condition, which is read "
                "but not computed",
                span=equation.span,
                hint="the line element defines the metric only where it is a quadratic form "
                "in the differentials, as in ds^2 = -dt^2 + dx^2",
            )
        if self._names_coordinate(lhs, scope):
            # t = \pm r^* + c, or \theta = \pi/2: where something holds, not a new t
            raise CompileError(
                f"an equation for the coordinate {lhs.span.text} is a condition, which "
                "is read but not computed",
                span=equation.span,
                hint=r"to compute where it holds, substitute it, as in f |_{\theta=\pi/2}",
            )
        if _bracketed(equation):  # a = ... \quad (k = 0): the case it holds in
            raise CompileError(
                f"{equation.span.text}, in brackets, is a condition, which is read but not "
                "computed",
                span=equation.span,
                hint=f"to compute with it, write it without the brackets: {_unbracketed(equation)}",
            )
        reading, note = self._reading(equation, scope, ctx)
        if reading == "function":
            assert isinstance(lhs, ast.Call)
            assert isinstance(lhs.func, ast.Name)
            params = tuple(arg.id for arg in lhs.args if isinstance(arg, ast.Name))
            definition = ast.FunctionDefinition(
                lhs.func.id, params, rhs, span=equation.span, latex=ctx.latex
            )
            failure = self._fails_as_function(definition, scope, ctx)
            if failure is None:
                self._define_function(definition, scope, ctx)
                return []
            # Not a definition: its value means nothing as a function. As an equation,
            # what fails says why.
            self._note(
                f"{lhs.span.text} = ... is read as an equation, not as the definition of a "
                f"function: as one, its value cannot be computed ({failure.message})"
            )
            note = None
        if reading == "define":
            if self._is_notation(rhs, scope, ctx, equation=True):  # nothing to define it
                return [Output(self._notation(equation, scope, ctx), equation.span)]
            self._define(lhs, rhs, scope, ctx)
            return []
        if reading == "notation" or self._is_notation(equation, scope, ctx, equation=True):
            return [Output(self._notation(equation, scope, ctx), equation.span, note=note)]
        return [Output(self.value(equation, scope, ctx), equation.span, note=note)]

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
        if not isinstance(target, ast.Scripted) or not isinstance(target.base, ast.Name):
            raise CompileError("cannot define this", span=target.span)
        entries = self._entries(target, "a definition")
        if target.base.id == self.session.names["metric"]:
            self._define_metric(target, entries, value, scope, ctx)
        else:
            self._define_tensor(target, entries, value, scope, ctx)

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

    def _fails_as_function(
        self, definition: ast.FunctionDefinition, scope: Scope, ctx: Context
    ) -> RelativisticPyError | None:
        """Why ``f(x) = body`` cannot define a function, if it cannot: the body compiled
        for its parameters as symbols, in a session of its own, fails. ``:=`` defines a
        function whatever its body, which may take tensors; ``=`` defines one only where
        the definition means something."""
        checker = Compiler(self.session.fork())
        checker._depth = 1  # its notes are not this statement's
        parameters = [
            checker.graph.const(self.session.symbol(param), SCALAR) for param in definition.params
        ]
        try:
            checker._inline(FunctionBinding(definition, scope), parameters, definition.span, ctx)
        except RelativisticPyError as error:
            return error
        return None

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

    # ======================================================================
    # Expressions
    # ======================================================================

    def value(self, node: ast.Expr, scope: Scope, ctx: Context) -> Node:
        """Compile an expression that must produce a value. A builtin's name with nothing
        to act on, as N is in N^2 (the lapse) or \\frac{N}{2}, is a symbol."""
        result = self.term(node, scope, ctx)
        if isinstance(result, FunctionReference) and scope.lookup(result.name) is None:
            return self._named_alone(result)
        if isinstance(result, Operator):
            raise self._nothing_to_act_on(result, scope)
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

    def _Dots(self, node: ast.Dots, scope: Scope, ctx: Context) -> Node:
        raise CompileError(
            f"{node.span.text} stands for terms that are not written, so it cannot be computed",
            span=node.span,
        )

    def _Relation(self, node: ast.Relation, scope: Scope, ctx: Context) -> Node:
        if node.op == "≡":  # A \equiv B, where A cannot be defined: an identity
            return self._Equation(ast.Equation(node.lhs, node.rhs, span=node.span), scope, ctx)
        if node.op == "∈":
            raise self._membership(node)
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
            raise self._star(node)
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
            return self._call("core.not", [operand], SCALAR, node.span)
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
            args = [self.value(node.left, scope, ctx), self.value(node.right, scope, ctx)]
            return self._call(f"core.{node.op}", args, SCALAR, node.span)
        left = self.value(_operand(node.left), scope, ctx)
        right = self.value(_operand(node.right), scope, ctx)
        if node.op == "/":
            return self._divide(left, right, node.span)
        return self._power(left, right, node.span)

    def _hadamard(self, node: ast.Binary, scope: Scope, ctx: Context) -> Node:
        """``A_{ta} \\odot B_{ta}``: the elementwise product, of tensors with the same free
        indices, summing nothing. Index notation has no other way to say it. Of lists or
        matrices, as ``[1, 2] \\odot [3, 4]``, it is elementwise too, never their product."""
        left = self.value(_operand(node.left), scope, ctx)
        right = self.value(_operand(node.right), scope, ctx)
        if any(isinstance(value.type, ValueType) for value in (left, right)) and not any(
            isinstance(value.type, TensorType) for value in (left, right)
        ):
            call = ops.Call("core.hadamard")
            return self.graph.add(call, (left, right), VALUE, node.span)
        if not (isinstance(left.type, TensorType) and isinstance(right.type, TensorType)):
            return self._multiply([left, right], node.span)
        right = self._align(right, left.type.indices, node.span, what="an elementwise product")
        return self.graph.add(ops.Call("linalg.hadamard"), (left, right), left.type, node.span)

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
        self._check_terms([(value, term) for _, value, term in values], node.span, scope)
        reference = tensors[0].type
        assert isinstance(reference, TensorType)
        kept: list[tuple[int, Node]] = []
        for sign, value, term in values:
            if isinstance(value.type, TensorType):
                kept.append((sign, self._align(value, reference.indices, term.span)))
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
                if position + 1 < len(factors):
                    self._check_statistic_name(factor, scope)
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
        items = operand.items if isinstance(operand, ast.Tuple | ast.List) else (operand,)
        expected = 2 if kind == "covariance" else 1
        if len(items) != expected:
            written = span.text.split("(")[0].split("[")[0]
            hint = None
            if kind == "expectation" and len(items) == 2:  # E_t[X, Y]: a covariance
                arguments = ", ".join(item.span.text for item in items)
                hint = f"for a covariance, write \\operatorname{{Cov}}_{label}({arguments})"
            raise CompileError(
                f"{written} takes {'one expression' if expected == 1 else 'two expressions'}, "
                f"but {len(items)} {'is' if len(items) == 1 else 'are'} given",
                span=span,
                hint=hint,
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
        # Complex data is conjugated as NumPy's var and cov do: the variance is of
        # |X - E X|^2, and the covariance conjugates its second argument.
        centered = [self._centered(node, label, span) for node in nodes]
        if kind == "variance":
            assert isinstance(centered[0].type, TensorType)
            complex_space = self.graph.const(_complex(centered[0].type), VALUE)
            square = self.graph.add(
                ops.Call("core.squared_modulus"),
                (centered[0], complex_space),
                centered[0].type,
                span,
            )
            return self._expectation(square, label, span)
        with self._binding(label):
            product = self._multiply([centered[0], self._conjugate(centered[1], span)], span)
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
        measure, fused with the contraction that made ``node``, and taken term by term of a
        sum, so that the product of everything averaged is never formed: the average of
        R_{ta} R_{tb} - \\mu_a \\mu_b is two contractions, neither over days, assets and
        assets."""
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
        if (parts := self._averaged_parts(node, label, span)) is not None:
            if remaining:
                parts = [self._align(part, remaining, span) for part in parts]
            if isinstance(node.op, ops.Add):
                return self.graph.add(ops.Add(node.op.signs), parts, kind, span)
            if isinstance(node.op, ops.Scale):  # what multiplies it is constant in label
                return self.graph.add(ops.Scale(), [*node.args[:-1], parts[0]], kind, span)
            return parts[0]
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

    def _averaged_parts(self, node: Node, label: str, span: Span) -> list[Node] | None:
        """The averages over ``label`` of the tensors that a sum adds, a product with
        scalars scales or a permutation reorders, each with the indices it has in
        ``node``; ``None`` for any other tensor."""
        assert isinstance(node.type, TensorType)
        indices = node.type.indices
        parts: list[tuple[Node, Indices]]
        if isinstance(node.op, ops.Permute):  # the result's axis i is the argument's axes[i]
            unpermuted = list(indices)
            for position, axis in enumerate(node.op.axes):
                unpermuted[axis] = indices[position]
            parts = [(node.args[0], Indices(unpermuted))]
        elif isinstance(node.op, ops.Add):
            parts = [(term, indices) for term in node.args]
        elif isinstance(node.op, ops.Scale):
            parts = [(node.args[-1], indices)]
        else:
            return None
        if not all(isinstance(part.type, TensorType) for part, _ in parts):
            return None
        averaged = []
        for part, named in parts:
            assert isinstance(part.type, TensorType)
            renamed = self.graph.retype(part, TensorType(named, part.type.spaces))
            averaged.append(self._expectation(renamed, label, span))
        return averaged

    def _measure(self, space: Space, index: Index, span: Span) -> Node:
        """The weights an expectation over ``space`` averages with: those declared for it,
        or the same for every element."""
        kind = TensorType(Indices([index]), (space,))
        declared = self.session.measures.get(space)
        if declared is not None:
            return self.graph.retype(declared, kind)
        return self.graph.add(
            ops.Call("core.uniform"), (self.graph.const(space.dim, VALUE),), kind, span
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
        measure = self.session.measures.get(node.type.space(label))
        if measure is not None and not _uniform_weights(measure):
            raise CompileError(
                "scikit-learn's Ledoit–Wolf estimate weighs every sample alike, but the "
                f"samples over {latex_label(label)} are weighted by a measure",
                span=span,
                hint="take the covariance with the measure: \\operatorname{Cov}_t(X_{ta}, X_{tb})",
            )
        other = 1 - names.index(label)
        if other == 0:
            node = self._permute(node, (1, 0), Indices(reversed(node.type.indices)))
        assert isinstance(node.type, TensorType)
        space = node.type.spaces[1]
        kind = TensorType(Indices([Index("#a"), Index("#b")]), (space, space))
        return self.graph.add(ops.Call("stats.ledoit_wolf"), (node,), kind, span)

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
        if ctx.latex:
            factors = _slashed(factors)
        factors = [part for factor in factors for part in _unsigned(factor)]
        grouped = self._statistic_factors(factors, scope, ctx)
        if len(grouped) == 1 and isinstance(grouped[0], Node):
            return grouped[0]
        terms: list[Term] = []
        chosen: list[list[Node]] = []  # the components each factor selects
        previous: ast.Expr | Node | None = None
        for factor in grouped:
            mark = len(self._selections)
            if isinstance(factor, Node):  # a statistic, already compiled
                terms.append(factor)
                chosen.append([])
                previous = factor
                continue
            if isinstance(previous, ast.Expr) and terms and _an_index(previous, factor):
                last = terms[-1]
                if isinstance(last, Node) and last.type == VALUE:  # [1, 2][0], as Python does
                    raise CompileError(
                        "a list is not indexed with [ ]",
                        span=previous.span.to(factor.span),
                        notes=[
                            f"after a list, {factor.span.text} is a factor: "
                            f"{previous.span.text}{factor.span.text} multiplies the list by it"
                        ],
                        hint="to name the elements of a list, write them on the left of :=, "
                        "as in a, b := [1, 2]",
                    )
            previous = factor
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
                    raise self._nothing_to_act_on(term, scope)
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
            kind = VALUE if any(isinstance(o.type, ValueType) for o in scalars) else SCALAR
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
        and with a symmetric :math:`A` it is a Cholesky solve. What the inverse multiplies
        is every factor joined to it through one of its indices, contracted first: in
        :math:`(G^{-1})^{kl} X_{tl} y_t`, the vector :math:`X_{tl} y_t`. It is a solve only
        where that index is written once more and the other is not among those factors;
        otherwise, as in the trace :math:`(\\Sigma^{-1})^{ab} \\Sigma_{ab}`, the product
        is contracted as it is written."""
        tensors = list(tensors)
        written: set[int] = set()  # inverses that are no solve
        while not self._bound:
            position = next(
                (
                    k
                    for k, t in enumerate(tensors)
                    if isinstance(t.op, ops.Inverse)
                    and t.id not in written
                    and any(j != k and _shares(o, t) for j, o in enumerate(tensors))
                ),
                None,
            )
            if position is None:
                break
            inverse = tensors[position]
            plan = self._solve_plan(tensors, position)
            if plan is None:
                written.add(inverse.id)
                continue
            along, kept, transpose, joined = plan
            rhs = tensors[joined[0]]
            if len(joined) > 1:
                rhs = self._contract([(tensors[k], _indices(tensors[k])) for k in joined], span)
            assert isinstance(rhs.type, TensorType)
            assert isinstance(inverse.type, TensorType)
            names = rhs.type.indices.names
            axis = names.index(along.name)
            order = [axis, *(i for i in range(len(names)) if i != axis)]
            rhs = self._permute(rhs, order, Indices(rhs.type.indices[i] for i in order))
            assert isinstance(rhs.type, TensorType)
            space = inverse.type.space(kept.name)
            indices = Indices([kept, *rhs.type.indices[1:]])
            kind = TensorType(indices, (space, *rhs.type.spaces[1:]))
            solution = self.graph.add(
                ops.Call("linalg.solve"),
                (inverse.args[0], rhs, self.graph.const(transpose, VALUE)),
                kind,
                span,
            )
            used = {position, *joined}
            tensors = [t for k, t in enumerate(tensors) if k not in used] + [solution]
        return tensors

    def _solve_plan(
        self, tensors: Sequence[Node], position: int
    ) -> tuple[Index, Index, bool, list[int]] | None:
        """How the inverse at ``position`` is a solve: the index it sums with what it
        multiplies, the index it keeps, whether that is its transpose's solve, and the
        positions of the factors joined to it through the index; ``None`` where it is
        no solve."""
        first, second = _indices(tensors[position])
        if first.name == second.name:
            return None
        others = [k for k in range(len(tensors)) if k != position]
        for along, kept, transpose in ((second, first, False), (first, second, True)):
            carriers = [k for k in others if along.name in _indices(tensors[k]).names]
            if sum(_indices(tensors[k]).names.count(along.name) for k in carriers) != 1:
                continue
            joined, pending = [carriers[0]], [carriers[0]]
            while pending:  # every factor joined to it through shared indices
                names = set(_indices(tensors[pending.pop()]).names)
                for k in others:
                    if k not in joined and names & set(_indices(tensors[k]).names):
                        joined.append(k)
                        pending.append(k)
            if not any(kept.name in _indices(tensors[k]).names for k in joined):
                return along, kept, transpose, sorted(joined)
        return None

    def _linear_algebra(self, name: str, node: ast.Call, scope: Scope, ctx: Context) -> Node | None:
        """Linear algebra on a tensor: ``eig`` and ``spec`` (its spectrum), ``svd``,
        ``det``, ``tr``, ``inv``, ``diag`` of a vector, and ``exp``, ``sqrt`` and ``log``
        of a map named without indices, :math:`e^{-iHt}`. ``None`` leaves the call to the
        builtins, as for a matrix written as a list."""
        argument = node.args[0]
        if name in ("exp", "sqrt", "log") and _mentions_tensor(argument):
            # Written with its indices: one component at a time, by the builtins, except
            # log, whose builtin takes a base and no tensor, as \ln does.
            if name != "log" or len(node.args) > 2:
                return None
            args = [self.value(arg, scope, ctx) for arg in node.args]
            if not isinstance(args[0].type, TensorType):
                return None
            return self.graph.add(ops.Map("log"), args, args[0].type, node.span)
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
            return self.graph.add(ops.Call("linalg.diag"), (tensor,), kind, span)
        if name == "inv":
            if tensor.type.rank != 2:
                raise IndexNotationError("only a tensor with two indices has an inverse", span=span)
            return self._inverse(tensor, span)
        if name in ("eig", "svd"):
            parts = self._decomposition(node, scope, ctx)
            assert parts is not None
            return self.graph.add(ops.Call("core.tuple"), parts, VALUE, span)
        operator, rows, metric = self._as_map(tensor, span, form=name == "spec")
        if name in ("det", "tr", "trace"):
            return self.graph.add(
                ops.Call("linalg.det" if name == "det" else "linalg.trace"),
                (operator, self.graph.const(rows, VALUE)),
                SCALAR,
                span,
            )
        if name in ("exp", "sqrt", "log"):
            return self.graph.add(
                ops.Call("linalg.function"),
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
        size = math.prod(space.dim for space in operator.type.spaces[:rows])
        count = None
        if len(call.args) > 1:
            number = self.value(call.args[1], scope, ctx)
            count = self._constant(number)
            if count is None or not sympy.sympify(count).is_integer or count == 0:
                raise CompileError(
                    "the number of eigenvalues is a whole number: the largest that many, "
                    "or the smallest if it is negative",
                    span=call.args[1].span,
                    hint=None if isinstance(number.op, ops.Const) else _KNOWN_FIRST,
                )
            count = int(count)
            if abs(count) > size:
                raise CompileError(
                    f"this map has {size} eigenvalues, not {abs(count)}", span=call.args[1].span
                )
        modes = Space(f"eigenvectors of {call.args[0].span.text}", abs(count) if count else size)
        args = [operator, self.graph.const(rows, VALUE), metric or self.graph.const(None, VALUE)]
        args.append(self.graph.const(count, VALUE))
        both = self.graph.add(ops.Call("linalg.eig"), args, VALUE, call.span)
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
            ops.Call("linalg.svd"), (tensor, self.graph.const(1, VALUE)), VALUE, call.span
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
        return self.graph.add(ops.Call("core.part"), (whole, index), kind, span)

    def _divide(self, numerator: Node, denominator: Node, span: Span) -> Node:
        if isinstance(denominator.type, TensorType):
            raise IndexNotationError("cannot divide by a tensor", span=span)
        minus_one = self.graph.const(sympy.S.NegativeOne, SCALAR)
        kind = _passed(denominator.type)
        reciprocal = self.graph.add(ops.Pow(), (denominator, minus_one), kind, span)
        return self._multiply([numerator, reciprocal], span)

    def _power(self, base: Node, exponent: Node, span: Span) -> Node:
        if isinstance(base.type, TensorType):
            inverse = isinstance(exponent.op, ops.Const) and exponent.op.value == -1
            metric = isinstance(base.op, ops.Geometry) and base.op.quantity == "metric"
            raise self._tensor_power(
                self.session.names["metric"] if metric else None,
                base.type.indices,
                "-1" if inverse else None,
                span,
            )
        if isinstance(exponent.type, TensorType):
            raise IndexNotationError("cannot raise to the power of a tensor", span=span)
        kind = (
            VALUE if any(isinstance(t, ValueType) for t in (base.type, exponent.type)) else SCALAR
        )
        return self.graph.add(ops.Pow(), (base, exponent), kind, span)

    # -- relations -----------------------------------------------------------

    def _Equation(self, node: ast.Equation, scope: Scope, ctx: Context) -> Node:
        lhs = self._components_of(self.value(node.lhs, scope, ctx), scope, ctx, node.lhs.span)
        rhs = self._components_of(self.value(node.rhs, scope, ctx), scope, ctx, node.rhs.span)
        tensor = next((v.type for v in (lhs, rhs) if isinstance(v.type, TensorType)), None)
        if tensor is not None:
            assert isinstance(tensor, TensorType)
            sides = []
            for v, side in ((lhs, node.lhs), (rhs, node.rhs)):
                if isinstance(v.type, TensorType):
                    sides.append(self._align_sides(v, tensor, lhs, side.span, node.span))
                else:
                    sides.append(self._zero_side(v, tensor, side, lhs is v, scope))
            return self.graph.add(ops.Equation(), sides, EquationType(tensor), node.span)
        return self.graph.add(ops.Equation(), (lhs, rhs), EquationType(SCALAR), node.span)

    def _align_sides(
        self, value: Node, tensor: TensorType, lhs: Node, span: Span, equation: Span
    ) -> Node:
        """One side of a tensor equation, its axes in the order of the first side's."""
        self._check_sides(value, tensor, lhs, equation)
        return self._align(value, tensor.indices, span)

    def _zero_side(
        self, value: Node, tensor: TensorType, side: ast.Expr, left: bool, scope: Scope
    ) -> Node:
        """The side of a tensor equation that is not a tensor: zero, or components."""
        if value.type == VALUE:
            return self.graph.add(ops.ToArray(tensor.shape), (value,), tensor, side.span)
        self._check_zero_side(value, tensor, side, left, scope)
        return value

    def _Comparison(self, node: ast.Comparison, scope: Scope, ctx: Context) -> Node:
        operands = [self.value(o, scope, ctx) for o in node.operands]
        relations = [
            self._call(_COMPARE[op], [a, b], SCALAR, node.span)
            for op, a, b in zip(node.ops, operands, operands[1:], strict=False)
        ]
        if len(relations) == 1:
            return relations[0]
        return self._call("core.and", relations, SCALAR, node.span)

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
        trace = binding is None and name in ("trace", "tr") and len(node.args) == 1
        if trace and (traced := self._metric_trace(node.args[0], scope, ctx)) is not None:
            return traced
        if binding is None and name in _LINEAR_ALGEBRA and node.args:
            done = self._linear_algebra(name, node, scope, ctx)
            if done is not None:
                return done
        names = {c.name for c in self.session.coordinates or ()}
        if isinstance(binding, VariableBinding) and all(
            isinstance(a, ast.Name) and (a.id in names or _mentions_symbol(binding.node, a.id))
            for a in node.args
        ):  # \alpha(r), once \alpha is defined, or a(t) of a value in t: what it depends on
            return binding.node
        one_name = len(node.args) == 1 and isinstance(node.args[0], ast.Name)
        product = (name in names and not one_name) or self._a_product(func, node.args, binding, ctx)
        if product:  # r(\partial_1 \beta - 1), with r a coordinate: as typeset, a product
            group = node.args[0] if len(node.args) == 1 else ast.Tuple(node.args, span=node.span)
            times = ast.Binary("*", func, group, implicit=True, span=node.span)
            return self.value(times, scope, ctx)
        if binding is not None:
            raise CompileError(f"{name} is not a function", span=func.span)
        if name == "geodesic_equations":
            return self._geodesic_equations(node, scope, ctx)
        if name == "geodesic":  # integrated numerically, from Python
            raise CompileError(
                "geodesics are integrated from Python, with relativisticpy.numeric.geodesic",
                span=node.span,
                hint="in the language, geodesic_equations(\\tau) gives the equations they solve",
            )
        if name in ("eig", "spec", "svd"):  # what _linear_algebra leaves: no tensor
            count = ", and how many eigenvalues if not all" if name != "svd" else ""
            raise CompileError(
                f"{name}() takes a tensor written with its indices{count}",
                span=node.span,
                hint=f"for example {name}(\\Sigma^{{a}}{{}}_{{b}})",
            )
        builtin = _names_builtin(func, latex=ctx.latex)  # \gamma(r) is an unknown function
        if builtin:
            self._check_keywords(BUILTINS[name], node)
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
        self._check_unknown_function(func, node)
        args = [self.value(a, scope, ctx) for a in node.args]
        for arg, source in zip(args, node.args, strict=True):
            if isinstance(arg.type, TensorType):
                raise CompileError(f"cannot pass a tensor to {name}()", span=source.span)
            if isinstance(arg.type, ValueType):  # f([1, 2]): of numbers, not of a list
                raise CompileError(
                    f"{func.span.text} is a function RelativisticPy does not know, so it takes "
                    "numbers and symbols, not a list",
                    span=source.span,
                )
        values = [arg.op.value if isinstance(arg.op, ops.Const) else None for arg in args]
        coordinates = self.session.coordinates or ()
        # Later, a bare name means this; but a coordinate, as r in r(t), stays a coordinate.
        if values and all(value in coordinates for value in values) and name not in names:
            self.session.functions[name] = tuple(values)
        return self._call("core.function", args, SCALAR, node.span, (("name", name),))

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
            op = ops.Map(builtin.name)
            return self.graph.add(op, args, first.type, span)
        args = [self._as_object(arg) for arg in args]
        kind = SCALAR if builtin.scalar and all(a.type == SCALAR for a in args) else VALUE
        return self._call(builtin.name, args, kind, span)

    def _as_object(self, node: Node) -> Node:
        """Wrap tensor components in a :class:`Tensor` for a function that wants one."""
        if isinstance(node.type, TensorType):
            wrap = ops.Call("core.tensor", self._wrapped(node.type.indices))
            return self.graph.add(wrap, (node,), VALUE, node.span)
        if isinstance(node.type, EquationType) and isinstance(node.type.of, TensorType):
            wrap = ops.Call("core.tensor_equation", self._wrapped(node.type.of.indices))
            return self.graph.add(wrap, (node,), VALUE, node.span)
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
        elif _names_builtin(prime.func, latex=ctx.latex):
            body = self._builtin(BUILTINS[name], [dummy], span)
        else:
            body = self._call("core.function", [dummy], SCALAR, span, (("name", name),))
        order = self.graph.const(prime.order, VALUE)
        return self._call("core.derivative_at", [body, dummy, order, point], SCALAR, span)

    def _geodesic_equations(self, node: ast.Call, scope: Scope, ctx: Context) -> Node:
        if len(node.args) > 1:
            raise CompileError(
                f"geodesic_equations(\\tau) takes at most 1 argument, not {len(node.args)}",
                span=node.span,
            )
        parameter = (
            self.value(node.args[0], scope, ctx)
            if node.args
            else self.graph.const(sympy.Symbol("tau"), SCALAR)
        )
        christoffel = self._geometry("christoffel", node.span)
        coordinates = self.graph.const(self.session.coordinates, VALUE)
        return self._call(
            "core.geodesic_equations", [christoffel, coordinates, parameter], VALUE, node.span
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
            return self._dalembertian(operand, span, operator_.span.text)
        if isinstance(operator_, LieDerivative):
            return self._lie_derivative(operator_.vector, operand, span, scope)
        assert isinstance(operator_, IndexDerivative)
        result = operand
        for item, upper in reversed(operator_.items):
            result = self._index_derivative(result, item, upper, operator_.covariant, span)
        return result

    # -- calculus ------------------------------------------------------------

    def _Derivative(self, node: ast.Derivative, scope: Scope, ctx: Context) -> Term:
        if (coordinate := self._along_a_path(node, scope)) is not None:
            raise CompileError(
                f"the derivative of the coordinate {coordinate} along a path is read "
                "but not computed",
                span=node.span,
                hint=r"write the path's coordinates as functions of its parameter, as in r(\tau)",
            )
        variables = tuple((self.value(v, scope, ctx), order) for v, order in node.variables)
        if node.expr is None:
            return LeibnizDerivative(node.span, variables)
        expr = node.expr
        binding = scope.lookup(expr.id) if isinstance(expr, ast.Name) else None
        if isinstance(binding, FunctionBinding) and isinstance(expr, ast.Name):
            # \frac{d^2 f}{dx^2}, with f(x) defined: the derivative of f(x)
            expr = self._applied(expr, binding, node)
        elif isinstance(expr, ast.Name) and self._undefined(expr, scope):
            by = [v for v, _ in node.variables if isinstance(v, ast.Name)]
            written = list(dict.fromkeys(v.span.text for v in by))
            if written and expr.id not in {v.id for v in by}:
                shown = expr.span.text
                self._note(
                    f"{shown} does not depend on {_listed(written)}, so its derivative is 0; "
                    f"for a function, write {shown}({', '.join(written)})"
                )
        return self._differentiate(self.value(expr, scope, ctx), variables, node.span)

    @staticmethod
    def _applied(name: ast.Name, binding: FunctionBinding, node: ast.Derivative) -> ast.Expr:
        """A function written alone in a derivative, as f in \\frac{df}{dx}, applied to
        what it is differentiated by: f(x), or f(x, y) by its parameters' names."""
        params = binding.definition.params
        variables = [v for v, _ in node.variables if isinstance(v, ast.Name)]
        names = list(dict.fromkeys(v.id for v in variables))
        if len(params) == 1 and len(names) == 1:
            args: tuple[ast.Expr, ...] = (variables[0],)
        elif names and set(names) <= set(params):
            args = tuple(ast.Name(param, span=name.span) for param in params)
        else:
            return name
        return ast.Call(name, args, span=name.span)

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
            return self.graph.add(ops.Call("core.gradient"), (expr, vector), kind, span)
        orders = self.graph.const(tuple(order for _, order in variables), VALUE)
        args = [expr, orders, *(v for v, _ in variables)]
        if isinstance(expr.type, TensorType):
            return self.graph.add(ops.Map("core.diff"), args, expr.type, span)
        return self._call("core.diff", args, _passed(expr.type), span)

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
            limits.append(self._call("core.limits", [symbol, lower, upper], VALUE, bound.span))
        integrand = self.value(node.integrand, inner, ctx)
        if isinstance(integrand.type, TensorType):
            return self.graph.add(
                ops.Map("integrate"), [integrand, *limits], integrand.type, node.span
            )
        return self._call("integrate", [integrand, *limits], _passed(integrand.type), node.span)

    def _BigOperator(self, node: ast.BigOperator, scope: Scope, ctx: Context) -> Node:
        if node.lower is None or node.upper is None:
            hint = f"for example \\{node.op}_{{n=0}}^{{N}} a_n"
            label = node.variable.id if isinstance(node.variable, ast.Name) else None
            indexed = [  # the tensors that carry the index, as w in \sum_a w^a
                current
                for current in _walk(node.body)
                if isinstance(current, ast.Scripted)
                and (pattern := _tensor_pattern(current))
                and any(name == label for name, _ in pattern[1])
                and isinstance(scope.lookup(pattern[0]), TensorBinding)
            ]
            if node.op == "sum" and label is not None and indexed:  # \sum_a w^a
                pattern = _tensor_pattern(indexed[0])
                assert pattern is not None
                written = indexed[0].span.text
                ones = f"1_{label}" if dict(pattern[1])[label] else f"1^{label}"
                hint = (
                    f"to sum the components of {written}, contract it with the covector of "
                    f"ones: {written} {ones}"
                )
            raise CompileError(f"\\{node.op} needs limits", span=node.span, hint=hint)
        inner = scope.child()
        variable = self._bound_variable(node.variable, inner)
        lower = self.value(node.lower, scope, ctx)
        upper = self.value(node.upper, scope, ctx)
        body = self.value(node.body, inner, ctx)
        name = "core.sum" if node.op == "sum" else "core.prod"
        args = [body, variable, lower, upper]
        if isinstance(body.type, TensorType):  # component by component, as an integral is
            return self.graph.add(ops.Map(name), args, body.type, node.span)
        return self._call(name, args, _passed(body.type), node.span)

    def _Limit(self, node: ast.Limit, scope: Scope, ctx: Context) -> Node:
        inner = scope.child()
        variable = self._bound_variable(node.variable, inner)
        point = self.value(node.point, scope, ctx)
        body = self.value(node.body, inner, ctx)
        direction = self.graph.const(node.direction or "+-", VALUE)
        if isinstance(body.type, TensorType):
            return self.graph.add(
                ops.Map("core.limit"), [body, variable, point, direction], body.type, node.span
            )
        return self._call("core.limit", [body, variable, point, direction], SCALAR, node.span)

    def _bound_variable(self, variable: ast.Expr, scope: Scope) -> Node:
        """Bind a summation, integration or limit variable as a plain symbol."""
        if not isinstance(variable, ast.Name):
            raise CompileError("expected a variable", span=variable.span)
        symbol = self.graph.const(self.session.symbol(variable.id), SCALAR, variable.span)
        scope.bind(variable.id, VariableBinding(symbol))
        return symbol

    def _Root(self, node: ast.Root, scope: Scope, ctx: Context) -> Node:
        if node.degree is None:  # \sqrt{\Sigma}, of a tensor named without indices: a matrix's
            call = ast.Call(ast.Name("sqrt", span=node.span), (node.radicand,), span=node.span)
            if (root := self._linear_algebra("sqrt", call, scope, ctx)) is not None:
                return root
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
        :math:`\\sqrt{g_{ab} \\bar v^a v^b}`, with the conjugate of complex components."""
        assert isinstance(vector.type, TensorType)
        index = vector.type.indices[0]
        lowered, _ = self._move_index(vector, vector.type.indices, 0, span)
        twin = Indices([Index(index.name, not index.upper)])
        conjugate = self._conjugate(vector, span)
        square = self._contract([(conjugate, vector.type.indices), (lowered, twin)], span)
        square = self.graph.add(ops.Call("core.real"), (square,), SCALAR, span)
        half = self.graph.const(sympy.Rational(1, 2), SCALAR)
        return self.graph.add(ops.Pow(), (square, half), SCALAR, span)

    def _conjugate(self, tensor: Node, span: Span) -> Node:
        """The complex conjugate of a tensor: of components that are numbers, always, and
        of SymPy's over a complex space, where a symbol need not be real."""
        assert isinstance(tensor.type, TensorType)
        complex_space = self.graph.const(_complex(tensor.type), VALUE)
        call = ops.Call("core.conjugate")
        return self.graph.add(call, (tensor, complex_space), tensor.type, span)

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
            return self.graph.add(ops.Map("core.subs"), [expr, *pairs], expr.type, span)
        return self._call("core.subs", [expr, *pairs], _passed(expr.type), span)

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
            return self.graph.add(ops.Map("core.time_derivative"), (value, order), value.type, span)
        return self._call("core.time_derivative", [value, order], _passed(value.type), span)

    # -- collections -----------------------------------------------------------

    def _List(self, node: ast.List, scope: Scope, ctx: Context) -> Node:
        items = [self.value(item, scope, ctx) for item in node.items]
        return self._call("core.list", items, VALUE, node.span)

    def _Tuple(self, node: ast.Tuple, scope: Scope, ctx: Context) -> Node:
        items = [self.value(item, scope, ctx) for item in node.items]
        return self._call("core.tuple", items, VALUE, node.span)

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
        return self._call("core.piecewise", args, SCALAR, node.span)

    def _Prime(self, node: ast.Prime, scope: Scope, ctx: Context) -> Node:
        """A primed name written alone. Of a function written once with its one argument,
        as A(r) is, it is the derivative there: A' is dA/dr. Otherwise it is a name of its
        own, as t' is in \\int f(t')\\, dt', or x' in a change of coordinates."""
        name = node.func.id
        primed = name + "'" * node.order
        if isinstance(binding := scope.lookup(primed), VariableBinding):
            return binding.node
        binding = scope.lookup(name)
        arguments = self.session.functions.get(name, ())
        if binding is None and len(arguments) == 1:
            at = ast.Name(str(arguments[0]), verbatim=True, span=node.span)
            return self._prime(node, (at,), node.span, scope, ctx)
        if binding is not None or _names_builtin(node.func, latex=ctx.latex):
            raise CompileError(f"{primed} needs an argument, as in {primed}(x)", span=node.span)
        return self.graph.const(self.session.symbol(primed), SCALAR, node.span)

    # ======================================================================
    # Scripts and tensors
    # ======================================================================

    def _Scripted(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Term:
        if isinstance(node.base, ast.Prime) and isinstance(named := _unprimed(node), ast.Scripted):
            node = named  # R'_{\mu\nu}: the tensor R'
        decorated = _named(node)
        if isinstance(decorated, ast.Name) and all(
            not isinstance(s.body, ast.Name) or not _bracketed_order(s.body) for s in node.scripts
        ):  # L^{(2)} or r^*, a name of its own even where L is a tensor; V^{(a)} is an index
            return self.term(decorated, scope, ctx)
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
            if source is None:  # R_{(\mu\nu\rho)} is checked as R_{\mu\nu\rho} is
                strict = rank is not None and all(
                    item.label is not None and item.value is None and item.derivative is None
                    for s in node.scripts
                    for item in s.indices.items  # type: ignore[union-attr]
                )
                source = self._tensor_source(
                    name, binding, rank, node.span, strict=strict, node=node
                )
            differentiated = _has_derivative(node)
            if source is None and rank == 0 and differentiated:  # \phi_{;\mu}
                source = self._scalar_source(base, binding, scope, ctx)
            if source is not None:
                self._check_component(node, base, source)
                return self._access(source, node, scope, ctx)
            if differentiated:  # never a symbol named phi_mu
                raise self._underivable(node, base, binding, rank)
            if rank is None:  # g_{\mu\nu}^{-1}: a tensor with its indices, then a power
                self._check_tensor_power(node, base, binding)
        if (ones := _tensor_pattern_of_one(node)) is not None:  # 𝟙_a, or 1_a
            return self._ones(ones, node.span)
        return self._script_math(node, scope, ctx)

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

    def _ones(self, label: tuple[str, bool], span: Span, space: Space | None = None) -> Node:
        """𝟙_a: the covector whose components are all one, which sums what it meets."""
        name, upper = label
        space = space or self._space_of(name, span)
        kind = TensorType(Indices([Index(name, upper)]), (space,))
        return self.graph.add(
            ops.Call("core.ones"), (self.graph.const(space.dim, VALUE),), kind, span
        )

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
                try:
                    result, current = self._move_index(result, current, position, node.span)
                except CompileError as error:  # V_a of V^{a}, with no metric to lower it
                    if self.session.metric is not None:
                        raise
                    defined = self._written_label(source.name) + self._written_indices(
                        source.indices
                    )
                    way = "raises" if upper else "lowers"
                    raise CompileError(
                        f"{source.name} is defined as {defined}; writing {node.span.text} "
                        f"{way} an index, which needs a metric",
                        span=node.span,
                        hint=error.hint,
                        notes=error.notes,
                    ) from None
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
        """``value``'s components with axes in the order of ``indices``: those of the
        other terms of ``what``, a sum, the other side of an equation or a definition, or
        the other factor of an elementwise product."""
        assert isinstance(value.type, TensorType)
        axes = self._alignment(value.type.indices, indices, span, what)
        return self._permute(value, axes, indices)

    def _contract(
        self,
        factors: Sequence[tuple[Node, Indices]],
        span: Span,
        output: Indices | None = None,
    ) -> Node:
        """Multiply tensors, summing each index that appears once up and once down, or,
        in a space whose metric is the identity, twice in any position (see
        :meth:`~relativisticpy.typing.typer.Typer._contraction`)."""
        op, kind = self._contraction(factors, span, output)
        return self.graph.add(op, [node for node, _ in factors], kind, span)

    def _inverse(self, node: Node, span: Span) -> Node:
        """The inverse of a two-index tensor, whose components are the inverse matrix. Of
        a form, with both indices down, it has both up, as :math:`(\\Sigma^{-1})^{ab}`
        does; of a map it is a map, :math:`(A^{-1})^a{}_b`. It goes the other way between
        two spaces: its first index ranges over the space of the tensor's second."""
        assert isinstance(node.type, TensorType)
        assert node.type.rank == 2
        first, second = node.type.indices
        if first.upper == second.upper:
            first, second = first.flip(), second.flip()
        indices = Indices([Index(second.name, first.upper), Index(first.name, second.upper)])
        kind = TensorType(indices, node.type.spaces[::-1])
        return self.graph.add(ops.Inverse(), (node,), kind, span)

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
        axes = self._symmetrized(node.type, labels, span)
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

    def _dalembertian(self, operand: Node, span: Span, written: str = "\\Box") -> Node:
        """``□X = ∇^a ∇_a X``."""
        self.session.require_coordinates(span, written)
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
        for script in scripts:  # \partial_{}: no derivative at all
            if script.body is None and (script.indices is None or not script.indices.items):
                where = "subscript" if script.kind == "_" else "superscript"
                raise CompileError(
                    f"empty {where}",
                    span=script.span,
                    hint=f"write the index of the derivative, as in {node.base.span.text}_\\mu",
                )
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
            if item.label in GREEK:  # \partial_\mu \phi, nothing declared: by a variable \mu
                label = self._written_label(item.label)
                self._note(
                    f"there are no coordinates, so a derivative with the index {label} is one "
                    f"with respect to a variable named {label}; declare coordinates to make "
                    f"{label} an index"
                )
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

    # -- abstract index notation -------------------------------------------------------
    #
    # An equation with a tensor that has no components, such as T_{\mu\nu} in Einstein's
    # equations before T is defined, is kept as notation (relativisticpy.abstract), and
    # solve, trace, simplify and subs work on it as notation. The compiler emits the
    # operations of notation, whose types their rules give (relativisticpy.typing.notation),
    # and the engines compute them as the statement runs. Where components are needed, in
    # a definition or a component equation, notation is turned into components with the
    # metric in force, by the lowerer, once the engines have computed it.

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

    def _notation(self, node: ast.Expr, scope: Scope, ctx: Context) -> Node:
        """Notation: what it is, as a value of the program."""
        return self._abstract(node, scope, ctx)

    def _algebra(self) -> Node:
        """The algebra of notation in force, which the engines make the tensors of
        notation in."""
        algebra = self.session.algebra()
        attributes = (("dim", algebra.dim), ("metric", algebra.metric), ("number", algebra.number))
        return self.graph.add(ops.Call("notation.algebra", attributes), (), VALUE)

    def _notation_call(
        self,
        name: str,
        args: Sequence[Node],
        kind: Type,
        span: Span | None,
        attributes: ops.Attributes = (),
    ) -> Node:
        """The operation of notation ``notation.<name>``, of the type its rule gives. It is
        computed as the statement that writes it runs, as the check of a definition is
        (see :class:`~relativisticpy.compiler.scope.Check`), in the order it is written:
        so notation that cannot be computed, as a solve for a tensor it does not contain,
        is refused there, with how the statement was read, before what is written after
        it."""
        node = self.graph.add(ops.Call(f"notation.{name}", attributes), list(args), kind, span)
        self._checks.append(Check(node, span, notes=tuple(self._notes)))
        return node

    def _abstract(self, node: ast.Expr, scope: Scope, ctx: Context) -> Node:
        node = _operand(node)
        if isinstance(node, ast.Name):
            binding = scope.lookup(node.id)
            if (
                isinstance(binding, VariableBinding)
                and (found := self._kept(binding.node)) is not None
            ):
                return found
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _NOTATION_OPERATIONS
        ):
            return self._notation_operation(node, scope, ctx)
        if isinstance(node, ast.Chain):
            raise CompileError("give the links of a chain one at a time", span=node.span)
        self._index_structure(node, scope)  # what breaks the rules, before it is built
        algebra = self._algebra()
        if isinstance(node, ast.Equation):
            lhs = self._tensorial(node.lhs, scope, ctx)
            rhs = self._tensorial(node.rhs, scope, ctx)
            kind = rules.equation(_notation_type(lhs), _notation_type(rhs))
            return self._notation_call("equation", (algebra, lhs, rhs), kind, node.span)
        term = self._tensorial(node, scope, ctx)
        kind = rules.expression(_notation_type(term))
        return self._notation_call("expression", (algebra, term), kind, node.span)

    def _notation_operation(self, node: ast.Call, scope: Scope, ctx: Context) -> Node:
        assert isinstance(node.func, ast.Name)
        name, args = node.func.id, node.args
        if not args:
            raise CompileError(f"{name} needs an equation", span=node.span)
        try:
            subject = self._abstract(args[0], scope, ctx)
            kind = _notation_type(subject)
            if name == "solve":
                if len(args) != 2 or not kind.equation:
                    raise CompileError(
                        "solve takes an equation and the tensor to solve it for",
                        span=node.span,
                        hint="for example solve(E, T_{\\mu\\nu})",
                    )
                self._index_structure(args[1], scope)
                target = self._tensorial(args[1], scope, ctx)
                unknown = self._notation_call(
                    "unknown", (subject, target), target.type, args[1].span
                )
                solved = rules.solve(kind, _notation_type(unknown))
                written = (("unknown", args[1].span.text),)
                return self._notation_call("solve", (subject, unknown), solved, node.span, written)
            if name in ("LHS", "RHS"):
                if not kind.equation:
                    raise CompileError(f"{name} takes an equation", span=node.span)
                which = rules.side(kind, 0 if name == "LHS" else 1)
                return self._notation_call(name.lower(), (subject,), which, node.span)
            if name == "subs":
                rule = self._abstract(args[1], scope, ctx) if len(args) == 2 else None
                if rule is None or not _notation_type(rule).equation:
                    raise CompileError(
                        "subs takes an expression and a rule",
                        span=node.span,
                        hint="for example subs(E, u^\\alpha u_\\alpha = -1)",
                    )
                return self._notation_call("subs", (subject, rule), rules.kept(kind), node.span)
            if name == "trace":
                return self._notation_call("trace", (subject,), rules.trace(kind), node.span)
            return self._notation_call("simplify", (subject,), rules.kept(kind), node.span)
        except CompileError as error:
            if error.span is None:
                error.span = node.span
            raise

    def _tensorial(self, node: ast.Expr, scope: Scope, ctx: Context) -> Node:
        """An expression read as abstract index notation: a term of notation, which the
        engines make a SymPy tensor expression."""
        node = _operand(node)
        algebra = self.session.algebra()
        if isinstance(node, ast.Binary) and node.op in ("+", "-"):
            signed = _signed_terms(node)
            terms = [self._tensorial(term, scope, ctx) for _, term in signed]
            kind = rules.add([_notation_type(term) for term in terms])
            signs = (("signs", tuple(sign for sign, _ in signed)),)
            return self._notation_call("add", terms, kind, node.span, signs)
        if isinstance(node, ast.Binary) and node.op == "*":
            factors = [self._tensorial(factor, scope, ctx) for factor in _product_factors(node)]
            kind = rules.mul([_notation_type(factor) for factor in factors])
            return self._notation_call("mul", factors, kind, node.span)
        if isinstance(node, ast.Binary) and node.op == "/":
            by = self._tensorial(node.right, scope, ctx)
            denominator = self._notation_call("denominator", (by,), by.type, node.right.span)
            numerator = self._tensorial(node.left, scope, ctx)
            kind = rules.quotient(_notation_type(numerator))
            return self._notation_call("div", (numerator, denominator), kind, node.span)
        if isinstance(node, ast.Unary) and node.op in ("-", "+"):
            value = self._tensorial(node.operand, scope, ctx)
            if node.op == "+":
                return value
            return self._notation_call("neg", (value,), value.type, node.span)
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
            if (
                isinstance(binding, VariableBinding)
                and (found := self._kept(binding.node)) is not None
            ):
                kept = _notation_type(found)
                if kept.equation:
                    raise CompileError(f"{node.id} is an equation", span=node.span)
                return self._notation_call("value", (found,), rules.value(kept), node.span)
            if binding is None and node.id == "R":
                kind = rules.term(algebra)
                return self._notation_call("ricci_scalar", (self._algebra(),), kind, node.span)
            if isinstance(binding, TensorBinding):  # a tensor without indices: its trace
                symbol = self.graph.const(Symbol(node.id), SCALAR, node.span)
                return self._notation_call("scalar", (symbol,), rules.term(algebra), node.span)
        if isinstance(node, ast.Equation):
            raise CompileError("an equation cannot be part of an expression", span=node.span)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in _NOTATION_OPERATIONS:
                found = self._notation_operation(node, scope, ctx)
                kept = _notation_type(found)
                if kept.equation:
                    raise CompileError(
                        "an equation cannot be part of an expression", span=node.span
                    )
                return self._notation_call("value", (found,), rules.value(kept), node.span)
            builtin = _names_builtin(node.func, latex=ctx.latex)
            function = isinstance(scope.lookup(name), FunctionBinding) or builtin
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
        if not rules.a_factor(value):
            raise CompileError(
                "this factor is computed from tensors, but tensor notation is read before "
                "anything is computed",
                span=node.span,
                hint="give the tensors of the notation components, so that the equation is "
                "computed with components",
            )
        return self._notation_call("scalar", (value,), rules.term(algebra), node.span)

    def _abstract_tensor(
        self, pattern: tuple[str, list[tuple[str, bool]]], scope: Scope, span: Span
    ) -> Node:
        name, entries = pattern
        binding = scope.lookup(name)
        if isinstance(binding, VariableBinding) and rules.notation_of(binding.node) is not None:
            raise IndexNotationError(
                f"{name} stands for notation, whose indices are its own",
                span=span,
                hint=f"write it without indices, as in trace({name})",
            )
        quantity = None
        if not isinstance(binding, TensorBinding):
            quantity = self._builtin_tensor(name, len(entries))
        head = _NOTATION_NAMES.get(quantity or "", name)
        indices = Indices(Index(label, upper) for label, upper in entries)
        algebra = self.session.algebra()
        attributes: ops.Attributes = (("name", head), ("indices", indices))
        if quantity is not None:
            attributes += (("quantity", quantity),)
        else:  # a tensor of one's own, whose components notation's may need
            self.session.notation_tensors.setdefault(algebra, set()).add((head, len(entries)))
        kind = rules.tensor(algebra, indices)
        return self._notation_call("tensor", (self._algebra(),), kind, span, attributes)

    def _components_of(self, node: Node, scope: Scope, ctx: Context, span: Span) -> Node:
        """A value kept as notation, turned into components with the metric in force."""
        found = self._kept(node)
        if found is None:
            return node
        kept = _notation_type(found)
        if kept.equation:
            raise CompileError(
                "an equation in index notation has no components",
                span=span,
                hint="take one side of it, as in RHS(E)",
            )
        return self._lower(found, kept, scope, span)

    def _lower(self, notation: Node, kept: NotationType, scope: Scope, span: Span) -> Node:
        """Notation turned into components with the metric in force. Each term of the
        notation is written in the language, with its scalar coefficient bound to a name,
        and compiled; but the terms are known only once the notation is computed, so the
        lowerer does this as the statement runs (see :func:`lower_notation`), given the
        metric and the tensors in scope that the notation may be written with. The
        components' type comes from notation's rules: its free indices, in the order
        they are written."""
        spaces = [self._notation_space(index.name, span) for index in kept.free]
        for index, space in zip(kept.free, spaces, strict=True):
            if space.dim or space.basis is None:  # not the space of no coordinates
                self._check_label(index.name, space, span)
        kind = rules.components(kept.free, spaces)
        session = self.session
        references = [session.metric.node] if session.metric is not None else []
        tensors: dict[str, Indices | None] = {}
        for name, _ in sorted(session.notation_tensors.get(kept.algebra, ())):
            binding = scope.lookup(name)
            if name in tensors or not isinstance(binding, TensorBinding | VariableBinding):
                continue
            tensors[name] = binding.indices if isinstance(binding, TensorBinding) else None
            references.append(binding.node)
        labels = {
            label: declared
            for label in kept.free.names
            if (declared := self._labels.get(label) or session.labels.get(label)) is not None
        }
        context = Lowering(
            kind=kind,
            span=span,
            text=self._text,
            coordinates=session.coordinates,
            names=tuple(session.names.items()),
            metric=session.metric.names if session.metric is not None else None,
            dropped=session.dropped,
            tensors=tuple(tensors.items()),
            labels=tuple(labels.items()),
            bound=self._bound,
        )
        lower = ops.Call("notation.lower", (("context", context),))
        lowered = self.graph.add(lower, (notation, *references), VALUE, span)
        self._checks.append(Check(lowered, span, notes=tuple(self._notes)))
        return self.graph.add(ops.Call("notation.components"), (lowered,), kind, span)

    def _notation_components(self, kept: Any, scope: Scope, ctx: Context, span: Span) -> Node:
        """Notation, computed, turned into components: each term written in the language,
        with its scalar coefficient bound to a name, and compiled. The lowerer does this
        as the statement runs (see :func:`lower_notation`)."""
        value, space = kept.value, kept.space
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
        # The built-in tensors and R are bound to the geometry itself, not reached by name:
        # the metric may use R as a symbol, and the Einstein tensor may be called E.
        used: set[str] = set()
        for coefficient, tensors in abstract.terms(value):
            used.update(q for t in tensors if (q := _quantity(t.head, space)) is not None)
            if coefficient.has(space.ricci_scalar):
                used.add("ricci_scalar")
        for quantity in sorted(used):
            node = self._geometry(quantity, span)
            binding: VariableBinding | TensorBinding = VariableBinding(node)
            if isinstance(node.type, TensorType):
                binding = TensorBinding(node, node.type.indices)
            inner.bind(f"#{quantity}", binding)
        total = reduce(lambda a, b: ast.Binary("+", a, b, span=span), summands)
        return self.value(total, inner, replace(ctx, latex=False))

    def _kept(self, node: Node) -> Node | None:
        """The notation a node holds, if any, in the algebra in force: notation kept before
        the coordinates changed, or the metric was renamed, is read in this one."""
        kept = rules.notation_of(node)
        if kept is None:
            return None
        algebra = self.session.algebra()
        if kept.algebra == algebra:
            return node
        tensors = self.session.notation_tensors
        tensors.setdefault(algebra, set()).update(tensors.get(kept.algebra, ()))
        kind = rules.rebased(kept, algebra)
        return self._notation_call("rebase", (self._algebra(), node), kind, node.span)

    # -- helpers -------------------------------------------------------------

    def _wrapped(self, indices: Indices) -> ops.Attributes:
        """The attributes of an operation that gives tensor components to a function that
        wants a :class:`~relativisticpy.tensors.Tensor`: its indices, as they are shown,
        and the coordinates."""
        return (("indices", _public(indices)), ("coordinates", self.session.coordinates or ()))


# ---------------------------------------------------------------------------
# Syntax helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Lowering:
    """What turning notation into components needs of the statement that writes it,
    besides the nodes of the metric and of the tensors it may be written with: the
    components' type, where they are written, the statement's text, to write labels as it
    does, the coordinates, the names of the built-in tensors, the names the metric is
    written with (``None`` without a metric), where the metric was dropped, the tensors
    bound to the names notation may write (each with its indices, or ``None`` for a
    value), the spaces declared for the free indices, and the labels an expectation
    binds there."""

    kind: Type
    span: Span
    text: str
    coordinates: tuple[Symbol, ...] | None
    names: tuple[tuple[str, str], ...]
    metric: frozenset[str] | None
    dropped: Span | None
    tensors: tuple[tuple[str, Indices | None], ...]
    labels: tuple[tuple[str, Space], ...]
    bound: frozenset[str]

    def __str__(self) -> str:
        """How the IR shows it: the components' type and the tensors they may need."""
        written = ", ".join(name for name, _ in self.tensors)
        return f"{self.kind}" + (f" of {written}" if written else "")


def lower_notation(kept: IndexExpression, references: Sequence[Node], context: Lowering) -> Node:
    """The IR of the components of ``kept``, notation the engines have computed, as the
    statement in ``context`` would have them: compiled in a session of its own, with the
    metric and tensors the ``references`` are the nodes of. Of the components' type, as
    notation's rules gave it; zero, of notation that vanishes."""
    session = Session(
        graph=Graph(),
        coordinates=context.coordinates,
        names=dict(context.names),
        dropped=context.dropped,
    )
    nodes = list(references)
    if context.metric is not None:
        session.metric = MetricState(nodes.pop(0), context.metric)
    scope = Scope()
    for (name, indices), node in zip(context.tensors, nodes, strict=True):
        binding = TensorBinding(node, indices) if indices is not None else VariableBinding(node)
        scope.bind(name, binding)
    compiler = Compiler(session)
    compiler._depth = 1  # a statement's part, whose notes are the statement's
    compiler._text, compiler._labels, compiler._bound = (
        context.text,
        dict(context.labels),
        context.bound,
    )
    node = compiler._notation_components(kept, scope, Context(), context.span)
    kind = context.kind
    if not isinstance(kind, TensorType):
        return node
    if not isinstance(node.type, TensorType):
        if _is_zero(node):  # notation that vanishes, whatever indices it has
            return session.graph.add(ops.ToArray(kind.shape), (node,), kind, context.span)
        raise IndexNotationError(
            f"the components of this notation have no free indices, where it has "
            f"{compiler._written_indices(kind.indices)}",
            span=context.span,
        )
    return compiler._align(node, kind.indices, context.span, what="notation")


def _notation_type(node: Node) -> NotationType:
    """The type of a value or term of notation."""
    assert isinstance(node.type, NotationType)
    return node.type


def _passed(kind: Type) -> Type:
    """The type of what an operation gives of a value of this type, as a derivative of
    it: the same, but of notation, which only notation's own operations give."""
    return VALUE if isinstance(kind, NotationType) else kind


def _unsigned(factor: ast.Expr) -> list[ast.Expr]:
    """Take the sign off a factor, so that ``-\\partial_\\mu \\phi`` and ``-d t^2`` let
    the operator act on what follows: ``-x`` becomes ``-1``, ``x``."""
    if isinstance(factor, ast.Unary) and factor.op == "-":
        return [ast.Number("-1", span=factor.span), *_unsigned(factor.operand)]
    return [factor]


def _uniform_weights(measure: Node) -> bool:
    """Whether a measure declared for a space weighs its elements alike."""
    from relativisticpy.tensors import numerical

    if not isinstance(measure.op, ops.Const):
        return False
    weights = numerical.numpy().asarray(measure.op.value, dtype=float)
    return bool(numerical.numpy().allclose(weights, weights.mean(), rtol=1e-12, atol=0.0))


# -- abstract index notation ---------------------------------------------------------

# The names built-in tensors have in notation, whatever they were written as.
_NOTATION_NAMES = {
    "riemann": "R", "ricci": "R", "einstein": "G", "weyl": "C", "christoffel": "Gamma",
}  # fmt: skip


def _tensor_syntax(
    tensor: Any, space: Any, labels: dict[str, str], fresh: Iterator[int], span: Span
) -> ast.Scripted:
    """A tensor of notation, written as the language writes it, dummies renamed. A
    built-in tensor is written with the name its geometry is bound to, as #riemann."""
    quantity = _quantity(tensor.head, space)
    name = f"#{quantity}" if quantity is not None else tensor.head.name
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


def _quantity(head: Any, space: Any) -> str | None:
    """The built-in tensor a head of notation stands for: the metric, the Kronecker delta,
    or a tensor of the metric's geometry; ``None`` for a tensor of one's own."""
    if head == space.metric:
        return "metric"
    if head == space.delta:
        return "kronecker"
    quantity: str | None = space.quantity(head)
    return quantity


def _trace_syntax(symbol: Symbol, head: Any, fresh: Iterator[int], span: Span) -> ast.Expr:
    """The trace a scalar of notation stands for: the Ricci scalar, bound to the name
    #ricci_scalar, or T^{a}{}_{a}."""
    if head is None:
        return ast.Name("#ricci_scalar", verbatim=True, span=span)
    label = f"#t{next(fresh)}"
    up = ast.Script(
        "^", None, ast.IndexList((ast.IndexItem(label, span=span),), span=span), span=span
    )
    down = ast.Script(
        "_", None, ast.IndexList((ast.IndexItem(label, span=span),), span=span), span=span
    )
    return ast.Scripted(ast.Name(head.name, verbatim=True, span=span), (up, down), span=span)
