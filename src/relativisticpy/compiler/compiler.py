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
import difflib
import inspect
import itertools
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import reduce
from typing import Any, Literal

import sympy
from sympy import Symbol
from sympy.core.function import AppliedUndef
from sympy.tensor import tensor as symbolic

from relativisticpy import abstract, display, spacetimes
from relativisticpy.abstract import IndexEquation, IndexExpression
from relativisticpy.compiler.builtins import _SPACETIMES, BUILTINS, CONSTANTS, SPACETIMES, Builtin
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
    GEOMETRY_POSITIONS,
    RENAMING_STATEMENTS,
    MetricState,
    Session,
    describe,
)
from relativisticpy.diffgeom import differential
from relativisticpy.errors import (
    CompileError,
    EvaluationError,
    IndexNotationError,
    MetricError,
    ParseError,
    RelativisticPyError,
)
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
from relativisticpy.language.vocabulary import GREEK, UNICODE_LETTERS
from relativisticpy.runtime.executor import fold
from relativisticpy.runtime.kernels import refuse_quoted
from relativisticpy.tensors import Index, Indices, arrays
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
        # How often the statement being compiled writes each label as a sub- and superscript.
        self._scripts: dict[str, Counter[str]] = {"_": Counter(), "^": Counter()}
        # The names the statement being compiled writes as factors, not as functions
        # called: G in 2GM, which makes G(p^2 + q^2) in the same statement a product.
        self._factors: frozenset[str] = frozenset()
        # The statement's text, to write index labels as it does.
        self._text = ""
        self._bound: frozenset[str] = frozenset()
        # How the statement being compiled was read, where that could surprise, and the
        # notes of statements that showed nothing to carry them, such as definitions.
        self._notes: list[str] = []
        self._unshown: list[str] = []
        self._depth = 0

    # ======================================================================
    # Statements
    # ======================================================================

    def compile(self, program: ast.Program) -> list[Output]:
        """Compile a program; return the values it shows, in order."""
        outputs: list[Output] = []
        for statement in program.statements:
            outputs.extend(self.statement(statement, self.session.globals))
        return outputs

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

    def _note(self, note: str) -> None:
        """Say how the statement being compiled was read: once, however often it is."""
        if note not in self._notes:
            self._notes.append(note)

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

    def _reading(
        self, equation: ast.Equation, scope: Scope, ctx: Context
    ) -> tuple[Literal["define", "function", "claim", "notation"], str | None]:
        """Whether ``lhs = rhs`` defines its left side, and if not, the note that says why
        it is a claim where it could have been read as a definition.

        ``=`` defines only what has no meaning yet: a new name, a new function written
        with plain names as its arguments, ``f(x) = x^2``, a new tensor, or the metric,
        by a line element or while there is none. A name the metric in force is written
        with has no value, so ``\\Delta = ...`` defines it, as "where" does after a line
        element. Everything else is a claim, to check: a name, tensor or function that
        already has a value (``:=`` gives it a new one), a component such as ``g_{tt}``, a
        built-in such as ``R``, and a constant such as ``\\pi``. A name in its own value, as
        in ``R = 2/R^2``, makes an equation, never a definition."""
        target, value = _named(_unprimed(equation.lhs)), equation.rhs
        written = equation.lhs.span.text
        if _is_line_element(target):
            return "define", None
        if isinstance(target, ast.Prime):  # t' = \gamma(t - vx), a new coordinate
            primed = target.func.id + "'" * target.order
            if isinstance(scope.lookup(primed), VariableBinding):
                return "claim", _has_a_value(written)
            if self._names_a_symbol(target, scope) and not self._in_own_value(
                primed, value, scope, ctx
            ):
                return "define", None
            return "claim", None
        if isinstance(target, ast.Name):
            return self._name_reading(target, written, value, scope, ctx)
        if isinstance(target, ast.Call):
            return self._function_reading(target, value, scope, ctx)
        if not isinstance(target, ast.Scripted) or not isinstance(target.base, ast.Name):
            return "claim", None
        if any(s.indices is None for s in target.scripts):
            return "claim", None
        items = [item for s in target.scripts for item in s.indices.items]  # type: ignore[union-attr]
        if not all(_plain_label(item) for item in items):
            return "claim", None
        labels = [str(item.label) for item in items]
        if len(set(labels)) != len(labels) or any(map(self._names_component, labels)):
            return "claim", None  # a component, or a trace: an equation about what is there
        name = target.base.id
        if name == self.session.names["metric"] and len(items) == 2:
            space = self._labels.get(labels[0]) or self.session.labels.get(labels[0])
            if space is not None and not space.spacetime:
                in_force = space in self.session.space_metrics
            else:
                in_force = self.session.metric is not None
            if not in_force:
                return "define", None
            return "claim", _built_in(written, "the metric in force", f"{written} := ...")
        binding = scope.lookup(name)
        if binding is not None:  # u^\mu = [2, 0] after u^\mu = [1, 0]
            tensor = isinstance(binding, TensorBinding) or (
                isinstance(binding, VariableBinding) and isinstance(binding.node.type, TensorType)
            )
            return "claim", _has_a_value(written) if tensor else None
        if self._builtin_tensor(name, len(items)) is not None:
            return "claim", None
        if _writes_tensor(value, name):  # T_{\mu\nu} = T_{\nu\mu}: a symmetry, not a definition
            return "claim", None
        return "define", None

    def _name_reading(
        self, target: ast.Name, written: str, value: ast.Expr, scope: Scope, ctx: Context
    ) -> tuple[Literal["define", "claim", "notation"], str | None]:
        """How ``x = ...`` is read: see :meth:`_reading`."""
        name = target.id
        if name in _DECLARATIONS:  # Coordinates = [t, r] declares, as := does
            return "define", None
        if name in _CLAIMED:  # \pi = 3.14159: a claim about the constant
            return "claim", _built_in(written, "a constant", f"{written} := ...")
        if scope.lookup(name) is not None:  # x = 3 after x := 2
            return "claim", _has_a_value(written)
        if _indexed_name(target, scope) is not None:  # y_t of a tensor y: a component
            return "claim", None
        quantity = self._builtin_scalar(name)
        if quantity is not None:  # R = 0 with a metric in force
            return "claim", _built_in(written, _BUILT_IN_SCALARS[quantity], f"{written} := ...")
        ricci = name in ("R", self.session.names["ricci"])
        if ricci and self.session.metric is None:  # R = -\kappa T, as notation
            return "notation", _built_in(written, "the Ricci scalar", f"{written} := ...")
        if self._in_own_value(name, value, scope, ctx):
            return "claim", (
                f"{written} appears on both sides, so this is an equation, not a "
                f"definition of {written}"
            )
        return "define", None

    def _function_reading(
        self, target: ast.Call, value: ast.Expr, scope: Scope, ctx: Context
    ) -> tuple[Literal["function", "claim"], str | None]:
        """How ``f(x) = ...`` is read: a definition of a function with the arguments as
        its parameters, where they are plain names and ``f`` has no meaning yet; a claim
        otherwise, as ``f(2) = 4`` and ``r(\\tau) = ...`` of a coordinate are."""
        func = target.func
        if not isinstance(func, ast.Name):
            return "claim", None
        name = func.id
        binding = scope.lookup(name)
        if binding is not None:
            params = binding.definition.params if isinstance(binding, FunctionBinding) else ()
            again = f"{name}({', '.join(params)}) := ..." if params else f"{name} := ..."
            note = (
                f"{func.span.text} already has a value, so = compares it; to give "
                f"{func.span.text} a new value, write {again}"
            )
            return "claim", note
        names = [arg.id for arg in target.args if isinstance(arg, ast.Name)]
        if not names or len(names) != len(target.args) or len(set(names)) != len(names):
            return "claim", None  # f(2) = 4: a claim about the value at 2
        if ctx.latex and _splittable(func) and name not in BUILTINS:
            return "claim", None  # 2Gm(r) = ...: a product of letters
        coordinates = {c.name for c in self.session.coordinates or ()}
        if name in coordinates or self._never_a_function(func, ctx):
            return "claim", None  # r(\tau) = ...: a coordinate along a path
        if name in self._factors:  # G(p^2 + q^2) beside 2GM: a product
            return "claim", None
        if any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
            for node in _walk(value)
        ):
            note = (
                f"{func.span.text} appears on both sides, so this is an equation, not a definition"
            )
            return "claim", note
        return "function", None

    def _never_a_function(self, func: ast.Name, ctx: Context) -> bool:
        """Whether a name, bound to nothing in the program, is never a function of what is
        in brackets after it: a builtin is one already, and a constant, such as ``e``, ``i``
        or one declared with ``Constants := [G]``, multiplies them."""
        name = func.id
        if _names_builtin(func, latex=ctx.latex) or name in CONSTANTS:
            return True
        declared = self.session.assumptions.get(name)
        return declared is not None and bool(declared.assumptions0.get("constant"))

    def _in_own_value(self, name: str, value: ast.Expr, scope: Scope, ctx: Context) -> bool:
        """Whether ``name`` is written in ``value`` other than as a function called there,
        as R is in 2/R^2 but \\beta is not in \\beta(r), or is in the value of a name
        written there, as \\rho is in p after p := w\\rho."""
        other = _other_names(value)  # f in f(x), x in x' and h in \bar h name other things
        for node in _walk(value):
            if isinstance(node, ast.Prime):  # x' is a name of its own, maybe written with x
                written = node.func.id + "'" * node.order
            elif isinstance(node, ast.Accent | ast.Scripted) and isinstance(
                named := _named(node), ast.Name
            ):
                written = named.id  # \bar h, E^{(2)}
            elif isinstance(node, ast.Name) and id(node) not in other:
                written = node.id
            else:
                continue
            if written == name:
                return True
            binding = scope.lookup(written)
            if binding is None and isinstance(node, ast.Name) and ctx.latex and _splittable(node):
                known = node.id in BUILTINS or node.id in CONSTANTS
                if not known and name in split_index_labels(node.id):
                    return True  # K in 2GK
            if isinstance(binding, VariableBinding | TensorBinding) and _mentions_symbol(
                binding.node, name
            ):
                return True
        return False

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

    def _names_component(self, label: str) -> bool:
        """Whether an index label names a component: a coordinate's name, as ``t`` in
        ``g_{tt}``, unless the label ranges over a space other than spacetime."""
        space = self._labels.get(label) or self.session.labels.get(label)
        if space is not None and not space.spacetime:
            return False
        return any(c.name == label for c in self.session.coordinates or ())

    def _along_a_path(self, node: ast.Derivative, scope: Scope) -> str | None:
        """The coordinate a derivative is of, when it is by a parameter that is no
        coordinate, as dt/d\\lambda along a geodesic is: the coordinate is no function
        of it, so the derivative would be zero, which is not what is meant."""
        coordinates = {c.name for c in self.session.coordinates or ()}

        def free(node: ast.Expr | None) -> str | None:
            if isinstance(node, ast.Name) and scope.lookup(node.id) is None:
                return node.id
            return None

        coordinate = free(node.expr)
        if coordinate is None or coordinate not in coordinates:
            return None
        parameters = [free(variable) for variable, _ in node.variables]
        along = all(p is not None and p not in coordinates for p in parameters)
        return node.expr.span.text if along and node.expr is not None else None

    def _names_coordinate(self, target: ast.Expr, scope: Scope) -> bool:
        """Whether ``target`` is the bare name of a coordinate in force, bound to nothing
        else in the program."""
        return (
            isinstance(target, ast.Name)
            and scope.lookup(target.id) is None
            and any(c.name == target.id for c in self.session.coordinates or ())
        )

    def _names_a_symbol(self, prime: ast.Prime, scope: Scope) -> bool:
        """Whether a primed name stands for a symbol of its own, as t' does, rather than
        the derivative of a function, as A' does where A(r) is written."""
        name = prime.func.id
        if isinstance(scope.lookup(name + "'" * prime.order), VariableBinding):
            return True
        return (
            scope.lookup(name) is None
            and len(self.session.functions.get(name, ())) != 1
            and not _names_builtin(prime.func)
        )

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

    def _written_in_metric(self, name: str) -> bool:
        """Whether the metric in force is written with ``name`` as a symbol or a function
        that has no value, as with \\Delta in ``ds^2 = -\\Delta dt^2 + ...``: a definition of
        it then gives it that value there too. A coordinate never is. It is read from what
        the metric is computed from, without computing it (see :func:`_written_with`), so a
        name that cancels, as x in 1 + x - x, counts too."""
        metric = self.session.metric
        if metric is None or name not in metric.names:
            return False
        if any(c.name == name for c in self.session.coordinates or ()):
            return False
        return name in _written_with(metric.node)

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

    def _check_spacetime_name(self, value: ast.Expr, scope: Scope, ctx: Context) -> None:
        """Refuse, where a metric is defined, a spacetime misspelt, as ``Schwarzchild(M)``,
        spelt as in Python, ``schwarzschild(M)``, or named without being called."""
        func = value.func if isinstance(value, ast.Call) else value
        if not isinstance(func, ast.Name) or scope.lookup(func.id) is not None:
            return
        name, written = func.id, func.span.text
        listed = ", ".join(list(_SPACETIME_NAMES)[:-1]) + " and " + list(_SPACETIME_NAMES)[-1]
        known = f"the spacetimes are {listed}"
        if not isinstance(value, ast.Call):
            if name in _SPACETIME_NAMES:
                raise CompileError(
                    f"{written} is a spacetime: call it, as in {name}()", span=value.span
                )
            spelt = _close(name, _SPACETIME_NAMES)
            if spelt is not None:
                raise CompileError(
                    f"unknown spacetime {written}",
                    span=value.span,
                    hint=f"did you mean {spelt}()?",
                    notes=[known],
                )
            return
        if name in BUILTINS or len(name) < 2 or name in GREEK:
            return
        args = ", ".join(arg.span.text for arg in value.args)
        same = next((s for s in _SPACETIME_NAMES if s.lower() == name.lower()), None)
        if same is not None:
            raise CompileError(
                f"unknown spacetime {written}",
                span=func.span,
                hint=f"in the language, spacetimes are capitalised: {same}({args})",
                notes=[known],
            )
        spelt = _close(name, _SPACETIME_NAMES)
        if spelt is not None or name[:1].isupper():
            raise CompileError(
                f"unknown spacetime {written}",
                span=func.span,
                hint=f"did you mean {spelt}({args})?" if spelt else None,
                notes=[known],
            )

    def _check_metric(self, metric: Node, span: Span | None) -> None:
        """Compute a metric where it is defined, so that one that cannot be is refused
        there, with the reason, rather than wherever it is first used: components of the
        wrong size, components that are not symmetric, or a determinant of zero."""
        try:
            self.session.evaluate(metric)
        except RelativisticPyError as error:
            raise self._metric_error(metric, span) or error from None

    def _metric_error(self, metric: Node, span: Span | None) -> EvaluationError | None:
        """Why the metric ``metric`` could not be computed, in the user's terms; ``None``
        where its components themselves could not be, which their own error says."""
        coordinates = self.session.coordinates or ()
        given = metric.args[0]
        if isinstance(given.op, ops.ToArray):  # diag(-1, 1, 1), checked for its shape
            given = given.args[0]
        try:
            components = arrays.to_array(self.session.evaluate(given))
        except Exception:
            return None
        names = [latex_label(c.name) for c in coordinates]
        listed = _listed(names)
        shape = tuple(components.shape)
        count = len(coordinates)
        if shape != (count, count):
            found = "×".join(map(str, shape)) if shape else "no"
            return EvaluationError(
                f"the metric has {found} components, but there are {count} coordinates",
                span=span,
                notes=[f"the coordinates are {listed}"],
                hint=f"give {count}×{count} components, a row and a column for each coordinate",
            )
        matrix = sympy.Matrix(components.tolist())
        for i, j in itertools.combinations(range(count), 2):
            if sympy.simplify(matrix[i, j] - matrix[j, i]) != 0:
                one, other = names[i], names[j]
                return EvaluationError(
                    f"the metric is not symmetric: g_{{{one}{other}}} = "
                    f"{display.text(matrix[i, j])} but g_{{{other}{one}}} = "
                    f"{display.text(matrix[j, i])}",
                    span=span,
                    hint="a metric has g_{\\mu\\nu} = g_{\\nu\\mu}",
                )
        if sympy.simplify(matrix.det()) != 0:
            return None
        missing = [
            names[i]
            for i in range(count)
            if all(sympy.simplify(entry) == 0 for entry in matrix.row(i))
        ]
        if missing:
            differentials = _listed([f"d{name}" for name in missing], "or")
            return MetricError(
                "the metric is degenerate: its determinant is zero",
                span=span,
                notes=[f"no term of it has {differentials}"],
                hint=f"give it a term with {_listed([f'd{name}' for name in missing])}",
            )
        return MetricError(
            "the metric is degenerate: its determinant is zero",
            span=span,
            notes=["so it has no inverse, and no curvature can be computed from it"],
        )

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

    def _constant(self, node: Node) -> Any:
        """The value of a node that is known when compiling: a constant, as a number written
        or a name given one, or data bound from Python; ``None`` for a value computed, which
        is not known until the program runs. Text in quotes is never a value."""
        if not isinstance(node.op, ops.Const):
            return None
        refuse_quoted(node, node.op.value)
        return node.op.value

    def _declares(self, relation: ast.Relation) -> bool:
        """``a, b \\in \\mathcal{A}``: declare that the labels range over a space, and say
        so; ``False`` for any other relation, which is read but not computed. Only a space
        defined by name declares its members: in a paper, ``x \\in \\mathbb{R}^3`` says
        what x is, and x may well be a coordinate."""
        names = relation.lhs.items if isinstance(relation.lhs, ast.Tuple) else (relation.lhs,)
        if relation.op != "∈" or not all(isinstance(n, ast.Name) for n in names):
            return False
        rhs = relation.rhs
        space = self.session.spaces.get(rhs.id) if isinstance(rhs, ast.Name) else None
        if space is None:
            return False
        for label in names:
            assert isinstance(label, ast.Name)
            self.session.labels[label.id] = space
        return True

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
        try:
            self._check_metric(metric, span)
        except EvaluationError as error:
            if replaceable and named and set(named) < {c.name for c in coordinates}:
                # the 2-sphere after a line element in three dimensions
                alone = _listed([latex_label(name) for name in named])
                in_force = _listed([latex_label(c.name) for c in coordinates])
                error.notes.append(
                    f"the coordinates in force are {in_force}, but this line element is "
                    f"written in {alone} alone"
                )
                error.hint = (
                    f"for a metric in {alone}, declare them first: Coordinates := "
                    f"[{', '.join(latex_label(name) for name in named)}]"
                )
            raise
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
            name = result.name
            if len(name) > 1 and name not in GREEK:  # simplify + 1, where N/2 is the lapse
                self._note(
                    f"{result.describe()} is a function, so alone it is read as a symbol; to "
                    f"apply it, write {result.describe()}(...)"
                )
            return self.graph.const(self.session.symbol(name), SCALAR, result.span)
        if isinstance(result, Operator):
            raise self._nothing_to_act_on(result, scope)
        return result

    def _nothing_to_act_on(self, operator_: Operator, scope: Scope) -> CompileError:
        """The error for an operator, or a function, written with nothing after it."""
        written = operator_.describe()
        if isinstance(operator_, FunctionReference):
            binding = scope.lookup(operator_.name)
            params = binding.definition.params if isinstance(binding, FunctionBinding) else ("x",)
            return CompileError(
                f"{written} is a function, so it needs an argument, as in "
                f"{written}({', '.join(params)})",
                span=operator_.span,
            )
        return CompileError(
            f"{written} needs something to act on, as in {written} f", span=operator_.span
        )

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
            kept = self._kept(binding.node)
            if kept is not None and kept is not _notation_of(binding.node):  # kept earlier
                return self.graph.const(kept, VALUE, node.span)
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
        self._note_symbol(node)
        if name in session.assumptions:
            return self.graph.const(session.assumptions[name], SCALAR, node.span)
        if name in CONSTANTS:
            return self.graph.const(CONSTANTS[name], SCALAR, node.span)
        if (arguments := session.functions.get(name)) is not None:  # \alpha is \alpha(t, r)
            args = [self.graph.const(argument, SCALAR) for argument in arguments]
            return self._call("core.function", args, SCALAR, node.span, (("name", name),))
        # In LaTeX a run of letters is a product, as typeset maths reads it: re^{x} is
        # r e^{x}. A builtin keeps its name when called, as in diff(f, x).
        if ctx.latex and _splittable(node):
            return self._latex_letters(node, scope, ctx)
        # A builtin applies to what follows it, as sin does in sin x; in LaTeX only if it
        # is written as a command, \operatorname{sign} x, so that N dt^2 is a product.
        if _names_builtin(node, latex=ctx.latex) and (node.verbatim or not ctx.latex):
            return FunctionReference(node.span, name)
        return self.graph.const(session.symbol(name), SCALAR, node.span)

    def _note_symbol(self, node: ast.Name) -> None:
        """Say so where a name bound to nothing is read as a symbol though it could be
        taken for something else: R and g where the metric is written with them, or
        where the metric was dropped, and the spelled-out names of built-ins, as
        RicciScalar."""
        name, written, session = node.id, node.span.text, self.session
        built_in = name in ("R", session.names["ricci"], session.names["metric"])
        if built_in and session.metric is not None and name in session.metric.names:
            meant = "the Ricci scalar is Ric" if name == "R" else "its determinant is det(g_{ab})"
            if name != session.names["ricci"]:
                self._note(
                    f"the metric is written with {written}, so {written} is that symbol; {meant}"
                )
        elif built_in and session.metric is None and session.dropped is not None:
            self._note(f"{written} is a symbol here: {session.dropped_note()[0]}")
        concept = _CONCEPTS.get(name.lower())
        if concept is not None and name not in BUILTINS:
            self._note(f"{written} is not defined, so it is a symbol; did you mean {concept}?")

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

    def _membership(self, node: ast.Relation) -> CompileError:
        """``a \\in \\mathcal{B}`` of no declared space, or ``x \\in \\mathbb{R}^3``: read but
        not computed, saying why, and how labels are made to range over a space."""
        member, space = node.lhs.span.text, node.rhs
        styled = _styled(space.span)  # \mathcal{B}, not B
        written = Span(node.span.source, node.span.start, styled.end).text
        if isinstance(space, ast.Name) and space.id not in self.session.spaces:
            name = styled.text
            return CompileError(
                f"{written} is read but not computed: {name} is not a declared space",
                span=node.span,
                hint=f"to have {member} range over a space, declare it first, as in "
                f"{name} := \\mathbb{{R}}^{{3}}",
            )
        return CompileError(
            f"{written} says what {member} is, which is read but not computed",
            span=node.span,
            hint=f"to have {member} range over a space of its own, name the space, as in "
            "\\mathcal{A} := \\mathbb{R}^{3}, and write "
            f"{member} \\in \\mathcal{{A}}",
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

    def _star(self, node: ast.Unary) -> RelativisticPyError:
        """A ``*`` with nothing on its left: the Hodge dual, *F, which is not supported,
        where a statement or an operand starts with it; otherwise an operator written
        twice or a \\cdot with nothing to multiply, a typo."""
        written = node.span.text
        star = next(
            (command for command in ("\\cdot", "\\times", "\\ast") if written.startswith(command)),
            written[:1],
        )
        before = node.span.source.text[: node.span.start].rstrip()
        doubled = before != "" and (
            before[-1] in "+-*/·×" or before.endswith(("\\cdot", "\\times"))
        )
        if star == "*" and not doubled:  # *F
            return CompileError("the Hodge dual is not supported yet", span=node.span)
        hodge = "a * before a form, as in *F, is the Hodge dual, which is not supported yet"
        return ParseError(
            f"'{star}' has nothing on its left",
            span=Span(node.span.source, node.span.start, node.span.start + len(star)),
            hint="remove it, or write what it multiplies",
            notes=[hodge] if star == "*" else [],
        )

    def _hadamard(self, node: ast.Binary, scope: Scope, ctx: Context) -> Node:
        """``A_{ta} \\odot B_{ta}``: the elementwise product, of tensors with the same free
        indices, summing nothing. Index notation has no other way to say it. Of lists or
        matrices, as ``[1, 2] \\odot [3, 4]``, it is elementwise too, never their product."""
        left = self.value(_operand(node.left), scope, ctx)
        right = self.value(_operand(node.right), scope, ctx)
        if VALUE in (left.type, right.type) and not any(
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

    def _check_terms(
        self, terms: Sequence[tuple[Node, ast.Expr]], span: Span, scope: Scope
    ) -> None:
        """That the terms of a sum have the same free indices, or say which term does
        not, and which index is wrong in it, where the others agree."""
        free = [
            frozenset(_indices(value)) if isinstance(value.type, TensorType) else frozenset()
            for value, _ in terms
        ]
        written = [
            (value, term, indices)
            for (value, term), indices in zip(terms, free, strict=True)
            if indices or not _is_zero(value)
        ]
        kinds = Counter(indices for _, _, indices in written)
        if len(kinds) <= 1:
            return
        common, count = kinds.most_common(1)[0]
        odd = [(value, term, indices) for value, term, indices in written if indices != common]
        if count > 1 and len(odd) == 1:  # the others agree: say what is wrong in this one
            _, term, indices = odd[0]
            raise self._odd_term(term, indices, common, scope)
        first, second = list(kinds)[:2]
        shown = []
        for kind in (first, second):
            value, term, _ = next(entry for entry in written if entry[2] == kind)
            shown.append(str(_indices(value)) if kind else "none")
            where = term.span
        raise IndexNotationError(
            f"the free indices of a sum must agree: {shown[0]} and {shown[1]}",
            span=where,
            notes=self._undefined_notes(
                [term for _, term, indices in written if not indices], scope
            ),
        )

    def _odd_term(
        self,
        term: ast.Expr,
        indices: frozenset[Index],
        common: frozenset[Index],
        scope: Scope,
    ) -> IndexNotationError:
        """The error for the one term of a sum whose free indices are not the others':
        ``\\rho is free in this term but in no other``, and what the others have."""
        mine = {index.name: index.upper for index in indices}
        theirs = {index.name: index.upper for index in common}
        extra = sorted(name for name in mine if name not in theirs)
        missing = sorted(name for name in theirs if name not in mine)
        moved = sorted(name for name in mine if name in theirs and mine[name] != theirs[name])
        others = self._written_free(common)
        notes = [f"the other terms have {others} free"]
        hint = None
        if extra:
            message = f"{self._are(extra)} free in this term but in no other"
            if missing:
                notes.append(f"{self._are(missing)} missing here")
            if len(extra) == 1 and len(missing) == 1:
                text = term.span.text
                wrong, right = self._written_label(extra[0]), self._written_label(missing[0])
                if text.count(wrong) == 1:
                    hint = f"did you mean {text.replace(wrong, right)}?"
        elif moved:
            name = moved[0]
            here, there = ("up" if mine[name] else "down"), ("up" if theirs[name] else "down")
            message = (
                f"{self._written_label(name)} is {here} in this term, but {there} in the others"
            )
        elif not indices:
            message = "this term has no free indices, but the others do"
            notes.extend(self._undefined_notes([term], scope))
        else:
            message = f"{self._are(missing)} free in the other terms, but not in this one"
        return IndexNotationError(message, span=term.span, notes=notes, hint=hint)

    def _are(self, labels: Sequence[str]) -> str:
        """``\\rho is`` or ``\\rho and \\sigma are``, as the statement writes the labels."""
        shown = _listed([self._written_label(label) for label in labels])
        return f"{shown} {'is' if len(labels) == 1 else 'are'}"

    def _undefined_notes(self, terms: Sequence[ast.Expr], scope: Scope) -> list[str]:
        """Notes for terms that are scalars only because a name is not defined, as y_t is
        one symbol where y is no tensor."""
        notes = []
        for term in terms:
            if isinstance(term, ast.Name) and "_" in term.id:
                base = term.id.partition("_")[0]
                if scope.lookup(base) is None:
                    notes.append(
                        f"{base} is not defined, so {term.span.text} is read as one symbol, "
                        f"{term.span.text}"
                    )
            elif (
                isinstance(term, ast.Scripted)
                and isinstance(term.base, ast.Name)
                and scope.lookup(term.base.id) is None
                and _tensor_pattern(term)
            ):
                written = term.base.span.text
                notes.append(f"{written} is not defined, so {term.span.text} is not a tensor")
        return notes

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

    def _check_statistic_name(self, factor: ast.Expr, scope: Scope) -> None:
        """``\\operatorname{E}_t`` or ``\\operatorname{Covv}_t`` before what it acts on: a
        statistic misspelt is refused with the right one; any other unknown operator
        written so is read as a symbol, and says so."""
        if not (
            isinstance(factor, ast.Scripted)
            and isinstance(factor.base, ast.Name)
            and factor.base.verbatim
            and scope.lookup(factor.base.id) is None
            and len(factor.scripts) == 1
            and factor.scripts[0].kind == "_"
            and factor.base.id not in BUILTINS
        ):
            return
        name, label = factor.base.id, factor.scripts[0].span.text.lstrip("_")
        meant = _STATISTIC_ALIASES.get(name)
        if meant is None and (close := _close(name, _STATISTICS, shortest=3)) is not None:
            meant = f"\\operatorname{{{close}}}"
        if meant is not None:
            raise CompileError(
                f"\\operatorname{{{name}}} is not a statistic RelativisticPy knows",
                span=factor.span,
                hint=f"did you mean {meant}_{label}?",
            )
        if len(name) > 1:
            self._note(
                f"\\operatorname{{{name}}} is not an operator RelativisticPy knows, so "
                f"{factor.span.text} is read as a symbol, {name}_{label}; the statistics are "
                "\\mathbb{E}, \\operatorname{Var}, \\operatorname{Cov} and "
                "\\operatorname{LedoitWolf}"
            )

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
        reciprocal = self.graph.add(ops.Pow(), (denominator, minus_one), denominator.type, span)
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
        kind = VALUE if VALUE in (base.type, exponent.type) else SCALAR
        return self.graph.add(ops.Pow(), (base, exponent), kind, span)

    def _tensor_power(
        self, metric: str | None, indices: Sequence[Index], exponent: str | None, span: Span
    ) -> IndexNotationError:
        """A tensor raised to a power: to -1, as for an inverse, or to any other."""
        flipped = self._written_indices([index.flip() for index in indices])
        if exponent == "-1":
            if metric is not None:
                hint = f"the inverse metric is {metric}{flipped}"
            elif len(indices) == 2:
                tensor = f"S{self._written_indices(indices)}"
                hint = f"the inverse of a tensor {tensor} is (S^{{-1}}){flipped}"
            else:
                hint = "only a tensor with two indices has an inverse"
            return IndexNotationError("cannot raise a tensor to the power -1", span=span, hint=hint)
        return IndexNotationError(
            "cannot raise a tensor to a power",
            span=span,
            hint="contract it with itself instead, as in V_{a} V^{a}",
        )

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
        assert isinstance(value.type, TensorType)
        if sorted(value.type.indices, key=str) == sorted(tensor.indices, key=str):
            return self._align(value, tensor.indices, span)
        left = _indices(lhs) if isinstance(lhs.type, TensorType) else value.type.indices
        right = value.type.indices if value is not lhs else tensor.indices
        raise IndexNotationError(
            f"the two sides have different free indices: {self._written_indices(left)} on the "
            f"left, {self._written_indices(right)} on the right",
            span=equation,
        )

    def _zero_side(
        self, value: Node, tensor: TensorType, side: ast.Expr, left: bool, scope: Scope
    ) -> Node:
        """The side of a tensor equation that is not a tensor: zero, or components."""
        if value.type == VALUE:
            return self.graph.add(ops.ToArray(tensor.shape), (value,), tensor, side.span)
        if not _is_zero(value):
            here = self._written_indices(tensor.indices)
            left_text, right_text = ("none", here) if left else (here, "none")
            raise IndexNotationError(
                f"the two sides have different free indices: {left_text} on the left, "
                f"{right_text} on the right",
                span=side.span,
                notes=[
                    *self._undefined_notes([side], scope),
                    "a scalar side can only be 0, which has whatever indices the other has",
                ],
            )
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
            if arg.type == VALUE:  # f([1, 2]): a function of numbers, not of a list
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

    def _check_unknown_function(self, func: ast.Name, node: ast.Call) -> None:
        """A call of a function no one defined, which stays an unknown function: refused
        where its name is a builtin's or a spacetime's misspelt, as simplfy, or a built-in
        quantity's spelled out, as RicciScalar(g); said so in a note where its name is a
        word rather than a letter, as foo(x); as it is for f(x) or \\alpha(t, r)."""
        name, written = func.id, func.span.text
        if name == "print":
            arguments = ", ".join(arg.span.text for arg in node.args)
            raise CompileError(
                "print is not needed: each result is shown as it is computed",
                span=func.span,
                hint=f"write {arguments}" if arguments else None,
            )
        if name in GREEK or name in BUILTINS or len(name) == 1:  # \gamma(r), f(x)
            letter = name in GREEK or len(name) == 1  # not a builtin here, as \gamma is not
            if letter and not all(isinstance(a, ast.Name) for a in node.args):
                # \gamma(t - vx): a function of t - vx, or \gamma times it, as typeset?
                inside = ", ".join(arg.span.text for arg in node.args)
                self._note(
                    f"{written} is not defined, so {node.span.text} is an unknown function "
                    f"applied to {inside}; for a product, write {written} \\cdot ({inside})"
                )
            return
        concept = _CONCEPTS.get(name.lower())
        close = _close(name, [*BUILTINS, *_SPACETIME_NAMES]) if concept is None else None
        if concept is not None or close is not None:
            suggestion = concept if concept is not None else f"{close}(...)"
            raise CompileError(
                f"unknown function {written}", span=func.span, hint=f"did you mean {suggestion}?"
            )
        if len(name) > 1 and name not in GREEK and name.rstrip("'") not in GREEK:
            self._note(
                f"{written} is not a function RelativisticPy knows, so {node.span.text} is an "
                "unknown function"
            )

    def _check_keywords(self, builtin: Builtin, node: ast.Call) -> None:
        """Refuse an argument given by name, as n=5 in series(sin(x), x, 0, n=5), which
        would be an equation passed in its place: arguments are given in order."""
        try:
            parameters = list(inspect.signature(builtin.func).parameters)
        except (TypeError, ValueError):  # a function whose signature cannot be read
            return
        for position, arg in enumerate(node.args):
            if not (
                isinstance(arg, ast.Equation)
                and isinstance(arg.lhs, ast.Name)
                and arg.lhs.id in parameters
            ):
                continue
            hint = None
            if parameters.index(arg.lhs.id) == position:
                hint = f"write {node.span.text.replace(arg.span.text, arg.rhs.span.text)}"
            raise CompileError(
                f"{builtin.name}() takes its arguments in order, not by name, as "
                f"{arg.span.text} gives one",
                span=arg.span,
                hint=hint,
            )

    def _index_written(self, script: ast.Script) -> str | None:
        """The label of a superscript that the statement shows to be an index: one it
        also writes as a subscript, as in x^\\mu x_\\mu, or on a tensor, or a Greek
        letter it raises on more than one name, as in p^\\mu N^\\nu = n m U^\\mu U^\\nu.
        A Greek superscript written once is a power, as in t^\\alpha."""
        items = script.indices.items if script.indices is not None else ()
        if len(items) != 1 or not _plain_label(items[0]):
            return None
        label = str(items[0].label)
        lowered, raised = self._scripts["_"][label], self._scripts["^"][label]
        index = lowered or label in self._labels or (label in GREEK and raised > 1)
        return label if index else None

    def _a_product(
        self, func: ast.Name, args: tuple[ast.Expr, ...], binding: object, ctx: Context
    ) -> bool:
        """Whether ``name(...)`` multiplies what is in the brackets rather than calls a
        function. A constant never is a function, ``e`` and ``i`` as much as a name declared
        with ``Constants := [G]``; nor is a name the statement writes as a factor too, as G
        is in ``2GM + G(p^2 + q^2)``, unless it is called with the coordinates, as
        \\alpha(t, r) is; nor, with more than names in the brackets, one written as a factor
        before, as \\rho in \\rho(1 + 3w) after p = w\\rho. In LaTeX a name with a value is
        a factor too, as are the cases of :meth:`_a_factor`."""
        name = func.id
        if binding is not None:
            return ctx.latex and isinstance(binding, VariableBinding)  # y(3), y defined
        if _names_builtin(func, latex=ctx.latex):
            return False  # sin(x)
        if self._never_a_function(func, ctx):
            return True  # e(x + 1)
        coordinates = {c.name for c in self.session.coordinates or ()}
        names = [arg.id for arg in args if isinstance(arg, ast.Name)]
        if name in coordinates:
            return False  # r(\tau): a coordinate along a path, as the caller reads it
        if names and len(names) == len(args) and set(names) <= coordinates:
            return False  # \alpha(t, r): a function of the coordinates
        if name in self._factors:
            return True
        written = name in self.session.factors and name not in self.session.functions
        if written and len(names) != len(args):
            return True
        one_name = len(args) == 1 and isinstance(args[0], ast.Name)
        return ctx.latex and self._a_factor(name, args, one_name)

    def _a_factor(self, name: str, args: tuple[ast.Expr, ...], one_name: bool) -> bool:
        """Whether, in LaTeX, ``name(...)`` is a product: where the name is written inside
        its own brackets, as M is in M(r - 2M), since nothing is its own argument; or
        where it is a parameter of the metric, as M is in Schwarzschild's, and the
        brackets hold more than a name: M(1 - 2M/r) is M times 1 - 2M/r."""
        if any(_names_inside(name, arg) for arg in args):
            return True
        metric = self.session.metric
        return (
            metric is not None
            and name in metric.names
            and name not in self.session.functions
            and not one_name
        )

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
        return self._call("core.diff", args, expr.type, span)

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
        return self._call("integrate", [integrand, *limits], integrand.type, node.span)

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
        return self._call(name, args, body.type, node.span)

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
        return self._call("core.subs", [expr, *pairs], expr.type, span)

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
        return self._call("core.time_derivative", [value, order], value.type, span)

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

    def _check_component(self, node: ast.Scripted, base: ast.Name, source: TensorSource) -> None:
        """Refuse a tensor with one index written with a bare number of 2 or more as its
        superscript, as L^2: its component with upper index 2, or L squared? Where there is
        no such component, as with L^4 and four coordinates, it is a power, refused as one.
        In braces, L^{2}, the number is an index, and L^0 and L^1 are never powers."""
        if len(source.indices) != 1 or len(node.scripts) != 1:
            return
        script = node.scripts[0]
        if script.kind != "^" or not re.fullmatch(r"\^\s*[2-9]", script.span.text):
            return
        name, number = base.span.text, script.span.text.lstrip("^ ")
        if len(split_index_labels(name.lstrip("\\"))) > 1:  # L of GML^2, a run of letters
            name = self._written_label(base.id)
        kind = source.node.type
        dim = kind.spaces[0].dim if isinstance(kind, TensorType) and kind.spaces else None
        if isinstance(dim, int) and int(number) >= dim:
            raise self._tensor_power(None, source.indices, None, node.span)
        raise IndexNotationError(
            f"{name} is a tensor, so {name}^{number} could be its component with upper index "
            f"{number} or a power of it",
            span=node.span,
            hint=f"for the component, write {name}^{{{number}}}; to multiply {name} by itself, "
            f"contract it, as in {name}_{{a}} {name}^{{a}}",
        )

    def _check_tensor_power(self, node: ast.Scripted, base: ast.Name, binding: object) -> None:
        """Refuse a tensor written with its indices and then raised to a power, as
        g_{\\mu\\nu}^{-1}, which would otherwise be read as a symbol g_{\\mu\\nu}."""
        indexed = [s for s in node.scripts if s.indices is not None and s.indices.items]
        powers = [s for s in node.scripts if s.indices is None and s.kind == "^"]
        if not indexed or not powers or len(indexed) + len(powers) != len(node.scripts):
            return
        items = [(item, s.kind == "^") for s in indexed for item in s.indices.items]  # type: ignore[union-attr]
        if not all(_plain_label(item) for item, _ in items):
            return
        tensor = isinstance(binding, TensorBinding) or (
            isinstance(binding, VariableBinding) and isinstance(binding.node.type, TensorType)
        )
        quantity = self._builtin_tensor(base.id, len(items)) if binding is None else None
        if not tensor and quantity is None:
            return
        indices = [Index(str(item.label), upper) for item, upper in items]
        exponent = powers[0].body.span.text if powers[0].body is not None else None
        metric = base.id if quantity == "metric" else None
        raise self._tensor_power(metric, indices, exponent, node.span)

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
        written = base.span.text
        if binding is not None:
            return CompileError(
                f"{written} is not a scalar or a tensor, so it has no derivative", span=node.span
            )
        if not rank:
            return CompileError(
                f"{written} is not defined, so it has no derivative",
                span=node.span,
                hint=f"define it first, for example {written} := ...",
            )
        # T^{\mu\nu}_{;\nu}: the tensor as written, without the derivative's index
        own = [
            Index(str(item.label), script.kind == "^")
            for script in node.scripts
            if script.indices is not None
            for item in script.indices.items
            if item.derivative is None and item.label is not None
        ]
        example = (
            self._written_indices(own) if own else "_{" + "".join(_EXAMPLE_LABELS[:rank]) + "}"
        )
        return CompileError(
            f"{written} is not a tensor",
            span=node.span,
            hint=f"define it first, for example {written}{example} := [...]",
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
        identity, or the metric declared for the space, as in g_{ab} := \\Sigma_{ab}. Without
        coordinates, the metric written with labels of no known space is that of the one
        space given a metric, if one is."""
        quantity = self._builtin_tensor(name, len(entries))
        if quantity not in ("kronecker", "metric"):
            return None
        space = self._labels.get(entries[0][0]) or self.session.labels.get(entries[0][0])
        if space is None and quantity == "metric":
            space = self._only_space_metric()
        if space is None:  # spacetime's, or an error that says what the index needs
            what = "the metric" if quantity == "metric" else "the Kronecker delta"
            space = self._space_of(entries[0][0], span, what)
        if space.spacetime:
            return None
        declared = self.session.space_metrics.get(space)
        if quantity == "metric" and declared is not None:
            assert isinstance(declared.type, TensorType)
            return TensorSource(declared, declared.type.indices, name)
        upper = quantity == "kronecker"
        indices = Indices([Index("a", upper), Index("b")])
        node = self.graph.add(
            ops.Call("core.identity"),
            (self.graph.const(space.dim, VALUE),),
            TensorType(indices, (space, space)),
            span,
        )
        return TensorSource(node, indices, name)

    def _only_space_metric(self) -> Space | None:
        """The one space given a metric, where there are no coordinates and so no metric
        of spacetime that g could also be; ``None`` where there are none or several."""
        defined = list(self.session.space_metrics)
        return defined[0] if len(defined) == 1 and self.session.coordinates is None else None

    def _ones(self, label: tuple[str, bool], span: Span, space: Space | None = None) -> Node:
        """𝟙_a: the covector whose components are all one, which sums what it meets."""
        name, upper = label
        space = space or self._space_of(name, span)
        kind = TensorType(Indices([Index(name, upper)]), (space,))
        return self.graph.add(
            ops.Call("core.ones"), (self.graph.const(space.dim, VALUE),), kind, span
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
        self,
        name: str,
        binding: object,
        rank: int | None,
        span: Span,
        *,
        strict: bool = False,
        node: ast.Scripted | None = None,
    ) -> TensorSource | None:
        """What ``name`` with ``rank`` indices refers to, if it is a tensor. With
        ``strict`` (the scripts are index labels), a built-in tensor given the wrong
        number of indices is an error rather than an ordinary symbol. A name that is also
        common for a tensor of one's own, as C for a covariance, says what it is built in
        as, and how to use it for one's own."""
        if isinstance(binding, TensorBinding):
            return TensorSource(binding.node, binding.indices, name)
        if isinstance(binding, VariableBinding) and isinstance(binding.node.type, TensorType):
            return TensorSource(binding.node, binding.node.type.indices, name)
        if binding is not None or rank is None:
            return None
        written = node.base.span.text if node is not None else name
        quantity = self._builtin_tensor(name, rank)
        if quantity is None:
            ranks = [n for n in range(1, 5) if self._builtin_tensor(name, n)]
            if ranks and strict:
                expected = " or ".join(str(n) for n in ranks)
                notes: list[str] = []
                hint = None
                if name in _SHARED_NAMES:  # W_{ij}, a matrix of weights
                    built = _listed([self._built_in_as(name, n) for n in ranks], "or")
                    notes.append(f"{written} is a built-in name: {built}")
                    own = node.span.text if node is not None else written
                    hint = f"to use {written} for a tensor of your own, define it first: "
                    hint += f"{own} := [...]"
                raise IndexNotationError(
                    f"{written} has {expected} indices, but {rank} "
                    f"{'is' if rank == 1 else 'are'} given",
                    span=span,
                    notes=notes,
                    hint=hint,
                )
            return None
        try:
            geometry = self._geometry(quantity, span)
        except CompileError as error:  # G_{kl} = X_{tk} X_{tl}: G is the Einstein tensor
            here = node.span.text if node is not None else written
            error.notes.append(f"{here} is {describe(quantity)}, which is built in")
            pattern = _tensor_pattern(node) if node is not None else None
            if pattern is not None and any(
                (space := self._labels.get(label) or self.session.labels.get(label)) is not None
                and not space.spacetime
                for label, _ in pattern[1]
            ):  # its indices range over data: a tensor of one's own is meant
                error.hint = f"to use {written} for a tensor of your own, define it: {here} := ..."
            raise
        assert isinstance(geometry.type, TensorType)
        return TensorSource(geometry, geometry.type.indices, name)

    def _built_in_as(self, name: str, rank: int) -> str:
        """What a built-in tensor's name means with ``rank`` indices, as it is written:
        the Christoffel connection C^{a}_{bc}."""
        quantity = self._builtin_tensor(name, rank) or ""
        positions = GEOMETRY_POSITIONS.get(quantity, (False,) * rank)
        labels = "abcd"[:rank]
        indices = [Index(label, upper) for label, upper in zip(labels, positions, strict=True)]
        return f"{describe(quantity)} {name}{self._written_indices(indices)}"

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
                        f"{item.span.text} stands for indices that are not written, so this "
                        "cannot be computed",
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

    def _index_value(self, item: ast.IndexItem, space: Space) -> int | None:
        """The component an index item fixes: ``a=0``, ``a:t``, or the name of an element
        of the space's basis, a coordinate's name in spacetime."""
        names = list(space.basis or ())
        what = "coordinates" if space.spacetime else f"elements of {space.name}"
        value = item.value
        if value is None:
            return names.index(item.label) if item.label in names else None
        if isinstance(value, ast.Number) and value.text in map(str, names):
            return list(map(str, names)).index(value.text)  # a label such as 2020, as .loc
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
        """``value``'s components with axes in the order of ``indices``: those of the
        other terms of ``what``, a sum, the other side of an equation or a definition, or
        the other factor of an elementwise product."""
        assert isinstance(value.type, TensorType)
        have = value.type.indices
        if sorted(have, key=str) != sorted(indices, key=str):
            wanted, found = self._written_indices(indices), self._written_indices(have)
            if what in ("an equation", "a definition"):
                message = (
                    f"the two sides have different free indices: {wanted} on the left, "
                    f"{found} on the right"
                )
            elif what == "a sum":
                message = f"the free indices of a sum must agree: {indices} and {have}"
            else:
                message = f"the free indices of {what} must agree: {wanted} and {found}"
            raise IndexNotationError(message, span=span)
        return self._permute(value, [have.index(i) for i in indices], indices)

    # -- the user's notation ----------------------------------------------------------

    def _written_label(self, label: str) -> str:
        """An index label as the statement writes it: ``\\mu`` in LaTeX, ``μ`` in
        Unicode, ``mu`` in plain text, and a Latin letter as it is."""
        name = _display(label)
        if name not in GREEK:
            return name
        text = self._text
        if "\\" + name in text:
            return "\\" + name
        character = _GREEK_CHARACTERS.get(name)
        if character is not None and character in text:
            return character
        return name if re.search(rf"(?<![\\A-Za-z]){name}(?![A-Za-z])", text) else "\\" + name

    def _written_indices(self, indices: Sequence[Index]) -> str:
        """Indices as the statement writes them, grouped by position: ``_{\\mu\\nu}``,
        ``^{a}_{b}``, or ``none``."""
        groups: list[tuple[bool, list[str]]] = []
        for index in indices:
            label = self._written_label(index.name)
            if groups and groups[-1][0] == index.upper:
                groups[-1][1].append(label)
            else:
                groups.append((index.upper, [label]))
        rendered = []
        for upper, labels in groups:
            plain = any(len(label) > 1 and label.isalpha() for label in labels)
            joined = " ".join(labels) if plain else "".join(labels)
            rendered.append(("^" if upper else "_") + "{" + joined + "}")
        return "".join(rendered) or "none"

    def _written_free(self, free: Iterable[Index]) -> str:
        """A set of free indices, upper first and in order of their labels."""
        return self._written_indices(sorted(free, key=lambda index: (not index.upper, index.name)))

    # -- the rules of index notation, one message each ------------------------------
    #
    # Each rule has one phrasing, on every path that checks it: as components and as
    # notation. The phrasings of README.md, docs/spaces.md and docs/getting-started.md
    # are kept as their examples show them.

    def _twice(self, name: str, upper: bool, span: Span) -> IndexNotationError:
        """An index written twice in one term in the same position."""
        label = self._written_label(name)
        notes = []
        coordinates = [c.name for c in self.session.coordinates or ()]
        if (
            coordinates
            and _display(name) in _COORDINATE_NAMES
            and _display(name) not in coordinates
        ):  # R_{zz} in (t, r, \theta, \phi): a component that is none
            shown = _listed([self._written_label(c) for c in coordinates])
            notes.append(
                f"{label} is not a coordinate, so it is an index; the coordinates are {shown}"
            )
        position = "an upper" if upper else "a lower"
        return IndexNotationError(
            f"index {label} appears twice as {position} index; a summed index must appear "
            "once up and once down",
            span=span,
            notes=notes,
        )

    def _thrice(self, name: str, count: int, span: Span) -> IndexNotationError:
        """An index written three times or more in one term."""
        return IndexNotationError(
            f"index {self._written_label(name)} appears {count} times in one term; an index "
            "appears once, or twice to be summed",
            span=span,
        )

    def _two_spaces(self, name: str, one: Space, other: Space, span: Span) -> IndexNotationError:
        """An index written on axes over two spaces."""
        ranges, written = _apart(one, other)
        return IndexNotationError(
            f"the index {self._written_label(name)} ranges over {ranges}, but is written here "
            f"on an axis over {written}",
            span=span,
        )

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
            over = list(dict.fromkeys(space for _, space in found))
            if len(over) > 1:
                raise self._two_spaces(name, over[0], over[1], span)
            if name in self._bound:
                continue
            if len(found) > 2:
                raise self._thrice(name, len(found), span)
            positions = {index.upper for index, _ in found}
            if len(found) == 2 and len(positions) == 1 and not self._euclidean(over[0]):
                raise self._twice(name, found[0][0].upper, span)
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

    # -- spaces ----------------------------------------------------------------------
    #
    # An index label ranges over a space: one it was declared in (a, b \in \mathcal{A}),
    # or, within a statement, the space of the first axis it is written on. Labels that
    # neither declare nor infer range over spacetime.

    def _space_of(self, name: str, span: Span, why: str = "index notation") -> Space:
        space = self._labels.get(name) or self.session.labels.get(name)
        if space is not None:
            return space
        if self.session.coordinates is None:
            label = self._written_label(name)
            data = bool(self.session.spaces) or any(
                not space.spacetime
                for _, binding in self.session.globals.bindings()
                if isinstance(binding, TensorBinding)
                for space in _spaces(binding.node) or ()
            )
            hint = (
                "name a space for it, as in \\mathcal{A} := \\mathbb{R}^{3}, and write "
                f"{label} \\in \\mathcal{{A}}"
                if data
                else "declare coordinates first, as in Coordinates := [t, r, \\theta, \\phi]"
            )
            raise CompileError(
                f"{why} needs coordinates, or a space for the index {label}",
                span=span,
                notes=[
                    "an index ranges over spacetime, over a space declared for it, or over the "
                    "axis of data it is written on in the same statement"
                ],
                hint=hint,
            )
        return self.session.spacetime(span, why)

    def _spaces_of(self, names: Sequence[str], span: Span, why: str) -> tuple[Space, ...]:
        return tuple(self._space_of(name, span, why) for name in names)

    def _check_label(self, name: str, space: Space, span: Span) -> None:
        """That an index written on an axis over ``space`` may range over it; the first
        time in a statement, it then does, for the rest of the statement. On an axis of
        spacetime, a coordinate's name, as t in \\Gamma^{t}_{tr}, names a component and is
        no index, whatever t is elsewhere."""
        if space.spacetime and name in (space.basis or ()):
            return
        expected = self._labels.get(name) or self.session.labels.get(name)
        if expected is None:
            self._labels[name] = space
        elif expected != space:
            raise self._two_spaces(name, expected, space, span)

    def _infer_labels(self, statement: ast.Statement, scope: Scope) -> dict[str, Space]:
        """The spaces of the labels written on tensors with spaces of their own in a
        statement, found before it is compiled, in the order they are written, so that
        𝟙_c in 𝟙_c (\\Sigma^{-1})^{cd}, and 1^k in 1^k \\operatorname{spec}(S)_k, know their
        spaces from what comes after them."""
        labels: dict[str, Space] = {}
        candidates = (
            node for node in _walk(statement) if isinstance(node, ast.Scripted | ast.Name)
        )
        for current in sorted(candidates, key=lambda node: node.span.start):
            found = self._written_on(current, scope, statement.latex, labels)
            if found is None:
                continue
            entries, spaces = found
            for (label, _), space in zip(entries, spaces, strict=True):
                if space.spacetime:
                    continue
                expected = labels.get(label) or self.session.labels.get(label)
                if expected is not None and expected != space:
                    raise self._two_spaces(label, expected, space, current.span)
                labels[label] = space
        return labels

    def _written_on(
        self, node: ast.Scripted | ast.Name, scope: Scope, latex: bool, labels: dict[str, Space]
    ) -> tuple[list[tuple[str, bool]], tuple[Space, ...]] | None:
        """The index labels written on a tensor with spaces of its own, and those spaces:
        a tensor defined or bound, ``y_t`` for ``y_{t}``, an inverse, whose spaces go the
        other way, or the result of a call such as ``\\operatorname{diag}(k)_{ij}``, which
        is compiled to find them; ``None`` for anything else."""
        if isinstance(node, ast.Name):
            indexed = _indexed_name(node, scope)
            if indexed is None:
                return None
            node = indexed
        base, inverse = node.base, False
        if isinstance(base, ast.Scripted) and len(base.scripts) == 1:  # (\Sigma^{-1})^{cd}
            if not _is_minus_one(base.scripts[0].body):
                return None
            base, inverse = base.base, True
        pattern = _tensor_pattern(replace(node, base=ast.Name("#", span=node.span)))
        spaces: tuple[Space, ...] | None = None
        if isinstance(base, ast.Name):
            binding = scope.lookup(base.id)
            if isinstance(binding, VariableBinding | TensorBinding):
                spaces = _spaces(binding.node)
        elif isinstance(base, ast.Call) and pattern is not None and self._names_tensor(base, scope):
            from relativisticpy.errors import RelativisticPyError

            saved, self._labels = self._labels, labels  # its labels are this statement's
            try:
                spaces = _spaces(self.value(base, scope, Context(latex=latex)))
            except RelativisticPyError:
                return None  # said where it is compiled
            finally:
                self._labels = saved
        if inverse:  # of two indices, and going the other way
            spaces = spaces[::-1] if spaces is not None and len(spaces) == 2 else None
        if not spaces or pattern is None or len(pattern[1]) != len(spaces):
            return None
        return pattern[1], spaces

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

    def _script_math(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Node:
        """Scripts on something that is not a tensor: subscripts name a symbol
        (``r_{s}`` is ``r_s``) and superscripts are powers."""
        base, scripts = node.base, list(node.scripts)
        if ctx.latex and isinstance(base, ast.Name) and _splittable(base):
            whole = self.term(base, scope, ctx)
            labels = split_index_labels(base.id)
            if (
                isinstance(whole, Node)
                and isinstance(whole.op, ops.Mul | ops.Scale)  # Scale: a letter is a tensor
                and len(whole.args) == len(labels)
            ):
                # re^{x} is r e^{x}: in LaTeX a script binds to the last letter only, which
                # may be a tensor, as in Rg_{\mu\nu}, or GML^2 where L is one
                last = _letter(labels[-1], base.span, first=False)
                rest = self.value(replace(node, base=last), scope, ctx)
                return self._multiply([*whole.args[:-1], rest], node.span)
        decorations = [s for s in scripts if s.kind == "^" and _decoration(s)]
        if decorations and isinstance(base, ast.Name):  # r^* is a name, not a power
            marks = "".join(_mark(s) for s in decorations)
            base = ast.Name(f"{base.id}^{marks}", verbatim=True, span=base.span)
            scripts = [s for s in scripts if s not in decorations]
        for script in scripts:  # x_{} or x^{}: nothing to read
            if script.body is None and (script.indices is None or not script.indices.items):
                where = "subscript" if script.kind == "_" else "superscript"
                raise CompileError(
                    f"empty {where}", span=script.span, hint="write what it holds, or remove it"
                )
        subscripts = [s for s in scripts if s.kind == "_"]
        powers = [s for s in scripts if s.kind == "^"]
        written_base = base.span.text if isinstance(base, ast.Name) else "this"
        if subscripts:
            if not isinstance(base, ast.Name):
                raise CompileError("only a name can have a subscript", span=node.span)
            items = [i for s in subscripts if s.indices for i in s.indices.items]
            if len(items) > 1 and all(_plain_label(i) for i in items):
                raise self._not_a_tensor(node, base, subscripts, items)
            if self._undefined(base, scope) and _greek_labels(subscripts):  # u_\alpha
                shown = node.span.text if not powers else written_base + _written(subscripts)
                self._note(
                    f"{written_base} is not defined, so {shown} is one symbol, not a "
                    f"component; to use a tensor, define it first: {shown} := [...]"
                )
            symbol_name = _subscripted_name(ast.Scripted(base, tuple(subscripts), span=node.span))
            value = self.value(ast.Name(symbol_name, verbatim=True, span=node.span), scope, ctx)
        else:
            value = self.value(base, scope, ctx)
        exponents = []
        for script in powers:
            if script.body is None or isinstance(script.body, ast.Equation):
                raise CompileError(f"{written_base} is not a tensor", span=script.span)
            if isinstance(base, ast.Name) and (label := self._index_written(script)):
                shown = self._written_label(label)
                raise CompileError(
                    f"{written_base} is not a tensor, so {shown} on it is not an index",
                    span=node.span,
                    hint=f"define it first, for example {written_base}^{{{shown}}} := [...]",
                )
            if not subscripts and isinstance(base, ast.Name) and self._undefined(base, scope):
                self._note_power(base, script)
            if (
                isinstance(script.body, ast.Name)
                and script.body.id in _TRANSPOSES
                and scope.lookup(script.body.id) is None
                and value.type == VALUE
            ):  # A^T of a matrix: a power, which is not what is meant
                raise CompileError(
                    f"{node.span.text} is read as {written_base} to the power "
                    f"{script.body.span.text}",
                    span=node.span,
                    notes=["with indices, the transpose of A_{ab} is A_{ba}"],
                    hint=f"for the transpose, write transpose({written_base})",
                )
            if (
                isinstance(base, ast.Name)
                and scope.lookup(base.id) is None
                and self._builtin_scalar(base.id) == "determinant"
            ):  # g^{-1} is 1/det g, which is easily taken for the inverse metric
                self._note(
                    f"{written_base} alone is the determinant of the metric, so {node.span.text} "
                    f"is a power of it; the inverse metric is {written_base}^{{\\mu\\nu}}"
                )
            exponents.append(self.value(script.body, scope, ctx))
        if not exponents:
            return value
        exponent = exponents[-1]
        for previous in reversed(exponents[:-1]):  # x^a^b is x^(a^b)
            exponent = self._power(previous, exponent, node.span)
        return self._power(value, exponent, node.span)

    def _undefined(self, base: ast.Expr, scope: Scope) -> bool:
        """Whether a name means nothing yet, so that scripts on it name a symbol or raise it
        to a power: bound to nothing, and no coordinate, constant, declared symbol, function
        of the coordinates or built-in scalar."""
        if not isinstance(base, ast.Name) or scope.lookup(base.id) is not None:
            return False
        name, session = base.id, self.session
        return not (
            any(c.name == name for c in session.coordinates or ())
            or name in CONSTANTS
            or name in session.assumptions
            or name in session.functions
            or self._builtin_scalar(name) is not None
        )

    def _note_power(self, base: ast.Name, script: ast.Script) -> None:
        """Say so where a superscript on a name that means nothing looks like indices, a
        Greek label or several, as in T^{\\mu\\nu}, but is read as a power; or where the
        statement also writes the name with the label as a subscript, as u^a u_a."""
        written = base.span.text
        items = script.indices.items if script.indices is not None else ()
        if not items or not all(_plain_label(item) for item in items):
            return
        labels = [str(item.label) for item in items]
        power = f"{written}{script.span.text}"
        tensor = f"{written}^{{{''.join(self._written_label(label) for label in labels)}}}"
        if any(label in GREEK for label in labels) or len(labels) > 1:
            self._note(
                f"{written} is not defined, so {power} is read as {written} to a power; to use "
                f"a tensor, define it first: {tensor} := [...]"
            )
        elif len(labels) == 1 and f"{base.id}_{labels[0]}" in self._factors:  # u^a u_a
            self._note(
                f"{written} is not defined, so {power} is a power and {written}_{labels[0]} one "
                f"symbol; to use a tensor, define it first: {tensor} := [...]"
            )

    def _not_a_tensor(
        self,
        node: ast.Scripted,
        base: ast.Name,
        subscripts: Sequence[ast.Script],
        items: Sequence[ast.IndexItem],
    ) -> CompileError:
        """``T_{ab}`` where ``T`` names no tensor: define it first, or, where the
        subscript is a word, as in ``T_{eff}``, it names a symbol, written so."""
        name = base.span.text
        written = _written(subscripts)
        run = written.lstrip("_").strip("{}")
        if len(subscripts) == 1 and _a_word(run):  # T_{eff}: a name, not three indices
            labels = _listed([str(item.label) for item in items])
            return CompileError(
                f"{name} is not a tensor",
                span=node.span,
                notes=[f"in braces each letter is an index: {labels}"],
                hint=f"to name a symbol, write {name}_{{\\rm {run}}} or {name}_{{\\text{{{run}}}}}",
            )
        if any(self._names_component(str(item.label)) for item in items):  # T_{tt}
            written = "_{" + "".join(_EXAMPLE_LABELS[: len(items)]) + "}"
        return CompileError(
            f"{name} is not a tensor",
            span=node.span,
            hint=f"define it first, for example {name}{written} := [...]",
        )

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
                notation = (
                    isinstance(binding, VariableBinding) and _notation_of(binding.node) is not None
                )
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
                    quantity = self._builtin_tensor(name, len(entries))
                    first = entries[0][0]
                    over = self._labels.get(first) or self.session.labels.get(first)
                    if quantity in ("kronecker", "metric") and over and not over.spacetime:
                        pass  # the identity or metric of a space of data: it has components
                    elif quantity is not None:
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
                if len(positions) > 2:
                    raise self._thrice(label, len(positions), span)
                if len(positions) == 2 and positions[0] == positions[1]:
                    raise self._twice(label, positions[0], span)
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
                        f"{self._written_free(_as_indices(lhs))} on the left, "
                        f"{self._written_free(_as_indices(rhs))} on the right",
                        span=node.span,
                    )
                either = lhs if lhs is not None else rhs
                return None if either is None else sorted(either)
            if isinstance(node, ast.Binary) and node.op in ("+", "-"):
                common: frozenset[tuple[str, bool]] | None = None
                kinds: list[tuple[ast.Expr, frozenset[tuple[str, bool]]]] = []
                for _, term in _signed_terms(node):
                    free = side(term)
                    if free is None:
                        continue
                    kinds.append((term, free))
                    common = free if common is None else common
                if len({free for _, free in kinds}) > 1:
                    counted = Counter(free for _, free in kinds)
                    most, count = counted.most_common(1)[0]
                    odd = [(term, free) for term, free in kinds if free != most]
                    if count > 1 and len(odd) == 1:
                        raise self._odd_term(
                            odd[0][0], _as_indices(odd[0][1]), _as_indices(most), scope
                        )
                    first, second = list(counted)[:2]
                    raise IndexNotationError(
                        "the free indices of a sum must agree: "
                        f"{self._written_free(_as_indices(first))} and "
                        f"{self._written_free(_as_indices(second))}",
                        span=node.span,
                    )
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
                if (
                    isinstance(binding, VariableBinding)
                    and (kept := _notation_of(binding.node)) is not None
                ):
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
                if isinstance(target, Symbol) and not any(
                    isinstance(side, symbolic.TensExpr) for side in (subject.lhs, subject.rhs)
                ):  # solve(trace(EFE), R): an equation of scalars, for a scalar
                    found = sympy.solve(subject.lhs - subject.rhs, target)
                    if len(found) != 1:
                        raise CompileError(
                            f"the equation does not fix {args[1].span.text}", span=node.span
                        )
                    return IndexEquation(target, found[0], space)
                if not isinstance(target, symbolic.Tensor):
                    raise CompileError(
                        "give the tensor to solve for with its indices, as in T_{\\mu\\nu}, or "
                        "a scalar the equation is of",
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
            if (
                isinstance(binding, VariableBinding)
                and (found := self._kept(binding.node)) is not None
            ):
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
        folded = fold(value)  # notation's algebra is done now, while compiling
        if folded is NotImplemented:
            raise CompileError(
                "this factor is computed from tensors, but tensor notation is read before "
                "anything is computed",
                span=node.span,
                hint="give the tensors of the notation components, so that the equation is "
                "computed with components",
            )
        scalar = sympy.sympify(folded)
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
        if isinstance(binding, VariableBinding) and _notation_of(binding.node) is not None:
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
        found = self._kept(node)
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

    def _kept(self, node: Node) -> IndexEquation | IndexExpression | None:
        """The notation a node holds, if any, in the tensors of the coordinates in force:
        notation kept before they changed is read in their number of dimensions."""
        found = _notation_of(node)
        return None if found is None else abstract.rebased(found, self.session.algebra())

    # -- helpers -------------------------------------------------------------

    def _call(
        self,
        name: str,
        args: Sequence[Node],
        kind: Type,
        span: Span,
        attributes: ops.Attributes = (),
    ) -> Node:
        """The operation registered as ``name`` (see :mod:`relativisticpy.runtime.registry`)
        applied to ``args``, with ``attributes``."""
        return self.graph.add(ops.Call(name, attributes), list(args), kind, span)

    def _wrapped(self, indices: Indices) -> ops.Attributes:
        """The attributes of an operation that gives tensor components to a function that
        wants a :class:`~relativisticpy.tensors.Tensor`: its indices, as they are shown,
        and the coordinates."""
        return (("indices", _public(indices)), ("coordinates", self.session.coordinates or ()))


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
    """``ds^2`` (also ``ds^{2}``, ``(ds)^2``, ``d s^2`` and, in Python's way, ``ds**2``)."""

    def squared(node: ast.Expr, name: str) -> bool:
        if (
            isinstance(node, ast.Binary)
            and node.op == "**"
            and isinstance(node.left, ast.Name)
            and node.left.id == name
        ):
            return isinstance(node.right, ast.Number) and node.right.text == "2"
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


def _named(target: ast.Expr) -> ast.Expr:
    """A decorated name, as ``\\tilde u`` or ``E^{(2)}``, as the name it is read as,
    ``utilde`` or ``E^(2)``; a dot is a time derivative, and anything else is left as it
    is."""
    if (
        isinstance(target, ast.Accent)
        and target.accent not in ("dot", "ddot")
        and isinstance(target.base, ast.Name)
    ):
        return ast.Name(target.base.id + target.accent, verbatim=True, span=target.span)
    if (
        isinstance(target, ast.Scripted)
        and isinstance(target.base, ast.Name)
        and all(script.kind == "^" and _decoration(script) for script in target.scripts)
    ):
        marks = "".join(_mark(script) for script in target.scripts)
        return ast.Name(f"{target.base.id}^{marks}", verbatim=True, span=target.span)
    return target


def _mark(script: ast.Script) -> str:
    """How a decoration marks a name: ``*`` in ``r^*``, ``(2)`` in ``E^{(2)}``."""
    body = script.body
    if isinstance(body, ast.Name) and not _bracketed_order(body):
        return body.id
    return body.span.text if body is not None else ""


def _script_labels(statement: ast.Node) -> dict[str, Counter[str]]:
    """How often a statement writes each index label as a subscript (``"_"``), as
    \\mu in x_\\mu, and as a superscript (``"^"``)."""
    counts: dict[str, Counter[str]] = {"_": Counter(), "^": Counter()}
    for node in _walk(statement):
        if isinstance(node, ast.Script) and node.indices is not None:
            counts[node.kind].update(
                str(item.label) for item in node.indices.items if _plain_label(item)
            )
    return counts


def _names_inside(name: str, node: ast.Expr) -> bool:
    """Whether ``name`` is written in ``node`` other than as a function called there: M
    is in r - 2M, and f is not in f(x)."""
    called = {id(n.func) for n in _walk(node) if isinstance(n, ast.Call)}
    return any(
        isinstance(n, ast.Name) and n.id == name and id(n) not in called for n in _walk(node)
    )


def _unprimed(node: ast.Expr) -> ast.Expr:
    """A primed name with subscripts, as ``R'_{\\mu\\nu}``, read as the name ``R'`` with
    them: a tensor or symbol of its own. A power of a primed name, ``A'^2``, is left as
    it is, to be the power of what ``A'`` means."""
    if (
        isinstance(node, ast.Scripted)
        and isinstance(node.base, ast.Prime)
        and any(script.kind == "_" for script in node.scripts)
    ):
        prime = node.base
        name = ast.Name(prime.func.id + "'" * prime.order, verbatim=True, span=prime.span)
        return replace(node, base=name)
    return node


def _names_builtin(node: ast.Name, *, latex: bool = False) -> bool:
    """Whether a name, bound to nothing in the program, is a builtin function: never a
    Greek letter written as one, so that \\gamma(r) in a metric is an unknown function,
    and in LaTeX never a single letter, so that N(t) there is the lapse, not ``N``."""
    if latex and len(node.id) == 1:
        return False
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
    ``v_{a}``, with its indices where the value has them. A value whose indices no one
    named, as the eigenvalues of s_k := \\operatorname{spec}(S), takes them in order."""
    base, underscore, tail = name.partition("_")
    if not (underscore and tail.isalpha() and isinstance(value.type, TensorType)):
        return None
    labels = split_index_labels(tail)
    indices = value.type.indices
    if len(labels) == len(indices) and all(index.name.startswith("#") for index in indices):
        return base, indices  # named by the labels where they are written, s_k or s_{k}
    have = {index.name: index for index in indices}
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


def _complex(kind: TensorType) -> bool:
    """Whether a tensor has an axis over a complex space."""
    return any(space.field == "complex" for space in kind.spaces)


def _uniform_weights(measure: Node) -> bool:
    """Whether a measure declared for a space weighs its elements alike."""
    from relativisticpy.tensors import numerical

    if not isinstance(measure.op, ops.Const):
        return False
    weights = numerical.numpy().asarray(measure.op.value, dtype=float)
    return bool(numerical.numpy().allclose(weights, weights.mean(), rtol=1e-12, atol=0.0))


#: Names of statistics written as operators, for the statistic they mean.
_STATISTIC_ALIASES = {
    "E": "\\mathbb{E}",
    "Mean": "\\mathbb{E}",
    "mean": "\\mathbb{E}",
    "Expectation": "\\mathbb{E}",
    "Variance": "\\operatorname{Var}",
    "Covariance": "\\operatorname{Cov}",
}

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


def _apart(one: Space, other: Space) -> tuple[str, str]:
    """Two different spaces in words that tell them apart: their names, or, where they
    share a name, as the spaces of one label in two sets of data do, their sizes,
    elements or fields."""
    if str(one) != str(other):
        return str(one), str(other)
    if one.dim != other.dim:
        return f"{one}, of dimension {one.dim}", f"{other}, of dimension {other.dim}"
    if one.basis != other.basis:
        return f"{one}, {_elements(one)}", f"{other}, {_elements(other)}"
    return f"{one}, {one.field}", f"{other}, {other.field}"


def _elements(space: Space) -> str:
    if space.basis is None:
        return "its elements unnamed"
    shown = ", ".join(space.basis[:3])
    return f"of {shown}{', ...' if len(space.basis) > 3 else ''}"


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
            if (
                isinstance(current, ast.Call)
                and isinstance(current.func, ast.Name)
                and current.func.id == "solve"
                and len(current.args) == 2
            ):
                pending.extend((current.func, current.args[0]))
            else:
                pending.extend(vars(current).values())


def _written_with(root: Node) -> frozenset[str]:
    """The names of the symbols and of the functions that have no value that ``root``'s
    value is written with, read from the IR without computing it: those of its constants
    and of the functions it applies, but not the differentials a line element takes away,
    the variables a sum, a product, an integral, a limit or an evaluation bar binds, a name
    a definition made after the metric gives its value to, or a parameter a spacetime is
    not written with, as G of flat space."""
    found: dict[int, frozenset[str]] = {}
    for node in dependencies([root]):
        found[node.id] = _names_written(node, found)
    return found[root.id]


def _names_written(node: Node, found: Mapping[int, frozenset[str]]) -> frozenset[str]:
    """The names ``node`` is written with (see :func:`_written_with`), from those of its
    arguments, in ``found``."""
    op = node.op
    if isinstance(op, ops.Const):
        return _symbol_names(op.value)
    args = [found[arg.id] for arg in node.args]
    every = frozenset().union(*args)
    if isinstance(op, ops.LineElement):
        return every - {differential.name for differential in op.differentials}
    if not isinstance(op, ops.Call | ops.Map):
        return every
    name, rest = op.name, frozenset().union(*args[2:])
    if name == "core.function":
        return every | {str(dict(op.attributes)["name"])}
    if name in ("core.sum", "core.prod", "core.limit"):  # body, variable, then the rest
        return (args[0] - args[1]) | rest
    if name == "core.derivative_at":  # body, the variable, the order and the point
        return (args[0] - args[1]) | args[3]
    if name == "core.subs":  # expr, then pairs of what and what for
        return (args[0] - frozenset().union(*args[1::2])) | frozenset().union(*args[2::2])
    if name == "integrate":  # what a definite integral's limits bind, as core.limits gives
        bound = frozenset().union(
            *(found[arg.args[0].id] for arg in node.args[1:] if _calls(arg, "core.limits"))
        )
        return every - bound
    if name in ("core.where", "core.where_function"):  # the metric, the name, the value
        defined = node.args[1].op
        assert isinstance(defined, ops.Const)
        parameters: frozenset[str] = frozenset()
        if name == "core.where_function":
            defined_name, symbols = defined.value
            parameters = frozenset(symbol.name for symbol in symbols)
        else:
            defined_name = defined.value
        return (args[0] - {defined_name}) | (args[2] - parameters)
    if name in _SPACETIMES:
        absent = spacetimes.absent_from(_SPACETIMES[name])
        written = BUILTINS[name].parameters
        return frozenset().union(
            *(names for names, p in zip(args, written, strict=False) if p not in absent)
        )
    return every


def _calls(node: Node, name: str) -> bool:
    """Whether ``node`` applies the operation registered as ``name``."""
    return isinstance(node.op, ops.Call) and node.op.name == name


def _symbol_names(value: Any) -> frozenset[str]:
    """The names of the symbols and of the functions that have no value in a constant."""
    if isinstance(value, list | tuple):
        return frozenset().union(*(_symbol_names(item) for item in value))
    if not isinstance(value, sympy.Basic):
        return frozenset()
    names = {symbol.name for symbol in value.free_symbols if isinstance(symbol, Symbol)}
    names.update(function.func.__name__ for function in value.atoms(AppliedUndef))
    return frozenset(names)


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

# Constants that an equation can only make a claim about: \pi = 3.14159 is not a new \pi.
# (e and i are often named afresh, as a charge or an index.)
_CLAIMED = frozenset({"pi", "infty", "oo"})

# Statements that declare, as := does, whether written with := or =.
_DECLARATIONS = frozenset({"Coordinates", *ASSUMPTION_STATEMENTS, *RENAMING_STATEMENTS})

# The built-in scalars, by the geometric quantity a name denotes.
_BUILT_IN_SCALARS = {
    "determinant": "the determinant of the metric",
    "ricci_scalar": "the Ricci scalar",
    "kretschmann": "the Kretschmann scalar",
}

# The spacetimes, by the names the language calls them.
_SPACETIME_NAMES = tuple(name for name in BUILTINS if name in SPACETIMES)


def _has_a_value(written: str) -> str:
    """The note on ``K = ...`` once K has a value: a claim, not a new value."""
    return (
        f"{written} already has a value, so = compares it; to give {written} a new value, "
        f"write {written} := ..."
    )


def _built_in(written: str, what: str, again: str) -> str:
    """The note on ``R = 0``: a claim about what a built-in name already means."""
    return (
        f"{written} is {what}, so = compares it; to give {written} a meaning of your own, "
        f"write {again}"
    )


def _late_note(written: str, metric: MetricState | None) -> str:
    """The note on ``\\Delta = ...`` after a line element written with \\Delta."""
    span = metric.node.span if metric is not None else None
    where = f"on line {span.line}" if span is not None else "before"
    return (
        f"the metric defined {where} is written with {written}, which has this value there "
        "from now on"
    )


def _shown(name: str, latex: bool) -> str:
    """A name as the user writes it: ``\\Delta`` in LaTeX, ``Delta`` in plain text."""
    return latex_label(name) if latex else name


def _listed(items: Sequence[str], conjunction: str = "and") -> str:
    """``t, r, \\theta and \\phi``."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


def _close(name: str, candidates: Iterable[str], shortest: int = 4) -> str | None:
    """The candidate a name is likely a misspelling of, as Schwarzchild is of
    Schwarzschild or simplfy of simplify; ``None`` where none is close. Names of fewer
    than four letters are never matched, and candidates shorter than ``shortest`` never
    offered, so that ln is no misspelt N, nor erfc a misspelt erf."""
    if len(name) < 4:
        return None
    known: dict[str, str] = {}
    for candidate in candidates:  # the first spelling of each, as Cov before cov
        if len(candidate) >= shortest:
            known.setdefault(candidate.lower(), candidate)
    found = difflib.get_close_matches(name.lower(), list(known), n=1, cutoff=0.8)
    return known[found[0]] if found else None


def _factor_names(statement: ast.Node) -> frozenset[str]:
    """The names a statement writes as factors rather than as functions it calls: G and
    M in 2GM, where in LaTeX a run of letters is a product of them, but neither f in
    f(x) or f'(x), nor what a substitution sets, as r in |_{r=r(\\tau)}."""
    other = _other_names(statement)
    for node in _walk(statement):
        if isinstance(node, ast.Substitution):
            other.update(id(target) for target, _ in node.bindings)
    latex = bool(getattr(statement, "latex", False))
    found: set[str] = set()
    for node in _walk(statement):
        if isinstance(node, ast.Name) and id(node) not in other:
            found.add(node.id)
            known = node.id in BUILTINS or node.id in CONSTANTS
            if latex and _splittable(node) and not known:
                found.update(split_index_labels(node.id))
    return frozenset(found)


def _other_names(node: ast.Node) -> set[int]:
    """The ids of the names in an expression that are part of another name, or of a
    function's: f in f(x), x in x', h in \\bar h, E in E^{(2)}."""
    found: set[int] = set()
    for current in _walk(node):
        if isinstance(current, ast.Call | ast.Prime):
            found.add(id(current.func))
        elif isinstance(current, ast.Accent | ast.Scripted) and isinstance(
            _named(current), ast.Name
        ):  # \bar h is hbar, and E^{(2)} is E^(2); \dot h is h, differentiated
            found.add(id(current.base))
    return found


def _writes_tensor(value: ast.Expr, name: str) -> bool:
    """Whether ``value`` writes the tensor ``name`` with index labels, or ``name`` alone,
    other than as the tensor solve is asked to solve for: T_{\\nu\\mu} in
    T_{\\mu\\nu} = T_{\\nu\\mu}, but not k in k^3, a power or a component."""
    bases = {id(node.base) for node in _used(value) if isinstance(node, ast.Scripted)}
    for node in _used(value):
        if isinstance(node, ast.Scripted) and (pattern := _tensor_pattern(node)):
            if pattern[0] == name:
                return True
        elif isinstance(node, ast.Name) and node.id == name and id(node) not in bases:
            return True
    return False


def _slashed(factors: list[ast.Expr]) -> list[ast.Expr]:
    """Factors with an inline slash read as in physics: ``x/4GM`` is x/(4GM), and
    ``e^{-r/2GM}`` is e^{-r/(2GM)}. The letters and numbers written right after the
    denominator, with no space, belong to it; ``x/4 GM``, with a space, and
    ``\\frac{x}{4}GM`` are (x/4) GM, and a tensor never joins a denominator."""

    def simple(node: ast.Expr) -> bool:
        if isinstance(node, ast.Number | ast.Name):
            return True
        return (
            isinstance(node, ast.Scripted)
            and isinstance(node.base, ast.Name)
            and _tensor_pattern(node) is None
            and all(script.kind == "^" for script in node.scripts)
        )

    result: list[ast.Expr] = []
    position = 0
    while position < len(factors):
        factor = factors[position]
        slash = (
            isinstance(factor, ast.Binary)
            and factor.op == "/"
            and Span(factor.span.source, factor.left.span.end, factor.right.span.start).text.strip()
            == "/"
            and simple(factor.right)
        )
        run: list[ast.Expr] = []
        end = factor.span.end
        while slash and position + 1 + len(run) < len(factors):
            following = factors[position + 1 + len(run)]
            if following.span.start != end or not simple(following):
                break
            run.append(following)
            end = following.span.end
        if not run:
            result.append(factor)
            position += 1
            continue
        assert isinstance(factor, ast.Binary)
        denominator = reduce(
            lambda a, b: ast.Binary("*", a, b, implicit=True, span=a.span.to(b.span)),
            [factor.right, *run],
        )
        span = factor.span.to(run[-1].span)
        result.append(ast.Binary("/", factor.left, denominator, span=span))
        position += 1 + len(run)
    return result


def _greek_labels(scripts: Sequence[ast.Script]) -> bool:
    """Whether scripts are index labels, all Greek, as \\alpha in u_\\alpha."""
    items = [item for s in scripts if s.indices is not None for item in s.indices.items]
    return bool(items) and all(_plain_label(item) and item.label in GREEK for item in items)


def _styled(span: Span) -> Span:
    """A span grown over the style command around it, as B to \\mathcal{B}."""
    text = span.source.text
    for command in ("\\mathcal{", "\\mathbb{", "\\mathscr{", "\\mathfrak{", "\\mathrm{"):
        start = span.start - len(command)
        if text[max(start, 0) : span.start] == command and text[span.end : span.end + 1] == "}":
            return Span(span.source, start, span.end + 1)
    return span


def _an_index(before: ast.Expr, factor: ast.Expr) -> bool:
    """Whether ``factor`` is written as Python indexes a list: a whole number in square
    brackets, right after what it would index, as [0] in [1, 2][0]."""
    return (
        isinstance(factor, ast.List)
        and len(factor.items) == 1
        and isinstance(factor.items[0], ast.Number)
        and factor.items[0].text.isdigit()
        and factor.span.text.startswith("[")
        and before.span.end == factor.span.start
    )


def _bracketed(equation: ast.Equation) -> bool:
    """Whether an equation is written in brackets of its own, as the case ``(k = 0)``
    beside a formula is."""
    text = equation.span.text
    for opener, closer in (("(", ")"), ("\\left(", "\\right)")):
        if (
            text.startswith(opener)
            and text.endswith(closer)
            and equation.lhs.span.start > equation.span.start
            and equation.rhs.span.end < equation.span.end
        ):
            return True
    return False


def _unbracketed(equation: ast.Equation) -> str:
    return Span(equation.span.source, equation.lhs.span.start, equation.rhs.span.end).text


def _mentions_symbol(node: Node, name: str) -> bool:
    """Whether a value is written with the symbol ``name``: a constant it is computed
    from has it among its free symbols."""
    for current in dependencies([node]):
        if not isinstance(current.op, ops.Const):
            continue
        value = current.op.value
        written = isinstance(value, sympy.Basic | sympy.NDimArray)
        if written and any(symbol.name == name for symbol in value.free_symbols):
            return True
    return False


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
    """A script that marks a name rather than raising it to a power: ``r^*``, ``i^+``, or
    an order in brackets, as of a perturbation, ``E^{(2)}``."""
    body = script.body
    if isinstance(body, ast.Name) and body.verbatim and body.id in ("*", "+", "-", "±", "∓"):
        return True
    return _bracketed_order(body)


def _bracketed_order(body: ast.Expr | None) -> bool:
    """``(2)`` or ``(n)``: a whole number or one letter in brackets."""
    if isinstance(body, ast.Number):
        whole = body.text.isdigit()
    else:
        whole = isinstance(body, ast.Name) and (len(body.id) == 1 or body.id in GREEK)
    text = body.span.text if body is not None else ""
    return whole and text.startswith("(") and text.endswith(")")


_DIFFERENTIAL_NAME = re.compile(
    r"d(" + "|".join(sorted(GREEK, key=len, reverse=True)) + r"|[A-Za-z][0-9]*)"
)


def _line_element_coordinates(value: ast.Expr, *, solid_angle: bool = True) -> list[str]:
    r"""The coordinates a line element's differentials name, in the order they appear:
    ``-dt^2 + dr^2 + r^2 d\Omega^2`` names t, r, θ and φ, or, without ``solid_angle``, t, r
    and Ω. Empty if a differential is abstract, as in ``g_{\mu\nu} dx^\mu dx^\nu``, or if
    there are none."""
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
        found = ["theta", "phi"] if target.id == "Omega" and solid_angle else [target.id]
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


#: Superscripts that write a transpose, as in A^T.
_TRANSPOSES = frozenset({"T", "top", "intercal"})

#: The built-in tensors and scalars by the names they are spelled out with, lower case,
#: as an agent may guess them: what to write instead.
_CONCEPTS = {
    "ricci": "Ric, the Ricci scalar, or R_{\\mu\\nu}, the Ricci tensor",
    "ricciscalar": "R or Ric, the Ricci scalar",
    "riccitensor": "R_{\\mu\\nu}, the Ricci tensor",
    "riemann": "R^{\\rho}{}_{\\sigma\\mu\\nu}, the Riemann tensor",
    "riemanntensor": "R^{\\rho}{}_{\\sigma\\mu\\nu}, the Riemann tensor",
    "christoffel": "\\Gamma^{\\lambda}_{\\mu\\nu}, the Christoffel symbols",
    "christoffelsymbols": "\\Gamma^{\\lambda}_{\\mu\\nu}, the Christoffel symbols",
    "einstein": "G_{\\mu\\nu}, the Einstein tensor",
    "einsteintensor": "G_{\\mu\\nu}, the Einstein tensor",
    "weyl": "C_{\\mu\\nu\\rho\\sigma}, the Weyl tensor",
    "weyltensor": "C_{\\mu\\nu\\rho\\sigma}, the Weyl tensor",
    "metric": "g_{\\mu\\nu}, the metric",
    "metrictensor": "g_{\\mu\\nu}, the metric",
    "curvaturescalar": "R or Ric, the Ricci scalar",
    "kretschmannscalar": "Kretschmann, or R_{abcd} R^{abcd}",
}

#: Names of built-in tensors that are as often names of one's own: a covariance C,
#: weights W, a matrix G.
_SHARED_NAMES = frozenset({"C", "G", "W"})

#: Words written as subscripts, as in T_{eff}: names, not one index per letter.
_SUBSCRIPT_WORDS = frozenset(
    {
        "eff", "vac", "max", "min", "tot", "obs", "crit", "irr", "ext", "int", "rad",
        "kin", "pot", "rel", "ret", "adv", "out", "in", "cm", "lab", "rest", "sun", "orb",
        "com", "phys", "BH", "ADM", "Pl", "em", "dm", "de",
    }
)  # fmt: skip


def _a_word(run: str) -> bool:
    """Whether a subscript written as a run of letters is a word that names something,
    as eff in T_{eff}, rather than one index per letter, as in T_{abc}: a word often
    written as a subscript, or three letters or more of which some repeat."""
    if not (run.isalpha() and run.isascii()):
        return False
    return run in _SUBSCRIPT_WORDS or (len(run) >= 3 and len(set(run)) < len(run))


def _as_indices(free: Iterable[tuple[str, bool]]) -> frozenset[Index]:
    """Free indices read from syntax, ``(label, upper)``, as indices."""
    return frozenset(Index(label, upper) for label, upper in free)


#: Greek letters as Unicode writes them, to write a label as the user did.
_GREEK_CHARACTERS: dict[str, str] = {}
for _character, _name in UNICODE_LETTERS.items():
    if _name in GREEK:
        _GREEK_CHARACTERS.setdefault(_name, _character)

#: Names usual for coordinates: written twice down on a tensor, as z in R_{zz}, and
#: not a coordinate in force, such a label is more likely a component than a sum.
_COORDINATE_NAMES = frozenset({"t", "x", "y", "z", "r", "theta", "phi", "rho", "chi", "psi"})


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
