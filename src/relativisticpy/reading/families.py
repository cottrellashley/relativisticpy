r"""Families: names defined with a free label (``design/notation.md``, decision 5.2), each
reading recorded with why, and a note where another reading was possible:

- on the left of ``:=``, a one-letter subscript that the value is written with makes a
  family: ``c_n := ...``, ``E_k := ...``, ``T_a := \sigma_a/2`` or ``\psi_n(x) := ...``.
  Its members are the family at their subscripts: ``c_1``, ``a_{n+1}``, ``E_{-1}``, and
  ``\psi_3(x)`` or ``\psi_m(x)`` called. A value written with a letter nothing binds, as
  c_k, is a symbol of its own, as before. In the family's value a name with its label as
  the subscript is the name at the member's, as \sigma_a is \sigma_1 in T_1; and a name
  the value writes with another subscript, as x_0 of ``x_n := x_0 + n d``, is a name of
  its own;
- a family defined through its own members, with its first members given, is a
  recurrence: ``a_{n+2} := a_{n+1} + a_n``, ``T_n = 2 T_{n-1} + 1`` or
  ``C_{n+1} = \sum_{k=0}^{n} C_k C_{n-k}``. Its members are computed by
  :mod:`relativisticpy.runtime.recurrences`: at a whole number one after another, and
  elsewhere in closed form where SymPy's ``rsolve`` finds one that agrees with them;
- a finite sum or product whose index is the subscript of names the program defines, as a
  is in ``\sum_{a=1}^{8} \lambda_a^2`` after \lambda_1, ..., \lambda_8, is taken term by
  term.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import Literal

import sympy

from relativisticpy.compiler.builtins import BUILTINS, CONSTANTS
from relativisticpy.compiler.passes import Term
from relativisticpy.compiler.scope import (
    Context,
    FunctionBinding,
    Scope,
    TensorBinding,
    VariableBinding,
)
from relativisticpy.compiler.session import Session
from relativisticpy.errors import CompileError, RelativisticPyError
from relativisticpy.integrations import _forms as forms
from relativisticpy.ir import SCALAR, VALUE, Node, TensorType, ValueType, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.parser import split_index_labels
from relativisticpy.language.source import Span
from relativisticpy.reading.labels import Labels
from relativisticpy.reading.names import (
    _STATISTICS,
    _mentions_symbol,
    _recurrence,
    _shown,
    _splittable,
    _subscripted,
    _with_arguments,
)
from relativisticpy.reading.subscripts import (
    _WORD,
    Kind,
    Reading,
    _raised,
    family_of,
    label_name,
    member_written,
    subscripted,
)
from relativisticpy.reading.subscripts import Family as Family
from relativisticpy.typing.notation import _passed
from relativisticpy.typing.typer import _listed

#: The decision of whether a name with a subscript is a family, or a member of one.
FAMILY = "family or name"

#: A subscript a family may be defined with: its label, maybe shifted, as n+2 is.
_LABEL = re.compile(r"(?P<label>[A-Za-z])(?P<shift>[-+]\d+)?")

#: How many terms a sum or a product is taken term by term at most.
_TERMS = 100


def _family_target(target: ast.Expr) -> tuple[str, str, int] | None:
    """The base, the label and the shift of a name a family may be defined with:
    ``("c", "n", 0)`` of ``c_n`` and ``c_{n}``, ``("a", "n", 2)`` of ``a_{n+2}``, and
    ``("E^(0)", "n", 0)`` of ``E^{(0)}_n`` and ``E_n^{(0)}``, an order raised on it;
    ``None`` for any other name."""
    named = _subscripted(target) or _ordered(target)
    if named is None:
        return None
    if isinstance(target, ast.Scripted) and _WORD.search(target.scripts[0].span.text):
        return None  # C_\mathrm{A}: a word, never a label that takes values
    base, subscript = named
    match = _LABEL.fullmatch(subscript)
    if match is None or not re.fullmatch(r"[A-Za-z]+", base.partition("^")[0]):
        return None
    return base, match["label"], int(match["shift"] or 0)


def _ordered(node: ast.Expr) -> tuple[str, str] | None:
    """A name with an order raised on it and one subscript, as ``E^{(0)}_n`` and
    ``E_n^{(0)}``, which TeX sets alike: the name with its order, E^(0), and the subscript
    as written, n; ``None`` for any other name."""
    if not (isinstance(node, ast.Scripted) and isinstance(node.base, ast.Name)):
        return None
    lowered = [s for s in node.scripts if s.kind == "_"]
    marks = [_raised(s) for s in node.scripts if s.kind == "^"]
    if not marks or any(mark is None for mark in marks):
        return None
    head, _, joined = node.base.id.partition("_")
    if len(lowered) == 1 and not joined:
        subscript = lowered[0].span.text.lstrip("_").strip("{}").replace(" ", "")
    elif not lowered and joined:
        subscript = joined
    else:
        return None
    return head + "".join(f"^{mark}" for mark in marks), subscript


def _fixed_members(value: ast.Expr, base: str) -> frozenset[str]:
    """The names a family's value writes with its base at a whole number, as x_0 is in
    ``x_n := x_0 + n d``: names of their own, never members."""
    found = set()
    for node in _nodes(value):
        named = _subscripted(node) if isinstance(node, ast.Name | ast.Scripted) else None
        if named is not None and named[0] == base and re.fullmatch(r"-?\d+", named[1]):
            found.add(f"{base}_{named[1]}")
    return frozenset(found)


def _nodes(node: ast.Node) -> Iterator[ast.Node]:
    """Every node of a syntax tree, as :func:`relativisticpy.typing.syntax._walk` gives
    them."""
    pending: list[object] = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, tuple):
            pending.extend(current)
        elif isinstance(current, ast.Node):
            yield current
            pending.extend(vars(current).values())


def _subscript_written(subscript: str, span: Span) -> ast.Expr | None:
    """The subscript of a name the lexer joins, as an expression: 1 of ``c_1``, k of
    ``c_k``; ``None`` for one that is neither a whole number nor a letter."""
    if re.fullmatch(r"\d+", subscript):
        return ast.Number(subscript, span=span)
    if re.fullmatch(r"[A-Za-z]", subscript):
        return ast.Name(subscript, span=span)
    return None


def _is_name(node: ast.Node, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _whole_numbers(node: ast.Expr | None) -> int | None:
    """A whole number written as a limit, as 1 or -2."""
    if isinstance(node, ast.Number) and node.text.isdigit():
        return int(node.text)
    if isinstance(node, ast.Unary) and node.op == "-":
        inside = _whole_numbers(node.operand)
        return -inside if inside is not None else None
    return None


class Families(Labels):
    """Families of names, defined with a free label, and their members."""

    def __init__(self, session: Session) -> None:
        super().__init__(session)
        # The families defined through their own members whose value is being made, by
        # the base their members are written with, and the function each member stands
        # for while it is.
        self._recurring: dict[str, str] = {}

    def _inline(self, binding: FunctionBinding, args: list[Node], span: Span, ctx: Context) -> Node:
        """A function's value at its arguments, which the lowerer gives (see
        :class:`~relativisticpy.lowering.expressions.Expressions`)."""
        raise NotImplementedError

    # -- what a family is -----------------------------------------------------------------

    def _may_label(self, label: str, scope: Scope) -> bool:
        """Whether a letter may be a family's label: no index, no coordinate, and free,
        as no name with a value is."""
        coordinates = {c.name for c in self.session.coordinates or ()}
        return not (
            self._an_index_here(label) or label in coordinates or scope.lookup(label) is not None
        )

    def _written_with_label(self, value: ast.Expr, label: str, scope: Scope, latex: bool) -> bool:
        """Whether a value is written with a family's label: as a name, as n is in 1/n^2, or
        as the subscript of a name, as a is in \\sigma_a and i in t_i; never as an index of
        a tensor, as a is in A_{ab} w^b."""
        pending: list[object] = [value]
        while pending:
            node = pending.pop()
            if isinstance(node, tuple):
                pending.extend(node)
            elif isinstance(node, ast.Name):
                if node.id == label or node.id.partition("_")[2] == label:
                    return True
                known = node.id in BUILTINS or node.id in CONSTANTS
                letters = split_index_labels(node.id) if latex and _splittable(node) else []
                if not known and label in letters:  # n of nx
                    return True
            elif isinstance(node, ast.Scripted) and self._indexed_tensor(node, scope):
                pending.append(node.base)
            elif isinstance(node, ast.Call):
                pending.extend(node.args)
            elif isinstance(node, ast.Node):
                pending.extend(vars(node).values())
        return False

    def _indexed_tensor(self, node: ast.Scripted, scope: Scope) -> bool:
        """Whether a name with scripts is a tensor written with its indices: one the
        program defines, a built-in one in force, or a derivative's index."""
        base = node.base
        if isinstance(base, ast.Operator):
            return True
        if not isinstance(base, ast.Name):
            return False
        binding = scope.lookup(base.id)
        if isinstance(binding, TensorBinding):
            return True
        if isinstance(binding, VariableBinding):
            return isinstance(binding.node.type, TensorType)
        rank = sum(len(s.indices.items) for s in node.scripts if s.indices is not None)
        in_force = self.session.coordinates is not None
        return binding is None and in_force and self._builtin_tensor(base.id, rank) is not None

    # -- definitions ------------------------------------------------------------------------

    def _define_family(self, target: ast.Expr, value: ast.Expr, scope: Scope, ctx: Context) -> bool:
        """Define a family where the name being defined has a one-letter subscript its value
        is written with, or a recurrence, written with its own members (decision 5.2):
        ``c_n := ...``, ``\\psi_n(x) := ...``, ``a_{n+2} := a_{n+1} + a_n``. ``False``
        for any other definition, which is made as before."""
        parameters: tuple[str, ...] = ()
        called = ""
        if (applied := _with_arguments(target)) is not None:  # \psi_n(x)
            target, group = applied
            items = group.items if isinstance(group, ast.Tuple) else (group,)
            parameters = tuple(item.id for item in items if isinstance(item, ast.Name))
            called = group.span.text
        found = _family_target(target)
        if found is None:
            return False
        base, label, shift = found
        recurring = _recurrence(target, value) is not None
        if label in parameters or not self._may_label(label, scope):
            if recurring:  # n has a value: a_n of its own value would be a symbol of its own
                raise CompileError(
                    f"{target.span.text} is written with its own members, as a recurrence in "
                    f"{label} is, but {label} already means something here",
                    span=target.span.to(value.span),
                    hint=f"write the recurrence with a label that means nothing yet, as in "
                    f"{_shown(base, ctx.latex)}_k := ...",
                )
            return False
        if not recurring and not self._written_with_label(value, label, scope, ctx.latex):
            return False
        family = Family(
            f"{base}_{label}",
            (label, *parameters),
            value,
            span=target.span.to(value.span),
            latex=ctx.latex,
            base=base,
            fixed=_fixed_members(value, base),
            shift=shift,
            recurring=recurring,
        )
        self._defined_family(family, target.span.text, called, scope, ctx)
        return True

    def _define_function_family(
        self, definition: ast.FunctionDefinition, scope: Scope, ctx: Context
    ) -> bool:
        """Define ``psi_n(x) := \\sin(n x)``, written in plain text, as a family of functions
        where its label is written in its value; ``False`` for any other function."""
        base, underscore, label = definition.name.partition("_")
        if not (underscore and re.fullmatch(r"[A-Za-z]+", base) and _LABEL.fullmatch(label)):
            return False
        if len(label) != 1 or label in definition.params or not self._may_label(label, scope):
            return False
        if _WORD.search(definition.span.text.partition("(")[0]):
            return False  # C_\mathrm{A}(t): a word, never a label that takes values
        body = definition.body
        if not isinstance(body, ast.Expr) or not self._written_with_label(
            body, label, scope, definition.latex
        ):
            return False
        family = Family(
            definition.name,
            (label, *definition.params),
            body,
            span=definition.span,
            latex=definition.latex,
            base=base,
            fixed=_fixed_members(body, base),
        )
        called = f"({', '.join(_shown(p, definition.latex) for p in definition.params)})"
        self._defined_family(family, _shown(definition.name, definition.latex), called, scope, ctx)
        return True

    def _defined_family(
        self, family: Family, written: str, called: str, scope: Scope, ctx: Context
    ) -> None:
        """Bind a family, written as ``written`` and, of functions, ``called`` with its
        parameters, checking that its value means something at its label, and say so."""
        label, shown = family.label, _shown(family.base, ctx.latex)
        written += called
        if family.recurring:
            why = "its value is written with its own members"
            reading = f"a sequence defined by a recurrence in {label}"
            note = (
                f"{written} is defined by a recurrence: its members are computed from those "
                "given before them"
            )
        else:
            why = f"{label} is written in its value"
            reading = f"a family in {label}"
            note = (
                f"{written} is a family in {label}: {shown}_1{called} is {written} at {label} = 1"
            )
        self._decide(FAMILY, family.span, reading, why, note, written=f"{written} := ...")
        self._note(note)
        binding = FunctionBinding(family, scope)
        if not family.recurring:  # what it means at its label is said where it is written
            symbol = self.graph.const(self.session.symbol(label), SCALAR)
            arguments = [self._symbol_of(p) for p in family.parameters]
            with self._members_at(label, symbol):
                self._inline(binding, [symbol, *arguments], family.span, ctx)
        scope.bind(family.name, binding)

    def _symbol_of(self, name: str) -> Node:
        return self.graph.const(self.session.symbol(name), SCALAR)

    # -- members ----------------------------------------------------------------------------

    def _member(
        self,
        binding: FunctionBinding,
        point: Node,
        arguments: Sequence[Node] | None,
        span: Span,
        scope: Scope,
        ctx: Context,
    ) -> Node:
        """A family at a member's subscript, ``point``, and, of a family of functions, at
        its ``arguments``, or as a function of its parameters where none are given."""
        family = binding.definition
        assert isinstance(family, Family)
        if family.base in self._recurring:  # a_{n-1} in the value of a_{n+2} := ...
            name = (("name", self._recurring[family.base]),)
            return self._call("core.function", [point], SCALAR, span, name)
        if family.recurring:
            return self._recurrence_member(binding, point, span, scope, ctx)
        at = self._shifted(point, -family.shift, span)
        if arguments is None:
            arguments = [self._symbol_of(p) for p in family.parameters]
        with self._members_at(family.label, at):
            return self._inline(binding, [at, *arguments], span, ctx)

    def _shifted(self, point: Node, by: int, span: Span) -> Node:
        if not by:
            return point
        return self.graph.add(ops.Add((1, 1)), [point, self.graph.const(by, SCALAR)], SCALAR, span)

    def _recurrence_member(
        self, binding: FunctionBinding, point: Node, span: Span, scope: Scope, ctx: Context
    ) -> Node:
        """A member of a family defined by a recurrence, computed from its first members,
        which the program gives as names of their own, as a_0 and a_1."""
        family = binding.definition
        assert isinstance(family, Family)
        given = self._given_members(family.base, scope)
        written = f"{_shown(family.base, ctx.latex)}_{family.label}"
        if not given:
            raise CompileError(
                f"{written} is defined by a recurrence, but none of its members is given",
                span=span,
                hint=f"give its first members, as in {_shown(family.base, ctx.latex)}_0 := ...",
            )
        label = self._symbol_of(family.label)
        member = f"{family.base}_"  # the function a member stands for, which no name is
        saved = dict(self._recurring)
        self._recurring[family.base] = member
        try:
            body = self._inline(binding, [label], span, ctx)
        finally:
            self._recurring = saved
        self._check_members_held(family, body, scope, ctx)
        pairs = [n for k, node in sorted(given.items()) for n in (self.graph.const(k, VALUE), node)]
        attributes = (("name", member), ("shift", family.shift))
        kind = _passed(body.type)
        return self._call("core.recurrence", [body, label, point, *pairs], kind, span, attributes)

    def _check_members_held(self, family: Family, body: Node, scope: Scope, ctx: Context) -> None:
        """Refuse a recurrence written with names whose values hold its member as a symbol of
        their own, as the stages k_1 := h f(t_i, w_i) of Runge and Kutta hold w_i where they
        are defined, before ``w_{i+1} := w_i + ...``: computed, it would keep w_i."""
        if not _mentions_symbol(body, family.name):
            return
        names = sorted(_nodes(family.body), key=lambda node: node.span.start)
        holders = [
            name.span.text
            for name in names
            if isinstance(name, ast.Name)
            and isinstance(binding := scope.lookup(name.id), VariableBinding)
            and _mentions_symbol(binding.node, family.name)
        ]
        written = f"{_shown(family.base, ctx.latex)}_{family.label}"
        held = _listed(list(dict.fromkeys(holders))) if holders else "the names it is written with"
        raise CompileError(
            f"{written} is computed from {held}, whose values hold {written} as a symbol of "
            "their own, not as the member the recurrence computes",
            span=family.span,
            hint="write what they stand for in the recurrence itself, or define each with the "
            f"label {family.label} too",
        )

    def _given_members(self, base: str, scope: Scope) -> dict[int, Node]:
        """The members of a family the program gives as names of their own, as a_0 := 1."""
        given: dict[int, Node] = {}
        current: Scope | None = scope
        while current is not None:
            for name, binding in current.bindings():
                head, _, index = name.partition("_")
                numbered = head == base and re.fullmatch(r"-?\d+", index)
                if numbered and isinstance(binding, VariableBinding):
                    given.setdefault(int(index), binding.node)
            current = current.parent
        return given

    def _member_written(
        self,
        node: ast.Expr,
        base: str,
        subscript: ast.Expr,
        arguments: Sequence[ast.Expr] | None,
        scope: Scope,
        ctx: Context,
        written: str | None = None,
    ) -> Node | None:
        """A name of ``base`` with a ``subscript``, which nothing defines, where the
        resolver reads it as a member of a family (see
        :meth:`~relativisticpy.reading.subscripts.Subscripts._reads_as_member`); ``None``
        where it does not."""
        shown = written or subscript.span.text.replace(" ", "")
        form = member_written(node, base, subscript, shown)
        reading = self._reads_as_member(form, scope, ctx.latex, arguments is not None)
        if reading is None or reading.refusal is not None:
            return None
        return self._member_read(reading, arguments, scope, ctx)

    def _member_read(
        self, reading: Reading, arguments: Sequence[ast.Expr] | None, scope: Scope, ctx: Context
    ) -> Node | None:
        """A member of a family, as the resolver read it: the member the program gives as
        a name of its own at the subscript, where it gives one, and else the family at the
        subscript, and at the ``arguments`` where it is called. Of a label at a value, the
        name at that value, as \\sigma_a is \\sigma_1 in T_1. ``None`` where that names no
        member."""
        written = reading.written
        assert written is not None
        node, base, subscript = written.node, written.base, written.subscript
        assert subscript is not None
        if reading.at_a_value:
            assert isinstance(subscript, ast.Name)
            return self._at_member(node, base, subscript.id, arguments, scope, ctx)
        binding = reading.binding
        assert isinstance(binding, FunctionBinding)
        family = binding.definition
        assert isinstance(family, Family)
        bound = isinstance(subscript, ast.Name) and scope.lookup(subscript.id) is not None
        own = isinstance(subscript, ast.Name) and subscript.id == family.label
        if own and not bound:  # c_n as it stands: at its label, never at i = \sqrt{-1}
            point = self._symbol_of(family.label)
        else:
            point = self.value(subscript, scope, ctx)
        at = self._folded(point)
        given = self._member_given(base, at, arguments, node, scope, ctx)
        if given is not None:
            return given
        args = [self.value(a, scope, ctx) for a in arguments] if arguments is not None else None
        member = f"{_shown(base, ctx.latex)}_{family.label}"
        why = f"{member} is a family in {family.label}, defined on line {family.span.line}"
        shown = str(at) if isinstance(at, sympy.Integer) else written.shown
        self._decide(FAMILY, node.span, f"{member} at {family.label} = {shown}", why)
        return self._member(binding, point, args, node.span, scope, ctx)

    def _member_given(
        self,
        base: str,
        at: object,
        arguments: Sequence[ast.Expr] | None,
        node: ast.Expr,
        scope: Scope,
        ctx: Context,
    ) -> Node | None:
        """The member the program gives as a name of its own at a subscript whose value is
        known where it is read, a whole number or a symbol, as ``c_2`` is of ``c_{2k}``
        where k is 1: the name's value, which the member written out has too. ``None``
        where the program gives none there."""
        if isinstance(at, sympy.Integer):
            name = f"{base}_{at}"
        elif isinstance(at, sympy.Symbol):
            name = f"{base}_{at.name}"
        else:
            return None
        given = scope.lookup(name)
        if given is None:
            return None
        if isinstance(given, FunctionBinding) and isinstance(given.definition, Family):
            return None  # c_n at n itself: the family, not a member it is given
        named = ast.Name(name, verbatim=True, span=node.span)
        if arguments is not None:
            return self.value(ast.Call(named, tuple(arguments), span=node.span), scope, ctx)
        return self.value(named, scope, ctx)

    def _at_member(
        self,
        node: ast.Expr,
        base: str,
        label: str,
        arguments: Sequence[ast.Expr] | None,
        scope: Scope,
        ctx: Context,
    ) -> Node | None:
        """A name written with a family's label as its subscript, in the family's value
        made at a member: the name with the member's subscript, as \\sigma_a is \\sigma_1
        in T_1 of ``T_a := \\sigma_a/2``. ``None`` where the member's subscript is no
        whole number or symbol, and the name has no family to be a member of."""
        point = self._at_members[label]
        value = point.op.value if isinstance(point.op, ops.Const) else None
        if isinstance(value, sympy.Symbol) and value.name == label:
            return None  # \sigma_a, as written, where the member is at a itself
        if isinstance(value, sympy.Integer | sympy.Symbol):
            shown = str(value) if isinstance(value, sympy.Integer) else value.name
            named = ast.Name(f"{base}_{shown}", verbatim=True, span=node.span)
            if scope.lookup(named.id) is not None or family_of(base, scope) is None:
                if arguments is not None:
                    return self.value(ast.Call(named, tuple(arguments), span=node.span), scope, ctx)
                return self.value(named, scope, ctx)
            if isinstance(value, sympy.Integer):
                written: ast.Expr = ast.Number(shown, span=node.span)
            else:
                written = ast.Name(shown, span=node.span)
            return self._member_written(node, base, written, arguments, scope, ctx, shown)
        binding = family_of(base, scope)
        if binding is None:
            return None
        args = [self.value(a, scope, ctx) for a in arguments] if arguments is not None else None
        return self._member(binding, point, args, node.span, scope, ctx)

    # -- names in expressions -------------------------------------------------------------

    def _Name(self, node: ast.Name, scope: Scope, ctx: Context) -> Term:
        binding = scope.lookup(node.id)
        if isinstance(binding, FunctionBinding) and isinstance(binding.definition, Family):
            return self._family_alone(node, binding, None, scope, ctx)  # c_n
        written = subscripted(node)
        if written is not None:  # c_1, t_i, f_x, X_i, H_n, y_t
            term = self._subscripted_term(written, scope, ctx)
            if term is not None:
                return term
        return super()._Name(node, scope, ctx)

    def _family_alone(
        self,
        node: ast.Expr,
        binding: FunctionBinding,
        arguments: Sequence[ast.Expr] | None,
        scope: Scope,
        ctx: Context,
    ) -> Node:
        """A family written by its own name, as ``c_n`` or ``a_n``: its member at the label
        as it stands, a symbol, or the variable a sum or a limit binds."""
        family = binding.definition
        assert isinstance(family, Family)
        label = ast.Name(family.label, span=node.span)
        member = self._member_written(node, family.base, label, arguments, scope, ctx, family.label)
        if member is None:  # its label made a name of its own: the family at the label
            point = self.value(label, scope, ctx)
            args = [self.value(a, scope, ctx) for a in arguments] if arguments else None
            return self._member(binding, point, args, node.span, scope, ctx)
        return member

    def _labelled(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Term | None:
        """A name with one subscript, as ``a_{n+1}``, ``E_{-1}``, ``X_{i}`` or ``f_{xy}``,
        read as the resolver reads it, with what is raised on a member its power, as
        ``T_1^2``; or as a name of its own (see :meth:`Labels._labelled`)."""
        if (member := self._ordered_member(node, scope, ctx)) is not None:  # E^{(0)}_1
            return member
        written = subscripted(node)
        if written is not None:
            term = self._subscripted_term(written, scope, ctx)
            if term is not None:
                return term
        return super()._labelled(node, scope, ctx)

    def _ordered_member(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Node | None:
        """``E^{(0)}_1`` or ``E_1^{(0)}``: a member of a family with an order raised on it,
        as ``E^{(0)}_n := ...`` defines; ``None`` for any other name."""
        ordered = _ordered(node)
        if ordered is None or family_of(ordered[0], scope) is None:
            return None
        base, at = ordered
        if scope.lookup(f"{base}_{at}") is not None:
            return None  # a member the program names, a name of its own
        lowered = [s for s in node.scripts if s.kind == "_"]
        subscript = lowered[0].body if lowered else _subscript_written(at, node.span)
        if subscript is None:
            return None
        return self._member_written(node, base, subscript, None, scope, ctx, at)

    def _member_call(self, node: ast.Call, scope: Scope, ctx: Context) -> Node | None:
        """A call of a member of a family of functions, as ``\\psi_3(x)``, which the parser
        writes ``psi_3(x)``, or ``\\psi_n(x)`` itself; ``None`` for any other call."""
        func = node.func
        written = subscripted(func)
        if written is None or written.raised or not isinstance(func, ast.Name | ast.Scripted):
            return None
        reading = self._subscript_reading(written, scope, latex=ctx.latex, called=True)
        if reading.kind is Kind.DEFINED and reading.name is not None:
            binding = scope.lookup(reading.name)
            if isinstance(binding, FunctionBinding) and isinstance(binding.definition, Family):
                alone = func if isinstance(func, ast.Name) else node
                return self._family_alone(alone, binding, node.args, scope, ctx)
            return None
        member = reading.kind in (Kind.MEMBER, Kind.RECURRENCE) and reading.refusal is None
        if not (member or reading.at_a_value):
            return None
        called = replace(reading, written=replace(written, node=node))
        return self._member_read(called, node.args, scope, ctx)

    def _calls_of_names(self, factors: list[ast.Expr], scope: Scope) -> list[ast.Expr]:
        """Factors with a name written with scripts and, right after it, its arguments in
        brackets, made one call where the name is a function the program defines or a
        family of them, as ``\\psi_n(x)`` and ``R_{nl}(r)``, which the parser reads as
        products."""
        merged: list[ast.Expr] = []
        position = 0
        while position < len(factors):
            factor = factors[position]
            following = factors[position + 1] if position + 1 < len(factors) else None
            group: ast.Expr | None = following
            scripts: tuple[ast.Script, ...] = ()
            if isinstance(following, ast.Scripted):  # \psi_n(x)^*: the call's script
                group, scripts = following.base, following.scripts
            arguments = forms.bracketed(group) if group is not None else None
            if (
                following is not None
                and group is not None
                and arguments is not None
                and factor.span.end == group.span.start
                and self._names_a_function(factor, scope)
            ):
                call: ast.Expr = ast.Call(factor, arguments, span=factor.span.to(group.span))
                if scripts:
                    call = ast.Scripted(call, scripts, span=factor.span.to(following.span))
                merged.append(call)
                position += 2
                continue
            merged.append(factor)
            position += 1
        return merged

    def _names_a_function(self, node: ast.Expr, scope: Scope) -> bool:
        """Whether a name with scripts is a function the program defines under that name,
        or a member of a family of functions, as ``\\psi_n`` of ``\\psi_n(x) := ...``."""
        if not (isinstance(node, ast.Scripted) and isinstance(node.base, ast.Name)):
            return False
        for name in (label_name(node), label_name(node, raised=False)):
            if name is not None and isinstance(scope.lookup(name), FunctionBinding):
                return True
        lowered = [s for s in node.scripts if s.kind == "_"]
        if len(lowered) != 1 or len(node.scripts) != 1 or "_" in node.base.id:
            return False
        binding = family_of(node.base.id, scope)
        family = binding.definition if binding is not None else None
        return isinstance(family, Family) and bool(family.parameters)

    # -- recurrences ------------------------------------------------------------------------

    def _check_recurrence(self, target: ast.Expr, value: ast.Expr, span: Span) -> None:
        """A recurrence of a family, as ``a_{n+2} := a_{n+1} + a_n``, is defined (see
        :meth:`_define_family`); one of another shape is refused, as before."""
        if _family_target(target) is not None and _recurrence(target, value) is not None:
            return
        super()._check_recurrence(target, value, span)

    def _reading(
        self, equation: ast.Equation, scope: Scope, ctx: Context
    ) -> tuple[Literal["define", "function", "claim", "notation"], str | None, str]:
        """``T_n = 2 T_{n-1} + 1`` or ``C_{n+1} = \\sum ...``: a recurrence, written with =,
        defines a family as := does, where its name has no meaning yet."""
        found = _family_target(equation.lhs)
        if found is not None and _recurrence(equation.lhs, equation.rhs) is not None:
            base, label, _ = found
            if scope.lookup(f"{base}_{label}") is None and self._may_label(label, scope):
                return "define", None, f"{equation.lhs.span.text} is defined by a recurrence"
        return super()._reading(equation, scope, ctx)

    # -- sums of names ----------------------------------------------------------------------

    def _term_by_term(self, node: ast.BigOperator, scope: Scope, ctx: Context) -> Node | None:
        """A sum or a product whose index runs through the subscript of a name, as n does
        in ``\\sum_{n=0}^{3} a_n`` and k in ``\\sum_{k=1}^{3} c_{2k}``: with limits that are
        whole numbers it is taken term by term, each name read at the value the index has
        there as it is read written out, the member the program gives, the family at the
        subscript, or a name of its own, so that the sum agrees with its terms. With other
        limits it is refused where its terms cannot be told apart from its formula (see
        :meth:`_check_runs`). ``None`` for any other sum, and for one of tensors."""
        variable = node.variable
        if not isinstance(variable, ast.Name) or node.lower is None or node.upper is None:
            return None
        runs = self._runs_through(node.body, variable.id, scope, ctx)
        if not runs:
            return None
        first = self._whole_limit(node.lower, scope, ctx)
        last = self._whole_limit(node.upper, scope, ctx)
        if first is not None and last == first - 1:
            return None  # no terms: its value does not depend on them
        if first is None or last is None or not 0 < last - first + 1 <= _TERMS:
            self._check_runs(node, runs, first, last, scope, ctx)
            return None
        terms = []
        ranges = dict(self._term_ranges)
        self._term_ranges[variable.id] = (first, last)
        try:
            for k in range(first, last + 1):
                inner = scope.child()
                point = self.graph.const(sympy.Integer(k), SCALAR)
                inner.bind(variable.id, VariableBinding(point))
                with self._members_at(variable.id, point):
                    terms.append(self.value(node.body, inner, ctx))
        finally:
            self._term_ranges = ranges
        if any(isinstance(term.type, TensorType) for term in terms):
            return None
        ran = runs[0].written
        assert ran is not None
        shown = _shown(ran.base, ctx.latex)
        what = "sum" if node.op == "sum" else "product"
        why = f"{variable.span.text} is the subscript of {shown}_{first}, …, {shown}_{last}"
        if not isinstance(ran.subscript, ast.Name):  # c_{2k}
            why = f"{variable.span.text} runs through the subscript of {ran.node.span.text}"
        self._decide(FAMILY, node.span, f"the {what} of its {len(terms)} terms", why)
        for reading in runs:
            self._say_names_of_a_book_letter(reading, variable, (first, last), scope, ctx)
        if node.op == "prod":
            return self._multiply(terms, node.span)
        kind = VALUE if any(isinstance(term.type, ValueType) for term in terms) else SCALAR
        return self.graph.add(ops.Add((1,) * len(terms)), terms, kind, node.span)

    def _whole_limit(self, limit: ast.Expr, scope: Scope, ctx: Context) -> int | None:
        """A limit of a sum or a product, where it is a whole number known as the program
        is read: written as one, as 3 or -2, or computed from names given whole numbers, as
        n and n - 1 after ``n := 8``; ``None`` for any other, as N or \\infty."""
        written = _whole_numbers(limit)
        if written is not None:
            return written
        read = dict(self._decided), dict(self._resolved), list(self._notes)
        try:  # compiled to see whether it is known, not to be read so
            value = self._folded(self.value(limit, scope, ctx))
        except RelativisticPyError:
            return None
        finally:
            self._decided, self._resolved, self._notes = read
        return int(value) if isinstance(value, sympy.Integer) else None

    def _say_names_of_a_book_letter(
        self,
        reading: Reading,
        index: ast.Name,
        ends: tuple[int, int],
        scope: Scope,
        ctx: Context,
    ) -> None:
        """Say so where a name a sum runs through is one books write for a number, as
        ``H_n`` after ``n \\in \\mathbb{N}``: under the sum over n it is a name for each n, as
        the coefficients of a series are (decisions 34 and 35), though alone it is the
        number."""
        written = reading.written
        assert written is not None
        book = self._reads_as_book(written, scope) if reading.kind is Kind.LABEL else None
        call = book.book.call if book is not None and book.book is not None else None
        if call is None:
            return
        base, capability = _shown(written.base, ctx.latex), call.capability
        self._note(
            f"{written.node.span.text} is a name for each {index.span.text} of the sum, "
            f"{base}_{ends[0]}, …, {base}_{ends[1]}, as the coefficients of a series are; "
            f"{capability.brief} is {capability.name}({index.span.text})"
        )

    def _runs_through(
        self, body: ast.Expr, index: str, scope: Scope, ctx: Context
    ) -> list[Reading]:
        """The names a sum's body writes with a subscript its index runs through, as
        ``a_n``, ``\\lambda_a`` and ``c_{2k}`` are, each as the resolver reads it there: a
        member of a family, or a name of its own. A tensor's index, an element of a list, a
        member of a sample and a book name are no such names: each is summed as it is."""
        found = []
        inner = scope.child()  # as the sum binds its index, an integer with no value yet
        symbol = self.session.symbol(index)
        inner.bind(index, VariableBinding(self.graph.const(symbol, SCALAR)))
        for node in _nodes(body):
            written = subscripted(node) if isinstance(node, ast.Expr) else None
            if written is None or written.subscript is None:
                continue
            if written.labels is not None and len(written.labels) > 1:
                continue  # T_{\mu\nu}: indices
            if written.base in _STATISTICS or written.base in BUILTINS:
                continue  # \mathbb{E}_t, a statistic over the index: no name
            if not any(_is_name(each, index) for each in _nodes(written.subscript)):
                continue
            self._sum_indices.append(symbol)
            try:
                reading = self._subscript_reading(written, inner, latex=ctx.latex)
            finally:
                self._sum_indices.pop()
            if reading.refusal is not None:
                continue
            if reading.kind is Kind.DEFINED and reading.name is not None:
                binding = scope.lookup(reading.name)
                if isinstance(binding, FunctionBinding) and isinstance(binding.definition, Family):
                    found.append(replace(reading, kind=Kind.MEMBER, binding=binding))  # a_n
            elif reading.kind in (Kind.MEMBER, Kind.RECURRENCE, Kind.LABEL):
                found.append(reading)
        return found

    def _check_runs(
        self,
        node: ast.BigOperator,
        runs: Sequence[Reading],
        first: int | None,
        last: int | None,
        scope: Scope,
        ctx: Context,
    ) -> None:
        """Refuse a sum or a product that cannot be taken term by term, its limits no whole
        numbers or its terms too many, whose index runs through a name of its own, as
        ``a_n`` where nothing defines a: read as written, a_n would be one symbol in every
        term. And one that runs past a member the program gives apart from its family, as
        a_0 beside ``a_n := 1/n``, which the family's value alone would miss. A sum over a
        recurrence, whose members follow from those given, is read as before."""
        what = "sum" if node.op == "sum" else "product"
        index = node.variable.span.text
        backwards = first is not None and last is not None and last < first
        if first is None or last is None:
            cannot = "has limits that are no whole numbers"
        else:
            cannot = "has limits that run backwards" if backwards else f"has over {_TERMS} terms"
        for reading in runs:
            written = reading.written
            assert written is not None
            text, base = written.node.span.text, _shown(written.base, ctx.latex)
            if reading.kind is Kind.LABEL:
                raise CompileError(
                    f"{text} names nothing, and the {what} runs over its subscript: read as "
                    f"written, it would be one symbol in every term, and the {what} {cannot}",
                    span=node.span,
                    hint=f"define the family, as in {base}_{index} := ..., or give the {what} "
                    "limits that are whole numbers, so that it is taken term by term",
                )
            binding = reading.binding
            family = binding.definition if isinstance(binding, FunctionBinding) else None
            if not isinstance(family, Family) or family.recurring or backwards:
                continue
            bare = isinstance(written.subscript, ast.Name)
            given = [
                k
                for k in sorted(self._given_members(written.base, scope))
                if not (bare and first is not None and k < first)
                and not (bare and last is not None and k > last)
            ]
            if given:
                member = f"{base}_{given[0]}"
                raise CompileError(
                    f"the {what} runs over {member}, which the program gives apart from "
                    f"{base}_{family.label} := ..., and the {what} {cannot}: it cannot be taken "
                    "term by term",
                    span=node.span,
                    hint=f"write the members given apart from the {what}, as {member} + "
                    f"\\{node.op}_{{{index}={given[0] + 1}}}^{{...}} {text}",
                )
