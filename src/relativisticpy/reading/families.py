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

import contextlib
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
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
from relativisticpy.errors import CompileError
from relativisticpy.integrations import _forms as forms
from relativisticpy.ir import SCALAR, VALUE, Node, TensorType, ValueType, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.parser import split_index_labels
from relativisticpy.language.source import Span
from relativisticpy.reading.labels import _WORD, Labels, _raised, label_name
from relativisticpy.reading.names import (
    _a_member,
    _mentions_symbol,
    _recurrence,
    _shown,
    _splittable,
    _subscripted,
    _with_arguments,
)
from relativisticpy.typing.notation import _passed
from relativisticpy.typing.typer import _listed

#: The decision of whether a name with a subscript is a family, or a member of one.
FAMILY = "family or name"

#: A subscript a family may be defined with: its label, maybe shifted, as n+2 is.
_LABEL = re.compile(r"(?P<label>[A-Za-z])(?P<shift>[-+]\d+)?")

#: How many terms a sum or a product is taken term by term at most.
_TERMS = 100


@dataclass(frozen=True)
class Family(ast.FunctionDefinition):
    """A family defined with a free label, as ``c_n := 1/n^2`` or ``\\psi_n(x) := ...``:
    the function of its label, then of its parameters, that each of its members is, bound
    under the family's name as written, ``c_n``. ``base`` is what its members are written
    with, c; ``fixed`` the names its value writes with the base at a subscript free of the
    label, as x_0 of ``x_n := x_0 + n d``, which are names of their own; ``shift`` how far
    the member it defines is from its label, as 2 of ``a_{n+2} := ...``; and ``recurring``
    whether its value is written with its own members."""

    base: str = field(default="", kw_only=True)
    fixed: frozenset[str] = field(default=frozenset(), kw_only=True)
    shift: int = field(default=0, kw_only=True)
    recurring: bool = field(default=False, kw_only=True)

    @property
    def label(self) -> str:
        """The label a member's subscript gives a value, n of ``c_n``."""
        return self.params[0]

    @property
    def parameters(self) -> tuple[str, ...]:
        """The parameters of each member, x of ``\\psi_n(x)``; none of ``c_n``."""
        return self.params[1:]


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
        # The labels of the families whose members are being made, each with the member's
        # subscript: \sigma_a of T_a := \sigma_a/2 is \sigma_1 in T_1.
        self._at_members: dict[str, Node] = {}
        # The families defined through their own members whose value is being made, by
        # the base their members are written with, and the function each member stands
        # for while it is.
        self._recurring: dict[str, str] = {}

    def _inline(self, binding: FunctionBinding, args: list[Node], span: Span, ctx: Context) -> Node:
        """A function's value at its arguments, which the lowerer gives (see
        :class:`~relativisticpy.lowering.expressions.Expressions`)."""
        raise NotImplementedError

    # -- what a family is -----------------------------------------------------------------

    def _family_of(self, base: str, scope: Scope) -> FunctionBinding | None:
        """The family the program defines whose members are written with ``base``, as c_n
        is for c."""
        current: Scope | None = scope
        while current is not None:
            for _, binding in reversed(current.bindings()):
                definition = binding.definition if isinstance(binding, FunctionBinding) else None
                if isinstance(definition, Family) and definition.base == base:
                    return binding  # type: ignore[return-value]
            current = current.parent
        return None

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

    @contextlib.contextmanager
    def _members_at(self, label: str, point: Node) -> Iterator[None]:
        """Make a family's value at a member: its label's subscripts are the member's."""
        saved = dict(self._at_members)
        self._at_members[label] = point
        try:
            yield
        finally:
            self._at_members = saved

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
        """A name with a subscript that nothing defines read as a member of a family of its
        base, as ``c_1`` and ``a_{n+1}`` are; or the name of the member's subscript inside
        a family's value, as \\sigma_a is \\sigma_1 in T_1. ``None`` for a name that is
        neither, or that is one of the names a family's value writes, as x_0; and, not
        called, for a subscript that is a letter nothing binds, as k of c_k, a symbol of
        its own as before."""
        written_subscript = written or subscript.span.text.replace(" ", "")
        if isinstance(subscript, ast.Name) and subscript.id in self._at_members:
            return self._at_member(node, base, subscript.id, arguments, scope, ctx)
        binding = self._family_of(base, scope)
        if binding is None:
            return None
        family = binding.definition
        assert isinstance(family, Family)
        name = f"{base}_{written_subscript}"
        if name in family.fixed:
            return None
        bound = isinstance(subscript, ast.Name) and scope.lookup(subscript.id) is not None
        own = isinstance(subscript, ast.Name) and subscript.id == family.label
        if arguments is None and not (own or _a_member(written_subscript, bound)):
            return None
        if own and not bound:  # c_n as it stands: at its label, never at i = \sqrt{-1}
            point = self._symbol_of(family.label)
        else:
            point = self.value(subscript, scope, ctx)
        args = [self.value(a, scope, ctx) for a in arguments] if arguments is not None else None
        member = f"{_shown(base, ctx.latex)}_{family.label}"
        why = f"{member} is a family in {family.label}, defined on line {family.span.line}"
        reading = f"{member} at {family.label} = {written_subscript}"
        self._decide(FAMILY, node.span, reading, why)
        return self._member(binding, point, args, node.span, scope, ctx)

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
            if scope.lookup(named.id) is not None or self._family_of(base, scope) is None:
                if arguments is not None:
                    return self.value(ast.Call(named, tuple(arguments), span=node.span), scope, ctx)
                return self.value(named, scope, ctx)
            if isinstance(value, sympy.Integer):
                written: ast.Expr = ast.Number(shown, span=node.span)
            else:
                written = ast.Name(shown, span=node.span)
            return self._member_written(node, base, written, arguments, scope, ctx, shown)
        binding = self._family_of(base, scope)
        if binding is None:
            return None
        args = [self.value(a, scope, ctx) for a in arguments] if arguments is not None else None
        return self._member(binding, point, args, node.span, scope, ctx)

    # -- names in expressions -------------------------------------------------------------

    def _Name(self, node: ast.Name, scope: Scope, ctx: Context) -> Term:
        binding = scope.lookup(node.id)
        if isinstance(binding, FunctionBinding) and isinstance(binding.definition, Family):
            return self._family_alone(node, binding, None, scope, ctx)  # c_n
        if binding is None:  # c_1, t_i
            base, underscore, subscript = node.id.partition("_")
            written = _subscript_written(subscript, node.span) if underscore and base else None
            if written is not None:
                member = self._member_written(node, base, written, None, scope, ctx, subscript)
                if member is not None:
                    return member
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
        """A name with one subscript, as ``a_{n+1}``, ``E_{-1}`` or ``c_{n}``, read as a
        member of a family, with what is raised on it a power, as ``T_1^2``; or as a name of
        its own (see :meth:`Labels._labelled`)."""
        lowered = [s for s in node.scripts if s.kind == "_"]
        raised = [s for s in node.scripts if s.kind == "^"]
        base = node.base
        if (member := self._ordered_member(node, scope, ctx)) is not None:  # E^{(0)}_1
            return member
        if len(lowered) != 1 or not isinstance(base, ast.Name) or "_" in base.id:
            return super()._labelled(node, scope, ctx)
        name = label_name(node, raised=False)
        if name is not None and scope.lookup(name) is not None:
            return super()._labelled(node, scope, ctx)  # E_\pi, defined
        subscript = lowered[0].body
        if subscript is None:
            return super()._labelled(node, scope, ctx)
        member = self._member_written(node, base.id, subscript, None, scope, ctx)
        if member is None:
            return super()._labelled(node, scope, ctx)
        for script in raised:
            if script.body is None:
                return super()._labelled(node, scope, ctx)
            member = self._power(member, self.value(script.body, scope, ctx), node.span)
        return member

    def _ordered_member(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Node | None:
        """``E^{(0)}_1`` or ``E_1^{(0)}``: a member of a family with an order raised on it,
        as ``E^{(0)}_n := ...`` defines; ``None`` for any other name."""
        ordered = _ordered(node)
        if ordered is None or self._family_of(ordered[0], scope) is None:
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
        if isinstance(func, ast.Name):
            binding = scope.lookup(func.id)
            if isinstance(binding, FunctionBinding) and isinstance(binding.definition, Family):
                return self._family_alone(func, binding, node.args, scope, ctx)
            base, underscore, subscript = func.id.partition("_")
            written = _subscript_written(subscript, func.span) if underscore else None
            if binding is not None or written is None:
                return None
            return self._member_written(node, base, written, node.args, scope, ctx, subscript)
        if not (isinstance(func, ast.Scripted) and isinstance(func.base, ast.Name)):
            return None
        lowered = [s for s in func.scripts if s.kind == "_"]
        if len(lowered) != 1 or len(func.scripts) != 1 or lowered[0].body is None:
            return None
        name = label_name(func, raised=False)
        binding = scope.lookup(name) if name is not None else None
        if isinstance(binding, FunctionBinding) and isinstance(binding.definition, Family):
            return self._family_alone(node, binding, node.args, scope, ctx)
        if binding is not None:
            return None
        return self._member_written(node, func.base.id, lowered[0].body, node.args, scope, ctx)

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
        binding = self._family_of(node.base.id, scope)
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
        """``\\sum_{a=1}^{8} \\lambda_a^2`` after \\lambda_1, ..., \\lambda_8 are defined: a
        finite sum or product whose index is the subscript of names the program defines, one
        or more of them, taken term by term; ``None`` for any other."""
        variable = node.variable
        first, last = _whole_numbers(node.lower), _whole_numbers(node.upper)
        if not isinstance(variable, ast.Name) or first is None or last is None:
            return None
        if not 0 < last - first + 1 <= _TERMS or scope.lookup(variable.id) is not None:
            return None
        bases = self._named_by_index(node.body, variable.id, scope)
        defined = [
            base
            for base in bases
            if any(scope.lookup(f"{base}_{k}") is not None for k in range(first, last + 1))
        ]
        if not defined:
            return None
        terms = []
        for k in range(first, last + 1):
            inner = scope.child()
            point = self.graph.const(sympy.Integer(k), SCALAR)
            inner.bind(variable.id, VariableBinding(point))
            with self._members_at(variable.id, point):
                terms.append(self.value(node.body, inner, ctx))
        if any(isinstance(term.type, TensorType) for term in terms):
            return None
        shown = _shown(defined[0], ctx.latex)
        what = "sum" if node.op == "sum" else "product"
        why = f"{variable.span.text} is the subscript of {shown}_{first}, …, {shown}_{last}"
        self._decide(FAMILY, node.span, f"the {what} of its {len(terms)} terms", why)
        if node.op == "prod":
            return self._multiply(terms, node.span)
        kind = VALUE if any(isinstance(term.type, ValueType) for term in terms) else SCALAR
        return self.graph.add(ops.Add((1,) * len(terms)), terms, kind, node.span)

    def _named_by_index(self, body: ast.Expr, label: str, scope: Scope) -> list[str]:
        """The bases of the names a body writes with ``label`` as their subscript and that
        name nothing else: \\lambda of \\lambda_a."""
        found = []
        for node in _nodes(body):
            named = _subscripted(node) if isinstance(node, ast.Name | ast.Scripted) else None
            if named is None or named[1] != label:
                continue
            base = named[0]
            if scope.lookup(base) is None and self._family_of(base, scope) is None:
                found.append(base)
        return list(dict.fromkeys(found))
