r"""The lowering of sequences, as :mod:`relativisticpy.reading.sequences` reads them
(design/notation.md, decision 31): a sequence defined with brackets, ``x[n] := \ldots``, and
its members, ``x[n - 1]``; the unit impulse and the unit step, ``\delta[n]`` and ``u[n]``;
sums over a sequence's index, ``\sum_k x[k] h[n - k]``, over every integer where no limits
are written; difference equations, kept at initial rest, and solved where a program asks;
the frequency response defined where it is written, ``H(e^{j\omega}) := \ldots``; and the
z-transforms of the sequences of a difference equation, written with their capital
letters, ``Y(z)``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import sympy

from relativisticpy import integrations
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
from relativisticpy.integrations._transforms import Written
from relativisticpy.ir import SCALAR, VALUE, Node, TensorType, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.parser import python_indexing
from relativisticpy.language.source import Span
from relativisticpy.lowering.arithmetic import INDEX_SET, Arithmetic
from relativisticpy.reading import sequences
from relativisticpy.reading.names import _shown
from relativisticpy.reading.sequences import SAMPLE, Response, Sample, Sequence
from relativisticpy.reading.subscripts import BRACKET_BOOKS, Kind, Reading
from relativisticpy.runtime.sequences import capital
from relativisticpy.typing.notation import _passed
from relativisticpy.typing.syntax import _walk
from relativisticpy.typing.typer import _listed

#: The decision of what = between sequences is, and of what := with brackets defines.
DEFINITION = "definition or claim"

#: Where a program's difference equation is kept, among the conventions it states.
SYSTEM = "difference equation"


@dataclass(frozen=True)
class System:
    """A difference equation a program states between sequences nothing defines, as
    ``y[n] - \\frac{3}{4} y[n-1] = x[n]``: its ``equation``, the sequence it gives, its
    ``output``, the others, its ``inputs``, the ``label`` its points are written with, and
    the line it is written on."""

    equation: Node
    output: str
    inputs: tuple[str, ...]
    label: str
    line: int

    def named(self, written: str) -> str | None:
        """The sequence whose transform a capital letter writes, y of Y."""
        for name in (self.output, *self.inputs):
            if capital(name) == written and capital(name) != name:
                return name
        return None


class Sequences(Arithmetic):
    """The lowering of sequences."""

    def __init__(self, session: Session) -> None:
        super().__init__(session)
        # The sequences defined through their own members whose value is being made, with
        # the function each member stands for while it is.
        self._sequence_members: dict[str, str] = {}
        # How many sequences at points have been lowered, so that a sum knows its body
        # writes one.
        self._samples_lowered = 0
        # The last difference equation compiled, with the unknown sequences it relates.
        self._difference: tuple[Node, ast.Equation, frozenset[str]] | None = None
        # How deep the values of sequences being used are, whose notes their definitions
        # said.
        self._used_sequences = 0

    # -- what brackets after a name are ---------------------------------------------------

    def _sampled(
        self,
        name: ast.Name,
        point: ast.Expr,
        scope: Scope,
        ctx: Context | None = None,
        span: Span | None = None,
    ) -> bool:
        """Whether ``name[point]`` is a sequence at a point: of a sequence the program
        defines, or a function of one variable; of an unknown sequence of the equation being
        compiled; or the impulse or the step, where nothing defines δ or u and the point is
        an integer. Otherwise it is read as before, which, given the statement's ``ctx`` and
        the ``span`` it is written over, is said: in plain text, refused as Python's
        indexing; in LaTeX a product, with a note where it could have been a sequence."""
        reading = self._bracket_reading(name, point, scope)
        sequence = reading.kind in (Kind.SEQUENCE, Kind.BOOK)
        if sequence or ctx is None or span is None:
            return sequence
        if reading.kind is Kind.OPERATOR:  # E[X]: an expectation, which probability reads
            return False
        if not ctx.latex:
            raise python_indexing(span.text, span)
        binding = scope.lookup(name.id)
        integral = sequences.integral(point, lambda label: self._integral_name(label, scope))
        if isinstance(binding, VariableBinding | TensorBinding) and integral:
            self._read_as_before(name, point, binding)
        elif binding is None and integral and sequences.integral(name, lambda label: None):
            note = (
                f"{span.text} is read as {name.span.text} times {point.span.text}, as nothing "
                f"defines {name.span.text}; for a sequence, define {name.span.text}[n] := ... "
                "first"
            )
            why = f"nothing defines {name.span.text}"
            self._decide(SAMPLE, span, "a product", why, note)
            self._note(note)
        return False

    def _a_sequence(self, name: ast.Name, point: ast.Expr, scope: Scope) -> bool:
        """Whether ``name[point]`` is a sequence at a point, as the resolver reads it (see
        :meth:`~relativisticpy.reading.subscripts.Subscripts._bracket_reading`)."""
        return self._bracket_reading(name, point, scope).kind in (Kind.SEQUENCE, Kind.BOOK)

    def _read_as_before(
        self, name: ast.Name, point: ast.Expr, binding: VariableBinding | TensorBinding
    ) -> None:
        """Record that brackets after a value the program gives a name are read as before:
        a list or a tensor is no sequence."""
        node = binding.node
        what = "a tensor" if isinstance(node.type, TensorType) else "a value"
        if node.type == VALUE:
            what = "a list or a value"
        shown = f"{name.span.text}[{point.span.text}]"
        why = f"{name.span.text} is {what}, not a sequence"
        self._decide(SAMPLE, name.span, "as before", why, written=shown)

    def _subscripted_brackets(self, name: ast.Name, point: ast.Expr, scope: Scope) -> bool:
        """Whether ``name[point]`` is read as ``name_{point}`` is: an element of a list, or
        a member of a family."""
        return self._bracket_reading(name, point, scope).kind in (Kind.ELEMENT, Kind.MEMBER)

    def _sequence_value(self, reading: Reading, scope: Scope, ctx: Context) -> Node:
        """``x_n`` or ``x_{n-1}`` of a sequence the program defines with brackets, read by
        the resolver as the sequence at the point: lowered as ``x[n]`` is, and said so."""
        written = reading.written
        assert written is not None
        assert written.subscript is not None
        node, shown = written.node, _shown(written.base, ctx.latex)
        if not self._used_sequences:  # where a sequence is used, its definition said it
            self._note(
                f"{node.span.text} is read as {shown}[{written.shown}], the sequence {shown} "
                f"at {written.shown}"
            )
        name = ast.Name(written.base, span=node.span)
        return self._Sample(Sample(name, (written.subscript,), span=node.span), scope, ctx)

    def _product(self, node: ast.Binary, scope: Scope, ctx: Context) -> Term:
        written = sequences.with_subscripts(
            node, lambda name, point: self._subscripted_brackets(name, point, scope)
        )
        if written is not node:  # x[n] of a list or of a family: as x_{n}
            return self.term(written, scope, ctx)
        rebuilt = sequences.merged(
            node, lambda name, point, span: self._sampled(name, point, scope, ctx, span)
        )
        if rebuilt is None:
            return super()._product(node, scope, ctx)
        if isinstance(rebuilt, ast.Binary):
            return super()._product(rebuilt, scope, ctx)
        return self.term(rebuilt, scope, ctx)

    def _Sample(self, node: Sample, scope: Scope, ctx: Context) -> Node:
        """A sequence at a point: a member of a sequence the program defines, the value of a
        function of one variable, an unknown sequence of an equation, or the unit impulse or
        the unit step."""
        name, point = node.func, node.point
        assert isinstance(name, ast.Name)
        self._samples_lowered += 1
        written, at = name.span.text, point.span.text
        binding = scope.lookup(name.id)
        if isinstance(binding, FunctionBinding):
            line = binding.definition.span.line
            if isinstance(binding.definition, Sequence):
                why = f"{written} is a sequence, defined on line {line}"
                self._decide(SAMPLE, node.span, f"{written} at {at}", why)
            else:
                called = f"{node.span.text} is read as {written}({at}), {written} being a function"
                why = f"{written} is a function of one variable, defined on line {line}"
                self._decide(SAMPLE, node.span, f"{written}({at})", why, called)
                self._note(called)
            return self._inline(binding, [self.value(point, scope, ctx)], node.span, ctx)
        if name.id in self._unknown_sequences:
            why = "an equation writes it at points, and nothing defines it"
            self._decide(SAMPLE, node.span, f"the sequence {written}, unknown, at {at}", why)
            value = self.value(point, scope, ctx)
            return self._call("core.function", [value], SCALAR, node.span, (("name", name.id),))
        capability = integrations.find(BRACKET_BOOKS[name.id])
        note: str | None = None
        if not self._used_sequences:  # where a sequence is used, its definition said it
            note = (
                f"{written}[{at}] is read as {capability.brief}, {capability.name}; for a "
                f"sequence of your own, define {written}[n] := ... first"
            )
            self._note(note)
        why = f"{capability.latex[0]} is how signal processing writes it, and nothing "
        why += f"defines {written}"
        self._decide(SAMPLE, node.span, f"{capability.name}, {capability.brief}", why, note)
        return self._builtin(BUILTINS[capability.name], [self.value(point, scope, ctx)], node.span)

    def _Call(self, node: ast.Call, scope: Scope, ctx: Context) -> Node:
        func = node.func
        if isinstance(func, ast.Name) and len(node.args) == 1 and not isinstance(node, Sample):
            binding = scope.lookup(func.id)
            square = _in_square_brackets(node)
            if isinstance(binding, FunctionBinding) and isinstance(binding.definition, Sequence):
                square = True
            if square and self._a_sequence(func, node.args[0], scope):
                return self._Sample(Sample(func, node.args, span=node.span), scope, ctx)
            if isinstance(binding, FunctionBinding) and isinstance(binding.definition, Response):
                return self._response_at(binding, node, scope, ctx)
            system = self.session.conventions.get(SYSTEM)
            if binding is None and isinstance(system, System):
                named = system.named(func.id)
                if named is not None and not self._has_a_meaning(func.id):
                    return self._transformed(system, named, node, scope, ctx)
        solving = isinstance(func, ast.Name) and func.id == "solve" and not scope.lookup("solve")
        if solving and (solved := self._solve_difference(node, scope, ctx)) is not None:
            return solved  # solve(y[n] = ..., y[n])
        return super()._Call(node, scope, ctx)

    # -- sequences defined ------------------------------------------------------------------

    def _define_now(self, target: ast.Expr, value: ast.Expr, scope: Scope, ctx: Context) -> None:
        found = sequences.target(target)
        if found is not None:
            self._define_sequence(found[0], found[1], target, value, scope, ctx)
            return
        response = self._response_target(target)
        if response is not None:
            self._define_response(target, response[0], response[1], value, scope, ctx)
            return
        super()._define_now(target, value, scope, ctx)

    def _define_sequence(
        self,
        name: ast.Name,
        point: ast.Expr,
        target: ast.Expr,
        value: ast.Expr,
        scope: Scope,
        ctx: Context,
    ) -> None:
        """``x[n] := a^n u[n]``: the sequence x, a function of its label; one written with
        its own members, as ``y[n] := \\frac{1}{2} y[n-1] + \\delta[n]``, a difference
        equation, at initial rest."""
        labelled = sequences.label(point)
        if labelled is None:
            raise CompileError(
                f"cannot define {target.span.text}: a sequence is defined at its label, a name, "
                "and its members follow from that",
                span=target.span,
                hint=f"define it at a label, as in {name.span.text}[n] := ...",
            )
        label, shift = labelled
        recurring = any(each.id == name.id for each, _ in sequences.samples(value))
        definition = Sequence(
            name.id,
            (label,),
            value,
            span=target.span.to(value.span),
            latex=ctx.latex,
            recurring=recurring,
            shift=shift,
        )
        written = target.span.text
        note = None
        if recurring:
            reading = f"a sequence of {label}, by a difference equation"
            why = f"its value writes {name.span.text} at other points"
            note = (
                f"{target.span.text} is defined by a difference equation, at initial rest, as "
                f"books take it: {name.span.text} is 0 before its input starts"
            )
            self._note(note)
        else:
            reading = f"a sequence of {label}"
            why = "a name with a label in square brackets is defined as a sequence"
        self._decide(DEFINITION, definition.span, reading, why, note, written=written)
        saved = scope.saved()
        binding = FunctionBinding(definition, scope)
        scope.bind(name.id, binding)
        at = self.graph.const(self.session.symbol(label), SCALAR, point.span)
        try:  # what it means at its label is said where it is written
            self._sequence_at(binding, at, definition.span, ctx)
        except BaseException:
            scope.restore(saved)
            raise

    def _inline(self, binding: FunctionBinding, args: list[Node], span: Span, ctx: Context) -> Node:
        definition = binding.definition
        if not isinstance(definition, Sequence | Response) or len(args) != 1:
            return super()._inline(binding, args, span, ctx)
        if definition.name in self._sequence_members:  # y[n - 1] in the value of y[n] := ...
            member = (("name", self._sequence_members[definition.name]),)
            return self._call("core.function", args, SCALAR, span, member)
        said = len(self._notes)
        self._used_sequences += 1
        try:  # how its value reads is said where it is defined
            if isinstance(definition, Response):
                return super()._inline(binding, args, span, ctx)
            return self._sequence_at(binding, args[0], span, ctx)
        finally:
            self._used_sequences -= 1
            del self._notes[said:]

    def _sequence_at(self, binding: FunctionBinding, point: Node, span: Span, ctx: Context) -> Node:
        """A sequence at a point: its value there, or, of one defined by a difference
        equation, its member at rest."""
        definition = binding.definition
        assert isinstance(definition, Sequence)
        if definition.recurring:
            return self._at_rest(binding, point, span, ctx)
        if definition.shift:
            by = self.graph.const(-definition.shift, SCALAR)
            point = self.graph.add(ops.Add((1, 1)), [point, by], SCALAR, span)
        return super()._inline(binding, [point], span, ctx)

    def _at_rest(self, binding: FunctionBinding, point: Node, span: Span, ctx: Context) -> Node:
        """A member of a sequence defined by a difference equation, at initial rest, computed
        as :mod:`relativisticpy.runtime.sequences` says."""
        definition = binding.definition
        assert isinstance(definition, Sequence)
        label = self._symbol_of(definition.label)
        saved = dict(self._sequence_members)
        self._sequence_members[definition.name] = definition.name
        try:
            body = super()._inline(binding, [label], span, ctx)
        finally:
            self._sequence_members = saved
        attributes = (("name", definition.name), ("shift", definition.shift))
        return self._call(
            "seq.recurrence", [body, label, point], _passed(body.type), span, attributes
        )

    # -- frequency responses ------------------------------------------------------------------

    def _response_target(self, target: ast.Expr) -> tuple[ast.Name, ast.Name] | None:
        """The name and the angle of ``H(e^{j\\omega})``, a frequency response defined
        where it is written; ``None`` for any other target."""
        if not (isinstance(target, ast.Call) and isinstance(target.func, ast.Name)):
            return None
        if len(target.args) != 1:
            return None
        found = sequences.unit_circle(target.args[0])
        if found is None or not isinstance(found[1], ast.Name):
            return None
        return target.func, found[1]

    def _define_response(
        self,
        target: ast.Expr,
        name: ast.Name,
        angle: ast.Name,
        value: ast.Expr,
        scope: Scope,
        ctx: Context,
    ) -> None:
        """``H(e^{j\\omega}) := \\sum_n h[n] e^{-j\\omega n}``: H, a function of ω, as books
        write the discrete-time Fourier transform at e^{jω}."""
        definition = Response(
            name.id,
            (angle.id,),
            value,
            span=target.span.to(value.span),
            latex=ctx.latex,
            written=target.span.text,
        )
        written = f"{target.span.text} := ..."
        point = target.args[0].span.text if isinstance(target, ast.Call) else target.span.text
        reading = f"a function of {angle.span.text}, written at {point}"
        why = "books write a frequency response at e^{j\\omega}, a function of \\omega"
        note = (
            f"{target.span.text} defines {name.span.text} as a function of {angle.span.text}: "
            f"{name.span.text} at e^{{j\\theta}} is its value at \\theta"
        )
        self._decide(DEFINITION, definition.span, reading, why, note, written=written)
        self._note(note)
        saved = scope.saved()
        binding = FunctionBinding(definition, scope)
        scope.bind(name.id, binding)
        at = self.graph.const(self.session.symbol(angle.id), SCALAR, angle.span)
        try:  # what it means at its angle is said where it is written
            super()._inline(binding, [at], definition.span, ctx)
        except BaseException:
            scope.restore(saved)
            raise

    def _response_at(
        self, binding: FunctionBinding, node: ast.Call, scope: Scope, ctx: Context
    ) -> Node:
        """``H(e^{j\\theta})`` of a response defined at e^{jω}: its value at θ."""
        definition = binding.definition
        assert isinstance(definition, Response)
        found = sequences.unit_circle(node.args[0])
        name = node.func.span.text
        at = definition.written.partition("(")[2][:-1]
        if found is None:
            raise CompileError(
                f"{name} is defined at {at}, a function of "
                f"{_shown(definition.params[0], definition.latex)}, so it is called at a point "
                "written so",
                span=node.span,
                hint=f"call it as {name}(e^{{j\\pi/2}}), or define it as a function of z, as "
                f"in {name}(z) := ..., to call it anywhere",
            )
        angle = found[1]
        why = f"{name} is defined at {at}, on line {definition.span.line}"
        self._decide(DEFINITION, node.span, f"{name} at {angle.span.text}", why)
        return self._inline(binding, [self.value(angle, scope, ctx)], node.span, ctx)

    # -- transforms at a sequence's label ----------------------------------------------------

    def _written_transform(self, written: Written, scope: Scope, ctx: Context) -> Node:
        """A transform written without its point, in the value of a sequence or a function
        whose parameter is the variable it takes to by convention, or in a sum over it, as n
        of ``x[n] := \\mathcal{Z}^{-1}\\{X\\}``: at that parameter, as the rest of the value
        is, so that x[3] is the sequence at 3."""
        if written.point is None:
            for _, taken in written.capability.variables:
                bound = scope.lookup(taken)
                if (
                    isinstance(bound, VariableBinding)
                    and self.session.globals.lookup(taken) is not bound
                ):
                    point = ast.Name(taken, span=written.span)
                    return super()._written_transform(replace(written, point=point), scope, ctx)
        return super()._written_transform(written, scope, ctx)

    # -- equations of sequences -------------------------------------------------------------

    def _unknown_samples(self, node: ast.Node, scope: Scope) -> list[str]:
        """The names an expression writes at points that nothing defines, once for each
        point, not the impulse and the step."""
        found = []
        for name, point in sequences.samples(node):
            if scope.lookup(name.id) is not None or self._has_a_meaning(name.id):
                continue
            if name.id in BRACKET_BOOKS or name.id in CONSTANTS or name.verbatim:
                continue
            if sequences.integral(point, lambda label: self._integral_name(label, scope)):
                found.append(name.id)
        return found

    def _unknown_in(self, node: ast.Node, scope: Scope) -> frozenset[str]:
        """The unknown sequences of an equation: the names it writes at points that nothing
        defines, where it writes such points twice at least, as a difference equation does,
        y[n] and y[n - 1], or y[n] and x[n]; none where it writes one, as the functional
        ``S[q] = \\int L\\, dt`` does."""
        found = self._unknown_samples(node, scope)
        return frozenset(found) if len(found) > 1 else frozenset()

    def _reading(
        self, equation: ast.Equation, scope: Scope, ctx: Context
    ) -> tuple[Literal["define", "function", "claim", "notation"], str | None, str]:
        """``y[n] = \\ldots`` defines the sequence y where nothing defines it, its value is
        written with sequences, and none of them is unknown: then it is a difference equation
        of y's own, or y's definition. With unknown sequences, it is a difference equation
        between them, a claim. ``H(e^{j\\omega}) = \\ldots`` defines H as := does."""
        found = sequences.target(equation.lhs)
        if found is not None and scope.lookup(found[0].id) is None:
            name, point = found
            if not self._has_a_meaning(name.id) and name.id not in BRACKET_BOOKS:
                written = sequences.samples(equation.rhs)
                of_sequences = any(
                    each.id == name.id or self._a_sequence(each, at, scope) for each, at in written
                )
                others = set(self._unknown_samples(equation.rhs, scope)) - {name.id}
                if of_sequences and not others and sequences.label(point) is not None:
                    return "define", None, f"{name.span.text} has no meaning yet: a sequence"
        if unknown := self._unknown_in(equation, scope):
            shown = _listed(sorted(unknown))
            return "claim", None, f"it relates sequences nothing defines, {shown}"
        response = self._response_target(equation.lhs)
        if response is not None and scope.lookup(response[0].id) is None:
            return "define", None, f"{response[0].span.text} has no meaning yet: a response"
        return super()._reading(equation, scope, ctx)

    def _Equation(self, node: ast.Equation, scope: Scope, ctx: Context) -> Node:
        unknown = self._unknown_in(node, scope)
        if not unknown:
            return super()._Equation(node, scope, ctx)
        shown = _listed([f"{name}" for name in sorted(unknown)])
        many = len(unknown) > 1
        note = (
            f"{node.span.text} is read as a difference equation in "
            f"{'the sequences' if many else 'the sequence'} {shown}, which nothing defines"
        )
        why = "it writes names nothing defines at points in square brackets"
        self._decide(SAMPLE, node.span, f"a difference equation in {shown}", why, note)
        self._note(note)
        saved = self._unknown_sequences
        self._unknown_sequences = saved | unknown
        try:
            result = super()._Equation(node, scope, ctx)
        finally:
            self._unknown_sequences = saved
        self._difference = (result, node, unknown)
        return result

    def _relate(self, equation: Node, span: Span, scope: Scope) -> None:
        """A difference equation between sequences nothing defines is the program's system,
        whose sequences' transforms its capital letters then write, as Y(z) and X(z)."""
        super()._relate(equation, span, scope)
        if self._difference is None or self._difference[0] is not equation:
            return
        _, written, unknown = self._difference
        left = [
            (name, point) for name, point in sequences.samples(written.lhs) if name.id in unknown
        ]
        if not left:
            return
        output, point = left[0]
        labelled = sequences.label(point)
        if labelled is None:
            return
        inputs = tuple(sorted(unknown - {output.id}))
        system = System(equation, output.id, inputs, labelled[0], span.line)
        self.session.conventions[SYSTEM] = system

    def _transformed(
        self, system: System, named: str, node: ast.Call, scope: Scope, ctx: Context
    ) -> Node:
        """``Y(z)``, the capital letter of a sequence of the program's difference equation,
        at a point: its z-transform, the output's the transfer function times the inputs'."""
        written = node.func.span.text
        note = (
            f"{node.span.text} is read as the z-transform of {named}, of the difference "
            f"equation on line {system.line}, its sequences at rest before they start"
        )
        why = f"{written} is the capital letter of the sequence {named}, and nothing defines it"
        self._decide(SAMPLE, node.span, f"the z-transform of {named}", why, note)
        self._note(note)
        attributes = (("name", named), ("output", system.output), ("label", system.label))
        point = self.value(node.args[0], scope, ctx)
        return self._call("seq.transform", [system.equation, point], SCALAR, node.span, attributes)

    def _solve_difference(self, node: ast.Call, scope: Scope, ctx: Context) -> Node | None:
        """``solve(y[n] = \\frac{1}{2} y[n-1] + \\delta[n], y[n])``: the member of y at its
        label, at initial rest, in closed form where SymPy finds one; ``None`` for any other
        solve."""
        if len(node.args) != 2 or not isinstance(node.args[0], ast.Equation):
            return None
        found = sequences.target(node.args[1])
        if found is None or scope.lookup(found[0].id) is not None:
            return None
        name, point = found
        equation = node.args[0]
        if self._unknown_in(equation, scope) != {name.id}:
            return None  # an input nothing defines too: no closed form
        labelled = sequences.label(point)
        if labelled is None or labelled[1] != 0:
            return None
        label = labelled[0]
        isolated = ast.Binary(
            "-", equation.lhs, equation.rhs, span=equation.lhs.span.to(equation.rhs.span)
        )
        saved = self._unknown_sequences
        self._unknown_sequences = saved | {name.id}
        try:
            difference = self.value(isolated, scope, ctx)
        finally:
            self._unknown_sequences = saved
        note = (
            f"{node.span.text} is solved at initial rest, as books take a difference "
            f"equation: {name.span.text} is 0 before its input starts"
        )
        why = f"{name.span.text}[{label}] is a sequence of a difference equation"
        self._decide(DEFINITION, node.span, "a difference equation, at initial rest", why, note)
        self._note(note)
        symbol = self._symbol_of(label)
        attributes = (("name", name.id),)
        return self._call("seq.solve", [difference, symbol], VALUE, node.span, attributes)

    # -- sums over a sequence's index -------------------------------------------------------

    def _BigOperator(self, node: ast.BigOperator, scope: Scope, ctx: Context) -> Node:
        body = sequences.with_subscripts(
            node.body, lambda name, point: self._subscripted_brackets(name, point, scope)
        )
        if body is not node.body:  # \sum_n x[n], of a list x: as \sum_n x_{n}
            node = replace(node, body=body)
        if (
            node.op == "sum"
            and node.lower is None
            and node.upper is None
            and isinstance(node.variable, ast.Name)
            and self._over_a_sequence(node, scope)
        ):  # \sum_n h[n] e^{-j\omega n}
            return self._over_every_integer(node, node.variable, scope, ctx)
        before = self._samples_lowered
        result = super()._BigOperator(node, scope, ctx)
        summed = isinstance(result.op, ops.Call) and result.op.name == "core.sum"
        if node.op == "sum" and summed and self._samples_lowered > before:
            assert isinstance(result.op, ops.Call)
            call = ops.Call("seq.sum", result.op.attributes)
            return self.graph.add(call, result.args, result.type, result.span)
        return result

    def _over_every_integer(
        self, node: ast.BigOperator, index: ast.Name, scope: Scope, ctx: Context
    ) -> Node:
        """``\\sum_n h[n] e^{-j\\omega n}``: a sum without limits whose index is the point of
        a sequence, over every integer, as signal processing writes it."""
        note = (
            f"\\sum_{{{index.span.text}}} runs over every integer, as signal processing sums "
            "over the index of a sequence"
        )
        why = f"{index.span.text} is written as the point of a sequence"
        self._decide(INDEX_SET, index.span, f"{index.span.text} over every integer", why, note)
        self._note(note)
        inner = scope.child()
        variable = self._bound_variable(index, inner)
        self._sum_indices.append(self.session.symbol(index.id))
        try:
            body = self.value(node.body, inner, ctx)
        finally:
            self._sum_indices.pop()
        ends = [self.graph.const(end, SCALAR) for end in (-sympy.oo, sympy.oo)]
        return self._call("seq.sum", [body, variable, *ends], _passed(body.type), node.span)

    def _over_a_sequence(self, node: ast.BigOperator, scope: Scope) -> bool:
        """Whether a sum's body writes a sequence at a point that its index is in."""
        index = node.variable
        assert isinstance(index, ast.Name)
        return any(
            any(isinstance(each, ast.Name) and each.id == index.id for each in _walk(point))
            and self._a_sequence(name, point, scope)
            for name, point in sequences.samples(node.body)
        )


def _in_square_brackets(node: ast.Call) -> bool:
    """Whether a call is written with square brackets, as ``u\\left[n\\right]`` is, which
    the parser reads as a call."""
    after = node.span.source.text[node.func.span.end : node.span.end].lstrip()
    return after.startswith(("[", "\\left[", "\\lbrack", "\\left\\lbrack"))
