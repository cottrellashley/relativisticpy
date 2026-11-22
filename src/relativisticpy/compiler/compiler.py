r"""The compiler: from syntax tree to IR, a statement at a time.

The compiler reads a statement in three passes, each a package of its own, over one walk
of its syntax (see :mod:`relativisticpy.compiler.passes`):

* the reader (:mod:`relativisticpy.reading`) decides what each name and each ambiguous
  form means: ``x^{a}`` is a component of a tensor ``x`` and a power of anything else;
  ``=`` defines what has no meaning yet and claims the rest; in LaTeX an unknown run of
  letters is a product, as in ``\frac{2GM}{r}``;
* the typer (:mod:`relativisticpy.typing`) holds the rules of index notation and spaces:
  the free indices of a sum agree, an index appears at most twice, a summed index once
  up and once down, and each axis ranges over a space;
* the lowerer (:mod:`relativisticpy.lowering`) emits the IR, computing nothing: raising
  an index, a covariant derivative or a trace become contractions with the metric or the
  connection, and ``G_{\mu\nu} = 8\pi G T_{\mu\nu}``, before T has components, the
  operations of index notation that the engines compute.

The :class:`Compiler` is the three together, and orchestrates statements: what each
shows, what it defines, the notes on how it was read and the checks that run as it does.
"""

from __future__ import annotations

from dataclasses import replace

from relativisticpy.compiler.scope import Check, Context, FunctionBinding, Output, Scope
from relativisticpy.compiler.session import Saved
from relativisticpy.errors import CompileError, RelativisticPyError
from relativisticpy.ir import SCALAR, ops
from relativisticpy.language import syntax as ast
from relativisticpy.lowering.arithmetic import Arithmetic
from relativisticpy.reading import assumptions

# The tables of the language, which its reference reads from here.
from relativisticpy.reading.names import _BUILTIN_TENSORS as _BUILTIN_TENSORS
from relativisticpy.reading.names import _LINEAR_ALGEBRA as _LINEAR_ALGEBRA
from relativisticpy.reading.names import _STATISTICS as _STATISTICS
from relativisticpy.reading.names import (
    _bracketed,
    _factor_names,
    _is_line_element,
    _operator_form,
    _script_labels,
    _unbracketed,
    _with_arguments,
)
from relativisticpy.reading.readings import Reading
from relativisticpy.typing.syntax import _NOTATION_OPERATIONS as _NOTATION_OPERATIONS
from relativisticpy.typing.syntax import _zero_literal

#: The decision of whether = defines or claims.
DEFINITION = "definition or claim"
#: How each reading of = is said.
_READ_AS = {
    "define": "a definition",
    "function": "a definition of a function",
    "claim": "a claim",
    "notation": "a claim, in index notation",
}


class Compiler(Arithmetic):
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
            self._decided, self._resolved = {}, {}
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

    def read(self, statement: ast.Statement) -> Reading:
        """How a statement is read, compiled as :meth:`statement` compiles it, in the
        session, computing nothing: how each name was resolved, each decision taken, with
        why, the notes, and the error the statement gives, if it gives one."""
        reading = Reading(statement)
        try:
            self.statement(statement, self.session.globals)
        except RelativisticPyError as error:
            reading.error = error
        reading.names, reading.decisions = list(self._resolved), list(self._decided)
        noted = [decision.note for decision in self._decided if decision.note is not None]
        reading.notes = list(dict.fromkeys([*noted, *self._notes]))
        return reading

    def _statement(
        self, statement: ast.Statement, scope: Scope, outer: Context | None
    ) -> list[Output]:
        ctx = Context(latex=statement.latex, calls=outer.calls if outer else ())
        if isinstance(statement, ast.FunctionDefinition):
            why = "it is written with :="
            written = f"{statement.name}({', '.join(statement.params)}) := ..."
            reading = "a definition of a function"
            self._decide(DEFINITION, statement.span, reading, why, written=written)
            if (head := _operator_form(statement.span.text)) is not None:  # \operatorname{sinc} x
                why = "a name written as an operator, with one name after it, is a function of it"
                call = f"{statement.name}({', '.join(statement.params)})"
                self._decide("product or call", statement.span, call, why, written=head)
            self._define_function(statement, scope, ctx)
            return []
        if isinstance(statement, ast.Define) and self._defines_transform(statement, scope, ctx):
            return []  # \mathcal{F}\{f\}(\omega) := \int ...: the convention of a transform
        if isinstance(statement, ast.Define) and self._defines_quantum(statement, scope, ctx):
            return []  # |0\rangle := ..., \langle x|p\rangle := ..., \hat{p} := -i\hbar d/dx
        if isinstance(statement, ast.Define) and self._states_identity(statement, scope):
            target = statement.targets[0]
            why = "it says, with \\equiv, what a built-in already is"
            written = f"{target.span.text} \\equiv ..."
            self._decide(DEFINITION, statement.span, "a claim", why, written=written)
            equation = ast.Equation(target, statement.value, span=statement.span)
            statement = ast.Evaluate(equation, span=statement.span, latex=statement.latex)
        if isinstance(statement, ast.Define):
            written = f"{', '.join(t.span.text for t in statement.targets)} := ..."
            why = "it is written with :="
            self._decide(DEFINITION, statement.span, "a definition", why, written=written)
            if len(statement.targets) > 1:
                self._destructure(statement.targets, statement.value, scope, ctx)
            else:
                self._check_recurrence(statement.targets[0], statement.value, statement.span)
                with self._over_ranges(statement):  # x_n := n h, \quad n = 0, \ldots, N-1
                    self._define(statement.targets[0], statement.value, scope, ctx)
            return []
        assert isinstance(statement, ast.Evaluate)
        expr = statement.expr
        if isinstance(expr, ast.Relation) and self._declares(expr):
            return []
        if self._states_rule(expr, scope, ctx):  # [\hat{x}, \hat{p}] = i\hbar, for what follows
            return []
        if assumptions.assume(self, expr, scope, ctx):  # a > 0, x \in \mathbb{R}
            return []
        if self._assumed_congruence(expr, scope):  # x \equiv 2 \pmod{3}
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
        self._check_recurrence(lhs, rhs, equation.span)  # T_n = 2 T_{n-1} + 1
        applied = _with_arguments(lhs)
        if (
            applied is not None
            and self._names_nothing(applied[0], scope)
            and self._read_as_product(lhs, scope, ctx)
        ):  # U_{\text{eff}}(r) = ...: a function, named with scripts
            if self._define_family(lhs, rhs, scope, ctx):
                return []
            if self._define_scripted_function(lhs, rhs, scope, ctx):
                return []
            raise self._cannot_define(lhs, rhs, equals=True)
        reading, note, why = self._reading(equation, scope, ctx)
        if reading == "function":
            assert isinstance(lhs, ast.Call)
            assert isinstance(lhs.func, ast.Name)
            params = tuple(arg.id for arg in lhs.args if isinstance(arg, ast.Name))
            definition = ast.FunctionDefinition(
                lhs.func.id, params, rhs, span=equation.span, latex=ctx.latex
            )
            failure = self._fails_as_function(definition, scope, ctx)
            if failure is None:
                read_as, written = "a definition of a function", f"{lhs.span.text} = ..."
                self._decide(DEFINITION, equation.span, read_as, why, written=written)
                self._define_function(definition, scope, ctx)
                return []
            # Not a definition: its value means nothing as a function. As an equation,
            # what fails says why.
            self._note(
                f"{lhs.span.text} = ... is read as an equation, not as the definition of a "
                f"function: as one, its value cannot be computed ({failure.message})"
            )
            note = None
            why = "as a function, its value cannot be computed"
            reading = "claim"
        written = f"{lhs.span.text} = ..."
        self._decide(DEFINITION, equation.span, _READ_AS[reading], why, note, written=written)
        if reading == "define":
            if self._is_notation(rhs, scope, ctx, equation=True):  # nothing to define it
                return [Output(self._notation(equation, scope, ctx), equation.span)]
            self._define(lhs, rhs, scope, ctx)
            return []
        if reading == "notation" or self._is_notation(equation, scope, ctx, equation=True):
            return [Output(self._notation(equation, scope, ctx), equation.span, note=note)]
        node = self.value(equation, scope, ctx)
        self._relate(node, equation.span, scope)  # x^3 + y^3 = 6xy ties y to x
        return [Output(node, equation.span, note=note)]

    def _read_as_product(self, lhs: ast.Expr, scope: Scope, ctx: Context) -> bool:
        """Whether the left side of an equation is read as a product, as
        ``U_{\\text{eff}}(r)`` is, U_eff times r, rather than as a call: compiled in a session
        of its own, which leaves this one as it was."""
        checker = Compiler(self.session.fork())
        checker._depth = 1  # its notes are not this statement's
        try:
            node = checker.value(lhs, scope, ctx)
        except RelativisticPyError:
            return False
        return isinstance(node.op, ops.Mul)

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
