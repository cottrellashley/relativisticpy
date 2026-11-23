r"""The lowering of quantum notation, as :mod:`relativisticpy.reading.quantum` reads it:
kets, bras, their brackets and matrix elements, what angle brackets hold, the commutators
of matrices and operators, operators as values, and the rules a program states for
operators and kets nothing defines.

An operator is applied to its operand by writing what it is written as applied to it: a
sum to each term, a product factor by factor from the right, a power that many times and a
commutator both ways, so that ``\frac{1}{\sqrt{2}}(\mp i\hat{p} + m\omega x)`` applied to ψ
is :math:`(\mp i\hat{p}\psi + m\omega x\psi)/\sqrt{2}`, which is compiled as any expression
is. A claim between operators is decided by applying both sides to a test function.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import sympy
from sympy.physics.quantum import Operator as AbstractOperator

from relativisticpy.compiler.passes import Term
from relativisticpy.compiler.scope import Context, FunctionBinding, Operator, Scope, VariableBinding
from relativisticpy.compiler.session import Session
from relativisticpy.errors import CompileError
from relativisticpy.ir import SCALAR, VALUE, EquationType, Node, dependencies, ops
from relativisticpy.ir.types import Type
from relativisticpy.language import syntax as ast
from relativisticpy.language.parser import split_index_labels
from relativisticpy.language.source import Span
from relativisticpy.language.syntax import to_sexpr
from relativisticpy.lowering.optima import Optima
from relativisticpy.reading.names import _a_matrix, _elements, _splittable
from relativisticpy.reading.quantum import (
    OPERAND,
    QUANTUM,
    RULE,
    BracketBinding,
    Operation,
    OperatorBinding,
    ShownBinding,
    ShownFunction,
    _bindings,
    _daggered,
    _power,
    _wavefunction,
    abstract,
    bracket_kind,
    bracket_name,
    column,
    expectation_name,
    groups_of,
    inner_name,
    items_of,
    ket_name,
    name_written,
    signed,
)
from relativisticpy.tensors.indices import latex_label
from relativisticpy.typing.syntax import _operand, _product_factors, _walk

#: The forms an operator may be written in, which are read as one where they act.
_OPERATOR_FORMS = (ast.Name, ast.Accent, ast.Scripted, ast.List, ast.Tuple, ast.Binary, ast.Unary)
#: The names a test function may take, the first the statement leaves free.
_TEST_NAMES = ("f", "g", "h", "u", "w")


@dataclass(frozen=True)
class State:
    """What a label names, as a state: its ``node``; its ``kind``, a ``column``, an
    ``abstract`` ket of the rules' algebra, or a ``wavefunction`` at its ``variable``; and
    the ``binding`` of a wavefunction, to give it another point."""

    node: Node
    kind: Literal["column", "abstract", "wavefunction"]
    variable: str | None = None
    binding: FunctionBinding | None = None


class Quantum(Optima):
    """The lowering of quantum notation."""

    def __init__(self, session: Session) -> None:
        super().__init__(session)
        # The labels of a rule's kets that stand for any label, while the rule is read.
        self._patterns: frozenset[str] = frozenset()

    # -- statements -------------------------------------------------------------------------

    def _defines_quantum(self, statement: ast.Define, scope: Scope, ctx: Context) -> bool:
        """Define what quantum notation writes, if a definition is of it: a ket, a bracket,
        what angle brackets hold, or an operator; ``False`` for any other definition."""
        if len(statement.targets) != 1:
            return False
        target, value, span = statement.targets[0], statement.value, statement.span
        if isinstance(target, ast.Ket):
            self._define_ket(target, value, span, scope, ctx)
        elif isinstance(target, ast.BraKet):
            self._define_bracket(target, value, span, scope, ctx)
        elif isinstance(target, ast.Expectation):
            self._define_expectation(target, value, span, scope, ctx)
        elif (names := _operator_names(target)) and self._acts(value, scope):
            self._define_operator(target, names, value, span, scope, ctx)
        else:
            return False
        return True

    def _define_ket(
        self, target: ast.Ket, value: ast.Expr, span: Span, scope: Scope, ctx: Context
    ) -> None:
        """``|0\\rangle := \\begin{pmatrix} 1 \\\\ 0 \\end{pmatrix}``, a column, or
        ``|\\alpha\\rangle := i|1\\rangle - 2|2\\rangle``, a sum of kets."""
        node = self.value(value, scope, ctx)
        written = target.span.text
        if not (column(node) or self._is_abstract(node)):
            raise CompileError(
                f"{written} is a ket, so its value is a column or a sum of kets, not "
                f"{value.span.text}",
                span=value.span,
                hint=f"write it as a column, as in {written} := \\begin{{pmatrix}} 1 \\\\ 0 "
                "\\end{pmatrix}, or as a sum of kets, as in "
                f"{written} := \\frac{{|0\\rangle + |1\\rangle}}{{\\sqrt{{2}}}}",
            )
        what = "a column" if column(node) else "a sum of kets"
        self._decide(QUANTUM, span, f"a definition of the ket {written}", f"its value is {what}")
        scope.bind(ket_name(target.label), ShownBinding(node, written))

    def _define_bracket(
        self, target: ast.BraKet, value: ast.Expr, span: Span, scope: Scope, ctx: Context
    ) -> None:
        """``\\langle x|p\\rangle := ...``: a function of the labels its value is written
        with; of one, the position, a wavefunction, as ``\\langle x|\\alpha\\rangle``."""
        written = target.span.text
        if target.operator is not None:
            raise CompileError(
                f"{written} is a matrix element, which := does not define",
                span=target.span,
                hint="define the operator, or the kets, and write the matrix element",
            )
        params = tuple(
            dict.fromkeys(
                label.id
                for label in (target.bra, target.ket)
                if isinstance(label, ast.Name) and _mentions(label.id, value, latex=ctx.latex)
            )
        )
        name = bracket_name(target.bra, target.ket)
        definition = ast.FunctionDefinition(name, params, value, span=span, latex=ctx.latex)
        binding = BracketBinding(definition, scope, written, target.bra, target.ket)
        symbols = [self.graph.const(self.session.symbol(param), SCALAR) for param in params]
        self._inline(binding, symbols, span, ctx)  # what cannot be computed is said here
        shown = " and ".join(params) if params else "nothing"
        reading = f"a definition of the bracket {written}, a function of {shown}"
        why = "its value is written with the labels it is a function of"
        self._decide(QUANTUM, span, reading, why)
        scope.bind(name, binding)

    def _define_expectation(
        self, target: ast.Expectation, value: ast.Expr, span: Span, scope: Scope, ctx: Context
    ) -> None:
        """``\\langle x^2 \\rangle := ...``, a name of its own, or
        ``\\langle p, q \\rangle := ...``, a function of the names it holds."""
        operand, written = target.operand, target.span.text
        items = operand.items if isinstance(operand, ast.Tuple) else ()
        if len(items) > 1 and all(isinstance(item, ast.Name) for item in items):
            params = tuple(item.id for item in items if isinstance(item, ast.Name))
            name = inner_name(len(params))
            definition = ast.FunctionDefinition(name, params, value, span=span, latex=ctx.latex)
            reading = f"a definition of the bracket {written}, a function of {', '.join(params)}"
            why = "angle brackets hold names separated by commas"
            self._decide(QUANTUM, span, reading, why)
            scope.bind(name, ShownFunction(definition, scope, written))
            return
        node = self.value(value, scope, ctx)
        why = (
            "a definition's angle brackets name what it defines (decision 6 of design/notation.md)"
        )
        self._decide(QUANTUM, span, f"a definition of {written}, a name of its own", why)
        scope.bind(expectation_name(operand), ShownBinding(node, written))

    def _define_operator(
        self,
        target: ast.Expr,
        names: list[tuple[str, bool | None]],
        value: ast.Expr,
        span: Span,
        scope: Scope,
        ctx: Context,
    ) -> None:
        """``\\hat{p} := -i\\hbar \\frac{d}{dx}``: an operator, a function of its operand. With
        ``\\pm`` in its name, as ``a_\\pm``, two, one for each sign. The functions defined
        before it are read without it, as they were where they are written."""
        before = tuple(b.definition for _, b in _bindings(scope) if isinstance(b, FunctionBinding))
        for name, upper in names:
            written = value if upper is None else signed(value, upper)
            operand = ast.Name(OPERAND, verbatim=True, span=value.span)
            body = self._acting_on(written, operand, scope)
            definition = ast.FunctionDefinition(name, (OPERAND,), body, span=span, latex=ctx.latex)
            test = self._test_function(self._variables_of(written, scope), value, scope)
            self._operate(written, test, scope, ctx)  # what cannot be applied is said here
            shown = target.span.text if upper is None else latex_label(name)
            scope.bind(name, OperatorBinding(definition, scope, shown, written, before))
        shown = " and ".join(name for name, _ in names)
        reading = f"a definition of the operator{'s' if len(names) > 1 else ''} {shown}"
        why = f"{value.span.text} acts on what follows it, and has nothing to act on"
        self._decide(QUANTUM, span, reading, why, written=f"{target.span.text} := ...")

    def _states_rule(self, expr: ast.Expr, scope: Scope, ctx: Context) -> bool:
        """State a rule for what follows, if an ``=`` states one (decision 20):
        ``[\\hat{x}, \\hat{p}] = i\\hbar``, ``a_-|n\\rangle = \\sqrt{n}|n - 1\\rangle`` or
        ``\\langle m|n\\rangle = \\delta_{mn}``, of operators and kets nothing defines, which
        are read as the rules' from then on; ``False`` for anything else."""
        if not isinstance(expr, ast.Equation):
            return False
        rule = self._rule_written(expr, scope)
        if rule is None:
            return False
        lhs = _operand(expr.lhs)
        operators: list[ast.Expr] = []
        if isinstance(lhs, ast.List | ast.Tuple):
            operators = list(lhs.items)
        elif isinstance(lhs, ast.Binary):
            operators = _product_factors(lhs)[:-1]
        named = [self._abstract_operator(operator_, scope) for operator_ in operators]
        inner = scope.child()
        for label in rule.labels:
            wild = self.graph.const(sympy.Wild(label), SCALAR, expr.span)
            inner.bind(label, VariableBinding(wild))
        saved, self._patterns = self._patterns, frozenset(rule.labels)
        try:
            if isinstance(lhs, ast.List | ast.Tuple):
                sides = [self.value(item, inner, ctx) for item in lhs.items]
            elif isinstance(lhs, ast.BraKet):
                bra = self._state(lhs.bra, lhs.span, inner, ctx)
                ket = self._state(lhs.ket, lhs.span, inner, ctx)
                dual = self._call("quantum.dagger", [bra.node], VALUE, lhs.span)
                sides = [self._call("quantum.braket", [dual, ket.node], VALUE, lhs.span)]
            else:
                sides = [self.value(lhs, inner, ctx)]
            sides.append(self.value(expr.rhs, inner, ctx))
        finally:
            self._patterns = saved
        number = len(self._rules(scope)) + 1
        node = self._call("quantum.rule", sides, VALUE, expr.span, (("kind", rule.kind),))
        scope.bind(f"{RULE}{number}⟩", ShownBinding(node, expr.span.text))
        free = [*named, *rule.labels]
        what = {
            "commutator": "the commutator of operators",
            "action": "operators acting on a ket",
            "inner": "the bracket of a bra and a ket",
        }[rule.kind]
        names = _listed_names(free)
        note = (
            f"{expr.span.text} is read as a rule for what follows, not as a claim: its left "
            f"side is {what}{f' written with {names}, which nothing defines' if names else ''}"
        )
        self._note(note)
        why = f"its left side is {what} nothing defines"
        self._decide(QUANTUM, expr.span, "a rule for what follows", why, note)
        return True

    def _abstract_operator(self, written: ast.Expr, scope: Scope) -> str:
        """Read an operator a rule is stated for as an operator of the rules' algebra from now
        on, if nothing defines it yet; the name it is written as."""
        if isinstance(written, ast.Scripted) and _daggered(written):
            return self._abstract_operator(written.base, scope)
        name = name_written(written)
        assert name is not None
        if scope.lookup(name) is None:
            operator_ = self.graph.const(AbstractOperator(name), VALUE, written.span)
            scope.bind(name, ShownBinding(operator_, written.span.text))
        return written.span.text

    # -- operators ----------------------------------------------------------------------------

    def term(self, node: ast.Expr, scope: Scope, ctx: Context) -> Term:
        if isinstance(node, _OPERATOR_FORMS) and self._acts(node, scope):
            return self._operation(node, scope, ctx)
        return super().term(node, scope, ctx)

    def _operation(self, node: ast.Expr, scope: Scope, ctx: Context) -> Operation:
        """An operator written, as a value that acts on what follows it."""
        binding = self._operator_binding(node, scope)
        if binding is not None:
            line = binding.definition.span.line
            self._resolve(node.span, f"an operator, defined on line {line}")
            closure, definition = binding.closure, binding.definition
            return Operation(node.span, binding.expression, closure, definition.latex, definition)
        if isinstance(node, ast.List | ast.Tuple):
            kind = bracket_kind(node)
            items = " and ".join(item.span.text for item in node.items)
            why = (
                f"{items} are operators"
                if all(self._acts(item, scope) for item in node.items)
                else f"one of {items} is an operator"
            )
            self._decide(QUANTUM, node.span, f"their {kind}, an operator", why)
        return Operation(node.span, node, scope, ctx.latex)

    def _apply(
        self, operator_: Operator, operand: Node, span: Span, scope: Scope, ctx: Context
    ) -> Node:
        if isinstance(operator_, Operation):
            free = self._vacuous(operator_.expression, operand, operator_.scope)
            if free:  # (a \frac{d}{d\lambda})(fg), of f and g nothing defines
                raise self._acts_in_vain(operator_, free, span)
            inner = replace(ctx, latex=operator_.latex)
            if operator_.definition is None:
                return self._operate(operator_.expression, operand, operator_.scope, inner)
            inner = replace(inner, body=operator_.definition)
            with self._as_written_in(operator_.definition):
                return self._operate(operator_.expression, operand, operator_.scope, inner)
        return super()._apply(operator_, operand, span, scope, ctx)

    def _acts_in_vain(self, operator_: Operation, free: list[str], span: Span) -> CompileError:
        """The error for an operator applied to names nothing defines, which do not depend
        on what it differentiates by."""
        written = operator_.span.text
        names = _listed_names([latex_label(name) for name in free])
        variables = ", ".join(self._variables_of(operator_.expression, operator_.scope))
        verb = "does" if len(free) == 1 else "do"
        return CompileError(
            f"{written} acts on {names}, which nothing defines, so as written it gives 0",
            span=span,
            notes=[f"as written, {names} {verb} not depend on {variables}"],
            hint=f"for functions, write them with their variables, as in "
            f"{latex_label(free[0])}({variables})",
        )

    def _operate(self, expression: ast.Expr, operand: Node, scope: Scope, ctx: Context) -> Node:
        """What an operator written as ``expression``, with the names of ``scope``, makes of
        ``operand``."""
        inner = scope.child()
        inner.bind(OPERAND, VariableBinding(operand))
        written = ast.Name(OPERAND, verbatim=True, span=expression.span)
        return self.value(self._acting_on(expression, written, scope), inner, ctx)

    def _acting_on(self, expression: ast.Expr, operand: ast.Expr, scope: Scope) -> ast.Expr:
        """``expression`` applied to ``operand``, written out: a sum to each term, a product
        factor by factor from the right, a quotient's numerator, a power that many times and
        a commutator both ways; anything else multiplies it, or, if it is an operator, acts
        on it."""
        expression = _operand(expression)
        span = expression.span
        if isinstance(expression, ast.Binary) and expression.op in ("+", "-"):
            left = self._acting_on(expression.left, operand, scope)
            right = self._acting_on(expression.right, operand, scope)
            return ast.Binary(expression.op, left, right, span=span)
        if isinstance(expression, ast.Unary) and expression.op in ("-", "+"):
            inside = self._acting_on(expression.operand, operand, scope)
            return ast.Unary(expression.op, inside, span=span)
        if (
            isinstance(expression, ast.Binary)
            and expression.op == "/"
            and self._acts(expression.left, scope)
        ):  # \frac{\hat{p}^2}{2m}
            numerator = self._acting_on(expression.left, operand, scope)
            return ast.Binary("/", numerator, expression.right, span=span)
        if isinstance(expression, ast.Binary) and expression.op == "*":
            result = operand
            for factor in reversed(_product_factors(expression)):
                result = self._acting_on(factor, result, scope)
            return result
        if isinstance(expression, ast.List | ast.Tuple) and self._acts(expression, scope):
            kind = bracket_kind(expression)
            first, second = expression.items
            one = self._acting_on(first, self._acting_on(second, operand, scope), scope)
            other = self._acting_on(second, self._acting_on(first, operand, scope), scope)
            return ast.Binary("-" if kind == "commutator" else "+", one, other, span=span)
        if isinstance(expression, ast.Scripted) and self._acts(expression, scope):
            count = _power(expression)
            if count is not None:
                result = operand
                for _ in range(count):
                    result = self._acting_on(expression.base, result, scope)
                return result
        return ast.Binary("*", expression, operand, implicit=True, span=span)

    def _test_function(self, variables: list[str], written: ast.Node, scope: Scope) -> Node:
        """A function of ``variables`` nothing defines, named as none of the names the
        statement is written with, to apply operators to."""
        used = {node.id for node in _walk(written) if isinstance(node, ast.Name)}
        name = next((n for n in _TEST_NAMES if scope.lookup(n) is None and n not in used), "f")
        symbols = [self.session.symbol(variable) for variable in variables]
        value = sympy.Function(name)(*symbols) if symbols else sympy.Symbol(name)
        return self.graph.const(value, SCALAR, written.span)

    def _Equation(self, node: ast.Equation, scope: Scope, ctx: Context) -> Node:
        if self._acts(node.lhs, scope) or self._acts(node.rhs, scope):
            return self._operator_claim(node, scope, ctx)
        equation = super()._Equation(node, scope, ctx)
        if isinstance(equation.op, ops.Equation) and any(
            self._is_abstract(side) for side in equation.args
        ):  # [\hat{x}^3, \hat{p}] = 3i\hbar\hat{x}^2, by the rules
            rules = self._rules(scope)
            sides = [
                self._call("quantum.ordered", [_unreduced(side), *rules], VALUE, node.span)
                for side in equation.args
            ]
            return self.graph.add(ops.Equation(), sides, equation.type, node.span)
        return equation

    def _operator_claim(self, node: ast.Equation, scope: Scope, ctx: Context) -> Node:
        """A claim between operators, as ``[L_x, L_y] = i\\hbar L_z``: both sides applied to
        a test function, a number or a function multiplying it."""
        test = self._test_function(self._variables_of(node, scope), node, scope)
        sides = []
        for side in (node.lhs, node.rhs):
            if self._acts(side, scope):
                sides.append(self._operate(side, test, scope, ctx))
            else:
                sides.append(self._multiply([self.value(side, scope, ctx), test], side.span))
        shown = _shown(test)
        note = (
            f"{node.span.text} is a claim about operators, decided by applying both sides "
            f"to a function {shown}"
        )
        self._note(note)
        why = "a side of it is an operator, which acts on what follows it"
        self._decide(QUANTUM, node.span, f"a claim about operators, on {shown}", why, note)
        return self.graph.add(ops.Equation(), sides, EquationType(SCALAR), node.span)

    def _Accent(self, node: ast.Accent, scope: Scope, ctx: Context) -> Node:
        variable = self._position(node, scope, ctx)
        if variable is not None:  # \hat{x} beside \hat{p} := -i\hbar \frac{d}{dx}
            note = (
                f"{node.span.text} is read as the position, {variable}, which acts by "
                f"multiplying: nothing defines it, and an operator differentiates by {variable}"
            )
            self._note(note)
            why = f"nothing defines it, and an operator differentiates by {variable}"
            self._decide(QUANTUM, node.span, f"the position {variable}", why, note)
            return self.value(ast.Name(variable, span=node.span), scope, ctx)
        return super()._Accent(node, scope, ctx)

    def _Scripted(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Term:
        if self._patterns and (delta := self._delta(node, scope, ctx)) is not None:
            return delta  # \delta_{mn} in a rule, of its labels
        state = node.scripts[0] if len(node.scripts) == 1 else None
        if isinstance(node.base, ast.Expectation) and state is not None and state.kind == "_":
            return self._expectation_value(node.base, state, node.span, scope, ctx)  # _\psi
        if _daggered(node):
            if isinstance(node.base, ast.Ket):  # |\psi\rangle^\dagger, a bra
                return self._Bra(ast.Bra(node.base.label, span=node.span), scope, ctx)
            name = name_written(node.base)
            binding = scope.lookup(name) if name is not None else None
            if isinstance(binding, VariableBinding) and abstract(binding.node):
                why = f"{node.base.span.text} is an operator a rule is stated for"
                self._decide(QUANTUM, node.span, f"the adjoint of {node.base.span.text}", why)
                return self._call("quantum.dagger", [binding.node], VALUE, node.span)
        return super()._Scripted(node, scope, ctx)

    def _delta(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Node | None:
        """``\\delta_{mn}`` in a rule whose kets are labelled m and n: the Kronecker delta of
        their labels; ``None`` for anything else."""
        base = node.base
        if not (isinstance(base, ast.Name) and base.id == "delta" and len(node.scripts) == 1):
            return None
        indices = node.scripts[0].indices
        if node.scripts[0].kind != "_" or indices is None:
            return None
        labels = [str(item.label) for item in indices.items if item.label is not None]
        if len(labels) != 2 or len(indices.items) != 2 or not set(labels) <= self._patterns:
            return None
        why = f"{' and '.join(labels)} label the kets of the rule"
        self._decide(QUANTUM, node.span, "the Kronecker delta of the labels", why)
        values = [self.value(ast.Name(label, span=node.span), scope, ctx) for label in labels]
        return self._call("quantum.delta", values, SCALAR, node.span)

    # -- brackets of two items ----------------------------------------------------------------

    def _List(self, node: ast.List, scope: Scope, ctx: Context) -> Node:
        if len(node.items) == 2:
            found = self._bracket(node, "commutator", scope, ctx)
            if found is not None:
                return found
        return super()._List(node, scope, ctx)

    def _Tuple(self, node: ast.Tuple, scope: Scope, ctx: Context) -> Node:
        kind = bracket_kind(node)
        if kind is not None and len(node.items) == 2:
            found = self._bracket(node, kind, scope, ctx)
            if found is not None:
                return found
        return super()._Tuple(node, scope, ctx)

    def _bracket(
        self,
        node: ast.List | ast.Tuple,
        kind: Literal["commutator", "anticommutator"],
        scope: Scope,
        ctx: Context,
    ) -> Node | None:
        """``[A, B]`` or ``\\{A, B\\}`` of matrices, ``AB - BA`` or ``AB + BA``, or of operators
        a rule is stated for; ``None`` for anything else, which is a list or a set, and said
        to be one where no item is a number, as of ``[t, r]`` but not of ``[0, 1]``."""
        first, second = (self.value(item, scope, ctx) for item in node.items)
        a, b = (item.span.text for item in node.items)
        sign = -1 if kind == "commutator" else 1
        written = f"{a} {b} {'-' if sign < 0 else '+'} {b} {a}"
        if len(a) + len(b) > 24:  # matrices written out: AB - BA, not them twice over
            written = "AB - BA" if sign < 0 else "AB + BA"
        if _a_matrix(first) and _a_matrix(second):
            shown = f"the matrices {a} and {b}" if len(a) + len(b) <= 24 else "two matrices"
            note = f"{node.span.text} is read as the {kind} of {shown}, {written}"
            if len(node.span.text) > 40:
                note = f"brackets of {shown} are read as their {kind}, {written}"
            self._note(note)
            why = f"{a} and {b} are matrices" if len(a) + len(b) <= 24 else "both are matrices"
            self._decide(QUANTUM, node.span, f"their {kind}, {written}", why, note)
            ab = self._multiply([first, second], node.span)
            ba = self._multiply([second, first], node.span)
            return self.graph.add(ops.Add((1, sign)), [ab, ba], VALUE, node.span)
        if self._is_abstract(first) or self._is_abstract(second):
            why = f"{a} and {b} are operators a rule is stated for"
            self._decide(QUANTUM, node.span, f"their {kind}, {written}", why)
            signed_by = self.graph.const(sign, VALUE)
            value = self._call("quantum.commutator", [first, second, signed_by], VALUE, node.span)
            return self._reduced(value, scope, VALUE, node.span)
        if _a_number(first) or _a_number(second):
            return None
        read = "a list" if kind == "commutator" else "a set"
        why = f"{a} and {b} are neither operators nor matrices"
        said: str | None = None
        if node.span.text.strip() == self._text.strip():  # alone, where [A, B] may be meant
            said = f"{node.span.text} is read as {read}, as {why}"
            self._note(said)
        self._decide(QUANTUM, node.span, read, why, said)
        return None

    # -- kets and bras -----------------------------------------------------------------------

    def _Ket(self, node: ast.Ket, scope: Scope, ctx: Context) -> Node:
        return self._state(node.label, node.span, scope, ctx).node

    def _Bra(self, node: ast.Bra, scope: Scope, ctx: Context) -> Node:
        state = self._state(node.label, node.span, scope, ctx)
        return self._dual(state, node.span)

    def _state(self, label: ast.Expr, span: Span, scope: Scope, ctx: Context) -> State:
        """What a ket's label names: a ket the program defines, the product of kets it
        defines, a column, a wavefunction, or else a ket of the rules' algebra."""
        binding = scope.lookup(ket_name(label))
        if isinstance(binding, VariableBinding):
            self._resolve_binding(label.span, binding)
            node = binding.node
            return State(node, "column" if column(node) else "abstract")
        product = self._product_of_kets(label, span, scope)
        if product is not None:
            return State(product, "column")
        name = name_written(label)
        named = scope.lookup(name) if name is not None else None
        if isinstance(named, VariableBinding) and column(named.node):
            why = f"{label.span.text} is a column the program defines"
            self._decide(QUANTUM, span, f"the column {label.span.text}", why)
            return State(named.node, "column")
        wavefunction = self._wavefunction_of(label, scope)
        if wavefunction is not None:
            function, variable = wavefunction
            at = self.graph.const(self.session.symbol(variable), SCALAR, span)
            why = f"{label.span.text} is a wavefunction the program defines"
            self._decide(QUANTUM, span, f"the wavefunction {label.span.text}({variable})", why)
            return State(
                self._inline(function, [at], span, ctx), "wavefunction", variable, function
            )
        values = [self._label_value(item, scope, ctx) for item in items_of(label)]
        return State(self._call("quantum.ket", values, VALUE, span), "abstract")

    def _label_value(self, item: ast.Expr, scope: Scope, ctx: Context) -> Node:
        """The value of an item of a label: a number written with a leading zero, as 00 of
        ``|00\\rangle``, as written."""
        if isinstance(item, ast.Number) and len(item.text) > 1 and item.text.startswith("0"):
            return self.graph.const(sympy.Symbol(item.text), SCALAR, item.span)
        return self.value(item, scope, ctx)

    def _product_of_kets(self, label: ast.Expr, span: Span, scope: Scope) -> Node | None:
        """``|00\\rangle``, of kets the program defines, ``|0\\rangle \\otimes |0\\rangle``;
        ``None`` for any other label."""
        parts: list[ast.Expr]
        if isinstance(label, ast.Number) and len(label.text) > 1 and label.text.isdigit():
            parts = [ast.Number(digit, span=label.span) for digit in label.text]
        elif isinstance(label, ast.Name) and _splittable(label):
            parts = [ast.Name(letter, span=label.span) for letter in split_index_labels(label.id)]
        else:
            return None
        bound = [scope.lookup(ket_name(part)) for part in parts]
        kets = [b.node for b in bound if isinstance(b, VariableBinding) and column(b.node)]
        if len(kets) != len(parts):
            return None
        shown = " \\otimes ".join(f"|{to_sexpr(part)}\\rangle" for part in parts)
        note = f"{span.text} is read as {shown}, of the kets the program defines"
        self._note(note)
        self._decide(
            QUANTUM, span, shown, "each of its labels names a ket the program defines", note
        )
        value = kets[0]
        for following in kets[1:]:
            value = self._of_a_matrix("kron", [value, following], span)
        return value

    def _dual(self, state: State, span: Span) -> Node:
        """The bra of a state: the conjugate transpose of a column, the adjoint of a ket of
        the rules' algebra, the conjugate of a wavefunction."""
        if state.kind == "column":
            return self._of_a_matrix("ctranspose", [state.node], span)
        if state.kind == "abstract":
            return self._call("quantum.dagger", [state.node], VALUE, span)
        return self._conjugated(state.node, [state.variable or ""], span)

    def _conjugated(self, value: Node, real: list[str], span: Span) -> Node:
        """The conjugate of a value, with the symbols named ``real`` taken to be real."""
        symbols = [self.graph.const(self.session.symbol(name), SCALAR) for name in real if name]
        return self._call("quantum.conjugate", [value, *symbols], value.type, span)

    # -- brackets ----------------------------------------------------------------------------

    def _BraKet(self, node: ast.BraKet, scope: Scope, ctx: Context) -> Node:
        if node.operator is None:
            defined = self._bracket_definition(node.bra, node.ket, scope)
            if defined is not None:
                return self._defined_bracket(node, *defined, scope=scope, ctx=ctx)
        coefficient = self._clebsch_gordan(node, scope, ctx)
        if coefficient is not None:
            return coefficient
        self._check_presentation(node)
        at = self._at_a_point(node, scope, ctx)
        if at is not None:
            return at
        bra = self._state(node.bra, node.span, scope, ctx)
        ket = self._state(node.ket, node.span, scope, ctx)
        if bra.kind == ket.kind == "column":
            return self._matrix_element(bra, node.operator, ket, node.span, scope, ctx)
        if bra.kind == ket.kind == "wavefunction":
            return self._overlap(bra, node, ket, scope, ctx)
        if bra.kind == ket.kind == "abstract":
            return self._abstract_element(bra, node.operator, ket, node.span, scope, ctx)
        kinds = {
            "column": "a column",
            "abstract": "a ket nothing defines",
            "wavefunction": "a wavefunction",
        }
        raise CompileError(
            f"{node.span.text} brings together {kinds[bra.kind]}, |{node.bra.span.text}\\rangle, "
            f"and {kinds[ket.kind]}, |{node.ket.span.text}\\rangle",
            span=node.span,
            notes=["a bracket is of two columns, two wavefunctions, or two kets of rules"],
        )

    def _defined_bracket(
        self,
        node: ast.BraKet,
        binding: BracketBinding,
        labels: list[ast.Expr],
        swapped: bool,
        *,
        scope: Scope,
        ctx: Context,
    ) -> Node:
        """A bracket the program defines, at the labels written; written the other way round,
        its conjugate, with the labels real, as eigenvalues are."""
        args = [self.value(label, scope, ctx) for label in labels]
        value = self._inline(binding, args, node.span, ctx)
        line = binding.definition.span.line
        if not swapped:
            self._resolve(node.span, f"a bracket, defined on line {line}")
            return value
        defined = f"\\langle {binding.bra.span.text}|{binding.ket.span.text}\\rangle"
        reals = [label.span.text for label in labels]
        note = (
            f"{node.span.text} is read as the conjugate of {defined}, defined on line {line}, "
            f"with {' and '.join(reals)} real"
        )
        self._note(note)
        why = f"{defined} is defined, and the bracket the other way round is its conjugate"
        self._decide(QUANTUM, node.span, f"the conjugate of {defined}", why, note)
        symbols = [arg for arg in args if isinstance(arg.op, ops.Const)]
        return self._call("quantum.conjugate", [value, *symbols], value.type, node.span)

    def _clebsch_gordan(self, node: ast.BraKet, scope: Scope, ctx: Context) -> Node | None:
        """``\\langle j_1, j_2; m_1, m_2|j_1, j_2; j, m\\rangle``, as Sakurai writes the overlap
        of an uncoupled and a coupled state: a Clebsch–Gordan coefficient; ``None`` for any
        other bracket."""
        bra, ket = groups_of(node.bra), groups_of(node.ket)
        if node.operator is not None or bra is None or ket is None:
            return None
        if not (len(bra) == len(ket) == 2 and all(len(group) == 2 for group in [*bra, *ket])):
            return None
        if [to_sexpr(item) for item in bra[0]] != [to_sexpr(item) for item in ket[0]]:
            return None
        (j1, j2), (m1, m2) = bra
        _, (j, m) = ket
        note = (
            f"{node.span.text} is read as a Clebsch–Gordan coefficient, "
            "⟨j₁ j₂; m₁ m₂|j₁ j₂; j m⟩, as Sakurai writes it"
        )
        self._note(note)
        why = "its bra and its ket share j₁ and j₂, and give m₁, m₂ and j, m"
        self._decide(QUANTUM, node.span, "a Clebsch–Gordan coefficient", why, note)
        values = [self.value(item, scope, ctx) for item in (j1, m1, j2, m2, j, m)]
        return self._call("quantum.clebsch_gordan", values, SCALAR, node.span)

    def _check_presentation(self, node: ast.BraKet) -> None:
        """Refuse ``\\langle a, b \\mid a b a^{-1} b\\rangle``, a group's presentation by its
        generators and relations, which no design reads yet."""
        between = node.span.source.text[node.bra.span.end : node.ket.span.start]
        if "\\mid" in between and isinstance(node.bra, ast.Tuple):
            raise CompileError(
                f"{node.span.text} is read as the presentation of a group, by its generators "
                "and relations, which is not supported",
                span=node.span,
                notes=["a group given by generators and relations is no value yet"],
            )

    def _at_a_point(self, node: ast.BraKet, scope: Scope, ctx: Context) -> Node | None:
        """``\\langle x|\\psi\\rangle``, of a wavefunction ψ the program defines and a label that
        names no state: ψ at it; the other way round, its conjugate. ``None`` otherwise."""
        if node.operator is not None:
            return None
        for point, label, conjugated in ((node.bra, node.ket, False), (node.ket, node.bra, True)):
            wavefunction = self._wavefunction_of(label, scope)
            if wavefunction is None or not isinstance(point, ast.Name):
                continue
            if self._names_a_state(point, scope):
                continue
            function, _ = wavefunction
            at = self.value(point, scope, ctx)
            value = self._inline(function, [at], node.span, ctx)
            reading = f"{label.span.text}({point.span.text})"
            if conjugated:
                reading = f"the conjugate of {reading}"
                value = self._call("quantum.conjugate", [value, at], value.type, node.span)
            why = f"{label.span.text} is a wavefunction, and {point.span.text} names no state"
            self._decide(QUANTUM, node.span, reading, why)
            return value
        return None

    def _matrix_element(
        self,
        bra: State,
        operator_: ast.Expr | None,
        ket: State,
        span: Span,
        scope: Scope,
        ctx: Context,
    ) -> Node:
        """``\\langle\\phi|A|\\psi\\rangle`` of columns: the row, the matrix and the column,
        multiplied, the number in the 1 × 1 matrix they make."""
        factors = [self._dual(bra, span)]
        if operator_ is not None:
            matrix = self.value(operator_, scope, ctx)
            if not (_a_matrix(matrix) or matrix.type == SCALAR):
                raise CompileError(
                    f"{operator_.span.text}, between a bra and a ket that are columns, is no "
                    "matrix",
                    span=operator_.span,
                    hint="define it as a matrix, as "
                    "\\begin{pmatrix} 1 & 0 \\\\ 0 & -1 \\end{pmatrix}",
                )
            factors.append(matrix)
        factors.append(ket.node)
        self._decide(QUANTUM, span, "a row times a matrix times a column", "its states are columns")
        product = self._multiply(factors, span)
        zero = self.graph.const(0, VALUE)
        return self._call("linalg.entry", [product, zero, zero], SCALAR, span)

    def _overlap(
        self, bra: State, node: ast.BraKet, ket: State, scope: Scope, ctx: Context
    ) -> Node:
        """``\\langle\\phi|A|\\psi\\rangle`` of wavefunctions: :math:`\\int \\phi^* A \\psi\\, dx`
        over the domain of x."""
        variable = ket.variable or "x"
        assert bra.binding is not None
        at = self.graph.const(self.session.symbol(variable), SCALAR, node.span)
        phi = self._inline(bra.binding, [at], node.span, ctx)
        applied = self._acted(node.operator, ket.node, scope, ctx)
        integrand = self._multiply(
            [self._conjugated(phi, [variable], node.span), applied], node.span
        )
        return self._integral(integrand, variable, node, "their overlap")

    def _acted(self, operator_: ast.Expr | None, operand: Node, scope: Scope, ctx: Context) -> Node:
        """What an operator makes of a wavefunction: applied, if it acts, or multiplying it."""
        if operator_ is None:
            return operand
        if self._acts(operator_, scope):
            return self._operate(operator_, operand, scope, ctx)
        return self._multiply([self.value(operator_, scope, ctx), operand], operator_.span)

    def _integral(self, integrand: Node, variable: str, node: ast.Expr, what: str) -> Node:
        """The integral of ``integrand`` over the domain of ``variable``, said in a note."""
        lower, upper = self._domain(variable)
        symbol = self.session.symbol(variable)
        shown = f"({sympy.latex(lower)}, {sympy.latex(upper)})"
        note = f"{node.span.text} is read as {what}, an integral over {variable} from {shown}"
        self._note(note)
        why = f"its states are wavefunctions of {variable}"
        self._decide(QUANTUM, node.span, f"{what}, an integral over {variable}", why, note)
        ends = [self.graph.const(end, SCALAR) for end in (symbol, lower, upper)]
        limits = self._call("core.limits", ends, VALUE, node.span)
        return self._call("integrate", [integrand, limits], SCALAR, node.span)

    def _abstract_element(
        self,
        bra: State,
        operator_: ast.Expr | None,
        ket: State,
        span: Span,
        scope: Scope,
        ctx: Context,
    ) -> Node:
        """``\\langle m|A|n\\rangle`` of kets nothing defines, reduced by the rules."""
        factors = [self._dual(bra, span)]
        if operator_ is not None:
            value = self.value(operator_, scope, ctx)
            if not self._is_abstract(value) and not _a_number(value):
                raise self._no_operator(operator_)
            factors.append(value)
        factors.append(ket.node)
        why = "nothing defines its kets, which the rules stated so far reduce"
        self._decide(QUANTUM, span, "a bracket of the rules' algebra", why)
        product = self._call("quantum.braket", factors, VALUE, span)
        return self._reduced(product, scope, SCALAR, span)

    def _no_operator(self, operator_: ast.Expr) -> CompileError:
        """The error for what stands between a bra and a ket and is no operator."""
        written = operator_.span.text
        if written.lstrip().startswith(("T\\{", "T\\left\\{", "T(", "\\mathcal{T}")):
            return CompileError(
                f"{written}, between a bra and a ket, is read as a time-ordered product of "
                "fields, which is not supported yet",
                span=operator_.span,
                notes=[
                    "Wick's theorem, and the fields of quantum field theory, have no design yet"
                ],
            )
        return CompileError(
            f"{written}, between a bra and a ket, is read as an operator, but nothing defines "
            "it as one",
            span=operator_.span,
            hint="define it, as \\hat{p} := -i\\hbar \\frac{d}{dx}, or state how it acts on "
            "kets, as a|n\\rangle = \\sqrt{n}|n - 1\\rangle",
        )

    def _Abs(self, node: ast.Abs, scope: Scope, ctx: Context) -> Node:
        """``|\\langle k|H'|0\\rangle|`` of kets nothing defines: the root of the bracket times
        its conjugate, its labels real, so that a sum over a label is taken; any other bars
        are what they always are."""
        if not isinstance(node.operand, ast.BraKet):
            return super()._Abs(node, scope, ctx)
        value = self.value(node.operand, scope, ctx)
        if not self._is_abstract(value):
            return self._absolute(node, value)
        items = (*items_of(node.operand.bra), *items_of(node.operand.ket))
        labels = [label for item in items if _a_symbol(label := self.value(item, scope, ctx))]
        why = "its kets are of the rules' algebra, whose labels are real"
        reading = f"the root of {node.operand.span.text} times its conjugate"
        self._decide(QUANTUM, node.span, reading, why)
        return self._call("quantum.modulus", [value, *labels], SCALAR, node.span)

    def _reduced(self, value: Node, scope: Scope, kind: Type, span: Span) -> Node:
        """A value of the rules' algebra, reduced by the rules stated so far."""
        if isinstance(value.op, ops.Call) and value.op.name == "quantum.reduced":
            return value
        return self._call("quantum.reduced", [value, *self._rules(scope)], kind, span)

    def _is_abstract(self, value: Node) -> bool:
        """Whether a value is of the rules' algebra: written with its kets or operators."""
        names = ("quantum.ket", "quantum.dagger", "quantum.commutator", "quantum.braket")
        for node in dependencies([value]):
            if isinstance(node.op, ops.Call) and node.op.name in names:
                return True
            if abstract(node):
                return True
        return False

    # -- angle brackets --------------------------------------------------------------------

    def _Expectation(self, node: ast.Expectation, scope: Scope, ctx: Context) -> Node:
        operand = node.operand
        binding = scope.lookup(expectation_name(operand))
        if isinstance(binding, VariableBinding):
            self._resolve_binding(node.span, binding)
            return binding.node
        items = list(operand.items) if isinstance(operand, ast.Tuple) else [operand]
        if len(items) > 1:
            return self._bracket_of_items(node, items, scope, ctx)
        return self._expectation_value(node, None, node.span, scope, ctx)

    def _bracket_of_items(
        self, node: ast.Expectation, items: list[ast.Expr], scope: Scope, ctx: Context
    ) -> Node:
        """``\\langle u, v\\rangle``: a bracket of items the program defines, called with
        them, or the inner product of two vectors."""
        defined = scope.lookup(inner_name(len(items)))
        written = ", ".join(item.span.text for item in items)
        if isinstance(defined, FunctionBinding):
            line = defined.definition.span.line
            why = f"the program defines the bracket of {len(items)} items on line {line}"
            self._decide(QUANTUM, node.span, f"a call of the bracket ⟨{written}⟩", why)
            args = [self.value(item, scope, ctx) for item in items]
            return self._inline(self._of_functions(defined, args), args, node.span, ctx)
        values = [self.value(item, scope, ctx) for item in items]
        if len(values) == 2 and all(_elements(v) is not None or column(v) for v in values):
            why = "it holds two vectors"
            reading = "their inner product, Σ uᵢ v̄ᵢ, linear in the first"
            self._decide(QUANTUM, node.span, reading, why)
            return self._call("quantum.inner", values, SCALAR, node.span)
        raise CompileError(
            f"{node.span.text} is angle brackets around {written}, which are not two vectors, "
            "and no bracket the program defines",
            span=node.span,
            notes=[
                "angle brackets around two vectors, as lists or columns, are their inner product"
            ],
            hint="define the bracket, as in "
            "\\langle p, q \\rangle := \\int_{-1}^{1} p(x) q(x) \\, dx",
        )

    def _of_functions(self, binding: FunctionBinding, args: list[Node]) -> FunctionBinding:
        """A bracket the program defines, whose body calls its parameters, as
        ``\\langle p, q \\rangle := \\int p(x) q(x) \\, dx`` does: each parameter it calls a
        function, whose value at its variable is the value it is given, so that
        ``\\langle 1, x^2 \\rangle`` is :math:`\\int 1 \\cdot x^2 \\, dx`, not 1 times x."""
        definition = binding.definition
        body = definition.body
        if not isinstance(body, ast.Expr):
            return binding
        closure = binding.closure.child()
        called = False
        for param, arg in zip(definition.params, args, strict=True):
            calls = [
                call
                for call in _walk(body)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == param
                and all(isinstance(a, ast.Name) for a in call.args)
            ]
            if not calls:
                closure.bind(param, VariableBinding(arg))
                continue
            called = True
            variables = tuple(a.id for a in calls[0].args if isinstance(a, ast.Name))
            hidden = f"·{param}"
            closure.bind(hidden, VariableBinding(arg))
            value = ast.Name(hidden, verbatim=True, span=calls[0].span)
            function = ast.FunctionDefinition(
                param, variables, value, span=calls[0].span, latex=definition.latex
            )
            closure.bind(param, FunctionBinding(function, closure))
        if not called:
            return binding
        renamed = replace(definition, params=tuple(f"·{p}·" for p in definition.params))
        return FunctionBinding(renamed, closure)

    def _expectation_value(
        self,
        node: ast.Expectation,
        named: ast.Script | None,
        span: Span,
        scope: Scope,
        ctx: Context,
    ) -> Node:
        """``\\langle A \\rangle``: the expectation value of A in the state a subscript names,
        or in the one state the program defines."""
        operand, written = node.operand, node.span.text
        states = self._states(scope)
        if named is not None:
            label = named.body if named.body is not None else ast.Name("", span=named.span)
            chosen = [
                (name, binding)
                for name, binding in states
                if name in (name_written(label), ket_name(label))
                or (
                    isinstance(binding, BracketBinding) and to_sexpr(binding.ket) == to_sexpr(label)
                )
            ]
            if not chosen:
                raise CompileError(
                    f"{span.text}: {named.span.text.lstrip('_')} names no state the program "
                    "defines",
                    span=named.span,
                    hint="define it, as \\psi(x) := ... or |\\psi\\rangle := ...",
                )
            states = chosen[-1:]
        if len(states) != 1:
            shown = ", ".join(_state_shown(name, binding) for name, binding in states)
            raise CompileError(
                f"{written} is an expectation value, but "
                + ("the program defines no state" if not states else f"in which state: {shown}?"),
                span=node.span,
                notes=["an expectation value is taken in a state, a wavefunction or a ket"],
                hint=f"name the state, as in {written}_\\psi, or define {written} itself, as in "
                f"{written} := \\int x |\\psi(x)|^2 \\, dx",
            )
        name, binding = states[0]
        where = _state_shown(name, binding)
        if isinstance(binding, FunctionBinding):
            variable = (
                _wavefunction(binding)
                if isinstance(binding, BracketBinding)
                else binding.definition.params[0]
            ) or "x"
            at = self.graph.const(self.session.symbol(variable), SCALAR, span)
            psi = self._inline(binding, [at], span, ctx)
            applied = self._acted(operand, psi, scope, ctx)
            integrand = self._multiply([self._conjugated(psi, [variable], span), applied], span)
            return self._integral(
                integrand,
                variable,
                node,
                f"the expectation value of {operand.span.text} in {where}",
            )
        assert isinstance(binding, VariableBinding)
        state = State(binding.node, "column" if column(binding.node) else "abstract")
        note = f"{written} is read as the expectation value of {operand.span.text} in {where}"
        self._note(note)
        why = f"{where} is the state the program defines"
        self._decide(QUANTUM, span, f"the expectation value in {where}", why, note)
        if state.kind == "column":
            return self._matrix_element(state, operand, state, span, scope, ctx)
        return self._abstract_element(state, operand, state, span, scope, ctx)


def _operator_names(target: ast.Expr) -> list[tuple[str, bool | None]]:
    """The names an operator's definition gives, each with the sign ``\\pm`` takes in its
    value, above (True) or below: ``a_+`` above and ``a_-`` below of ``a_\\pm``; none for a
    target that is no name."""
    if isinstance(target, ast.Scripted) and isinstance(target.base, ast.Name):
        script = target.scripts[-1]
        mark = (
            script.body.id if isinstance(script.body, ast.Name) and script.body.verbatim else None
        )
        if len(target.scripts) == 1 and script.kind == "_" and mark in ("±", "∓"):
            base = target.base.id
            signs = ("+", "-") if mark == "±" else ("-", "+")
            return [(f"{base}_{signs[0]}", True), (f"{base}_{signs[1]}", False)]
    if isinstance(target, ast.Accent) and target.accent in ("dot", "ddot"):
        return []
    name = name_written(target)
    return [(name, None)] if name is not None else []


def _mentions(name: str, value: ast.Expr, *, latex: bool) -> bool:
    """Whether ``value`` is written with ``name``, in LaTeX also as a letter of a run of
    them, as p is in ``e^{ipx/\\hbar}``."""
    for node in _walk(value):
        if isinstance(node, ast.Name):
            if node.id == name:
                return True
            if latex and _splittable(node) and name in split_index_labels(node.id):
                return True
    return False


def _unreduced(value: Node) -> Node:
    """A value of the rules' algebra before the rules reduced it, to reduce again."""
    if isinstance(value.op, ops.Call) and value.op.name == "quantum.reduced":
        return value.args[0]
    return value


def _a_symbol(value: Node) -> bool:
    """Whether a value is a symbol, as the variable a function is called with."""
    return isinstance(value.op, ops.Const) and isinstance(value.op.value, sympy.Symbol)


def _a_number(value: Node) -> bool:
    """Whether a value is a number, known as one where it is written."""
    return (
        value.type == SCALAR
        and isinstance(value.op, ops.Const)
        and sympy.sympify(value.op.value).is_number
    )


def _listed_names(names: list[str]) -> str:
    """Names in a sentence: ``a``, ``a and b``, ``a, b and c``."""
    names = list(dict.fromkeys(names))
    if len(names) < 2:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _shown(test: Node) -> str:
    """A test function as a note shows it: ``f(x, y, z)``."""
    value = test.op.value if isinstance(test.op, ops.Const) else test
    return str(value)


def _state_shown(name: str, binding: object) -> str:
    """A state as a message shows it: as it is written, as ``|+\\rangle``, or a function by
    its name in LaTeX, as ``\\psi_{1}``."""
    if isinstance(binding, BracketBinding):  # \langle x|\alpha\rangle := ..., of |\alpha\rangle
        return f"|{binding.ket.span.text}\\rangle"
    shown = getattr(binding, "shown", None)
    return str(shown) if shown is not None else latex_label(name)
