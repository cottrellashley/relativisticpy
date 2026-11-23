r"""The lowering of number theory: the arithmetic functions by their letters, as
``\pi(n)`` and ``B_n`` write them; the sums and products over the values a condition gives
their index, as ``\sum_{d \mid n}`` and ``\prod_{p \text{ prime}}`` write them; remainders
and congruences, ``a \bmod m`` and ``a \equiv b \pmod{m}``; and the coefficients of a
series, ``[x^n] F(x)``; each as :mod:`relativisticpy.reading.arithmetic` reads it
(design/notation.md, 19, 30 and 32 to 37).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sympy
from sympy import Symbol

from relativisticpy import assumptions as applying
from relativisticpy.compiler.builtins import BUILTINS, CONSTANTS
from relativisticpy.compiler.passes import Term
from relativisticpy.compiler.scope import Context, Scope, VariableBinding
from relativisticpy.compiler.session import Session
from relativisticpy.errors import CompileError
from relativisticpy.ir import SCALAR, VALUE, EquationType, Node, TensorType, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.source import Span
from relativisticpy.lowering.objects import PythonObjects
from relativisticpy.reading import arithmetic
from relativisticpy.reading.arithmetic import IndexSet
from relativisticpy.reading.assumptions import ASSUMPTION, _constant
from relativisticpy.typing.notation import _passed

#: The decision of what values a condition under a sum or a product gives its index.
INDEX_SET = "index set"
#: The decision of what a congruence is, and of the value congruences leave a name.
CONGRUENCE = "congruence"
#: The decision of what a bracket of one power before a function is.
COEFFICIENT = "product or call"


class Arithmetic(PythonObjects):
    """The lowering of number theory."""

    def __init__(self, session: Session) -> None:
        super().__init__(session)
        # The indices of the sums and products being lowered, innermost last: integers, of
        # which a letter books give an arithmetic function is that function.
        self._indices_summed: list[Symbol] = []

    def _Call(self, node: ast.Call, scope: Scope, ctx: Context) -> Node:
        func = node.func
        if isinstance(func, ast.Name) and scope.lookup(func.id) is None:
            if func.id in CONSTANTS and (book := self._book_call(node, scope)) is not None:
                return self._capability(book, node.span, scope, ctx)  # \pi(n), of an integer n
            if func.id == "solve" and (solved := self._solve_congruences(node, scope, ctx)):
                return solved
        return super()._Call(node, scope, ctx)

    def _solve_congruences(self, node: ast.Call, scope: Scope, ctx: Context) -> Node | None:
        """``solve(x \\equiv 2 \\pmod{3}, x \\equiv 3 \\pmod{5}, x)``: the remainders that every
        solution of congruences in one unknown leaves, modulo the least common multiple of
        their moduli, by the Chinese remainder theorem; ``None`` where solve is given no
        congruence, and solves as before."""
        listed = [
            item
            for argument in node.args
            for item in (
                argument.items if isinstance(argument, ast.List | ast.Tuple) else (argument,)
            )
        ]
        found = [arithmetic.congruence(item) for item in listed]
        congruences = [each for each in found if each is not None]
        if not congruences:
            return None
        unknowns = [item for item, each in zip(listed, found, strict=True) if each is None]
        if len(unknowns) != 1 or not isinstance(unknowns[0], ast.Name):
            raise CompileError(
                "solve solves congruences in one unknown, named after them",
                span=node.span,
                hint="for example solve(x \\equiv 2 \\pmod{3}, x \\equiv 3 \\pmod{5}, x)",
            )
        why = "solve is given congruences, whose solutions are remainders"
        self._decide(CONGRUENCE, node.span, "the remainders modulo their moduli", why)
        pairs: list[Node] = []
        for each in congruences:
            difference = ast.Binary("-", each.lhs, each.rhs, span=each.lhs.span.to(each.rhs.span))
            pairs += [self.value(difference, scope, ctx), self.value(each.modulus, scope, ctx)]
        unknown = self.value(unknowns[0], scope, ctx)
        return self._call("arith.solve", [unknown, *pairs], VALUE, node.span)

    def _Name(self, node: ast.Name, scope: Scope, ctx: Context) -> Term:
        if scope.lookup(node.id) is None and (pinned := self._pinned(node)) is not None:
            return pinned  # x, of which 0 \le x < 105 and congruences are assumed
        return super()._Name(node, scope, ctx)

    # -- the coefficients of a series ------------------------------------------------------

    def _product(self, node: ast.Binary, scope: Scope, ctx: Context) -> Term:
        found = arithmetic.coefficient(node)
        if found is None:
            return super()._product(node, scope, ctx)
        power = f"{found.variable.span.text}^{{{found.power.span.text}}}"
        why = "a bracket of one power before a function is how combinatorics writes it"
        note = f"{found.written} is read as the coefficient of {power} in what follows"
        self._decide(COEFFICIENT, node.span, f"the coefficient of {power}", why, note)
        self._note(note)
        inner = scope.child()  # x is the variable of the series, whatever it means outside
        variable = self.graph.const(Symbol(found.variable.id), SCALAR, found.variable.span)
        inner.bind(found.variable.id, VariableBinding(variable))
        args = [self.value(found.of, inner, ctx), variable, self.value(found.power, scope, ctx)]
        return self._call("arith.coefficient", args, SCALAR, node.span)

    # -- congruences and remainders -------------------------------------------------------

    def _Binary(self, node: ast.Binary, scope: Scope, ctx: Context) -> Term:
        if node.op == "mod":  # a \bmod m
            return self._remainder(node.left, node.right, node.span, scope, ctx)
        return super()._Binary(node, scope, ctx)

    def _Relation(self, node: ast.Relation, scope: Scope, ctx: Context) -> Node:
        found = arithmetic.congruence(node)
        if found is None:
            return super()._Relation(node, scope, ctx)
        why = "\\equiv with a modulus on its right is a congruence, as Gauss wrote it"
        self._decide(CONGRUENCE, node.span, "a claim, decided by the remainders", why)
        sides = [
            self._remainder(side, found.modulus, node.span, scope, ctx)
            for side in (found.lhs, found.rhs)
        ]
        return self.graph.add(ops.Equation(), sides, EquationType(SCALAR), node.span)

    def _remainder(
        self, value: ast.Expr, modulus: ast.Expr, span: Span, scope: Scope, ctx: Context
    ) -> Node:
        """The remainder of ``value`` on division by ``modulus``: of a power, ``a^{k}``, by
        repeated squaring; and of ``a^{-1}`` the inverse of a modulo m, not the remainder of
        the fraction 1/a (design/notation.md, 30)."""
        divisor = self.value(modulus, scope, ctx)
        power = arithmetic.power_of(value)
        if power is None:
            return self._builtin(BUILTINS["mod"], [self.value(value, scope, ctx), divisor], span)
        base, exponent = power
        if arithmetic.negative(exponent):
            self._note(
                f"{value.span.text} modulo {modulus.span.text} is read with the inverse of "
                f"{base.span.text} modulo {modulus.span.text}, as number theory writes it, not "
                "as the remainder of a fraction"
            )
        args = [self.value(base, scope, ctx), self.value(exponent, scope, ctx), divisor]
        return self._call("arith.power_mod", args, SCALAR, span)

    def _assumed_congruence(self, expr: ast.Expr, scope: Scope) -> bool:
        """``x \\equiv 2 \\pmod{3}``, standing alone, of a name nothing defines, with a whole
        remainder and modulus: an assumption, as ``x > 0`` is (decision 2), that x is an
        integer that leaves 2 on division by 3, for the rest of the program. Congruences
        assumed so are taken together, by the Chinese remainder theorem, and one that
        contradicts them is refused. ``False`` for anything else, which is read as before."""
        found = arithmetic.congruence(expr)
        if found is None or scope is not self.session.globals:
            return False
        target = found.lhs
        if not isinstance(target, ast.Name) or target.id in CONSTANTS:
            return False
        if scope.lookup(target.id) is not None:
            return False
        remainder, modulus = _constant(found.rhs), _constant(found.modulus)
        if not (isinstance(remainder, sympy.Integer) and isinstance(modulus, sympy.Integer)):
            return False
        if modulus <= 0:
            return False
        session, name = self.session, target.id
        written = " ".join(expr.span.text.split())
        try:
            symbol = applying.symbol(name, {"integer": True}, session.assumptions.get(name))
        except ValueError as error:
            raise CompileError(
                f"{written} contradicts what was assumed before", span=expr.span, notes=[str(error)]
            ) from None
        congruence = sympy.Eq(sympy.Mod(symbol, modulus), remainder % modulus)
        assumed = {**session.assumptions, name: symbol}
        bounds = applying.in_force(session.bounds, assumed)
        together = arithmetic.residues(symbol, [*bounds, congruence])
        if together is None:
            raise CompileError(
                f"{written} contradicts what was assumed before",
                span=expr.span,
                notes=[f"no integer meets both it and {_congruences(symbol, bounds)}"],
            )
        session.assumptions[name] = symbol
        session.bounds.append((congruence, expr.span))
        left, every = together
        said = f"{target.span.text} is an integer that leaves {left} on division by {every}"
        note = f"{written} is an assumption: from here on, {said}"
        why = "a congruence of a name nothing defines, standing alone, says what the name is"
        self._decide(ASSUMPTION, expr.span, "an assumption", why, note, written=written)
        self._note(note)
        self._resolve(
            target.span, f"a symbol, assumed an integer that leaves {left} on division by {every}"
        )
        return True

    def _pinned(self, node: ast.Name) -> Node | None:
        """A name of which congruences and bounds are assumed that leave it one value, as
        ``0 \\le x < 105`` and three congruences leave the 23 of Sunzi's problem: that
        value, by the Chinese remainder theorem, said so in a note."""
        session = self.session
        symbol = session.assumptions.get(node.id)
        if symbol is None or not session.bounds:
            return None
        found = arithmetic.pinned(symbol, applying.in_force(session.bounds, session.assumptions))
        if found is None:
            return None
        value, remainder, modulus, low, high = found
        note = (
            f"{node.span.text} is {value}, the one integer from {low} to {high} that leaves "
            f"{remainder} on division by {modulus}, as the congruences assumed of it say, by "
            "the Chinese remainder theorem"
        )
        why = "the congruences and the bounds assumed of it leave it one value"
        self._decide(CONGRUENCE, node.span, f"{value}", why, note)
        self._note(note)
        self._resolve(node.span, "the one value its assumptions leave", sympy.Integer(value))
        return self.graph.const(sympy.Integer(value), SCALAR, node.span)

    # -- sums and products over a condition --------------------------------------------

    def _BigOperator(self, node: ast.BigOperator, scope: Scope, ctx: Context) -> Node:
        if node.lower is None and node.upper is None and not isinstance(node.variable, ast.Name):
            given = arithmetic.index_set(node.variable)
            if given is not None:
                return self._over(node, given, scope, ctx)
        if (every := arithmetic.every_integer(node)) is not None:  # \sum_k \binom{r}{k} ...
            return self._over(node, every, scope, ctx)
        if not isinstance(node.variable, ast.Name):
            return super()._BigOperator(node, scope, ctx)
        self._indices_summed.append(self.session.symbol(node.variable.id))
        try:
            return super()._BigOperator(node, scope, ctx)
        finally:
            self._indices_summed.pop()

    def _over(self, node: ast.BigOperator, given: IndexSet, scope: Scope, ctx: Context) -> Node:
        """``\\sum_{d \\mid n} f(d)``, ``\\prod_{p \\le x}`` or ``\\prod_{p \\text{ prime}}``:
        the sum or the product of the body over the values the condition gives its index,
        an integer, positive where it divides, and prime where it ranges over primes."""
        written = node.variable.span.text
        why = "a condition under a sum or a product gives its index its values"
        self._decide(INDEX_SET, node.variable.span, _described(given), why, written=written)
        for note in given.notes:
            self._note(note)
        facts = {"prime": True} if given.primes else {"integer": True}
        if given.of is not None:
            facts["positive"] = True
        symbol = Symbol(given.index.id, **facts)
        inner = scope.child()
        index = self.graph.const(symbol, SCALAR, given.index.span)
        inner.bind(given.index.id, VariableBinding(index))
        self._indices_summed.append(symbol)
        try:
            body = self.value(node.body, inner, ctx)
            conditions = [self._condition(each, inner, ctx) for each in given.conditions]
        finally:
            self._indices_summed.pop()
        lower = self._bound_or(given.lower, -sympy.oo, scope, ctx)
        upper = self._bound_or(given.upper, sympy.oo, scope, ctx)
        of = self._bound_or(given.of, sympy.S.Zero, scope, ctx)
        excluded = [self.value(each, scope, ctx) for each in given.excluded]
        attributes: ops.Attributes = (
            ("op", node.op),
            ("primes", given.primes),
            ("divisors", given.of is not None),
            ("strict", given.strict),
            ("excluded", len(excluded)),
        )
        args = [body, index, lower, upper, of, *excluded, *conditions]
        if isinstance(body.type, TensorType):  # component by component, as a sum is
            return self.graph.add(ops.Map("arith.over", attributes), args, body.type, node.span)
        return self._call("arith.over", args, _passed(body.type), node.span, attributes)

    def _bound_or(self, bound: ast.Expr | None, default: Any, scope: Scope, ctx: Context) -> Node:
        if bound is None:
            return self.graph.const(default, SCALAR)
        return self.value(bound, scope, ctx)

    def _condition(self, condition: ast.Expr, scope: Scope, ctx: Context) -> Node:
        """A condition each value of an index meets, as a truth value of the index: an
        equation, as gcd(k, n) = 1, holds where its sides are equal; ``d \\mid n`` where d
        divides n; ``k \\text{ odd}`` where k leaves 1 on division by 2."""
        if isinstance(condition, ast.Equation):
            sides = [self.value(side, scope, ctx) for side in (condition.lhs, condition.rhs)]
            return self._call("core.eq", sides, SCALAR, condition.span)
        if isinstance(condition, ast.Relation) and condition.op == "∣":
            sides = [self.value(side, scope, ctx) for side in (condition.lhs, condition.rhs)]
            return self._call("arith.divides", sides, SCALAR, condition.span)
        described = arithmetic.described(condition)
        if described is not None and described[1] in ("odd", "even"):
            name, word = described
            value = self.value(name, scope, ctx)
            remainder = self.graph.const(sympy.Integer(1 if word == "odd" else 0), SCALAR)
            two = self.graph.const(sympy.Integer(2), SCALAR)
            mod = self._builtin(BUILTINS["mod"], [value, two], condition.span)
            return self._call("core.eq", [mod, remainder], SCALAR, condition.span)
        return self.value(condition, scope, ctx)


def _congruences(symbol: Symbol, bounds: Sequence[Any]) -> str:
    """The congruences assumed of a symbol before, as a note says them."""
    found = [
        f"{symbol} ≡ {bound.rhs} (mod {bound.lhs.args[1]})"
        for bound in bounds
        if isinstance(bound, sympy.Eq)
        and isinstance(bound.lhs, sympy.Mod)
        and bound.lhs.args[0] == symbol
    ]
    return ", ".join(found) if found else "what was assumed of it"


def _described(given: IndexSet) -> str:
    """The values of an index set, as a decision says them."""
    index = given.index.span.text
    if given.of is not None:
        what = "prime divisors" if given.primes else "divisors"
        return f"{index} over the {what} of {given.of.span.text}"
    ends = [
        f"from {given.lower.span.text}" if given.lower is not None else "",
        f"to {given.upper.span.text}" if given.upper is not None else "",
    ]
    kind = "the primes" if given.primes else "the integers"
    shown = " ".join(end for end in ends if end)
    return f"{index} over {kind}{' ' + shown if shown else ''}"
