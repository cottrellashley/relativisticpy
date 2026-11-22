r"""Quantum mechanics without a representation: kets and bras named by their labels,
operators nothing defines, and the rules a program states for them, as
``[\hat{x}, \hat{p}] = i\hbar``, ``a_-|n\rangle = \sqrt{n}|n - 1\rangle`` and
``\langle m|n\rangle = \delta_{mn}`` (design/notation.md, decision 20).

Values are those of :mod:`sympy.physics.quantum`. A value is reduced by the rules until none
applies: an operator acts on the ket after it, as a rule says it does; a bra meets a ket in
their inner product, which a rule gives; and operators are put in the order their
commutators give, as ``\hat{p}\hat{x}`` is ``\hat{x}\hat{p} - i\hbar``. What no rule reduces is
kept as it is. The compiler reads the notation (:mod:`relativisticpy.reading.quantum`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import sympy
from sympy.physics.quantum import Commutator, Dagger, InnerProduct, Ket, Operator
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.state import KetBase

from relativisticpy import assumptions, display

#: The most rewrites a value is given, past which what is left is kept as it is.
_MOST = 2000


@dataclass(frozen=True)
class Rule:
    """A rule a program states: its ``kind``, ``commutator``, ``action`` or ``inner``; its
    left side, ``lhs``, as the operators of a commutator, the operators and the ket they act
    on, or the bracket; and its right side, ``rhs``. Wild symbols in it stand for any label."""

    kind: str
    lhs: tuple[Any, ...]
    rhs: Any


def rule(*sides: Any, kind: str) -> Rule:
    """A rule of ``kind`` with its left side's parts and its right side, the last of
    ``sides``."""
    *lhs, rhs = sides
    return Rule(kind, tuple(lhs), sympy.sympify(rhs))


def ket(*labels: Any) -> Any:
    """The ket of ``labels``, ``|n\\rangle`` or ``|j, m\\rangle``."""
    return Ket(*labels)


def dagger(value: Any) -> Any:
    """The adjoint: the bra of a ket, and the adjoint of an operator, with the conjugates
    of their coefficients."""
    return Dagger(value)


def braket(*factors: Any) -> Any:
    """A bra, the operators after it and a ket, multiplied in order."""
    return sympy.Mul(*factors)


def commutator(left: Any, right: Any, sign: int) -> Any:
    """``AB - BA``, or with ``sign`` 1 the anticommutator ``AB + BA``."""
    return left * right + sign * right * left


def delta(first: Any, second: Any) -> Any:
    """The Kronecker delta of two labels."""
    return sympy.KroneckerDelta(first, second)


def clebsch_gordan(j1: Any, m1: Any, j2: Any, m2: Any, j: Any, m: Any) -> Any:
    """The Clebsch–Gordan coefficient :math:`\\langle j_1 j_2; m_1 m_2|j_1 j_2; j m\\rangle`."""
    return CG(j1, m1, j2, m2, j, m).doit()


def conjugate(value: Any, *real: Any) -> Any:
    """The complex conjugate of ``value``, with the symbols ``real`` real, as the labels of a
    bracket and the variable of a wavefunction are. Where a conjugate of a parameter the
    program says nothing of would remain, it is taken as books take it, real, or where that
    is not enough, as under a root, positive, and a remark says so (decision 1 of
    design/notation.md)."""
    return _as_books_take(sympy.sympify(value), real, sympy.conjugate)


def modulus(value: Any, *real: Any) -> Any:
    """:math:`|z|` of a value of the rules' algebra, with the symbols ``real`` real, as its
    labels are: the root of z times its conjugate, since SymPy sums that over a label where
    it leaves the absolute value of Kronecker deltas as it is. Parameters are taken as
    :func:`conjugate` takes them."""
    square = _as_books_take(sympy.sympify(value), real, _times_its_conjugate)
    return sympy.sqrt(square)


def _times_its_conjugate(value: Any) -> Any:
    """``value`` times its conjugate, expanded, so that products of deltas are summed."""
    return sympy.expand(value * sympy.conjugate(value))


def _as_books_take(value: Any, real: Sequence[Any], conjugated: Any) -> Any:
    """``conjugated`` of ``value``, a function that conjugates it, with the symbols ``real``
    real, and the parameters the program says nothing of as :func:`conjugate` takes them."""
    labels: dict[Any, Any] = {
        symbol: sympy.Dummy(symbol.name, real=True)
        for symbol in real
        if isinstance(symbol, sympy.Symbol) and symbol.is_real is not True
    }
    labels.update(  # a parameter assumed a >= 0, away from 0, where the value is defined
        (symbol, sympy.Dummy(symbol.name, positive=True))
        for symbol in value.free_symbols
        if isinstance(symbol, sympy.Symbol) and symbol.is_nonnegative and not symbol.is_positive
    )
    found = conjugated(value.xreplace(labels))
    left = _conjugated_parameters(found)
    reals = {symbol: sympy.Dummy(symbol.name, real=True) for symbol in left}
    if reals:
        found = conjugated(value.xreplace({**labels, **reals}))
        under = {symbol for each in found.atoms(sympy.conjugate) for symbol in each.free_symbols}
        still = [symbol for symbol in left if reals[symbol] in under]
        positive = {symbol: sympy.Dummy(symbol.name, positive=True) for symbol in still}
        reals = {symbol: dummy for symbol, dummy in reals.items() if symbol not in still}
        found = conjugated(value.xreplace({**labels, **reals, **positive}))
        if reals:
            shown = assumptions.listed([display.unicode(symbol) for symbol in reals])
            verb = "is" if len(reals) == 1 else "are"
            assumptions.remark(
                f"assuming {shown} real, as books take a parameter to be; the program does not "
                f"say what {shown} {verb}",
                assuming=[f"{display.unicode(symbol)} real" for symbol in reals],
            )
        found = assumptions._assumed(found, positive, reals)
    back = {dummy: symbol for symbol, dummy in labels.items()}
    return found.xreplace(back)


def _conjugated_parameters(value: Any) -> list[Any]:
    """The symbols the program says nothing of under a conjugate in ``value``, in order."""
    found = {
        symbol
        for each in value.atoms(sympy.conjugate)
        for symbol in each.free_symbols
        if isinstance(symbol, sympy.Symbol) and assumptions._general(symbol)
    }
    return sorted(found, key=str)


def inner(first: Any, second: Any) -> Any:
    """The inner product of two vectors, :math:`\\sum_i u_i \\bar v_i`, linear in the first,
    as mathematicians write it with a comma."""
    u, v = _entries(first), _entries(second)
    if len(u) != len(v):
        raise ValueError(
            f"an inner product is of vectors of one length, not of {len(u)} and {len(v)} entries"
        )
    return sympy.Add(*(a * sympy.conjugate(b) for a, b in zip(u, v, strict=True)))


def _entries(vector: Any) -> list[Any]:
    """The entries of a vector, written as a list, a tuple or a column."""
    if isinstance(vector, sympy.MatrixBase | sympy.NDimArray):
        return list(sympy.Matrix(vector).reshape(len(sympy.Matrix(vector)), 1))
    if isinstance(vector, list | tuple):
        return [sympy.sympify(entry) for entry in vector]
    raise ValueError("an inner product is of vectors, written as lists or columns")


def reduced(value: Any, *rules: Rule) -> Any:
    """``value`` with the ``rules`` applied until none applies: each operator acting on the
    ket after it, each bra meeting the ket after it, and operators in the order their
    commutators give; with commutators written out first."""
    if not isinstance(value, sympy.Basic):
        return value
    value = value.replace(lambda e: isinstance(e, Commutator), lambda e: e.doit())
    for _ in range(_MOST):
        rewritten = _rewritten(sympy.expand(value), rules)
        if rewritten is None:
            break
        value = rewritten
    return sympy.expand(value)


def ordered(value: Any, *rules: Rule) -> Any:
    """``value`` reduced by the ``rules``, and where they order every pair of the operators
    left in it, with each operator a commuting symbol of its name, so that the sides of a
    claim are compared as numbers are: in the order the commutators give, an expression of
    operators has one form."""
    value = reduced(value, *rules)
    if not isinstance(value, sympy.Basic):
        return value
    factors = {
        factor for term in sympy.Add.make_args(value) for factor in _spelled(term.args_cnc()[1])
    }
    if any(isinstance(factor, KetBase) or not _an_operator(factor) for factor in factors):
        return value
    ordering = {frozenset(rule.lhs) for rule in rules if rule.kind == "commutator"}
    pairs = {frozenset((a, b)) for a in factors for b in factors if a != b}
    if not pairs <= ordering:
        return value
    return value.xreplace({factor: sympy.Symbol(str(factor)) for factor in factors})


def _an_operator(factor: Any) -> bool:
    """Whether a noncommuting factor is an operator, or its adjoint."""
    return isinstance(factor, Operator) or (
        isinstance(factor, Dagger) and isinstance(factor.args[0], Operator)
    )


def _rewritten(value: Any, rules: Sequence[Rule]) -> Any:
    """``value`` with one rewrite made in each term that has one; ``None`` where none has."""
    terms = sympy.Add.make_args(value)
    done = [_term(term, rules) for term in terms]
    if all(found is None for found in done):
        return None
    return sympy.Add(
        *(term if found is None else found for term, found in zip(terms, done, strict=True))
    )


def _term(term: Any, rules: Sequence[Rule]) -> Any:
    """One term with one rewrite made, the rightmost first; ``None`` where none applies."""
    coefficient, factors = term.args_cnc()
    for position, factor in enumerate(coefficient):
        if isinstance(factor, InnerProduct):
            found = _inner(factor.bra, factor.ket, rules)
            if found is not None:
                rest = coefficient[:position] + coefficient[position + 1 :]
                return sympy.Mul(*rest, found, *factors)
    sequence = _spelled(factors)
    for position in range(len(sequence) - 1, 0, -1):
        if not isinstance(sequence[position], KetBase):
            continue
        for rule in rules:
            if rule.kind != "action":
                continue
            found, width = _action(sequence[: position + 1], rule)
            if found is not None:
                before, after = sequence[: position + 1 - width], sequence[position + 1 :]
                return sympy.Mul(*coefficient, *before, found, *after)
    for position in range(len(sequence) - 1):
        left, right = sequence[position], sequence[position + 1]
        for rule in rules:
            if rule.kind == "commutator" and (right, left) == rule.lhs and left != right:
                first, second = rule.lhs
                swapped = first * second - rule.rhs
                before, after = sequence[:position], sequence[position + 2 :]
                return sympy.Mul(*coefficient, *before, swapped, *after)
    return None


def _spelled(factors: Sequence[Any]) -> list[Any]:
    """Noncommuting factors one by one, a power of an operator as that many factors."""
    spelled: list[Any] = []
    for factor in factors:
        if isinstance(factor, sympy.Pow) and factor.exp.is_Integer and factor.exp > 0:
            spelled.extend([factor.base] * int(factor.exp))
        else:
            spelled.append(factor)
    return spelled


def _action(sequence: Sequence[Any], rule: Rule) -> tuple[Any, int]:
    """What the operators that end ``sequence`` before its last factor, a ket, make of it by
    an action ``rule``, and how many factors that replaces; ``None`` where the rule does not
    apply."""
    (product,) = rule.lhs
    _, parts = sympy.sympify(product).args_cnc()
    parts = _spelled(parts)
    *operators, pattern = parts
    width = len(operators) + 1
    if not isinstance(pattern, KetBase) or len(sequence) < width:
        return None, 0
    if list(sequence[-width:-1]) != operators:
        return None, 0
    matched = _labels(sequence[-1], pattern)
    if matched is None:
        return None, 0
    return rule.rhs.xreplace(matched), width


def _inner(bra: Any, written: Any, rules: Sequence[Rule]) -> Any:
    """The inner product of a bra and a ket an ``inner`` rule gives; ``None`` where none
    does."""
    for rule in rules:
        if rule.kind != "inner":
            continue
        (pattern,) = rule.lhs
        if not isinstance(pattern, InnerProduct):
            continue
        labels = sympy.Tuple(*bra.args, *written.args)
        wanted = sympy.Tuple(*pattern.bra.args, *pattern.ket.args)
        if len(labels) != len(wanted):
            continue
        matched = labels.match(wanted)
        if matched is not None:
            return rule.rhs.xreplace(matched)
    return None


def _labels(found: Any, pattern: Any) -> dict[Any, Any] | None:
    """How the labels of a ket match those of a ket in a rule, whose Wild symbols stand for
    any label; ``None`` where they do not."""
    if len(found.args) != len(pattern.args):
        return None
    matched = sympy.Tuple(*found.args).match(sympy.Tuple(*pattern.args))
    return dict(matched) if matched is not None else None
