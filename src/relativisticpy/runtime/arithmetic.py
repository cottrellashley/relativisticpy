r"""Number theory, computed: sums and products over the values a condition gives their
index, the remainders of powers and the solutions of congruences, and the coefficients of
a series (design/notation.md, 19, 30, 36 and 37), each registered by
:mod:`relativisticpy.runtime.registry` under a name of its own.

A set of values that is finite and known, as the divisors of 28 or the primes up to 100,
is summed term by term, each value meeting the conditions written beside it. One without
end is summed in closed form where SymPy finds one, as the sum over every integer but 0;
and a product over every prime, Euler's, is the value of ζ it is (Apostol (1976),
*Introduction to Analytic Number Theory*, theorem 11.7).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import sympy

from relativisticpy import assumptions, display
from relativisticpy.errors import EvaluationError

#: The most values a sum or a product over a range is carried out term by term for.
_MOST = 100_000


def divides(divisor: Any, number: Any) -> Any:
    """Whether ``divisor`` divides ``number``, as ``d \\mid n`` says."""
    return sympy.Eq(sympy.Mod(number, divisor), 0)


def coefficient(f: Any, variable: Any, power: Any) -> Any:
    """``[x^n] F(x)``: the coefficient of x^n in the power series of F about 0, as
    combinatorics reads numbers from a generating function (Graham, Knuth and Patashnik
    (1994), *Concrete Mathematics*, section 7.1), for a whole number n."""
    n = sympy.sympify(power)
    if not (n.is_Integer and n >= 0):
        shown = display.unicode(n)
        raise EvaluationError(
            f"the coefficient of {variable}^({shown}) is computed where the power is a whole "
            "number",
            hint=f"give the power a value, as [{variable}^{{10}}]",
        )
    found = sympy.series(sympy.sympify(f), variable, 0, int(n) + 1).removeO()
    return sympy.expand(found).coeff(variable, int(n))


def power_mod(base: Any, exponent: Any, m: Any) -> Any:
    """``a^k \\bmod m``, by repeated squaring, so that 3^{100} needs no 48 digits; of a
    negative k, the power of the inverse of a modulo m, as ``17^{-1} \\bmod 43`` is 38,
    refused where a has none."""
    a, k, n = (sympy.sympify(value) for value in (base, exponent, m))
    if not (a.is_Integer and k.is_Integer and n.is_Integer and n != 0):
        return sympy.Mod(a**k, n)
    try:
        return sympy.Integer(pow(int(a), int(k), int(n)))
    except ValueError:
        shared = sympy.gcd(a, n)
        raise EvaluationError(
            f"{a} has no inverse modulo {n}, as both are multiples of {shared}",
            hint="an inverse modulo m is of a number prime to m",
        ) from None


def solve_congruences(unknown: Any, *pairs: Any) -> Any:
    """The solutions of congruences in ``unknown``, each given as a difference of its sides
    and its modulus, ``pairs`` of them: the remainders modulo the least common multiple of
    the moduli, which every solution leaves, by the Chinese remainder theorem."""
    from sympy.ntheory.modular import solve_congruence
    from sympy.ntheory.residue_ntheory import polynomial_congruence

    choices: list[tuple[list[int], int]] = []
    for difference, modulus in zip(pairs[::2], pairs[1::2], strict=True):
        m = sympy.sympify(modulus)
        polynomial = sympy.sympify(difference).as_poly(unknown)
        if (
            not (m.is_Integer and m > 0)
            or polynomial is None
            or not _whole_coefficients(polynomial)
        ):
            raise EvaluationError(
                "solve solves congruences of polynomials with whole coefficients, to whole moduli",
                hint="as solve(x^2 \\equiv 1 \\pmod{8}, x)",
            )
        found = polynomial_congruence(polynomial.as_expr(), int(m))
        choices.append((sorted({int(r) % int(m) for r in found}), int(m)))
    solutions: set[int] = set()
    together = 1
    moduli = [m for _, m in choices]
    for picked in itertools.product(*(remainders for remainders, _ in choices)):
        joined = solve_congruence(*zip(picked, moduli, strict=True))
        if joined is not None:
            solutions.add(int(joined[0]))
            together = int(joined[1])
    if not solutions:
        assumptions.remark(f"no integer {unknown} meets the congruences")
        return []
    shown = ", ".join(map(str, sorted(solutions)))
    assumptions.remark(
        f"each solution is one of {shown} plus a multiple of {together}: they are the "
        f"remainders modulo {together}"
    )
    return [sympy.Integer(value) for value in sorted(solutions)]


def _whole_coefficients(polynomial: Any) -> bool:
    return all(sympy.sympify(c).is_Integer for c in polynomial.all_coeffs())


def over(
    body: Any,
    index: Any,
    lower: Any,
    upper: Any,
    of: Any,
    *rest: Any,
    op: str,
    primes: bool,
    divisors: bool,
    strict: tuple[bool, bool],
    excluded: int,
) -> Any:
    """The sum, or the product where ``op`` says so, of ``body`` over the values of
    ``index`` a condition gives: the divisors of ``of``, where ``divisors``, or the integers
    from ``lower`` to ``upper``, either excluded where it is ``strict``; the ``primes``
    alone where it says so; without the first ``excluded`` of ``rest``, and where each
    condition that follows holds."""
    left_out, conditions = rest[:excluded], rest[excluded:]
    low, high = _whole(lower, upper, strict)
    if not (divisors or primes):
        low, high = _support(body, index, low, high)
    if primes and not divisors and not left_out and not conditions:
        counted = _of_a_count(body, index, low, high, op)
        if counted is not None:
            return counted
    values = _divisors(of, primes, low, high) if divisors else _range(low, high, primes)
    if values is not None:  # each value left out a condition, decided where it is a number
        apart = [sympy.Ne(index, out) for out in left_out]
        return _term_by_term(body, index, values, [*apart, *conditions], op)
    if divisors:
        raise EvaluationError(
            f"a {_kind(op)} over the divisors of {of} is computed where {of} is a number",
            hint=f"give {of} a value first",
        )
    if primes:
        return _over_primes(body, index, low, high, conditions, op)
    return _over_integers(body, index, low, high, left_out, conditions, op)


def _kind(op: str) -> str:
    return "sum" if op == "sum" else "product"


def _whole(lower: Any, upper: Any, strict: tuple[bool, bool]) -> tuple[Any, Any]:
    """The least and the greatest integer the bounds allow, each excluded where it is
    strict: ``k < x`` allows up to ⌈x⌉ - 1."""
    low, high = sympy.sympify(lower), sympy.sympify(upper)
    if low.is_finite:
        low = sympy.floor(low) + 1 if strict[0] else sympy.ceiling(low)
    if high.is_finite:
        high = sympy.ceiling(high) - 1 if strict[1] else sympy.floor(high)
    return low, high


def _support(body: Any, index: Any, low: Any, high: Any) -> tuple[Any, Any]:
    """The bounds of a sum without end narrowed to where its binomial coefficients can be
    other than 0, as Concrete Mathematics leaves a sum's range to them: binomial(a, k + c)
    is 0 where k < -c, and binomial(a, c - k) where k > c, of an integer c, as their lower
    index is then a negative integer; and of a whole number a, where the lower index is
    above a."""
    for factor in sympy.Mul.make_args(sympy.sympify(body)):
        if not isinstance(factor, sympy.binomial):
            continue
        top, bottom = factor.args[0], sympy.expand(factor.args[1])
        slope = bottom.coeff(index)
        rest = bottom - slope * index
        if rest.has(index) or rest.is_integer is not True or slope not in (1, -1):
            continue
        whole = top.is_integer is True and top.is_nonnegative is True
        if slope == 1:  # k + c, from 0 to a
            low = -rest if not low.is_finite else low
            high = top - rest if whole and not high.is_finite else high
        else:  # c - k, from 0 to a
            high = rest if not high.is_finite else high
            low = rest - top if whole and not low.is_finite else low
    return low, high


def _divisors(of: Any, primes: bool, low: Any, high: Any) -> list[Any] | None:
    """The positive divisors of a whole number, or the primes that divide it, between the
    bounds; ``None`` of anything else, as of a symbol."""
    number = sympy.sympify(of)
    if not (number.is_Integer and number != 0):
        return None
    found = sympy.primefactors(number) if primes else sympy.divisors(abs(number))
    return [sympy.Integer(d) for d in found if _between(d, low, high)]


def _between(value: int, low: Any, high: Any) -> bool:
    return bool(
        (not low.is_finite or sympy.sympify(value >= low) is sympy.true)
        and (not high.is_finite or sympy.sympify(value <= high) is sympy.true)
    )


def _range(low: Any, high: Any, primes: bool) -> list[Any] | None:
    """The integers between two whole numbers, or the primes; ``None`` where one is no
    number or has no end. An error that says so where there are too many to sum one by
    one, as the primes up to 10^9 are."""
    start = sympy.Integer(2) if primes and not low.is_finite else low
    if not (start.is_Integer and high.is_Integer):
        return None
    if primes:
        first = max(int(start), 2)
        count = int(sympy.primepi(high)) - int(sympy.primepi(first - 1))
        if count > _MOST:
            raise EvaluationError(
                f"there are {count} primes from {first} to {high}, more than the {_MOST} a sum "
                "or a product is carried out over one by one",
                hint="take fewer of them, or the sum of a constant, which counts them",
            )
        return [sympy.Integer(p) for p in sympy.primerange(first, int(high) + 1)]
    if high - start > _MOST:
        return None
    return [sympy.Integer(k) for k in range(int(start), int(high) + 1)]


def _of_a_count(body: Any, index: Any, low: Any, high: Any, op: str) -> Any:
    """A sum or a product over the primes between two numbers of what does not depend on
    the prime: as many times it as there are primes, which π counts at once, as
    \\sum_{p \\le 10^9} 1 is π(10^9); ``None`` of anything else."""
    first = sympy.Integer(2) if not low.is_finite else low
    term = sympy.sympify(body)
    if term.has(index) or not (first.is_Integer and high.is_Integer):
        return None
    count = max(int(sympy.primepi(high)) - int(sympy.primepi(max(int(first), 2) - 1)), 0)
    return term * count if op == "sum" else term**count


def _term_by_term(
    body: Any, index: Any, values: Sequence[Any], conditions: Sequence[Any], op: str
) -> Any:
    """A sum or a product over finitely many values, a term at a time, within the time a
    closed form is given: exact fractions of many primes have denominators of many digits,
    which take long to add. An error that says so where it runs out."""
    from relativisticpy.runtime import core

    try:
        with core._budget(core._CLOSED_FORM_BUDGET):
            terms = [_term(body, index, value, conditions, op) for value in values]
            return sympy.Add(*terms) if op == "sum" else sympy.Mul(*terms)
    except core._OverBudget:
        raise EvaluationError(
            f"the {_kind(op)} of {len(values)} terms, exactly, takes longer than the "
            f"{core._CLOSED_FORM_BUDGET:g} s it is given",
            hint="write a number in it as a decimal, as 1.0, to compute it numerically",
        ) from None


def _term(body: Any, index: Any, value: Any, conditions: Sequence[Any], op: str) -> Any:
    """The term of one value: the body there, where each condition holds there; kept with
    its condition where that is not decided."""
    term = sympy.sympify(body).subs(index, value)
    held = [sympy.sympify(condition).subs(index, value) for condition in conditions]
    if any(each is sympy.false for each in held):
        return sympy.S.Zero if op == "sum" else sympy.S.One
    open_ = [each for each in held if each is not sympy.true]
    if not open_:
        return term
    otherwise = sympy.S.Zero if op == "sum" else sympy.S.One
    return sympy.Piecewise((term, sympy.And(*open_)), (otherwise, True))


def _over_primes(
    body: Any, index: Any, low: Any, high: Any, conditions: Sequence[Any], op: str
) -> Any:
    """A product over every prime that is one of Euler's for ζ, as ∏ 1/(1 - p^{-s}) is ζ(s);
    refused, saying why, where the primes have no end otherwise, or a bound is a symbol."""
    if op == "prod" and not conditions and high == sympy.oo and not low.is_finite:
        found = euler_product(sympy.sympify(body), index)
        if found is not None:
            return found
    bound = high if not high.is_number else low
    if high == sympy.oo:
        raise EvaluationError(
            f"a {_kind(op)} over every prime is computed where it is one of Euler's products for ζ",
            hint="as \\prod_{p \\text{ prime}} \\frac{1}{1 - p^{-s}}, which is ζ(s)",
        )
    raise EvaluationError(
        f"a {_kind(op)} over the primes up to {bound} is computed where {bound} is a number",
        hint=f"give {bound} a value first",
    )


def euler_product(body: Any, prime: Any) -> Any:
    """The value of ∏ f(p) over every prime p where f is one of Euler's factors for ζ, of a
    real part s > 1: 1/(1 - p^{-s}) gives ζ(s), 1 - p^{-s} its inverse, 1 + p^{-s} ζ(s)/ζ(2s)
    and 1/(1 + p^{-s}) ζ(2s)/ζ(s) (Apostol (1976), theorems 11.6 and 11.7); ``None`` of any
    other factor, and refused where the product diverges."""
    for factor, inverted in ((sympy.expand(1 / body), True), (sympy.expand(body), False)):
        rest = sympy.expand(factor - 1)
        sign, power = rest.as_coeff_Mul()
        if sign not in (1, -1) or not (power.is_Pow and power.base == prime):
            continue
        s = -power.exp
        if s.has(prime):
            continue
        if s.is_number and not sympy.re(s) > 1:
            raise EvaluationError(
                f"Euler's product over every prime diverges where s is {s}",
                hint="it converges where the real part of s is above 1, as ζ(s) does",
            )
        if not s.is_number:
            assumptions.remark(f"Euler's product converges where the real part of {s} is above 1")
        zeta, double = sympy.zeta(s), sympy.zeta(2 * s)
        if sign == -1:
            return zeta if inverted else 1 / zeta
        return double / zeta if inverted else zeta / double
    return None


def _over_integers(
    body: Any,
    index: Any,
    low: Any,
    high: Any,
    left_out: Sequence[Any],
    conditions: Sequence[Any],
    op: str,
) -> Any:
    """A sum or a product over integers without end, or between symbols: in closed form, as
    the language computes ``\\sum_{k=a}^{b}``, split where whole numbers are left out, as
    0 is by k ≠ 0, with each condition a factor that is 1 where it holds."""
    from relativisticpy.runtime import core

    whole = sympy.sympify(body)
    if conditions:
        held = sympy.And(*(sympy.sympify(condition) for condition in conditions))
        otherwise = sympy.S.Zero if op == "sum" else sympy.S.One
        whole = sympy.Piecewise((whole, held), (otherwise, True))
    symbolic = [out for out in map(sympy.sympify, left_out) if not out.is_Integer]
    if symbolic:
        whole, left_out = _shifted(whole, index, low, high, symbolic, left_out, op)
    pieces = [(low, high)]
    for out in sorted((sympy.sympify(v) for v in left_out if sympy.sympify(v).is_Integer), key=int):
        pieces = [
            part
            for start, end in pieces
            for part in ((start, out - 1), (out + 1, end))
            if not (part[0].is_number and part[1].is_number and part[0] > part[1])
        ]
    combine = core.summation if op == "sum" else core.product
    values = [combine(whole, index, start, end) for start, end in pieces]
    return sympy.Add(*values) if op == "sum" else sympy.Mul(*values)


def _shifted(
    body: Any,
    index: Any,
    low: Any,
    high: Any,
    symbolic: Sequence[Any],
    left_out: Sequence[Any],
    op: str,
) -> tuple[Any, list[Any]]:
    """A sum or a product over every integer that leaves out one, written with symbols, as
    ``\\sum_{k \\neq n}`` does: over the integers counted from it, k = n + j, which leaves
    out j = 0, so that the term it leaves out, which may have no value, is never taken.
    That value is an integer, as one the sum runs over, which a remark says where nothing
    else does. Refused, saying why, for any other range or more values left out."""
    (out, *rest) = symbolic
    shown = display.unicode(out)
    if rest or len(left_out) > 1 or low.is_finite or high.is_finite:
        raise EvaluationError(
            f"a {_kind(op)} over infinitely many integers that leaves out {shown}, which is "
            "not a number, is computed only over every integer, with that one left out",
            hint=f"give {shown} a value first",
        )
    if out.is_integer is not True:
        assumptions.remark(
            f"assuming {shown} is an integer, as the {_kind(op)} leaves out the term "
            f"{display.unicode(index)} = {shown}",
            assuming=[f"{shown} is an integer"],
        )
    return sympy.sympify(body).xreplace({index: index + out}), [sympy.S.Zero]
