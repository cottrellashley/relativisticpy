r"""Assumptions, applied: what a program assumes of its names, put where it decides
something (design/notation.md, decisions 1 and 2).

A relation on names nothing defines, as ``a > 0`` or ``x \in \mathbb{R}``, is an assumption
for the rest of the program (:mod:`relativisticpy.reading.assumptions` reads it). What SymPy
keeps on a symbol, that it is positive, nonnegative, real or an integer, is put on the
symbol, and what was defined with the plain symbol takes it from then on. What SymPy cannot
keep, a bound as x < 1 in 0 < x < 1 or a > r, is kept in the session, and given to what it
decides: the solutions of ``solve``, the conditions of an integral, and inequalities.

A computation says what it did to its value in a :class:`Remark`, which the workbook shows
as a note: that ``solve`` dropped the solutions an assumption rules out, or that a value is
given under an assumption books make, ``assuming a > 0``.
"""

from __future__ import annotations

import contextlib
import itertools
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import sympy
from sympy import Symbol
from sympy.core.facts import InconsistentAssumptions
from sympy.core.relational import Relational

from relativisticpy import display

#: The facts an assumption may put on a symbol, as SymPy names them, in the order they
#: are said.
FACTS = ("positive", "nonnegative", "negative", "nonpositive", "nonzero", "real", "integer",
         "rational", "complex")  # fmt: skip
#: How each fact is said.
_WORDS = {
    "positive": "positive",
    "nonnegative": "zero or positive",
    "negative": "negative",
    "nonpositive": "zero or negative",
    "nonzero": "nonzero",
    "real": "real",
    "rational": "rational",
    "complex": "complex",
}
#: The facts that order a value against zero, each of which says it is real.
_ORDERED = ("positive", "nonnegative", "negative", "nonpositive")
#: The operations a bound the program assumes decides something of, which are given the
#: bounds in force as their attribute ``assuming``.
DECIDED = frozenset(
    {"solve", "integrate", "core.sum", "core.lt", "core.gt", "core.le", "core.ge",
     "optimum.find", "optimum.point", "optimum.components", "seq.sum"}
)  # fmt: skip
#: The relation each fact of a sign is, for a coordinate, whose symbol stays as it is.
_SIGNS = {
    "positive": sympy.Gt,
    "nonnegative": sympy.Ge,
    "negative": sympy.Lt,
    "nonpositive": sympy.Le,
    "nonzero": sympy.Ne,
}


def symbol(name: str, facts: Mapping[str, bool], before: Symbol | None = None) -> Symbol:
    """The symbol called ``name`` with ``facts``, and whatever was assumed of it
    ``before``; a :class:`ValueError` that says why where they contradict each other."""
    given = dict(getattr(before, "_assumptions_orig", None) or {})
    given.update(facts)
    try:
        return Symbol(name, **given)
    except InconsistentAssumptions:
        was = described(getattr(before, "_assumptions_orig", None) or {})
        raise ValueError(f"{name} was assumed {was}, and cannot be {described(facts)}") from None


def described(facts: Mapping[str, bool]) -> str:
    """Facts as words: ``positive``, ``real``, ``a positive integer``."""
    held = [fact for fact in FACTS if facts.get(fact)]
    ordered = [fact for fact in held if fact in _ORDERED]  # each says the value is real
    if facts.get("integer"):
        if not ordered:
            return "a nonzero integer" if facts.get("nonzero") else "an integer"
        if ordered[0] in ("nonnegative", "nonpositive"):
            return f"an integer, {_WORDS[ordered[0]]}"
        return f"a {_WORDS[ordered[0]]} integer"
    shown = [_WORDS[fact] for fact in held if not (fact == "real" and ordered)]
    return " and ".join(shown) if shown else "general"


def as_bounds(name: Symbol, facts: Mapping[str, bool]) -> list[Any]:
    """The facts of a sign as relations, for a coordinate, whose symbol stays as it is:
    positive is name > 0."""
    return [_SIGNS[fact](name, 0) for fact in _SIGNS if facts.get(fact)]


def said(facts: Mapping[str, Mapping[str, bool]], bounds: Sequence[Any]) -> str:
    """What an assumption says of its names, as a note says it: ``a and b are positive``,
    ``x is positive and x < 1``."""
    by_facts: dict[str, list[str]] = {}
    for name, held in facts.items():
        if held:
            by_facts.setdefault(described(held), []).append(name)
    parts = []
    for words, names in by_facts.items():
        shown = listed([display.unicode(Symbol(name)) for name in names])
        verb = "is" if len(names) == 1 else "are"
        parts.append(f"{shown} {verb} {words}")
    parts.extend(display.unicode(bound) for bound in bounds)
    return listed(parts)


def listed(items: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    if len(items) < 2:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


# -- what was defined, with the symbols an assumption makes ---------------------------------


def substituted(value: Any, *, symbols: tuple[Symbol, ...]) -> Any:
    """``value`` with each symbol named as one of ``symbols`` put in its place, whatever
    was assumed of it before: what a value defined with a plain symbol is once an
    assumption gives the symbol facts. A value of another kind, as numbers, is as it is."""
    by_name = {each.name: each for each in symbols}

    def put(expr: Any) -> Any:
        found = {
            s: by_name[s.name]
            for s in getattr(expr, "free_symbols", ())
            if isinstance(s, Symbol) and not isinstance(s, sympy.Dummy) and s.name in by_name
        }
        return expr.xreplace(found) if found else expr

    return mapped(value, put)


def mapped(value: Any, put: Callable[[Any], Any]) -> Any:
    """``value`` with ``put`` applied to each SymPy expression in it, through lists,
    tuples, dictionaries and arrays."""
    if isinstance(value, sympy.Basic):
        return put(value)
    if isinstance(value, sympy.NDimArray | sympy.MatrixBase):
        return value.applyfunc(put)
    if isinstance(value, list | tuple):
        return type(value)(mapped(item, put) for item in value)
    if isinstance(value, dict):
        return {mapped(key, put): mapped(item, put) for key, item in value.items()}
    return value


def in_force(bounds: Sequence[tuple[Any, Any]], symbols: Mapping[str, Symbol]) -> tuple[Any, ...]:
    """The bounds a program assumes, as given with where each was stated, written with the
    symbols in force: with what was assumed of their names since."""
    return tuple(
        substituted(bound, symbols=tuple(symbols.values())) for bound, _ in dict(bounds).items()
    )


# -- what a computation says of its value ----------------------------------------------------


@dataclass(frozen=True)
class Remark:
    """What a computation says of the value it gave, as a note: that ``solve`` dropped
    the solutions an assumption rules out, or that a value is given under an assumption
    books make, which ``assuming`` holds, as ``("a > 0",)``."""

    note: str
    assuming: tuple[str, ...] = ()


_REMARKS: ContextVar[list[Remark] | None] = ContextVar("remarks", default=None)


def remark(note: str, assuming: Sequence[str] = ()) -> None:
    """Say ``note`` of the value being computed."""
    found = _REMARKS.get()
    said = Remark(note, tuple(assuming))
    if found is not None and said not in found:
        found.append(said)


@contextlib.contextmanager
def remarks() -> Iterator[list[Remark]]:
    """Collect what the computations inside say of their values."""
    found: list[Remark] = []
    token = _REMARKS.set(found)
    try:
        yield found
    finally:
        _REMARKS.reset(token)


# -- what the bounds decide ---------------------------------------------------------------


def holds(relation: Any, bounds: Sequence[Any]) -> bool | None:
    """Whether ``relation`` holds wherever ``bounds`` and what is assumed of its symbols
    do: ``True`` or ``False``, or ``None`` where that is not decided."""
    decided = sympy.sympify(relation).xreplace(within(bounds))
    if decided in (sympy.true, sympy.false):
        return bool(decided)
    if isinstance(decided, sympy.And | sympy.Or):  # each part, as SymPy leaves them
        verdicts = [holds(part, ()) for part in decided.args]
        every, some = (False, True) if isinstance(decided, sympy.Or) else (True, False)
        if every in verdicts:
            return every
        return some if all(verdict is some for verdict in verdicts) else None
    if isinstance(decided, Relational) and not isinstance(decided, sympy.Eq | sympy.Ne):
        return _compared(decided)
    return None


def _compared(relation: Any) -> bool | None:
    """Whether an inequality SymPy leaves as it is holds: by the sign of the difference of
    its sides, as one fraction; and, of two positive sides, by their logarithms, as
    e^{-a} < 1 holds of a positive a."""
    tries = [(relation.lhs, relation.rhs)]
    if relation.lhs.is_positive and relation.rhs.is_positive:
        tries.append(tuple(sympy.expand_log(sympy.log(side), force=True) for side in relation.args))
    for lhs, rhs in tries:
        settled = type(relation)(sympy.factor(sympy.together(lhs - rhs)), 0)
        if settled in (sympy.true, sympy.false):
            return bool(settled)
    return None


def within(bounds: Sequence[Any]) -> dict[Any, Any]:
    """Each symbol a bound limits, as an expression of new positive symbols whose values are
    exactly those the bounds and its own assumptions allow, as far as that can be written:
    x > 1 is 1 + p, a > r is r + p, and a positive x < 1 is 1/(1 + p). Where it cannot be,
    the values written for are more than are allowed, never fewer, so that what holds of
    them holds wherever the bounds do."""
    lower: dict[Any, list[tuple[Any, bool]]] = {}
    upper: dict[Any, list[tuple[Any, bool]]] = {}
    for bound in bounds:
        if not isinstance(bound, Relational) or isinstance(bound, sympy.Eq | sympy.Ne):
            continue
        small, large = (
            (bound.lhs, bound.rhs) if bound.rel_op in ("<", "<=") else (bound.rhs, bound.lhs)
        )
        strict = bound.rel_op in ("<", ">")
        if _limited(large):
            lower.setdefault(large, []).append((small, strict))
        if _limited(small) and not _limited(large):  # x < a is a > x
            upper.setdefault(small, []).append((large, strict))
    for name in [*lower, *upper]:  # and the sign each is assumed to have
        if name.is_positive or name.is_nonnegative:
            lower.setdefault(name, []).append((sympy.Integer(0), bool(name.is_positive)))
        if name.is_negative or name.is_nonpositive:
            upper.setdefault(name, []).append((sympy.Integer(0), bool(name.is_negative)))
    substitution: dict[Any, Any] = {}
    pending = list(dict.fromkeys([*lower, *upper]))
    while pending:
        ready = [
            name
            for name in pending
            if not any(end in pending for end, _ in lower.get(name, ()) if end != name)
        ]
        if not ready:  # bounds that go round, as a > b > a, which say nothing to rely on
            break
        for name in ready:
            pending.remove(name)
            below, above = (
                [(sympy.sympify(end).xreplace(substitution), strict) for end, strict in found]
                for found in (lower.get(name, []), upper.get(name, []))
            )
            substitution[name] = _between(below, above, name)
    return {name: value for name, value in substitution.items() if value is not name}


def _limited(value: Any) -> bool:
    """Whether a bound limits a value, written for in :func:`within`: a symbol, or the
    modulus of one, as |z| of |z| > 1/2."""
    return isinstance(value, Symbol) or (
        isinstance(value, sympy.Abs) and isinstance(value.args[0], Symbol)
    )


def _between(lower: list[tuple[Any, bool]], upper: list[tuple[Any, bool]], name: Any) -> Any:
    """A symbol between its first lower and upper bounds, each with whether it is strict,
    as an expression of a new symbol, positive or nonnegative; beside a bound, where it has
    one only. A closed interval is written for as the values from its lower end on."""
    if lower and upper:
        (bottom, open_below), (top, open_above) = lower[0], upper[0]
        width = top - bottom
        if open_below and open_above:  # (L, U)
            return bottom + width / (1 + _step(name, strict=True))
        if open_above:  # [L, U)
            step = _step(name, strict=False)
            return bottom + width * step / (1 + step)
        if open_below:  # (L, U]
            step = _step(name, strict=False)
            return top - width * step / (1 + step)
    if lower:
        bottom, strict = lower[0]
        return bottom + _step(name, strict=strict)
    if upper:
        top, strict = upper[0]
        return top - _step(name, strict=strict)
    return name


def _step(name: Any, *, strict: bool) -> Any:
    """A new symbol, positive or else nonnegative, named after ``name``, or after the
    symbol whose modulus it is."""
    named = str(name.args[0]) if isinstance(name, sympy.Abs) else name.name
    return sympy.Dummy(named, positive=True) if strict else sympy.Dummy(named, nonnegative=True)


# -- what is assumed of the unknowns of solve -------------------------------------------------


def ruled_out(solution: Mapping[Any, Any], bounds: Sequence[Any]) -> str | None:
    """Why ``solution``, the values of the unknowns of an equation, is not one the program
    allows: an unknown assumed positive that would be negative, or a bound that would not
    hold. ``None`` where it is allowed, or where that cannot be decided."""
    from sympy.core.assumptions import check_assumptions

    for unknown, value in solution.items():
        given: dict[str, bool] = dict(getattr(unknown, "_assumptions_orig", None) or {})
        held = isinstance(unknown, Symbol) and bool(given)
        if held and check_assumptions(sympy.sympify(value), **given) is False:
            return f"{display.unicode(unknown)} is assumed {described(given)}"
    unknowns = set(solution)
    for bound in bounds:
        if not bound.free_symbols & unknowns:
            continue
        others = [other for other in bounds if not other.free_symbols & unknowns]
        if holds(bound.xreplace(dict(solution)), others) is False:
            return f"{display.unicode(bound)} is assumed"
    return None


# -- integrals: what the bounds decide, and what books assume ------------------------------

#: What SymPy writes of a symbol it takes to be complex, which an assumption of its sign or
#: of its reality removes: |a|, arg a, Re a, Im a, the conjugate, sign a and polar_lift a.
_COMPLEX_PLANE = (
    sympy.Abs,
    sympy.arg,
    sympy.re,
    sympy.im,
    sympy.conjugate,
    sympy.sign,
    sympy.polar_lift,
)
#: The names books give a complex variable, which they never take to be real.
_COMPLEX_VARIABLES = frozenset({"z"})
#: The most parameters whose combinations are tried, to find the fewest to assume.
_MOST_TRIED = 8


def integrated(f: Any, limits: Sequence[Any], bounds: Sequence[Any] = ()) -> Any:
    """The integral of ``f`` over ``limits``, as SymPy gives it, with its conditions
    decided where the bounds the program assumes decide them. Where it would still hold
    conditions on its parameters, or a modulus of one, or be left undone for one, it is the
    value under the assumption books make, which a remark says: the variable of an integral
    over real limits is real, and a parameter a scale, positive (design/notation.md,
    decision 1). Nothing is assumed of a symbol the program said something of. An integral
    SymPy leaves undone is computed another way, where one applies (see :func:`_otherwise`).
    Where no closed form is found in the time an integral is given, it is kept as written,
    as a remark says (decision 4)."""
    from relativisticpy.runtime import core

    f = sympy.sympify(f)
    try:
        with core._budget(core._INTEGRAL_BUDGET):
            found = _computed(sympy.integrate, f, limits, bounds)
            if isinstance(found, sympy.Basic) and found.has(sympy.Integral):
                found = _otherwise(f, limits, bounds, found)
            return found
    except core._OverBudget:
        remark(
            f"SymPy found no closed form of the integral in the {core._INTEGRAL_BUDGET:g} s it "
            "is given"
        )
        return sympy.Integral(f, *limits)


#: How long simplifying the integrand of an integral SymPy left undone may take, in seconds.
_SIMPLIFYING = 5.0


def _otherwise(f: Any, limits: Sequence[Any], bounds: Sequence[Any], found: Any) -> Any:
    """An integral SymPy left undone, computed another way where one applies: with its
    integrand simplified where its variables lie (:func:`_where_it_lies`); as SymPy left
    it, ``found``, where that gives no value."""
    for route in (_where_it_lies,):
        try:
            value = route(f, limits, bounds)
        except Exception:  # a way that fails says nothing of the integral
            continue
        if isinstance(value, sympy.Basic) and not value.has(sympy.Integral, sympy.nan):
            return value
    return found


def _where_it_lies(f: Any, limits: Sequence[Any], bounds: Sequence[Any]) -> Any:
    """The integral of ``f`` with its integrand simplified where each variable lies: real,
    between real limits, and zero or positive, or zero or negative, where both its limits
    are, as √(1 + sinh² x) is cosh x for a real x, and √(x²) is x from 0 on. ``None`` where
    that changes nothing, or where a limit is not real."""
    from relativisticpy.runtime import core

    lying: dict[Any, Any] = {}
    for limit in reversed(list(limits)):  # the outermost first: an inner limit may hold it
        if not (isinstance(limit, tuple) and len(limit) == 3):
            return None
        variable = sympy.sympify(limit[0])
        ends = [sympy.sympify(end).xreplace(lying) for end in limit[1:]]
        if not isinstance(variable, Symbol) or not all(end.is_extended_real for end in ends):
            return None
        facts = {"real": True}
        if all(end.is_nonnegative for end in ends):
            facts["nonnegative"] = True
        elif all(end.is_nonpositive for end in ends):
            facts["nonpositive"] = True
        lying[variable] = sympy.Dummy(variable.name, **facts)
    integrand = f.xreplace(lying)
    try:
        with core._budget(_SIMPLIFYING):
            simpler = sympy.simplify(integrand)
    except core._OverBudget:
        return None
    if simpler == integrand:
        return None
    value = _computed(sympy.integrate, simpler, _put(limits, lying), bounds)
    return value.xreplace({each: variable for variable, each in lying.items()})


def summed(body: Any, limits: Sequence[Any], bounds: Sequence[Any] = ()) -> Any:
    """The sum of ``body`` over ``limits``, ``(n, a, b)``, in closed form where SymPy finds
    one, with the index an integer: as :func:`integrated` gives an integral, under the
    bounds the program assumes, and else the assumption books make where it converges only
    under conditions on its parameters (design/notation.md, decisions 1 and 4). A sum in
    an exponential of the index, as of e^{-a n}, is summed as a geometric series. A sum of
    finitely many terms, from an integer to an integer, as the binomial theorem's is, has
    the value SymPy finds without the conditions an infinite series would converge under.
    A closed form that is not defined where the terms are is none (see :func:`_defined`)."""
    ((index, lower, upper),) = limits
    whole = Symbol(index.name, **{**dict(getattr(index, "_assumptions_orig", {})), "integer": True})
    body = sympy.sympify(body).xreplace({index: whole})
    if _finitely_many(lower, upper):
        found = _converged(_sum(body, (whole, lower, upper)))
        if not _undone(found):
            return _defined(found, body, (whole, lower, upper))
    found = _computed(_sum, body, [(whole, lower, upper)], bounds)
    return _defined(found, body, (whole, lower, upper))


def _defined(found: Any, body: Any, limits: tuple[Any, Any, Any]) -> Any:
    """The closed form SymPy gives a sum, where it is one: not where it holds nan or zoo,
    which the terms do not, as SymPy's of Σ 1/(n(2n + 1)) does, which converges. Of a
    rational function of the index summed to infinity, the closed form by its partial
    fractions then; else the sum as written, which a remark says."""
    undefined = (sympy.nan, sympy.zoo)
    if not isinstance(found, sympy.Basic) or not found.has(*undefined) or body.has(*undefined):
        return found
    index, lower, upper = limits
    if upper == sympy.oo and sympy.sympify(lower).is_finite:
        by_parts = _by_partial_fractions(body, index, lower)
        if by_parts is not None and not by_parts.has(*undefined):
            return by_parts
    remark(
        "SymPy's closed form of the sum has no value where its terms have one, so the sum is "
        "kept as written"
    )
    return sympy.Sum(body, limits)


def _by_partial_fractions(body: Any, index: Any, lower: Any) -> Any:
    """The sum from ``lower`` to infinity of a rational function of the index that falls
    as 1/n² or faster, by its partial fractions: of the c/(n + r), whose c's add up to 0
    where the sum converges, -Σ c ψ(lower + r), by the series of the digamma function ψ;
    and of c/(n + r)^k, c ζ(k, lower + r), Hurwitz's zeta function. ``None`` of anything
    else, or where a term of the range is not defined."""
    if not body.is_rational_function(index):
        return None
    numerator, denominator = sympy.fraction(sympy.together(body))
    top, bottom = sympy.Poly(numerator, index), sympy.Poly(denominator, index)
    if bottom.degree() - top.degree() < 2:
        return None
    total, simple = sympy.S.Zero, sympy.S.Zero
    for term in sympy.Add.make_args(sympy.apart(body, index, full=True).doit()):
        coefficient, rest = term.as_independent(index, as_Add=False)
        base, power = rest.as_base_exp()
        linear = sympy.Poly(base, index)
        if not (power.is_Integer and power < 0) or linear.degree() != 1:
            return None
        slope, shift = linear.all_coeffs()
        order, root = -power, shift / slope  # c/(slope n + shift)^k is c'/(n + root)^k
        start = sympy.simplify(lower + root)
        if start.is_integer and start.is_nonpositive:
            return None  # the term at n = -root is not defined
        weight = coefficient / slope**order
        if order == 1:
            total -= weight * sympy.polygamma(0, start)
            simple += weight
        else:
            total += weight * sympy.zeta(order, start)
    if sympy.simplify(simple) != 0:
        return None
    return sympy.simplify(sympy.expand_func(total.rewrite(sympy.harmonic)))


def _finitely_many(lower: Any, upper: Any) -> bool:
    """Whether a sum from ``lower`` to ``upper`` has finitely many terms: both integers."""
    ends = [sympy.sympify(end) for end in (lower, upper)]
    return all(end.is_integer and end.is_finite for end in ends)


def _converged(value: Any) -> Any:
    """A sum of finitely many terms without the conditions SymPy finds it under, which say
    where the infinite series it sums converge, and nothing of it: each piecewise value
    under them its first case."""

    def first(piecewise: Any) -> Any:
        return piecewise.args[0][0] if _on_conditions(piecewise) else piecewise

    return value.replace(lambda e: isinstance(e, sympy.Piecewise), first)


def _sum(body: Any, *limits: Any) -> Any:
    """A sum carried out by SymPy, with each exponential of the index, e^{c n + d}, as
    e^d z^n, so that SymPy sums it as a geometric series in z = e^c."""
    ((index, lower, upper),) = limits
    ratios: dict[Any, Any] = {}

    def geometric(power: Any) -> Any:
        exponent = sympy.expand(power.args[0])
        rate = exponent.coeff(index)
        rest = exponent - rate * index
        if rate == 0 or rest.has(index):
            return power
        ratio = sympy.Dummy("z")
        ratios[ratio] = sympy.exp(rate)
        return sympy.exp(rest) * ratio**index

    rewritten = body.replace(lambda e: isinstance(e, sympy.exp) and e.has(index), geometric)
    return sympy.Sum(rewritten, (index, lower, upper)).doit().xreplace(ratios)


#: What SymPy raises where it cannot decide a relation of the parameters, as a - r >= 0 of
#: the integral of |a + r cos θ|, which the bounds the program assumes may decide.
_UNDECIDED = (TypeError, ValueError, NotImplementedError)


def _computed(
    compute: Callable[..., Any], f: Any, limits: Sequence[Any], bounds: Sequence[Any]
) -> Any:
    """What ``compute`` gives of ``f`` over ``limits``, an integral or a sum: where it holds
    conditions, or where SymPy fails for want of a relation of its parameters, as a > r,
    computed again under the bounds the program assumes, and else under the assumption
    books make (see :func:`integrated`)."""
    try:
        found = compute(f, *limits)
    except _UNDECIDED:
        if not bounds:
            raise
        found = _within_bounds(compute, f, limits, bounds, None)
        if found is None:
            raise
    if bounds and _conditional(found):
        found = _within_bounds(compute, f, limits, bounds, found)
    if not _conditional(found):
        return found
    return _as_books_take(compute, f, limits, found)


def _within_bounds(
    compute: Callable[..., Any], f: Any, limits: Sequence[Any], bounds: Sequence[Any], found: Any
) -> Any:
    """An integral or a sum that holds conditions, or that SymPy could not compute at all,
    ``found`` as ``None``, computed again where the bounds hold: each symbol they limit
    written as the new positive symbols that keep them, x > 1 as 1 + p, and the value
    written with the symbol again, p as x - 1. Its conditions decided by the bounds, where
    that gives no value."""
    written = f.free_symbols.union(*(sympy.sympify(limit).free_symbols for limit in limits))
    limited = {x: e for x, e in within(bounds).items() if x in written}
    steps = sorted(
        {s for e in limited.values() for s in e.free_symbols if isinstance(s, sympy.Dummy)}, key=str
    )
    if limited and steps:
        inverse = sympy.solve([sympy.Eq(x, e) for x, e in limited.items()], steps, dict=True)
        if len(inverse) == 1:
            try:
                again = compute(f.xreplace(limited), *_put(limits, limited))
            except _UNDECIDED:
                if found is not None:
                    raise
                return None  # SymPy fails here too: what it said first is the error
            if not _conditional(again):
                return sympy.simplify(again.xreplace(inverse[0]))
    return None if found is None else settled(found, bounds)


def settled(value: Any, bounds: Sequence[Any]) -> Any:
    """``value`` with each case of it that holds conditions decided where the bounds decide
    them: the first whose condition holds, past those whose do not, where it is a value,
    not an integral left undone."""

    def branch(piecewise: Any) -> Any:
        for expr, condition in piecewise.args:
            verdict = holds(condition, bounds)
            if verdict:
                return piecewise if _undone(expr) else expr
            if verdict is None:
                return piecewise
        return piecewise

    if not isinstance(value, sympy.Basic) or not value.has(sympy.Piecewise):
        return value
    return value.replace(lambda e: isinstance(e, sympy.Piecewise), branch)


def _conditional(value: Any) -> bool:
    """Whether a result holds what an assumption books make would remove: conditions on
    its parameters, an integral left undone, or a modulus or a conjugate of a symbol."""
    if not isinstance(value, sympy.Basic):
        return False
    if _undone(value) or any(_on_conditions(p) for p in value.atoms(sympy.Piecewise)):
        return True
    return any(each.free_symbols for each in value.atoms(*_COMPLEX_PLANE))


def _undone(value: Any) -> bool:
    """Whether a value holds an integral or a sum SymPy left undone."""
    return isinstance(value, sympy.Basic) and value.has(sympy.Integral, sympy.Sum)


def _on_conditions(piecewise: Any) -> bool:
    """Whether a piecewise value is a result under conditions SymPy puts on its parameters,
    as an integral's ``|arg a| < π/2`` or a geometric series' ``|e^{-a}| < 1``, or that a
    parameter is not 0, as of the integral of :math:`\\sin^2(\\pi x/a)` over the width a."""
    return any(
        condition.has(sympy.arg, sympy.re, sympy.im, sympy.polar_lift, sympy.Abs)
        or any(isinstance(part, sympy.Ne) for part in sympy.And.make_args(condition))
        for _, condition in piecewise.args
    )


def _as_books_take(compute: Callable[..., Any], f: Any, limits: Sequence[Any], found: Any) -> Any:
    """An integral or a sum that holds conditions, or a modulus, or is left undone, under the
    assumption books make, with a remark that says so; as it was where none removes them.
    A value under conditions is the case that holds with the fewest parameters positive;
    one with a modulus of a parameter, or left undone, is computed again with those
    parameters positive and the variable real."""
    variables = [limit[0] if isinstance(limit, tuple) else limit for limit in limits]
    written = f.free_symbols.union(*(sympy.sympify(limit).free_symbols for limit in limits))
    parameters = sorted(
        (s for s in written - set(variables) if isinstance(s, Symbol) and _general(s)), key=str
    )
    written_cases = {c for p in f.atoms(sympy.Piecewise) for _, c in p.args}
    pieces = [
        p
        for p in found.atoms(sympy.Piecewise)
        if _on_conditions(p) and not {c for _, c in p.args} <= written_cases
    ]
    if pieces:
        chosen = _positive_for(pieces, parameters)
        if chosen is None:
            return found
        positive = {s: sympy.Dummy(s.name, positive=True) for s in chosen}
        again = _cases_taken(compute(f.xreplace(positive), *_put(limits, positive)))
        if _conditional(again):  # SymPy's own value under the assumption, else the case
            again = _cases_taken(found.xreplace(positive))
        return _assumed(again, positive)
    moduli = {s for each in [*f.atoms(*_COMPLEX_PLANE), *found.atoms(*_COMPLEX_PLANE)]
              for s in each.free_symbols}  # fmt: skip
    positive = {s: sympy.Dummy(s.name, positive=True) for s in parameters if s in moduli}
    real = {
        v: sympy.Dummy(v.name, real=True)
        for v, limit in zip(variables, limits, strict=True)
        if isinstance(v, Symbol) and _general(v) and _over_real_limits(limit, positive)
    }
    if not positive and not (real and _undone(found)):
        return found
    assumed = {**positive, **real}
    again = compute(f.xreplace(assumed), *_put(limits, assumed))
    if _undone(again) or any(_on_conditions(p) for p in again.atoms(sympy.Piecewise)):
        return found
    return _assumed(again, positive, real)


def _general(symbol: Symbol) -> bool:
    """Whether books would assume something of a symbol the program says nothing of: not
    of one it said what it is, which decides, nor of z, which complex analysis keeps
    complex."""
    said = getattr(symbol, "_assumptions_orig", None) or isinstance(symbol, sympy.Dummy)
    return not said and symbol.name not in _COMPLEX_VARIABLES


def _over_real_limits(limit: Any, positive: Mapping[Any, Any]) -> bool:
    """Whether an integral's limits are real, as ``(x, 0, a)`` is of a positive ``a``."""
    if not isinstance(limit, tuple) or len(limit) != 3:
        return False
    return all(sympy.sympify(end).xreplace(positive).is_extended_real for end in limit[1:])


def _put(limits: Sequence[Any], symbols: Mapping[Any, Any]) -> list[Any]:
    return [
        tuple(sympy.sympify(part).xreplace(symbols) for part in limit)
        if isinstance(limit, tuple)
        else sympy.sympify(limit).xreplace(symbols)
        for limit in limits
    ]


def _positive_for(pieces: Sequence[Any], parameters: Sequence[Symbol]) -> list[Symbol] | None:
    """The fewest parameters which, taken positive, make a case of each piecewise value
    hold that is a value, not an integral left undone; ``None`` where no choice does."""
    involved = sorted(
        {s for p in pieces for _, c in p.args for s in c.free_symbols} & set(parameters), key=str
    )[:_MOST_TRIED]
    for size in range(1, len(involved) + 1):
        for chosen in itertools.combinations(involved, size):
            positive = {s: sympy.Dummy(s.name, positive=True) for s in chosen}
            cases = [_first_holding(p.xreplace(positive)) for p in pieces]
            if all(case is not None and not _undone(case) for case in cases):
                return list(chosen)
    return None


def _cases_taken(value: Any) -> Any:
    """``value`` with each piecewise value in it that has a case that holds as that case."""

    def taken(piecewise: Any) -> Any:
        case = _first_holding(piecewise)
        return piecewise if case is None else case

    return value.replace(lambda e: isinstance(e, sympy.Piecewise), taken)


def _first_holding(piecewise: Any) -> Any:
    """The value of the first case of a piecewise value whose condition holds, past those
    whose conditions do not; ``None`` where one before it is not decided."""
    if not isinstance(piecewise, sympy.Piecewise):
        return piecewise
    for expr, condition in piecewise.args:
        verdict = holds(condition, ())
        if verdict:
            return expr
        if verdict is None:
            return None
    return None


def _assumed(
    value: Any,
    positive: Mapping[Any, Any],
    real: Mapping[Any, Any] | None = None,
    away: Mapping[Any, Any] | None = None,
) -> Any:
    """A value computed with ``positive`` parameters, ``real`` variables and parameters
    real and ``away`` from 0, as SymPy's own symbols again, and the remark that it holds
    assuming them."""
    back = {d: s for s, d in [*positive.items(), *(real or {}).items(), *(away or {}).items()]}
    for assumed, form, books in (
        (positive, "{} > 0", (", as books take a scale to be", ", as books take scales to be")),
        (away or {}, "{} ≠ 0, real", ("", "")),
    ):
        if not assumed:
            continue
        one = len(assumed) == 1
        shown = [display.unicode(s) for s in assumed]
        said = [form.format(s) for s in shown]
        remark(
            f"assuming {listed(said)}{books[0] if one else books[1]}; the program does not "
            f"say what {listed(shown)} {'is' if one else 'are'}",
            assuming=said,
        )
    return value.xreplace(back)


def limited(
    take: Callable[[Any, Any, Any, str], Any], body: Any, variable: Any, point: Any, direction: str
) -> Any:
    """A limit ``take`` cannot take for not knowing the sign of a parameter, under the
    assumption books make, with a remark that says so: the variable going to infinity
    positive, and the parameters real and away from 0, as the heat kernel goes to 0 where
    x is not 0, or else the fewest of them positive, tried in turn; ``None`` where none of
    these settles it."""
    if not isinstance(body, sympy.Basic) or not isinstance(variable, Symbol):
        return None
    parameters = sorted(
        (s for s in body.free_symbols - {variable} if isinstance(s, Symbol) and _general(s)),
        key=str,
    )[:_MOST_TRIED]
    real: dict[Any, Any] = {}
    if _general(variable) and point in (sympy.oo, -sympy.oo):
        real = {variable: sympy.Dummy(variable.name, positive=point == sympy.oo, real=True)}
    away = {s: sympy.Dummy(s.name, real=True, nonzero=True) for s in parameters}
    tried = [list(c) for size in (1, 2) for c in itertools.combinations(parameters, size)]
    choices = [{s: sympy.Dummy(s.name, positive=True) for s in chosen} for chosen in tried]
    if len(parameters) > 2:
        choices.append({s: sympy.Dummy(s.name, positive=True) for s in parameters})
    across = real.get(variable, variable)
    for assumed in [away, *choices] if parameters else []:
        try:
            found = take(body.xreplace({**assumed, **real}), across, point, direction)
        except (NotImplementedError, ValueError):
            continue
        if isinstance(found, sympy.Basic) and not found.has(sympy.Limit):
            if assumed is away:
                return _assumed(found, {}, real, away)
            return _assumed(found, assumed, real)
    return None
