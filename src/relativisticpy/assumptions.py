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
DECIDED = frozenset({"solve", "core.lt", "core.gt", "core.le", "core.ge"})
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
    if isinstance(decided, Relational) and not isinstance(decided, sympy.Eq | sympy.Ne):
        difference = sympy.factor(sympy.together(decided.lhs - decided.rhs))
        settled = type(decided)(difference, 0)
        if settled in (sympy.true, sympy.false):
            return bool(settled)
    return None


def within(bounds: Sequence[Any]) -> dict[Any, Any]:
    """Each symbol a bound limits, as an expression of new positive symbols whose values are
    exactly those the bounds and its own assumptions allow, as far as that can be written:
    x > 1 is 1 + p, a > r is r + p, and a positive x < 1 is 1/(1 + p). Where it cannot be,
    the values written for are more than are allowed, never fewer, so that what holds of
    them holds wherever the bounds do."""
    lower: dict[Symbol, list[tuple[Any, bool]]] = {}
    upper: dict[Symbol, list[tuple[Any, bool]]] = {}
    for bound in bounds:
        if not isinstance(bound, Relational) or isinstance(bound, sympy.Eq | sympy.Ne):
            continue
        small, large = (
            (bound.lhs, bound.rhs) if bound.rel_op in ("<", "<=") else (bound.rhs, bound.lhs)
        )
        strict = bound.rel_op in ("<", ">")
        if isinstance(large, Symbol):
            lower.setdefault(large, []).append((small, strict))
        if isinstance(small, Symbol) and not isinstance(large, Symbol):  # x < a is a > x
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


def _between(lower: list[tuple[Any, bool]], upper: list[tuple[Any, bool]], name: Symbol) -> Any:
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


def _step(name: Symbol, *, strict: bool) -> Any:
    """A new symbol, positive or else nonnegative, named after ``name``."""
    return (
        sympy.Dummy(name.name, positive=True)
        if strict
        else sympy.Dummy(name.name, nonnegative=True)
    )


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
