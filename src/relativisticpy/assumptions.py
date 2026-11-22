r"""Assumptions, applied: what a program assumes of its names, put where it decides
something (design/notation.md, decisions 1 and 2).

A relation on names nothing defines, as ``a > 0`` or ``x \in \mathbb{R}``, is an assumption
for the rest of the program (:mod:`relativisticpy.reading.assumptions` reads it). What SymPy
keeps on a symbol, that it is positive, nonnegative, real or an integer, is put on the
symbol, and what was defined with the plain symbol takes it from then on. What SymPy cannot
keep, a bound as x < 1 in 0 < x < 1 or a > r, is kept in the session, and given to what it
decides: the solutions of ``solve``, the conditions of an integral, and inequalities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import sympy
from sympy import Symbol
from sympy.core.facts import InconsistentAssumptions

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

    return _each(value, put)


def _each(value: Any, put: Any) -> Any:
    """``value`` with ``put`` applied to each SymPy expression in it, through lists,
    tuples, dictionaries and arrays."""
    if isinstance(value, sympy.Basic):
        return put(value)
    if isinstance(value, sympy.NDimArray | sympy.MatrixBase):
        return value.applyfunc(put)
    if isinstance(value, list | tuple):
        return type(value)(_each(item, put) for item in value)
    if isinstance(value, dict):
        return {_each(key, put): _each(item, put) for key, item in value.items()}
    return value
