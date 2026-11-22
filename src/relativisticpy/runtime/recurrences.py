"""The members of a sequence defined by a recurrence, as ``a_{n+2} := a_{n+1} + a_n`` with
its first members given (design/notation.md, decision 5.2): at a whole number, computed one
after another from those given; at any other point, in closed form where SymPy's ``rsolve``
finds one within a time budget that agrees with the members computed; and otherwise a
member that is computed once its point is a whole number, as the terms of a finite sum
are."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

import sympy
from sympy.core.function import AppliedUndef
from sympy.core.symbol import Str

from relativisticpy.runtime.core import _budget, _OverBudget

#: How many members are computed one after another at most.
_MEMBERS = 10_000

#: How long SymPy may look for a closed form, in seconds.
_CLOSED_FORM_BUDGET = 5.0

#: How many members after those given a closed form is checked against.
_CHECKED = 3


def member(body: Any, label: Any, point: Any, *given: Any, name: str, shift: int) -> Any:
    """The member at ``point`` of the sequence whose members, written as the function
    ``name``, keep ``name(label + shift) = body``, and whose members ``given`` are pairs of
    a whole number and its value."""
    variable = sympy.Dummy(str(label), integer=True)  # not the variable of a sum around it
    step = sympy.Lambda(variable, sympy.sympify(body).xreplace({label: variable}))
    pairs = sympy.Tuple(*(sympy.Tuple(k, v) for k, v in zip(given[::2], given[1::2], strict=True)))
    return Member(point, Str(name), sympy.Integer(shift), step, pairs)


class Member(sympy.Function):  # type: ignore[misc]
    """``a_m`` of a sequence defined by a recurrence: ``Member(point, name, shift, step,
    given)``, where ``step`` maps the label n to the value of the member at n + shift, written
    with ``name`` applied to the members it needs, and ``given`` holds the first members.
    At a whole number it is that member's value; elsewhere, its closed form, if SymPy finds
    one that agrees with the members, or else it stays as it is, to be computed once its
    point is a whole number."""

    @classmethod
    def eval(cls, point: Any, name: Any, shift: Any, step: Any, given: Any) -> Any:
        sequence = _sequence(str(name), int(shift), step, given)
        if point.is_Integer:
            return sequence.at(int(point))
        if point.is_number:
            raise ValueError(f"{sequence.shown(point)} is no member: {point} is no whole number")
        closed = sequence.closed_form()
        return closed(point) if closed is not None else None

    def _relativisticpy_plain(self, printer: Any) -> str:
        return _shown_member(self, printer, latex=False)

    def _sympystr(self, printer: Any) -> str:
        return _shown_member(self, printer, latex=False)

    def _latex(self, printer: Any, exp: Any = None) -> str:
        shown = _shown_member(self, printer, latex=True)
        return f"{shown}^{{{printer._print(exp)}}}" if exp is not None else shown


def _shown_member(member: Member, printer: Any, *, latex: bool) -> str:
    """``a_{n + 1}``: a member as books write it, by its family's name and its point."""
    base = str(member.args[1]).partition("_")[0]
    point = printer._print(member.args[0])
    return f"{base}_{{{point}}}" if latex or len(point) > 1 else f"{base}_{point}"


@functools.lru_cache(maxsize=64)
def _sequence(name: str, shift: int, step: Any, given: Any) -> _Sequence:
    """The sequence of these, kept so that its members and its closed form are computed once,
    however often SymPy makes a member of it again."""
    return _Sequence(name, shift, step, given)


class _Sequence:
    """A sequence defined by a recurrence: its members computed one after another, each
    once, and its closed form."""

    def __init__(self, name: str, shift: int, step: Any, given: Any) -> None:
        self.name, self.shift, self.step = name, shift, step
        self.function = sympy.Function(name)
        self.given: dict[int, Any] = {int(k): v for k, v in given}
        self.known: dict[int, Any] = dict(self.given)
        self._closed: sympy.Lambda | bool | None = False  # False: not looked for yet

    def shown(self, point: Any) -> str:
        return f"{self.name.partition('_')[0]}_{{{point}}}"

    def at(self, point: int) -> Any:
        """The member at a whole number: one given, or computed from those before it."""
        known = self.known
        if point in known:
            return known[point]
        if not known or point < min(known):
            first = min(known) if known else 0
            raise ValueError(
                f"{self.shown(point)} comes before the first member given, {self.shown(first)}"
            )
        start = max(known) + 1
        if point - start >= _MEMBERS:
            raise ValueError(
                f"{self.shown(point)} is {point - start + 1} members on from those given, more "
                f"than the {_MEMBERS} computed one after another"
            )
        for index in range(start, point + 1):
            known[index] = self._member(index, known)
        return known[point]

    def _member(self, index: int, known: Mapping[int, Any]) -> Any:
        """The member at ``index``, from the step, with the members it needs put in."""
        value = self.step(sympy.Integer(index - self.shift))
        if isinstance(value, sympy.Basic) and value.has(sympy.Sum, sympy.Product):
            value = value.doit()

        def known_member(application: Any) -> Any:
            (at,) = application.args
            if not at.is_Integer or int(at) not in known:
                raise ValueError(
                    f"{self.shown(index)} needs {self.shown(at)}, which is neither given nor "
                    "computed before it"
                )
            return known[int(at)]

        if isinstance(value, sympy.Basic):
            needed = {
                a: known_member(a) for a in value.atoms(AppliedUndef) if a.func == self.function
            }
            value = value.xreplace(needed)
        return value

    def closed_form(self) -> sympy.Lambda | None:
        """The member as a function of its point, where SymPy's ``rsolve`` finds one within
        its time budget that gives the members given and the next ones computed; ``None``
        otherwise."""
        if self._closed is False:
            self._closed = self._closed_form()
        return self._closed if isinstance(self._closed, sympy.Lambda) else None

    def _closed_form(self) -> sympy.Lambda | None:
        if not self.given:
            return None
        n = sympy.Dummy("n", integer=True)
        recurrence = self.function(n + self.shift) - self.step(n)
        initial = {self.function(k): v for k, v in self.given.items()}
        try:
            with _budget(_CLOSED_FORM_BUDGET):
                found = sympy.rsolve(recurrence, self.function(n), initial)
        except (_OverBudget, ValueError, NotImplementedError, TypeError, AttributeError):
            return None
        if found is None or (isinstance(found, sympy.Basic) and found.has(self.function)):
            return None
        last = max(self.given)
        points = [*self.given, *range(last + 1, last + 1 + _CHECKED)]
        try:
            for point in points:
                if sympy.simplify(found.subs(n, point) - self.at(point)) != 0:
                    return None
        except ValueError:
            return None
        return sympy.Lambda(n, found)
