r"""Sequences, computed (design/notation.md, decision 31): sums over the index of a
sequence, the members of a sequence defined by a difference equation, at initial rest, and
the z-transforms of the sequences of a difference equation.

A sum over the index of a sequence is taken a term at a time
(:func:`relativisticpy.integrations.sequences.pieces`): an impulse leaves its term at the
one point where it is 1, and a step cuts its term's range to where it is 1; each part is
then summed as any sum is, in closed form where SymPy finds one within the time budget of
a sum, under the bounds the program assumes.

A sequence defined through its own members, as ``y[n] := \frac{1}{2} y[n-1] + \delta[n]``,
is at initial rest, as books assume: 0 before its input starts, the input being the rest of
its value, which its impulses and steps say where it starts. Its members at whole numbers
are computed one after another from there; at a symbol, in closed form where SymPy's
``rsolve`` finds one within a time budget: for the members after the input's last impulse,
from where its steps are 1, and those before as impulses, so that the closed form holds at
every integer.
"""

from __future__ import annotations

import functools
import itertools
from typing import Any

import sympy
from sympy.core.function import AppliedUndef
from sympy.core.symbol import Str

from relativisticpy import assumptions, display
from relativisticpy.integrations import sequences
from relativisticpy.integrations.sequences import impulse, step
from relativisticpy.runtime import core
from relativisticpy.runtime.core import _budget, _OverBudget

#: How many members are computed one after another at most.
_MEMBERS = 10_000

#: How many members after those the closed form starts from it is checked against.
_CHECKED = 4


# -- sums ------------------------------------------------------------------------------


def summation(
    body: Any, variable: Any, lower: Any, upper: Any, *, assuming: tuple[Any, ...] = ()
) -> Any:
    """A sum over the index of a sequence, each impulse and step of the index taken apart
    (see the module), and each part summed as :func:`relativisticpy.runtime.core.summation`
    sums; as that sums it where the body holds none, or has a whole number of terms."""
    index = (
        sympy.Symbol(str(variable), integer=True)
        if isinstance(variable, sympy.Symbol)
        else variable
    )
    written = sympy.sympify(body).xreplace({variable: index})
    ends = [sympy.sympify(end) for end in (lower, upper)]
    whole = all(end.is_Integer for end in ends)
    if whole or not sequences.of_sequences(written, index):
        return _tidied(core.summation(body, variable, lower, upper, assuming=assuming))
    try:
        with _budget(core._CLOSED_FORM_BUDGET):
            found = sequences.pieces(written, index, lower, upper)
            if found is None:
                return core.summation(body, variable, lower, upper, assuming=assuming)
            total = sympy.S.Zero
            for term, first, last in _merged(found):
                value = assumptions.summed(term, [(index, first, last)], assuming)
                total += sequences.emptied(value, first, last)
    except _OverBudget:
        core._past_the_budget("sum")
        return sympy.Sum(body, (variable, lower, upper))
    if total.has(sympy.nan) and not written.has(sympy.nan):
        assumptions.remark(
            "the sum of its parts has no value where its terms have one, so the sum is kept "
            "as written"
        )
        return sympy.Sum(body, (variable, lower, upper))
    return _general(assumptions.settled(total, assuming), written)


def _merged(pieces: list[tuple[Any, Any, Any]]) -> list[tuple[Any, Any, Any]]:
    """The parts of a sum (see :func:`relativisticpy.integrations.sequences.pieces`), with
    each two of opposite terms over ranges that share an end taken together, as the range
    one has and the other has not: u[n] - u[n - 4] is 1 from 0 to 3, not 1 from 0 on less 1
    from 4 on, two sums without end whose difference has no value."""
    kept = list(pieces)
    joined = True
    while joined:
        joined = False
        for one, other in itertools.combinations(kept, 2):
            together = _together(one, other)
            if together is not None:
                kept = [each for each in kept if each is not one and each is not other]
                kept.append(together)
                joined = True
                break
    return kept


def _together(
    one: tuple[Any, Any, Any], other: tuple[Any, Any, Any]
) -> tuple[Any, Any, Any] | None:
    """Two parts of opposite terms over ranges that share an end, as one part over the
    range one has and the other has not; ``None`` of any others."""
    (term, first, last), (opposite, start, end) = one, other
    if sympy.expand(term + opposite) != 0:
        return None
    if last == end and (first - start).is_number:  # from a on, less from b on
        return (term, first, start - 1) if first <= start else (opposite, start, first - 1)
    if first == start and (last - end).is_number:  # up to a, less up to b
        return (term, end + 1, last) if end <= last else (opposite, last + 1, end)
    return None


def _general(value: Any, body: Any) -> Any:
    """A sum whose closed form holds a case of its own where its parameters are equal, as
    the convolution of a^n and b^n does where a = b: the general case, as books give it,
    with a remark that says what is assumed, and what the sum is in the other case; not a
    case its ``body`` writes itself."""
    written = {c for each in body.atoms(sympy.Piecewise) for _, c in each.args}

    def general(piecewise: Any) -> Any:
        *cases, (last, otherwise) = piecewise.args
        conditions = [condition for _, condition in cases]
        if otherwise is not sympy.true or written & set(conditions):
            return piecewise
        if not all(isinstance(condition, sympy.Eq) for condition in conditions):
            return piecewise
        for case, condition in cases:
            apart = f"{display.unicode(condition.lhs)} ≠ {display.unicode(condition.rhs)}"
            equal = f"{display.unicode(condition.lhs)} = {display.unicode(condition.rhs)}"
            assumptions.remark(
                f"assuming {apart}, as the sum takes the case {equal} apart: there it is "
                f"{display.unicode(case)}",
                assuming=[apart],
            )
        return last

    if not isinstance(value, sympy.Basic) or not value.has(sympy.Piecewise):
        return value
    return value.replace(lambda e: isinstance(e, sympy.Piecewise), general)


def _tidied(value: Any) -> Any:
    """A sum of numbers with exponentials of i in it, as a bin of a discrete Fourier
    transform is, written out where that is shorter: 4, not 2 + √2 e^{-iπ/4}/2 + ...."""
    if isinstance(value, sympy.Expr) and value.is_number and value.has(sympy.exp):
        expanded = sympy.expand_complex(value)
        if sympy.count_ops(expanded) <= sympy.count_ops(value):
            return expanded
    return value


# -- circular convolution -----------------------------------------------------------------


def circular(first: Any, second: Any, label: Any, point: Any, *length: Any) -> Any:
    """The circular convolution at ``point`` of two sequences, written as expressions of
    ``label``: the sum over m from 0 to N - 1 of the first at m times the second at
    (point - m) mod N, where N is the ``length`` given, or else that of the longer sequence,
    from 0 to its last member that is not 0."""
    first, second = sympy.sympify(first), sympy.sympify(second)
    if length:
        size = sympy.sympify(length[0])
        if not (size.is_Integer and size > 0):
            raise ValueError(f"the length of a circular convolution is a whole number, not {size}")
        count = int(size)
    else:
        count = max(_length(first, label), _length(second, label))
    total = sympy.S.Zero
    for m in range(count):
        at = sympy.Mod(sympy.sympify(point) - m, count)
        total += first.xreplace({label: m}) * second.xreplace({label: at})
    return total


def _length(sequence: Any, label: Any) -> int:
    """How many members a sequence has, from 0 to its last member that is not 0, as its
    impulses and steps say; refused for one that starts before 0, or never ends."""
    shown = display.unicode(sequence)
    end = _settled_from(sequence, label)
    if end is None or sympy.simplify(_smoothed(sequence, label, end)) != 0:
        raise ValueError(
            f"{shown} does not end, so its circular convolution needs its length: write it "
            "as the subscript, as in x \\circledast_N h"
        )
    start = _start(sequence, label)
    if start is not None and start < min(end, 0):
        raise ValueError(f"{shown} starts before 0, where a circular convolution starts")
    return max(end, 0)


# -- difference equations, at initial rest ------------------------------------------------


def member(body: Any, label: Any, point: Any, *, name: str, shift: int) -> Any:
    """The member at ``point`` of the sequence whose members, written as the function
    ``name``, keep ``name(label + shift) = body``, at initial rest."""
    variable = sympy.Dummy(str(label), integer=True)
    rule = sympy.Lambda(variable, sympy.sympify(body).xreplace({label: variable}))
    return AtRest(point, Str(name), sympy.Integer(shift), rule)


def solved(difference: Any, label: Any, *, name: str) -> list[Any]:
    """The solution of a difference equation in the sequence ``name``, written as the
    difference of its sides, at initial rest: its member at the label, in closed form where
    SymPy finds one."""
    function = sympy.Function(name)
    members = [a for a in sympy.sympify(difference).atoms(AppliedUndef) if a.func == function]
    shifts = {each: sympy.simplify(each.args[0] - label) for each in members}
    if not members or not all(shift.is_Integer for shift in shifts.values()):
        raise ValueError(f"this is no difference equation in {name}, of {label}")
    newest = max(members, key=lambda each: int(shifts[each]))
    found = sympy.solve(difference, newest)
    if len(found) != 1:
        raise ValueError(f"the difference equation does not give {name}[{newest.args[0]}]")
    return [member(found[0], label, label, name=name, shift=int(shifts[newest]))]


class AtRest(sympy.Function):  # type: ignore[misc]
    """``y[m]`` of a sequence defined by a difference equation, at initial rest:
    ``AtRest(point, name, shift, rule)``, where ``rule`` maps the label n to the member at
    n + shift, written with ``name`` applied to the members it needs. At a whole number it is
    that member's value; elsewhere its closed form, where SymPy finds one, or else it stays
    as it is, to be computed once its point is a whole number, as the terms of a sum are."""

    @classmethod
    def eval(cls, point: Any, name: Any, shift: Any, rule: Any) -> Any:
        sequence = _rested(str(name), int(shift), rule)
        if point.is_Integer:
            return sequence.at(int(point))
        if point.is_number:
            raise ValueError(f"{sequence.shown(point)} is no member: {point} is no whole number")
        closed = sequence.closed_form()
        return closed(point) if closed is not None else None

    def _relativisticpy_plain(self, printer: Any) -> str:
        return f"{self.args[1]}[{printer._print(self.args[0])}]"

    def _sympystr(self, printer: Any) -> str:
        return f"{self.args[1]}[{printer._print(self.args[0])}]"

    def _latex(self, printer: Any, exp: Any = None) -> str:
        shown = f"{self.args[1]}\\left[{printer._print(self.args[0])}\\right]"
        return f"{shown}^{{{printer._print(exp)}}}" if exp is not None else shown


@functools.lru_cache(maxsize=64)
def _rested(name: str, shift: int, rule: Any) -> _Rest:
    """The sequence of these, kept so that its members and its closed form are computed
    once, however often SymPy makes a member of it again."""
    return _Rest(name, shift, rule)


class _Rest:
    """A sequence defined by a difference equation, at initial rest: its members, each
    computed once, and its closed form."""

    def __init__(self, name: str, shift: int, rule: Any) -> None:
        self.name, self.shift, self.rule = name, shift, rule
        self.function = sympy.Function(name)
        label = rule.variables[0]
        point = sympy.Dummy("m", integer=True)
        written = rule.expr.xreplace({label: point - shift})  # the member at m, of m
        members = [a for a in written.atoms(AppliedUndef) if a.func == self.function]
        lags = [sympy.simplify(point - each.args[0]) for each in members]
        if any(not lag.is_Integer or lag <= 0 for lag in lags):
            raise ValueError(
                f"{self.shown(point)} is defined through members at or after it, which a "
                "difference equation at rest cannot compute"
            )
        self.point, self.written = point, written
        self.order = max((int(lag) for lag in lags), default=0)
        self.input = written.xreplace(dict.fromkeys(members, sympy.S.Zero))
        self.start = _start(self.input, point)
        self.known: dict[int, Any] = {}
        self._closed: sympy.Lambda | bool | None = False  # False: not looked for yet

    def shown(self, point: Any) -> str:
        return f"{self.name}[{point}]"

    def at(self, point: int) -> Any:
        """The member at a whole number: 0 before the input starts, and computed one after
        another from there."""
        start = self.first()
        if point < start:
            return sympy.S.Zero
        if point - start >= _MEMBERS:
            raise ValueError(
                f"{self.shown(point)} is {point - start + 1} members on from where its input "
                f"starts, more than the {_MEMBERS} computed one after another"
            )
        for index in range(start, point + 1):
            if index not in self.known:
                self.known[index] = self._member(index)
        return self.known[point]

    def first(self) -> int:
        """Where the input starts: the first member that is not 0 at rest."""
        if self.start is None:
            raise ValueError(
                f"{self.shown('n')} is defined by a difference equation at rest, 0 before its "
                "input starts, but its input never starts: it is not 0 before any n"
            )
        return self.start

    def _member(self, index: int) -> Any:
        value = self.written.xreplace({self.point: index})

        def earlier(application: Any) -> Any:
            (at,) = application.args
            return self.at(int(at))

        needed = {a: earlier(a) for a in value.atoms(AppliedUndef) if a.func == self.function}
        value = value.xreplace(needed)
        return sympy.simplify(value) if isinstance(value, sympy.Basic) else value

    def closed_form(self) -> sympy.Lambda | None:
        """The member as a function of its point, at every integer, where SymPy's
        ``rsolve`` finds one within the time budget of a sum; ``None`` otherwise."""
        if self._closed is False:
            try:
                with _budget(core._CLOSED_FORM_BUDGET):
                    self._closed = self._closed_form()
            except (_OverBudget, ValueError, NotImplementedError, TypeError, AttributeError):
                self._closed = None
        return self._closed if isinstance(self._closed, sympy.Lambda) else None

    def _closed_form(self) -> sympy.Lambda | None:
        if self.start is None:
            return None
        m = self.point
        settled = _settled_from(self.input, m)
        if settled is None:
            return None
        begin = max(self.start, settled)
        smooth = sympy.nsimplify(_smoothed(self.written, m, begin), rational=True)
        if self.order == 0:
            found = smooth
        else:
            initial = {
                self.function(begin - k): sympy.nsimplify(self.at(begin - k), rational=True)
                for k in range(1, self.order + 1)
            }
            found = sympy.rsolve(self.function(m) - smooth, self.function(m), initial)
        if found is None or (isinstance(found, sympy.Basic) and found.has(self.function)):
            return None
        last = begin + self.order + _CHECKED
        agrees = [
            sympy.simplify(found.xreplace({m: k}) - self.at(k)) == 0
            for k in range(self.start, last + 1)
        ]
        if not all(agrees[begin - self.start :]):
            return None
        held = begin  # the first member the closed form gives, those before it as impulses
        while held > self.start and agrees[held - 1 - self.start]:
            held -= 1
        value = found * step(m - held)
        for k in range(self.start, held):
            value += self.at(k) * impulse(m - k)
        return sympy.Lambda(m, value)


def _start(expr: Any, point: Any) -> int | None:
    """Where an input built of impulses and steps starts: the first whole number at which it
    may not be 0, as 0 is of δ[n] + u[n - 2]; a large number where it is 0 everywhere, and
    ``None`` where it does not start, as an input without a step or an impulse does not."""
    starts = []
    for term in sympy.Add.make_args(sympy.expand_mul(sympy.sympify(expr))):
        if term == 0:
            continue
        begins = []
        for factor in sympy.Mul.make_args(term):
            if not (isinstance(factor, sequences.MEMBERS) and factor.has(point)):
                continue
            found = sequences.linear(factor.args[0], point)
            if found is None:
                return None
            coefficient, rest = found
            if isinstance(factor, step) and coefficient < 0:
                continue  # u[c - n] ends: it says nothing of where its term starts
            begins.append(-rest / coefficient)
        if not begins or not all(each.is_number for each in begins):
            return None
        starts.append(int(sympy.ceiling(max(begins))))
    return min(starts) if starts else _MEMBERS


def _settled_from(expr: Any, point: Any) -> int | None:
    """From where an input's impulses are all 0 and its steps all 1 or 0 for good: past its
    last impulse, and from where its last step starts or ends; ``None`` where that is not a
    whole number."""
    ends: list[Any] = []
    for factor in sympy.sympify(expr).atoms(*sequences.MEMBERS):
        if not factor.has(point):
            continue
        found = sequences.linear(factor.args[0], point)
        if found is None:
            return None
        coefficient, rest = found
        at = -rest / coefficient
        ends.append(at + 1 if isinstance(factor, impulse) or coefficient < 0 else at)
    if not all(end.is_number for end in ends):
        return None
    return int(sympy.ceiling(max(ends))) if ends else -_MEMBERS


def _smoothed(expr: Any, point: Any, begin: int) -> Any:
    """An expression of a point from ``begin`` on, where its impulses are 0 and its steps 1
    or 0, as they are from there."""

    def settled(factor: Any) -> Any:
        if not factor.has(point):
            return factor
        if isinstance(factor, impulse):
            return sympy.S.Zero
        coefficient, _ = sequences.linear(factor.args[0], point) or (1, 0)
        return sympy.S.One if coefficient > 0 else sympy.S.Zero

    return sympy.sympify(expr).replace(lambda e: isinstance(e, sequences.MEMBERS), settled)


# -- the z-transforms of a difference equation's sequences ---------------------------------


def capital(name: str) -> str:
    """The capital letter books write a sequence's transform with: Y of y, X_1 of x_1."""
    return name[:1].upper() + name[1:]


def transformed(equation: Any, z: Any, *, name: str, output: str, label: str) -> Any:
    """The z-transform at z of the sequence ``name`` of a difference equation, with
    constant coefficients, of the sequences nothing defines, written as functions of the
    label: a delay of k samples is z^{-k}, as books transform it, the sequences at rest
    before they start. The output's transform is the transfer function times the inputs',
    which the equation gives; an input's is its capital letter at z, as X(z)."""
    if not isinstance(equation, sympy.Eq):
        raise ValueError(f"the difference equation of {name} holds identically, or never")
    symbols = [s for s in equation.free_symbols if isinstance(s, sympy.Symbol) and s.name == label]
    n = symbols[0] if symbols else sympy.Symbol(label)
    difference = equation.lhs - equation.rhs
    delayed: dict[Any, Any] = {}
    for applied in difference.atoms(AppliedUndef):
        (argument,) = applied.args
        lag = sympy.simplify(argument - n)
        if not lag.is_Integer:
            raise ValueError(
                f"{applied.func.__name__}[{argument}] is no delay of a sequence of {label}"
            )
        delayed[applied] = z**lag * sympy.Function(capital(applied.func.__name__))(z)
    found = sympy.expand(difference.xreplace(delayed))
    if found.has(n):
        raise ValueError(
            "the difference equation's coefficients depend on "
            f"{label}, so it has no transfer function"
        )
    transform = sympy.Function(capital(name))(z)
    if name != output:
        return transform
    unknown = sympy.Function(capital(output))(z)
    solved = sympy.solve(found, unknown)
    if len(solved) != 1:
        raise ValueError(f"the difference equation does not give {capital(output)}({z})")
    return sympy.simplify(solved[0])
