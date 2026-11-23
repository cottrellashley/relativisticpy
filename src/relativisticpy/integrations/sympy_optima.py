r"""Optima, exactly, with SymPy.

The optimum of a function of one variable from its stationary points, the points where it
has no derivative and the ends of its domain; of several, from the Lagrange and the KKT
conditions, at each set of constraints that hold with equality, the stationary points
classified by the Hessian; a linear program by SymPy's simplex method; and an optimum over
finitely many points by trying each. Each says whether the optimum it finds is global: of
a convex problem, or over a bounded region, or one that compares every candidate.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import sympy
from sympy import Symbol
from sympy.functions.elementary.hyperbolic import (
    HyperbolicFunction,
    InverseHyperbolicFunction,
)
from sympy.functions.elementary.trigonometric import (
    InverseTrigonometricFunction,
    TrigonometricFunction,
)

from relativisticpy import assumptions
from relativisticpy.errors import EvaluationError
from relativisticpy.integrations import Example, Found, Problem, optimizer
from relativisticpy.integrations._optimizers import Undecided

x, y, z = sympy.symbols("x y z", real=True)
n = sympy.Symbol("n", integer=True, nonnegative=True)
a, b = sympy.symbols("a b")
k = sympy.Symbol("k", positive=True)

#: The most points an optimum over finitely many is tried at.
_MOST_POINTS = 4096
#: The most variables, and constraints that may hold with equality, the KKT conditions are
#: written for.
_MOST_VARIABLES = 8
_MOST_ACTIVE = 10
#: The most smooth pieces a function with a modulus or extremes is taken apart into.
_MOST_PIECES = 16
#: The most candidates compared with each other, for the same point written two ways.
_FEW = 64


# -- comparing values ------------------------------------------------------------------------


def _sign(value: Any, bounds: Sequence[Any] = ()) -> int | None:
    """The sign of a value, 1, 0 or -1, as its symbols' facts and the bounds the program
    assumes decide it; ``None`` where they do not. Numbers decide to rounding error."""
    value = sympy.sympify(value)
    if value.has(sympy.nan, sympy.zoo):
        return None
    if value.is_number:
        if value.is_infinite:
            return 1 if value.is_extended_positive else -1
        number = complex(sympy.N(value, 40))
        if abs(number.imag) > 1e-12 * max(1.0, abs(number.real)):
            return None
        if abs(number.real) <= 1e-12 * max(1.0, _size(value)):
            return 0
        return 1 if number.real > 0 else -1
    simpler = sympy.simplify(value)
    for verdict, sign in (
        (simpler.is_zero, 0),
        (simpler.is_positive, 1),
        (simpler.is_negative, -1),
    ):
        if verdict:
            return sign
    if bounds:
        if assumptions.holds(sympy.Gt(simpler, 0), bounds):
            return 1
        if assumptions.holds(sympy.Lt(simpler, 0), bounds):
            return -1
    return None


def _size(value: Any) -> float:
    return max(
        (abs(complex(sympy.N(term))) for term in sympy.Add.make_args(sympy.expand(value))),
        default=1.0,
    )


def _compare(one: Any, other: Any, bounds: Sequence[Any] = ()) -> int | None:
    """Whether one value is above another, 1, the same, 0, or below, -1."""
    one, other = sympy.sympify(one), sympy.sympify(other)
    if one == other:
        return 0
    if one.is_infinite or other.is_infinite:
        if one.is_infinite and other.is_infinite:
            return 0 if one == other else (1 if one.is_extended_positive else -1)
        return (
            (1 if one.is_extended_positive else -1)
            if one.is_infinite
            else (-1 if other.is_extended_positive else 1)
        )
    return _sign(one - other, bounds)


def _at_least(one: Any, other: Any, bounds: Sequence[Any] = ()) -> bool | None:
    """Whether one value is at least another, as their symbols' facts and the bounds the
    program assumes decide it: ``None`` where they do not."""
    compared = _compare(one, other, bounds)
    if compared is not None:
        return compared >= 0
    difference = sympy.simplify(sympy.sympify(one) - sympy.sympify(other))
    if difference.is_nonnegative:
        return True
    if difference.is_negative:
        return False
    if bounds and assumptions.holds(sympy.Ge(difference, 0), bounds):
        return True
    return None


@dataclass(frozen=True)
class _Candidate:
    """A value an optimum may be: at a ``point``, or, where it is not ``attained``, as a
    limit, toward an end of the domain or a point where the function is not defined. Where
    it is the value at infinitely many points, the ``family`` of them: a set, as of the
    points π/2 + 2πn, or a condition, as x² + y² = 1, of which the point is one."""

    value: Any
    point: tuple[Any, ...] | None
    attained: bool = True
    family: Any = None


def _best(
    candidates: Sequence[_Candidate], greatest: bool, bounds: Sequence[Any]
) -> tuple[_Candidate, list[_Candidate]] | None:
    """The greatest or least of the candidates, with those reached that share its value, in
    order (see :func:`_in_order`): the first of them, where one is reached; ``None`` where
    comparing them is not decided. A limit beats a point only where it is strictly
    better."""

    def beats(one: _Candidate, other: _Candidate) -> bool | None:
        if greatest:
            return _at_least(one.value, other.value, bounds)
        return _at_least(other.value, one.value, bounds)

    ordered = _in_order(candidates)
    front: list[_Candidate] = []  # those no other is known to be at least as good as
    for candidate in ordered:
        if any(beats(other, candidate) for other in front):
            continue
        front = [other for other in front if not beats(candidate, other)]
        front.append(candidate)
    if not front or any(_compare(front[0].value, other.value, bounds) != 0 for other in front):
        return None
    ties = [c for c in ordered if _compare(front[0].value, c.value, bounds) == 0]
    attained = [c for c in ties if c.attained]
    return (attained[0] if attained else front[0]), attained


def _in_order(candidates: Sequence[_Candidate]) -> list[_Candidate]:
    """The candidates without those at the same point twice, in an order that depends on
    nothing but them: by their points, the least first where they are numbers, then the
    limits. The first of several points that reach an optimum is so the same every time,
    whatever order SymPy found them in."""
    distinct = list({(c.value, c.point, c.attained, c.family): c for c in candidates}.values())
    if len(distinct) <= _FEW:  # the same point written two ways, as SymPy may find it twice
        kept: list[_Candidate] = []
        for candidate in distinct:
            twin = next((k for k, other in enumerate(kept) if _same(candidate, other)), None)
            if twin is None:
                kept.append(candidate)
            elif kept[twin].family is None:  # the one that stands for a family of them
                kept[twin] = candidate
        distinct = kept
    return sorted(distinct, key=_place)


def _same(one: _Candidate, other: _Candidate) -> bool:
    if one.point is None or other.point is None or one.attained != other.attained:
        return False
    return all(_compare(p, q) == 0 for p, q in zip(one.point, other.point, strict=True))


def _place(candidate: _Candidate) -> tuple[Any, ...]:
    """Where a candidate comes in order: points before limits, by their coordinates, as
    numbers where they are numbers."""
    if candidate.point is None:
        return (1, (), sympy.default_sort_key(sympy.sympify(candidate.value)))
    numbers = []
    for coordinate in candidate.point:
        coordinate = sympy.sympify(coordinate)
        if not coordinate.is_number or not coordinate.is_extended_real:
            return (0, (), sympy.default_sort_key(sympy.Tuple(*candidate.point)))
        numbers.append(float(sympy.N(coordinate, 30)))
    return (0, tuple(numbers), ())


# -- one variable ----------------------------------------------------------------------------


def _interval(constraint: Any, variable: Symbol, strict: bool) -> Any:
    """The values of a variable at which a constraint on it, ``g ≥ 0`` or ``g > 0``,
    holds: an interval where it is linear in it, with a number before it."""
    g = sympy.expand(constraint)
    slope = g.coeff(variable)
    rest = sympy.expand(g - slope * variable)
    if slope.is_number and slope != 0 and not rest.has(variable):
        end = -rest / slope
        if slope.is_positive:
            return sympy.Interval(end, sympy.oo, left_open=strict)
        return sympy.Interval(-sympy.oo, end, right_open=strict)
    relation = sympy.Gt(g, 0) if strict else sympy.Ge(g, 0)
    return sympy.solveset(relation, variable, sympy.S.Reals)


def _natural(f: Any, variable: Symbol) -> Iterator[tuple[Any, bool]]:
    """Where a function of a variable is real, as constraints on it, each with whether it
    is strict: what is under a root is not negative, and what a logarithm is of positive."""
    for power in sympy.sympify(f).atoms(sympy.Pow):
        base, exponent = power.args
        if base.has(variable) and exponent.is_rational and not exponent.is_integer:
            yield base, bool(exponent.is_negative)
    for logarithm in sympy.sympify(f).atoms(sympy.log):
        if logarithm.args[0].has(variable):
            yield logarithm.args[0], True


def _domain(problem: Problem, variable: Symbol) -> Any:
    """The values of the variable of a problem of one variable that its constraints allow,
    and at which what it is of is real, as a set of real numbers; ``None`` where it cannot
    be written as intervals and points, or its ends cannot be put in order."""
    bounds = problem.bounds
    sets: list[Any] = [_interval(g, variable, strict=False) for g in problem.inequalities]
    sets += [_interval(g, variable, strict=True) for g in problem.strict]
    sets += [_interval(g, variable, strict) for g, strict in _natural(problem.objective, variable)]
    sets += [sympy.FiniteSet(*_roots(h, variable)) for h in problem.equalities]
    domain: Any = sympy.S.Reals
    for each in sets:
        domain = _meet(domain, each, bounds)
        if domain is None:
            return None
    for g in problem.distinct:
        for point in _roots(g, variable):
            domain = _without(domain, point, bounds)
            if domain is None:
                return None
    return domain


def _meet(one: Any, other: Any, bounds: Sequence[Any]) -> Any:
    """Where two sets of real numbers meet, each intervals and points, with ends that may be
    written with parameters; ``None`` where that is not decided."""
    if not (_written_as_intervals(one) and _written_as_intervals(other)):
        return None
    parts = []
    for first in _parts(one):
        for second in _parts(other):
            met = _meet_parts(first, second, bounds)
            if met is None:
                return None
            if met is not sympy.S.EmptySet:
                parts.append(met)
    return _union(parts)


def _union(parts: Sequence[Any]) -> Any:
    kept = [part for part in parts if part is not sympy.S.EmptySet]
    if len(kept) > 1:
        return sympy.Union(*kept, evaluate=False)
    return kept[0] if kept else sympy.S.EmptySet


def _meet_parts(first: Any, second: Any, bounds: Sequence[Any]) -> Any:
    if isinstance(first, sympy.FiniteSet) or isinstance(second, sympy.FiniteSet):
        points, other = (first, second) if isinstance(first, sympy.FiniteSet) else (second, first)
        kept = []
        for point in points.args:
            inside = _within(point, other, bounds)
            if inside is None:
                return None
            if inside:
                kept.append(point)
        return sympy.FiniteSet(*kept)
    start, left = _end(first.start, first.left_open, second.start, second.left_open, 1, bounds)
    end, right = _end(first.end, first.right_open, second.end, second.right_open, -1, bounds)
    if start is None or end is None:
        return None
    order = _compare(start, end, bounds)
    if order is None:
        return None
    if order > 0 or (order == 0 and (left or right)):
        return sympy.S.EmptySet
    if order == 0:
        return sympy.FiniteSet(start)
    return sympy.Interval(start, end, left, right)


def _end(
    one: Any, open_one: bool, other: Any, open_other: bool, sign: int, bounds: Sequence[Any]
) -> tuple[Any, bool]:
    """The inner of two ends of intervals, the greater of two starts or the lesser of two
    ends, with whether it is open."""
    order = _compare(one, other, bounds)
    if order is None:
        return None, False
    if order == 0:
        return one, open_one or open_other
    return (one, open_one) if sign * order > 0 else (other, open_other)


def _within(point: Any, part: Any, bounds: Sequence[Any]) -> bool | None:
    """Whether a point belongs to an interval or a set of points."""
    if isinstance(part, sympy.FiniteSet):
        verdicts = [_compare(point, each, bounds) for each in part.args]
        if 0 in verdicts:
            return True
        return False if None not in verdicts else None
    low = _compare(point, part.start, bounds)
    high = _compare(point, part.end, bounds)
    if low is None or high is None:
        return None
    return (low > 0 or (low == 0 and not part.left_open)) and (
        high < 0 or (high == 0 and not part.right_open)
    )


def _without(domain: Any, point: Any, bounds: Sequence[Any]) -> Any:
    """A set of real numbers without a point: an interval it is inside in two, and one it
    ends open at it."""
    parts = []
    for part in _parts(domain):
        if isinstance(part, sympy.FiniteSet):
            kept = [p for p in part.args if _compare(p, point, bounds) != 0]
            parts.append(sympy.FiniteSet(*kept))
            continue
        inside = _inside(point, part, bounds)
        if inside is None:
            return None
        if inside:
            parts += [
                sympy.Interval(part.start, point, part.left_open, True),
                sympy.Interval(point, part.end, True, part.right_open),
            ]
            continue
        at_start, at_end = (_compare(point, end, bounds) == 0 for end in (part.start, part.end))
        if at_start or at_end:
            part = sympy.Interval(part.start, part.end, at_start or part.left_open,
                                  at_end or part.right_open)  # fmt: skip
        parts.append(part)
    return _union(parts)


def _written_as_intervals(domain: Any) -> bool:
    parts = domain.args if isinstance(domain, sympy.Union) else (domain,)
    return all(
        isinstance(part, sympy.Interval | sympy.FiniteSet) or part is sympy.S.EmptySet
        for part in parts
    )


def _parts(domain: Any) -> list[Any]:
    parts = domain.args if isinstance(domain, sympy.Union) else (domain,)
    return [part for part in parts if part is not sympy.S.EmptySet]


def _pieces(f: Any, variable: Symbol) -> list[Any]:
    """The smooth functions a function with a modulus, extremes or cases of the variable
    is made of, one for each choice of each: the stationary points of the function are
    among theirs."""
    rough = [
        each
        for each in sympy.sympify(f).atoms(sympy.Abs, sympy.Min, sympy.Max, sympy.Piecewise)
        if each.has(variable)
    ]
    if not rough:
        return [f]
    first = min(rough, key=lambda each: sympy.count_ops(each))
    if isinstance(first, sympy.Abs):
        choices = [first.args[0], -first.args[0]]
    elif isinstance(first, sympy.Piecewise):
        choices = [expr for expr, _ in first.args]
    else:
        choices = list(first.args)
    found: list[Any] = []
    for choice in choices:
        found.extend(_pieces(f.xreplace({first: choice}), variable))
        if len(found) > _MOST_PIECES:
            raise NotImplementedError("too many pieces")
    return found


@dataclass(frozen=True)
class _Zeros:
    """Where an expression of a variable is 0 within an interval: at ``points``, and at
    every point of each of the ``families``, sets of infinitely many, as π/2 + 2πn."""

    points: tuple[Any, ...] = ()
    families: tuple[Any, ...] = ()

    def __add__(self, other: _Zeros) -> _Zeros:
        return _Zeros(self.points + other.points, self.families + other.families)


def _stationary(piece: Any, variable: Symbol, part: Any, bounds: Sequence[Any]) -> _Zeros | None:
    """The stationary points of a smooth function of a variable within an interval: none
    of a straight one, whose optimum is at an end (see :func:`_zeros`)."""
    return _zeros(sympy.diff(piece, variable), variable, part, bounds)


def _zeros(expr: Any, variable: Symbol, part: Any, bounds: Sequence[Any]) -> _Zeros | None:
    """Where an expression of a variable is 0 within an interval, every point of it: those
    SymPy's solveset finds on the interval, the infinitely many of a periodic function, as
    cos x has, as families of them. Of a rational function, and of one written with
    parameters that is not periodic, whose solutions solveset leaves under conditions, the
    finitely many SymPy solves for. ``None`` where SymPy does not find every one, as of
    4x³ + sinh x - 3, or where it finds infinitely many on an interval with ends that are
    no numbers."""
    expr = sympy.sympify(expr)
    if not expr.has(variable):
        return _Zeros()
    periodic = any(each.has(variable) for each in expr.atoms(TrigonometricFunction))
    parameters = expr.free_symbols - {variable}
    if expr.is_rational_function(variable) or (parameters and not periodic):
        return _Zeros(tuple(_roots(expr, variable, bounds)))
    numbers = all(end.is_number for end in (part.start, part.end))
    try:
        found = sympy.solveset(expr, variable, part if numbers else sympy.S.Reals)
    except (NotImplementedError, ValueError, TypeError):
        return None
    zeros = _solutions(found, expr, variable, bounds)
    if zeros is not None and zeros.families and not numbers:
        return None
    return zeros


def _solutions(found: Any, expr: Any, variable: Symbol, bounds: Sequence[Any]) -> _Zeros | None:
    """The solutions solveset found, as points and families of them: those of a set it
    gives within another, as the points that are real, or the multiples of 2π past 0, kept
    for the domain to decide; ``None`` where it does not list them, as a ConditionSet."""
    if found is sympy.S.EmptySet:
        return _Zeros()
    if isinstance(found, sympy.FiniteSet):
        return _Zeros(tuple(p for p in found.args if _real_root(p, expr, variable, bounds)))
    if isinstance(found, sympy.ImageSet):
        return _Zeros(families=(found,))
    if isinstance(found, sympy.Union):
        parts = [_solutions(each, expr, variable, bounds) for each in found.args]
        if any(each is None for each in parts):
            return None
        return sum((each for each in parts if each is not None), _Zeros())
    if isinstance(found, sympy.Complement):
        kept, apart = found.args
        within = _solutions(kept, expr, variable, bounds)
        if within is None or within.families or not isinstance(apart, sympy.FiniteSet):
            return None
        return _Zeros(tuple(p for p in within.points if p not in apart))
    if isinstance(found, sympy.Intersection):  # of the real line, or an interval
        listed = [each for each in found.args if not isinstance(each, sympy.Interval)]
        if len(listed) != 1:
            return None
        if isinstance(listed[0], sympy.ImageSet):
            return _Zeros(families=(found,))
        within = _solutions(listed[0], expr, variable, bounds)
        return None if within is None or within.families else within
    return None


def _real_root(point: Any, expr: Any, variable: Symbol, bounds: Sequence[Any]) -> bool:
    """Whether a solution solveset found is real, and a root whatever the parameters are
    (see :func:`_a_root`)."""
    point = sympy.sympify(point)
    if point.is_extended_real is False:
        return False
    if point.is_number:
        number = complex(sympy.N(point, 30))
        if abs(number.imag) > 1e-12 * max(1.0, abs(number.real)):
            return False
    return _a_root(expr.subs(variable, point), bounds)


def _interior(
    point: Any, part: Any, kept: Sequence[tuple[Any, bool]], variable: Symbol, bounds: Sequence[Any]
) -> bool:
    """Whether a point is inside an interval of the domain, away from its ends: where each
    constraint, and what makes the function real, holds strictly; or else, of an interval
    whose ends can be put in order with it, between them. An error that says so where
    neither decides."""
    signs = [_sign(sympy.sympify(g).subs(variable, point), bounds) for g, _ in kept]
    if any(s is not None and s < 0 for s in signs):
        return False
    inside = _inside(point, part, bounds)
    if inside is not None:
        return inside
    if all(s == 1 for s in signs):
        return True
    raise Undecided(f"whether {point} is in the domain depends on the parameters")


def _breaks(f: Any, variable: Symbol, part: Any, bounds: Sequence[Any] = ()) -> _Zeros | None:
    """Where a function has no derivative for a modulus, extremes or cases of the variable,
    within an interval: where the modulus is of 0, two extremes are equal, or a case begins
    (see :func:`_zeros`)."""
    found = _Zeros()
    rough = sympy.sympify(f).atoms(sympy.Abs, sympy.Min, sympy.Max, sympy.Piecewise)
    for each in sorted(rough, key=sympy.default_sort_key):
        if not each.has(variable):
            continue
        if isinstance(each, sympy.Abs):
            zeros = [each.args[0]]
        elif isinstance(each, sympy.Piecewise):
            zeros = [
                c.lhs - c.rhs
                for _, c in each.args
                if isinstance(c, sympy.core.relational.Relational)
            ]
        else:
            zeros = [p - q for p, q in itertools.combinations(each.args, 2)]
        for zero in zeros:
            where = _zeros(zero, variable, part, bounds)
            if where is None:
                return None
            found += where
    return found


#: What a function of one variable may be built of for the points where it is not defined to
#: be found: the elementary functions, whose poles are where they divide by 0.
_ELEMENTARY = (
    sympy.exp,
    sympy.log,
    TrigonometricFunction,
    HyperbolicFunction,
    InverseTrigonometricFunction,
    InverseHyperbolicFunction,
    sympy.Abs,
    sympy.Piecewise,
    sympy.erf,
    sympy.erfc,
)
#: The functions that divide by a cosine, as tan x is sin x / cos x.
_OVER_A_COSINE = ([sympy.tan, sympy.cot, sympy.sec, sympy.csc], sympy.cos)
_OVER_A_COSH = ([sympy.tanh, sympy.coth, sympy.sech, sympy.csch], sympy.cosh)


def _undefined(f: Any, variable: Symbol, part: Any, bounds: Sequence[Any]) -> _Zeros | None:
    """Where a function of a variable is not defined within an interval, as tan is not at
    π/2, nor 1/x at 0: where it divides by 0. An optimum near such a point is approached,
    never reached there. ``None`` where that is not found: of a function built of more than
    the elementary ones, as Γ is, whose poles SymPy does not give, or where the zeros of
    what it divides by are not all found."""
    functions = [each for each in f.atoms(sympy.Function) if each.has(variable)]
    if not all(isinstance(each, _ELEMENTARY) for each in functions):
        return None
    written = f.rewrite(*_OVER_A_COSINE).rewrite(*_OVER_A_COSH)
    found = _Zeros()
    divided_by = [
        power.base
        for power in written.atoms(sympy.Pow)
        if power.base.has(variable) and power.exp.is_negative
    ]
    for base in sorted(set(divided_by), key=sympy.default_sort_key):
        zeros = _zeros(base, variable, part, bounds)
        if zeros is None:
            return None
        found += zeros
    return found


def _roots(expr: Any, variable: Symbol, bounds: Sequence[Any] = ()) -> list[Any]:
    """The real roots of an expression in a variable, as SymPy solves for them, without
    those that are roots only where the parameters take particular values: SymPy keeps a
    root it finds by squaring where it cannot rule it out, as -cp/√(c²m² + p²) of
    p - mv/√(1 - v²/c²), a root only where p = 0."""
    expr = sympy.sympify(expr)
    if not expr.has(variable):
        return []
    found = [r for r in sympy.solve(expr, variable) if sympy.sympify(r).is_real is not False]
    return [r for r in found if _a_root(expr.subs(variable, r), bounds)]


def _a_root(value: Any, bounds: Sequence[Any]) -> bool:
    """Whether what an expression is at a root SymPy found is 0: exactly, or, where that
    is not decided, at six points of the parameters drawn with a fixed seed, which keep what
    is assumed of them and the bounds the program assumes. Where it is 0 at some of them
    and not at others, whether it is a root depends on the parameters, which is said."""
    import random

    value = sympy.sympify(value)
    try:
        if sympy.simplify(value) == 0:
            return True
    except (TypeError, ValueError, NotImplementedError):
        return True
    symbols = sorted(value.free_symbols, key=str)
    draw = random.Random(0)
    verdicts: list[bool] = []
    for _ in range(80):
        if len(verdicts) == 6:
            break
        values = {s: _drawn(s, draw) for s in symbols}
        if bounds and not all(
            assumptions.holds(sympy.sympify(bound).xreplace(values), ()) is not False
            for bound in bounds
        ):
            continue
        try:
            number = complex(sympy.N(value.xreplace(values), 30))
        except (TypeError, ValueError):
            return True
        verdicts.append(abs(number) <= 1e-9 * max(1.0, _size(value.xreplace(values))))
    if all(verdicts):
        return True
    if not any(verdicts):
        return False
    raise Undecided("whether a stationary point is one depends on the signs of the parameters")


def _drawn(symbol: Any, draw: Any) -> Any:
    """A value for a parameter that keeps what is assumed of it."""
    size = sympy.Rational(draw.randrange(300, 3000), 1000)
    if symbol.is_integer:
        size = sympy.Integer(draw.randrange(1, 9))
    if symbol.is_positive or symbol.is_nonnegative:
        return size
    if symbol.is_negative or symbol.is_nonpositive:
        return -size
    return size if draw.random() < 0.5 else -size


def _inside(point: Any, interval: Any, bounds: Sequence[Any]) -> bool | None:
    """Whether a point is inside an interval, away from its ends."""
    above = _compare(point, interval.start, bounds)
    below = _compare(point, interval.end, bounds)
    if interval.start == -sympy.oo:
        above = 1
    if interval.end == sympy.oo:
        below = -1
    if above is None or below is None:
        return None
    return above > 0 and below < 0


@optimizer(
    "extrema",
    summary="optima of a function of one variable: at its stationary points, where it has "
    "no derivative, and at the ends of its domain, or toward them",
    examples=[
        Example(r"\min_{x \in \mathbb{R}} \left(x^2 - 4x + 7\right)", 3,
                "standard: x^2 - 4x + 7 = (x - 2)^2 + 3"),
        Example(r"\max_{0 \leq x \leq 3} \left(x^3 - 3x\right)", 18,
                "Stewart (2015), section 4.1, the closed interval method"),
        Example(r"\inf_{x > 0} \left(x + \frac{1}{x}\right)", 2,
                "the inequality of the means: x + 1/x >= 2, with equality at x = 1"),
        Example(r"\sup_{x \geq 0} \frac{x}{1 + x}", 1,
                "x/(1 + x) = 1 - 1/(1 + x), which rises toward 1 and never reaches it"),
        Example(r"\arg\max_{x \geq 0} \; x e^{-x}", 1, "standard: (1 - x) e^{-x} = 0 at x = 1"),
    ],
    problems=[
        Problem("min", x**2 - 4 * x + 7, (x,)),
        Problem("max", x**3 - 3 * x, (x,), inequalities=(x, 3 - x)),
        Problem("max", x * sympy.exp(-x), (x,), inequalities=(x,)),
        Problem("min", sympy.Abs(x - 1) + sympy.Abs(x + 2), (x,)),
        Problem("min", k * x**2 - 2 * x, (x,)),
    ],
)  # fmt: skip
class OneVariable:
    def takes(self, problem: Problem) -> bool:
        return len(problem.variables) == 1 and problem.continuous()

    def optimum(self, problem: Problem) -> Found | None:
        """The optimum, from every stationary point and point without a derivative inside
        the domain, as SymPy's solveset finds them on each interval of it, and the ends of
        each, or the limits toward them where they are open, as at a point where the
        function is not defined; ``None`` where those points are not all found."""
        (variable,) = problem.variables
        f = sympy.sympify(problem.objective)
        greatest = problem.sense == "max"
        bounds = problem.bounds
        domain = _domain(problem, variable)
        if domain is None:
            return None
        if domain.is_empty:
            raise EvaluationError(f"no value of {variable} keeps the constraints of {problem}")
        kept = [*((g, False) for g in problem.inequalities), *((g, True) for g in problem.strict)]
        kept += list(_natural(f, variable))
        undefined = _Zeros()
        for part in _parts(domain):
            where = _undefined(f, variable, part, bounds) if _an_interval(part) else _Zeros()
            if where is None:
                return None
            undefined += where
        if undefined.families:  # infinitely many: decided only where it is unbounded at one
            toward = _toward_a_pole(f, variable, undefined.families, domain, kept, problem)
            if toward is None:
                return None
            return Found(toward, None, _how(False, True), proven=True, attained=False)
        defined = domain
        for point in dict.fromkeys(undefined.points):
            defined = _without(defined, point, bounds) if _surely_real(point) else None
            if defined is None:
                raise Undecided(f"whether {point} is in the domain depends on the parameters")
        parts = _parts(defined)
        pieces = _pieces(f, variable)
        candidates: list[_Candidate] = []
        rough = False
        for part in parts:
            if isinstance(part, sympy.FiniteSet):
                candidates += [_Candidate(f.subs(variable, p), (p,)) for p in part.args]
                continue
            breaks = _breaks(f, variable, part, bounds)
            stationary = [_stationary(piece, variable, part, bounds) for piece in pieces]
            if breaks is None or any(each is None for each in stationary):
                return None  # not every point an optimum may be at is found
            rough = rough or bool(breaks.points or breaks.families)
            zeros = sum((each for each in stationary if each is not None), breaks)
            for point in zeros.points:
                if _interior(point, part, kept, variable, bounds):
                    candidates.append(_Candidate(f.subs(variable, point), (point,)))
            for point in breaks.points if f.has(sympy.Piecewise) else ():
                beside = _beside(f, variable, point)  # a case may begin with a jump
                if beside is None:
                    return None
                if _interior(point, part, kept, variable, bounds):
                    candidates += beside
            for family in zeros.families:
                candidate = _of_family(f, variable, family, part, kept, bounds)
                if candidate is None:
                    return None
                candidates.append(candidate)
            ends = _at_the_ends(f, variable, part)
            if ends is None:
                return None
            candidates += ends
        if not candidates or any(c.value.has(sympy.nan, sympy.zoo) for c in candidates):
            return None
        how = _how(rough, defined != domain)
        best = _best(candidates, greatest, bounds)
        if best is not None:
            chosen, tied = best
            reached = _tied_set(tied, f, variable, parts) if chosen.attained else None
            return Found(
                chosen.value, chosen.point, how, proven=True, attained=chosen.attained,
                tied=reached,
            )  # fmt: skip
        smooth = problem.smooth()
        return self._undecided(f, variable, parts, candidates, smooth, greatest, how, bounds)

    def _undecided(
        self,
        f: Any,
        variable: Symbol,
        parts: Sequence[Any],
        candidates: Sequence[_Candidate],
        smooth: bool,
        greatest: bool,
        how: str,
        bounds: Sequence[Any],
    ) -> Found:
        """The optimum where its candidates cannot all be compared: at the one stationary
        point of a smooth function on an interval, where the second derivative shows it is
        of the right kind, as no other point can then be better; or the greatest or least of
        the values at points, as a closed form, where every candidate is a point."""
        interior = [
            c for c in candidates if c.point is not None and not _at_an_end(c.point[0], parts)
        ]
        if (
            smooth
            and len(parts) == 1
            and isinstance(parts[0], sympy.Interval)
            and len(interior) == 1
        ):
            (point,) = interior[0].point  # type: ignore[misc]
            curvature = _sign(sympy.diff(f, variable, 2).subs(variable, point), bounds)
            if curvature == (-1 if greatest else 1):
                how += "; it is the only stationary point, and its second derivative shows its kind"
                return Found(interior[0].value, (point,), how, proven=True)
        if all(c.attained for c in candidates):
            values = list(dict.fromkeys(sympy.sympify(c.value) for c in candidates))
            value = sympy.Max(*values) if greatest else sympy.Min(*values)
            return Found(value, None, how, proven=True)
        raise Undecided(
            "which of "
            + assumptions.listed([str(c.value) for c in candidates])
            + " is "
            + ("greatest" if greatest else "least")
            + " depends on the parameters"
        )


def _at_an_end(point: Any, parts: Sequence[Any]) -> bool:
    return any(
        isinstance(part, sympy.Interval) and point in (part.start, part.end) for part in parts
    )


def _an_interval(part: Any) -> bool:
    return isinstance(part, sympy.Interval)


def _how(rough: bool, undefined: bool) -> str:
    """How the optimum of a function of one variable was found, in words."""
    sources = ["its stationary points"]
    if rough:
        sources.append("the points where it has no derivative")
    sources.append("the ends of its domain")
    if undefined:
        sources.append("the points where it is not defined")
    if len(sources) == 2:
        return "by SymPy, from " + " and ".join(sources)
    return "by SymPy, from " + ", ".join(sources[:-1]) + ", and " + sources[-1]


def _surely_real(point: Any) -> bool:
    """Whether a point is a real number, whatever its parameters are."""
    point = sympy.sympify(point)
    if not point.is_number:
        return point.is_real is True
    number = complex(sympy.N(point, 30))
    return abs(number.imag) <= 1e-12 * max(1.0, abs(number.real))


def _at_the_ends(f: Any, variable: Symbol, part: Any) -> list[_Candidate] | None:
    """The values a function of a variable takes at the ends of an interval: at an end it
    holds, the value there; toward one it does not, as an end at infinity or a point where
    the function is not defined, the limit, which no point reaches. Toward infinity a
    periodic function has no limit, and takes there no value it does not take at its
    stationary points. ``None`` where SymPy finds no limit otherwise."""
    found = []
    for end, closed, direction in (
        (part.start, not part.left_open, "+"),
        (part.end, not part.right_open, "-"),
    ):
        if end.is_finite and closed:
            found.append(_Candidate(f.subs(variable, end), (end,)))
            continue
        try:
            limit = sympy.limit(f, variable, end, direction)
            periodic = end.is_infinite and sympy.periodicity(f, variable) is not None
        except (NotImplementedError, ValueError, TypeError):
            return None
        if limit.has(sympy.AccumBounds) and periodic:
            continue  # the bounds SymPy gives its values there are no values it comes near
        if limit.has(sympy.AccumBounds, sympy.Limit, sympy.nan, sympy.zoo):
            return None
        if limit.is_extended_real is False:
            return None
        found.append(_Candidate(limit, None, attained=False))
    return found


def _beside(f: Any, variable: Symbol, point: Any) -> list[_Candidate] | None:
    """The values a function written in cases comes near on each side of a point where a
    case begins, which it need not take there, as it jumps: its limits from above and from
    below, which no point reaches. ``None`` where SymPy finds no limit."""
    step = sympy.Dummy("h", positive=True)
    found = []
    for side in (1, -1):
        try:
            limit = sympy.limit(f.subs(variable, point + side * step), step, 0, "+")
        except (NotImplementedError, ValueError, TypeError):
            return None
        if limit.has(sympy.AccumBounds, sympy.Limit, sympy.nan, sympy.zoo):
            return None
        if limit.is_extended_real is False:
            return None
        found.append(_Candidate(limit, None, attained=False))
    return found


#: The most whole numbers tried, either side of 0, for a point of a family inside an
#: interval.
_MOST_TRIED = 8


def _member(
    family: Any,
    part: Any,
    kept: Sequence[tuple[Any, bool]],
    variable: Symbol,
    bounds: Sequence[Any],
) -> tuple[Any, Any, Any]:
    """Of a family of points, as π/2 + 2πn of every integer n: its point as written with n,
    the n, and the point of it inside an interval at the whole number nearest 0, ``None``
    where none of the few tried is inside."""
    image = (
        family
        if isinstance(family, sympy.ImageSet)
        else next(each for each in family.args if isinstance(each, sympy.ImageSet))
    )
    (label,) = image.lamda.variables
    whole = sympy.Dummy(label.name, integer=True)
    point = image.lamda.expr.xreplace({label: whole})
    for k in sorted(range(-_MOST_TRIED, _MOST_TRIED + 1), key=lambda k: (abs(k), -k)):
        at = sympy.simplify(point.xreplace({whole: k}))
        if _interior(at, part, kept, variable, bounds):
            return point, whole, at
    return point, whole, None


def _of_family(
    f: Any,
    variable: Symbol,
    family: Any,
    part: Any,
    kept: Sequence[tuple[Any, bool]],
    bounds: Sequence[Any],
) -> _Candidate | None:
    """The value of a function at every point of a family of them, as sin x is 1 at each
    π/2 + 2πn, with the point of it at the whole number nearest 0 inside the interval as
    its point; ``None`` where the value is not the same at each, or no point of it is
    found inside."""
    point, whole, inside = _member(family, part, kept, variable, bounds)
    value = sympy.simplify(f.subs(variable, point))
    if value.has(whole) or inside is None:
        return None
    return _Candidate(value, (inside,), family=family)


def _toward_a_pole(
    f: Any,
    variable: Symbol,
    families: Sequence[Any],
    domain: Any,
    kept: Sequence[tuple[Any, bool]],
    problem: Problem,
) -> Any:
    """∞ or -∞, where a function grows without bound, the way its optimum is sought, toward
    a point of a family of points where it is not defined, as tan does toward π/2: no other
    point then matters. ``None`` where it does not."""
    wanted = sympy.oo if problem.sense == "max" else -sympy.oo
    for family in families:
        for part in filter(_an_interval, _parts(domain)):
            _, _, pole = _member(family, part, kept, variable, problem.bounds)
            if pole is None:
                continue
            for direction in "+-":
                try:
                    limit = sympy.limit(f, variable, pole, direction)
                except (NotImplementedError, ValueError, TypeError):
                    continue
                if limit == wanted:
                    return wanted
    return None


def _tied_set(tied: Sequence[_Candidate], f: Any, variable: Symbol, parts: Sequence[Any]) -> Any:
    """The points that reach an optimum, where more than one does, as a set: the families of
    them, and each run of points with the function the same at every point between each
    two, in one interval of the domain, as the interval they span, as |x - 1| + |x + 1| is
    2 on [-1, 1]; ``None`` where one alone reaches it."""
    families = [c.family for c in tied if c.family is not None]
    points = [c.point[0] for c in tied if c.family is None and c.point is not None]
    if not families and len(points) < 2:
        return None
    value = tied[0].value
    runs: list[list[Any]] = []
    for point in points:  # in order, the least first
        if runs and _constant_between(f, variable, runs[-1][-1], point, value, parts):
            runs[-1].append(point)
        else:
            runs.append([point])
    spans = [
        sympy.Interval(run[0], run[-1]) if len(run) > 1 else sympy.FiniteSet(run[0]) for run in runs
    ]
    return sympy.Union(*spans, *families)


def _constant_between(
    f: Any, variable: Symbol, one: Any, other: Any, value: Any, parts: Sequence[Any]
) -> bool:
    """Whether a function is ``value`` at every point between two, in one interval of the
    domain: at one + (other - one)/(1 + s), of every positive s."""
    if _compare(one, other) != -1:
        return False
    if not any(
        isinstance(part, sympy.Interval) and _within(one, part, ()) and _within(other, part, ())
        for part in parts
    ):
        return False
    step = sympy.Dummy("s", positive=True)
    between = f.subs(variable, one + (other - one) / (1 + step))
    try:
        return bool(sympy.simplify(between - value) == 0)
    except (TypeError, ValueError, NotImplementedError):
        return False


# -- several variables -----------------------------------------------------------------------


@optimizer(
    "kkt",
    summary="optima of smooth functions of several variables: where the gradient vanishes, "
    "or the Lagrange or the KKT conditions hold, at each set of constraints that hold with "
    "equality",
    examples=[
        Example(r"\min_{x_1, x_2} \; x_1 + x_2 \quad \text{subject to} \quad x_1^2 + x_2^2 = 2",
                -2, "Nocedal and Wright (2006), section 12.1"),
        Example(r"\arg\min_{x, y} \; (x - 1)^2 + (y - 2)^2 \quad \text{subject to} \quad "
                r"x + y \leq 1", [0, 1], "standard: the projection of (1, 2) on x + y <= 1"),
        Example(r"\max_{x, y} \; x y \quad \text{subject to} \quad x + y = 10", 25,
                "standard: x = y = 5, by the inequality of the means"),
    ],
    problems=[
        Problem("min", x + y, (x, y), equalities=(x**2 + y**2 - 2,)),
        Problem("min", (x - 1) ** 2 + (y - 2) ** 2, (x, y), inequalities=(1 - x - y,)),
        Problem("max", x * y, (x, y), equalities=(x + y - 10,)),
        Problem("min", x**2 + y**2 + z**2, (x, y, z), equalities=(x + y - 1, y + z - 2)),
        Problem("min", x**2 + 2 * y**2, (x, y), inequalities=(x + y - 1, x, y)),
    ],
)  # fmt: skip
class Stationary:
    def takes(self, problem: Problem) -> bool:
        return (
            len(problem.variables) >= 2
            and len(problem.variables) <= _MOST_VARIABLES
            and len(problem.inequalities) <= _MOST_ACTIVE
            and problem.continuous()
            and problem.smooth()
            and not problem.distinct
            and not problem.linear()
        )

    def optimum(self, problem: Problem) -> Found | None:
        """The optimum, from the points where the KKT conditions hold, at each set of
        constraints that hold with equality, but those the second-order conditions show
        to be no optimum of the kind; and from each family of such points, a curve or a
        surface, on which the function is one value. It is proven global of a convex
        problem, and of a polynomial one over a bounded region, where every such point is
        found. ``None`` where they are not all found, as of a periodic function between
        constraints, of whose stationary points SymPy gives the principal ones alone."""
        variables = problem.variables
        sign = 1 if problem.sense == "min" else -1
        least = sign * sympy.sympify(problem.objective)  # what is made least
        if _constrained(problem) and _periodic(problem):
            return None
        found: list[_Candidate] = []
        for active in _active_sets(problem):
            points = _kkt_points(problem, least, active)
            if points is None:
                return None  # not every point the optimum may be at is found
            found += points
        how = "by SymPy, from " + _conditions(problem)
        convex = problem.convex()
        bounded = problem.bounded() and _polynomial(problem)
        proven = convex or bounded
        if not _constrained(problem) and not (proven and found):
            origin = tuple(sympy.S.Zero for _ in variables)
            through = (found[0].point if found else None) or origin
            unbounded = _unbounded(least, variables, through)
            if unbounded is not None:
                return Found(sign * unbounded, None, how, proven=True, attained=False)
        if not found:
            return None
        best = _best(found, False, problem.bounds)
        if best is None:
            if len(_in_order(found)) > 1:
                values = list(dict.fromkeys(sign * c.value for c in found))
                value = sympy.Max(*values) if sign < 0 else sympy.Min(*values)
                if not proven:
                    raise Undecided(
                        "which of the stationary points is best depends on the parameters"
                    )
                return Found(value, None, how, proven=True)
            best = (found[0], [found[0]])
        chosen, tied = best
        if convex:
            how += "; the problem is convex"
        elif bounded:
            how += "; the constraints bound the region"
        reached = _tied_points(tied)
        return Found(sign * chosen.value, chosen.point, how, proven=proven, tied=reached)


def _periodic(problem: Problem) -> bool:
    """Whether a problem is written with a trigonometric function of its variables."""
    return any(
        each.has(*problem.variables)
        for expr in problem.expressions
        for each in sympy.sympify(expr).atoms(TrigonometricFunction)
    )


def _polynomial(problem: Problem) -> bool:
    """Whether what a problem is of and its constraints are polynomials in its variables:
    SymPy then finds every point where the KKT conditions hold, and what it is of is
    defined at every point."""
    return all(sympy.sympify(e).is_polynomial(*problem.variables) for e in problem.expressions)


def _conditions(problem: Problem) -> str:
    if problem.inequalities:
        return "the KKT conditions, at each set of constraints that hold with equality"
    if problem.equalities:
        return "the Lagrange conditions"
    return "the points where the gradient vanishes"


def _active_sets(problem: Problem) -> Iterator[tuple[int, ...]]:
    """The sets of inequalities that may hold with equality at an optimum: no more of
    them, with the equalities, than there are variables."""
    room = len(problem.variables) - len(problem.equalities)
    for size in range(min(room, len(problem.inequalities)) + 1):
        yield from itertools.combinations(range(len(problem.inequalities)), size)


def _kkt_points(problem: Problem, least: Any, active: tuple[int, ...]) -> list[_Candidate] | None:
    """The candidates where the KKT conditions hold with the inequalities ``active`` held at
    0: the gradient of what is made least a combination of the gradients of the
    constraints, the multipliers of the inequalities not negative, and every constraint
    kept; each point the second-order conditions do not rule out (see
    :func:`_not_a_minimum`), and each family of them, a line or a curve of points, where what
    is made least is one value along it (see :func:`_along`). ``None`` where SymPy cannot
    solve the conditions, or the value along a family is not one."""
    variables = problem.variables
    held = [problem.inequalities[k] for k in active]
    lagrange = [sympy.Dummy(f"lambda{k}") for k in range(len(problem.equalities))]
    kuhn = [sympy.Dummy(f"mu{k}") for k in range(len(held))]
    function = (
        least
        - sum((m * h for m, h in zip(lagrange, problem.equalities, strict=True)), sympy.Integer(0))
        - sum((m * g for m, g in zip(kuhn, held, strict=True)), sympy.Integer(0))
    )
    equations = [sympy.diff(function, v) for v in variables] + list(problem.equalities) + held
    unknowns = [*variables, *lagrange, *kuhn]
    try:
        solutions = sympy.solve(equations, unknowns, dict=True)
    except NotImplementedError:
        return None
    found: list[_Candidate] = []
    for solution in solutions:
        point = tuple(sympy.sympify(solution.get(v, v)) for v in variables)
        if any(p.is_real is False for p in point):
            continue
        if any(_sign(solution.get(m, 0), problem.bounds) == -1 for m in kuhn):
            continue
        if any(p.has(*variables) for p in point):  # a line or a curve of them
            along = _along(problem, least, solution, point)
            if along is None:
                return None
            found += along
            continue
        if not _kept(problem, point):
            continue
        multipliers = [solution.get(m, m) for m in (*lagrange, *kuhn)]
        if _not_a_minimum(problem, least, point, held, multipliers):
            continue
        found.append(_Candidate(least.xreplace(dict(zip(variables, point, strict=True))), point))
    return found


#: The values tried for the variables a family of points is written with, for a point of
#: it that keeps the constraints.
_TRIED = (0, 1, -1, sympy.Rational(1, 2), sympy.Rational(-1, 2), 2, -2)


def _along(
    problem: Problem, least: Any, solution: dict[Any, Any], point: tuple[Any, ...]
) -> list[_Candidate] | None:
    """A family of points where the KKT conditions hold, as the circle where x² + y² ≤ 1
    holds with equality is of x² + y²: what is made least there, where it is one value
    along it, at a point of it that keeps the constraints, which stands for all; nothing
    of a family a constraint rules out whole. ``None`` where the value is not one along
    it, or no point of it that keeps the constraints is found among the few tried."""
    variables = problem.variables
    along = dict(zip(variables, point, strict=True))
    for g in problem.inequalities:
        if _sign(sympy.simplify(sympy.sympify(g).xreplace(along)), problem.bounds) == -1:
            return []
    free = sorted(set().union(*(p.free_symbols for p in point)) & set(variables), key=str)
    value = sympy.simplify(least.xreplace(along))
    if value.has(*free):
        return None
    family = sympy.And(*(sympy.Eq(v, solution[v]) for v in variables if v in solution))
    for values in itertools.product(_TRIED, repeat=len(free)):
        at = dict(zip(free, values, strict=True))
        chosen = tuple(sympy.simplify(p.xreplace(at)) for p in point)
        if all(_real_number(p) for p in chosen) and _kept(problem, chosen):
            return [_Candidate(value, chosen, family=family)]
    return None


def _real_number(value: Any) -> bool:
    """Whether a value is a real number, or real for its symbols' facts."""
    value = sympy.sympify(value)
    if value.has(sympy.nan, sympy.zoo, sympy.oo, -sympy.oo):
        return False
    if not value.is_number:
        return value.is_real is not False
    number = complex(sympy.N(value, 30))
    return abs(number.imag) <= 1e-12 * max(1.0, abs(number.real))


def _not_a_minimum(
    problem: Problem,
    least: Any,
    point: tuple[Any, ...],
    held: Sequence[Any],
    multipliers: Sequence[Any],
) -> bool:
    """Whether a point where the KKT conditions hold is surely no minimum of what is made
    least, by the second-order conditions: the Hessian of the Lagrangian, on the directions
    along which every constraint that holds with equality stays so, has a direction of
    negative curvature, as the centre of x² + y² ≤ 1 has of -(x² + y²). Not where another
    constraint holds with equality there too, which the conditions with it held judge, nor
    where the multipliers, or the curvature, are not numbers."""
    variables = list(problem.variables)
    constraints = [*problem.equalities, *held]
    at = dict(zip(variables, point, strict=True))
    others = [g for g in problem.inequalities if not any(g is each for each in held)]
    if any(_sign(sympy.sympify(g).xreplace(at), problem.bounds) != 1 for g in others):
        return False
    if any(sympy.sympify(m).free_symbols for m in multipliers):
        return False
    lagrangian = least - sum(
        (m * c for m, c in zip(multipliers, constraints, strict=True)), sympy.Integer(0)
    )
    hessian = sympy.hessian(lagrangian, variables).xreplace(at)
    if constraints:
        gradients = [[sympy.diff(c, v).xreplace(at) for v in variables] for c in constraints]
        directions = sympy.Matrix(gradients).nullspace()
        if not directions:
            return False  # a vertex: no direction keeps every constraint that holds
        along = sympy.Matrix.hstack(*directions)
        hessian = along.T * hessian * along
    if hessian.free_symbols:
        return False
    try:
        curvatures = [complex(sympy.N(e)).real for e in hessian.eigenvals(multiple=True)]
        scale = max([1.0, *(abs(complex(sympy.N(e))) for e in hessian)])
    except (TypeError, ValueError, NotImplementedError):
        return False
    return min(curvatures) < -1e-12 * scale


def _tied_points(tied: Sequence[_Candidate]) -> Any:
    """The points that reach an optimum, where more than one does: the condition the points
    of each family of them keep, or the set of them, each a number, of one variable, or
    the values of several; ``None`` where one alone reaches it."""
    families = [c.family for c in tied if c.family is not None]
    points = [c.point for c in tied if c.family is None and c.point is not None]
    if families:
        return sympy.Or(*families) if len(families) > 1 else families[0]
    if len(points) < 2:
        return None
    return sympy.FiniteSet(*(p[0] if len(p) == 1 else sympy.Tuple(*p) for p in points))


def _kept(problem: Problem, point: tuple[Any, ...]) -> bool:
    """Whether no constraint is surely broken at a point."""
    at = dict(zip(problem.variables, point, strict=True))
    for g in problem.inequalities:
        if _sign(sympy.sympify(g).xreplace(at), problem.bounds) == -1:
            return False
    for g in problem.strict:
        if _sign(sympy.sympify(g).xreplace(at), problem.bounds) in (-1, 0):
            return False
    return True


def _unbounded(least: Any, variables: Sequence[Symbol], point: tuple[Any, ...]) -> Any:
    """-∞, where what is made least falls without bound along a line through a point: an
    axis, or the way it falls fastest there; ``None`` where it does not, or that is not
    decided."""
    at = dict(zip(variables, point, strict=True))
    step = sympy.Dummy("t", real=True)
    lines = []
    for k, variable in enumerate(variables):
        others = {v: p for j, (v, p) in enumerate(zip(variables, point, strict=True)) if j != k}
        lines.append((least.xreplace(others), variable))
    slope = [sympy.diff(least, v).xreplace(at) for v in variables]
    if all(each.is_number for each in slope) and any(each != 0 for each in slope):
        down = {v: p - step * s for v, p, s in zip(variables, point, slope, strict=True)}
        lines.append((least.xreplace(down), step))
    for line, along in lines:
        for end in (sympy.oo, -sympy.oo):
            try:
                found = sympy.limit(line, along, end)
            except (NotImplementedError, ValueError):
                continue
            if found == -sympy.oo:
                return -sympy.oo
    return None


def _constrained(problem: Problem) -> bool:
    return bool(problem.equalities or problem.inequalities or problem.strict)


# -- linear programs -------------------------------------------------------------------------


@optimizer(
    "simplex",
    summary="linear programs, by SymPy's simplex method",
    examples=[
        Example(r"\max_{x_1, x_2} \; 3x_1 + 5x_2 \quad \text{subject to} \quad x_1 \leq 4, \quad "
                r"2x_2 \leq 12, \quad 3x_1 + 2x_2 \leq 18, \quad x_1, x_2 \geq 0", 36,
                "Hillier and Lieberman (2015), chapter 3: the Wyndor Glass Company"),
        Example(r"\min_{x, y} \; x + y \quad \text{subject to} \quad x + 2y \geq 4, \quad "
                r"3x + y \geq 6, \quad x, y \geq 0", sympy.Rational(14, 5),
                "standard: the vertex (8/5, 6/5), where both constraints hold with equality"),
    ],
    problems=[
        Problem("max", 3 * x + 5 * y, (x, y),
                inequalities=(4 - x, 12 - 2 * y, 18 - 3 * x - 2 * y, x, y)),
        Problem("min", x + y, (x, y), inequalities=(x + 2 * y - 4, 3 * x + y - 6, x, y)),
        Problem("min", 2 * x + 3 * y + z, (x, y, z), equalities=(x + y + z - 1,),
                inequalities=(x, y, z)),
    ],
)  # fmt: skip
class Simplex:
    def takes(self, problem: Problem) -> bool:
        return problem.linear() and len(problem.variables) <= 40

    def optimum(self, problem: Problem) -> Found | None:
        from sympy.solvers.simplex import InfeasibleLPError, UnboundedLPError, lpmax, lpmin

        plain = {v: sympy.Dummy(v.name) for v in problem.variables}  # facts are constraints
        constraints = []
        for relation in problem.relations():
            kept = type(relation)(sympy.sympify(relation.lhs).xreplace(plain), relation.rhs)
            if kept is sympy.false:
                raise EvaluationError(f"no point keeps the constraints of {problem}")
            if kept is not sympy.true:
                constraints.append(kept)
        how = "by SymPy's simplex method"
        greatest = problem.sense == "max"
        objective = sympy.sympify(problem.objective).xreplace(plain)
        try:
            value, at = (lpmax if greatest else lpmin)(objective, constraints)
        except UnboundedLPError:
            infinite = sympy.oo if greatest else -sympy.oo
            return Found(infinite, None, how, proven=True, attained=False)
        except InfeasibleLPError:
            raise EvaluationError(f"no point keeps the constraints of {problem}") from None
        point = tuple(sympy.sympify(at.get(plain[v], 0)) for v in problem.variables)
        return Found(sympy.sympify(value), point, how, proven=True)


# -- finitely many points --------------------------------------------------------------------


def _choices(problem: Problem) -> list[tuple[Any, ...]] | None:
    """The values each variable takes, where each takes finitely many: those its set
    gives, or the whole numbers between its ends; ``None`` where one takes infinitely
    many."""
    given = problem.values
    ranges = problem.ranges()
    found: list[tuple[Any, ...]] = []
    for variable in problem.variables:
        if variable in given:
            found.append(tuple(given[variable]))
            continue
        low, high = ranges[variable]
        if not (variable.is_integer and low.is_finite and high.is_finite):
            return None
        if not (low.is_number and high.is_number):
            return None
        first, last = math.ceil(float(low)), math.floor(float(high))
        found.append(tuple(sympy.Integer(k) for k in range(first, last + 1)))
    return found


@optimizer(
    "enumeration",
    summary="optima over finitely many points, as whole numbers between bounds or values "
    "in \\{0, 1\\}, by trying each",
    examples=[
        Example(r"\max_{x \in \{0, 1\}^3} \; 60x_1 + 100x_2 + 120x_3 \quad \text{subject to} "
                r"\quad 10x_1 + 20x_2 + 30x_3 \leq 50", 220,
                "Cormen, Leiserson, Rivest and Stein (2009), section 16.2: the knapsack"),
        Example(r"\arg\min_{n \in \mathbb{Z}} \; \left(n - \frac{13}{5}\right)^2 \quad "
                r"\text{subject to} \quad 0 \leq n \leq 10", 3,
                "standard: the whole number nearest 13/5"),
    ],
    problems=[
        Problem("max", 60 * a + 100 * b + 120 * z, (a, b, z),
                inequalities=(50 - 10 * a - 20 * b - 30 * z,),
                finite=((a, (0, 1)), (b, (0, 1)), (z, (0, 1)))),
        Problem("min", (n - sympy.Rational(13, 5)) ** 2, (n,), inequalities=(10 - n,)),
    ],
)  # fmt: skip
class Enumeration:
    def takes(self, problem: Problem) -> bool:
        choices = _choices(problem)
        if choices is None or problem.parameters:
            return False
        return math.prod(len(each) for each in choices) <= _MOST_POINTS

    def optimum(self, problem: Problem) -> Found | None:
        choices = _choices(problem)
        assert choices is not None
        count = math.prod(len(each) for each in choices)
        found = [
            _Candidate(problem.at(point), tuple(point))
            for point in itertools.product(*choices)
            if not problem.violations(point, 0.0)
        ]
        if not found:
            raise EvaluationError(f"no point keeps the constraints of {problem}")
        best = _best(found, problem.sense == "max", ())
        if best is None:
            return None
        chosen, tied = best
        how = f"by SymPy, trying each of the {count} points of its domain"
        return Found(chosen.value, chosen.point, how, proven=True, tied=_tied_points(tied))
