r"""Results checked by another route, where that is cheap and decisive.

SymPy's closed form of a definite integral, or of a convergent infinite sum, of numbers is
compared with mpmath's numerical value; an antiderivative is differentiated back. Where
they confidently disagree, the closed form is not shown: the computation is refused,
saying what disagreed. A numerical value counts only where two methods of mpmath agree
with each other far more closely than with the closed form, so that a failure of the
numbers never turns a right answer into a refusal. Each check has a budget of its own,
past which nothing is said.

SymPy integrates sign(sin x) from 0 to 3π to 0, where it is π, and atan(tan x) from 0 to π
to -π²/2, where it is 0: its antiderivative jumps where the integrand does.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

import sympy

from relativisticpy import display
from relativisticpy.errors import EvaluationError

#: How long a check may take, in seconds: past it, the value is given unchecked.
_CHECKING = 8.0
#: Working digits of the numerical values.
_DIGITS = 25
#: How closely two numerical methods must agree for their value to count, relatively.
_RELIABLE = 1e-12
#: How far a closed form must be from a value that counts to be refused, relatively.
_DISAGREE = 1e-8
#: The most points an interval is cut at, where the integrand jumps or bends.
_MOST_POINTS = 60


def integral(value: Any, f: Any, limits: Sequence[Any]) -> None:
    """Refuse SymPy's closed form ``value`` of the integral of ``f`` over ``limits`` where
    numerical integration confidently gives another number: a definite integral of numbers
    over one variable, its interval cut where the integrand jumps or bends. An indefinite
    integral is differentiated back instead (see :func:`antiderivative`)."""
    if len(limits) == 1 and _variable_alone(limits[0]):
        antiderivative(value, f, limits[0] if not isinstance(limits[0], tuple) else limits[0][0])
        return
    if len(limits) != 1 or not (isinstance(limits[0], tuple) and len(limits[0]) == 3):
        return
    variable, lower, upper = (sympy.sympify(each) for each in limits[0])
    if (
        not _a_number(value)
        or f.free_symbols - {variable}
        or not isinstance(variable, sympy.Symbol)
    ):
        return
    if not all(end.is_extended_real and not end.free_symbols for end in (lower, upper)):
        return
    found = _within_budget(lambda: _quadrature(f, variable, lower, upper))
    if found is None:
        return
    closed = complex(sympy.N(value, _DIGITS))
    if _far(closed, found):
        raise EvaluationError(
            f"SymPy's closed form of the integral, {_closed(value, closed)}, disagrees with "
            f"numerical integration, which gives {_number(found)}",
            hint="SymPy's antiderivative may jump where the integrand jumps or has a pole; "
            "split the integral there, or ask for its numerical value with N",
        )


def series(value: Any, body: Any, index: Any, lower: Any, upper: Any) -> None:
    """Refuse SymPy's closed form ``value`` of a sum to infinity of numbers where mpmath's
    sum of the series, by two methods that agree, confidently gives another number."""
    if upper != sympy.oo or not _a_number(value) or body.free_symbols - {index}:
        return
    if not (sympy.sympify(lower).is_integer and not sympy.sympify(lower).free_symbols):
        return
    found = _within_budget(lambda: _summed(body, index, int(lower)))
    if found is None:
        return
    closed = complex(sympy.N(value, _DIGITS))
    if _far(closed, found):
        raise EvaluationError(
            f"SymPy's closed form of the sum, {_closed(value, closed)}, disagrees with the "
            f"series summed numerically, which gives {_number(found)}",
            hint="ask for its numerical value with N, or write the sum in parts",
        )


def antiderivative(value: Any, f: Any, variable: Any) -> None:
    """Refuse SymPy's antiderivative ``value`` of ``f`` where, differentiated back, it is
    confidently not ``f``: at three points of the variable and of the parameters, each a
    number, it differs from ``f`` by far more than rounding."""
    if not isinstance(value, sympy.Expr) or value.has(sympy.Integral, sympy.Piecewise):
        return
    if not isinstance(variable, sympy.Symbol) or not isinstance(f, sympy.Expr):
        return
    difference = _within_budget(lambda: sympy.diff(value, variable) - f)
    if difference is None or difference == 0:
        return
    rng = random.Random(1)
    symbols = sorted(difference.free_symbols | f.free_symbols | {variable}, key=str)
    if difference.atoms(sympy.core.function.AppliedUndef):
        return
    for _ in range(3):
        point = {s: sympy.Rational(rng.randrange(300, 1700), 1000) for s in symbols}
        try:
            off = complex(sympy.N(difference.xreplace(point), _DIGITS))
            scale = complex(sympy.N(f.xreplace(point), _DIGITS))
        except (TypeError, ValueError):
            return
        if not abs(off) > 1e-6 * max(1.0, abs(scale)):  # nan too says nothing
            return
    raise EvaluationError(
        f"SymPy's antiderivative, {display.unicode(value)}, differentiated back, is not "
        f"{display.unicode(f)}",
        hint="this is a failure inside SymPy; write the integral another way",
    )


def _variable_alone(limit: Any) -> bool:
    """Whether a limit names the variable alone, as of an indefinite integral."""
    if isinstance(limit, tuple):
        return len(limit) == 1
    return isinstance(sympy.sympify(limit), sympy.Symbol)


def _a_number(value: Any) -> bool:
    """Whether a value is a finite number, which numbers can check."""
    if not isinstance(value, sympy.Expr) or value.free_symbols:
        return False
    return not value.has(sympy.nan, sympy.zoo, sympy.oo, -sympy.oo, sympy.Integral, sympy.Sum)


def _within_budget(compute: Callable[[], Any]) -> Any:
    """What ``compute`` gives within the budget of a check; ``None`` past it, or where it
    fails, as numbers that cannot be computed say nothing."""
    from relativisticpy.runtime import core

    try:
        with core._budget(_CHECKING):
            return compute()
    except core._OverBudget:
        return None
    except Exception:  # numbers that cannot be computed say nothing
        return None


def _quadrature(f: Any, variable: Any, lower: Any, upper: Any) -> complex | None:
    """The integral by two of mpmath's quadratures, over the interval cut where the
    integrand jumps, bends or has a pole; ``None`` where they disagree with each other."""
    import mpmath

    with mpmath.workdps(_DIGITS):
        g = sympy.lambdify(variable, f, modules="mpmath")
        path = [
            _end(lower),
            *(_end(p) for p in _breakpoints(f, variable, lower, upper)),
            _end(upper),
        ]
        first = complex(mpmath.quad(g, path, method="tanh-sinh"))
        second = complex(mpmath.quad(g, path, method="gauss-legendre"))
    return first if not _far(first, second, _RELIABLE) else None


def _summed(body: Any, index: Any, lower: int) -> complex | None:
    """The sum of a series by two of mpmath's methods of acceleration; ``None`` where they
    disagree with each other."""
    import mpmath

    with mpmath.workdps(_DIGITS):
        term = sympy.lambdify(index, body, modules="mpmath")
        first = complex(mpmath.nsum(term, [lower, mpmath.inf], method="richardson+shanks"))
        second = complex(mpmath.nsum(term, [lower, mpmath.inf], method="levin"))
    return first if not _far(first, second, _RELIABLE) else None


def _end(value: Any) -> Any:
    """An end of an interval as mpmath takes it."""
    import mpmath

    if value == sympy.oo:
        return mpmath.inf
    if value == -sympy.oo:
        return -mpmath.inf
    return mpmath.mpf(str(sympy.N(value, _DIGITS + 5)))


def _breakpoints(f: Any, variable: Any, lower: Any, upper: Any) -> list[Any]:
    """The points inside an interval where an integrand jumps, bends or has a pole, in
    order: where what a modulus, a sign or a step is of is 0, where the arguments of a
    maximum or a minimum cross, where a condition of a piecewise value changes, where what
    a root is of is 0, and the poles SymPy finds; but none where there are infinitely many."""
    interval = sympy.Interval(lower, upper)
    where: list[Any] = []
    for node in f.atoms(sympy.Abs, sympy.sign, sympy.Heaviside):
        where.append(node.args[0])
    for node in f.atoms(sympy.Max, sympy.Min):
        where.extend(a - b for a, b in zip(node.args, node.args[1:], strict=False))
    for node in f.atoms(sympy.Piecewise):
        for _, condition in node.args:
            where.extend(r.lhs - r.rhs for r in condition.atoms(sympy.core.relational.Relational))
    for node in f.atoms(sympy.Pow):
        if node.exp.is_Rational and not node.exp.is_Integer:
            where.append(node.base)
    found: set[Any] = set()
    for expr in where:
        if expr.has(variable):
            found |= _zeros(expr, variable, interval)
    try:
        poles = sympy.singularities(f, variable, interval)
    except Exception:  # SymPy cannot tell: the interval is cut where it can
        poles = sympy.S.EmptySet
    if isinstance(poles, sympy.FiniteSet):
        found |= set(poles)
    inside = sorted(
        (
            p
            for p in found
            if p.is_extended_real and (p - lower).is_positive and (upper - p).is_positive
        ),
        key=lambda p: float(p),
    )
    return inside[:_MOST_POINTS]


def _zeros(expr: Any, variable: Any, interval: Any) -> set[Any]:
    """The zeros of ``expr`` in an interval, where they are finitely many."""
    try:
        zeros = sympy.solveset(expr, variable, interval)
    except Exception:  # SymPy cannot solve it: no point is known
        return set()
    return set(zeros) if isinstance(zeros, sympy.FiniteSet) else set()


def _far(value: complex, reference: complex, relative: float = _DISAGREE) -> bool:
    """Whether a number is farther from another than ``relative`` of its size, or of 1."""
    return abs(value - reference) > relative * max(1.0, abs(reference))


def _closed(value: Any, number: complex) -> str:
    """A closed form as a message shows it, with its number where that is not the same."""
    shown, digits = display.unicode(value), _number(number)
    return shown if shown == digits else f"{shown}, which is {digits}"


def _number(value: complex) -> str:
    """A number as a message shows it, to ten digits, and 0 where it is 0 to rounding."""
    size = max(1.0, abs(value))
    real, imag = (part if abs(part) > 1e-15 * size else 0.0 for part in (value.real, value.imag))
    shown = f"{real:.10g}".replace("-", "−")
    if imag:
        shown += f" {'+' if imag > 0 else '−'} {abs(imag):.10g}i"
    return shown
