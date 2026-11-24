r"""Optimizers: a kind of capability that finds the optimum of a problem, and its checks.

An optimum written as books write it, ``\min_{x} f(x)``, or ``\arg\max_{x, y} u(x, y) \quad
\text{subject to} \quad p_x x + p_y y = m``, is a :class:`Problem`: a function of variables
to make greatest or least, over the numbers each takes, subject to constraints. The
optimizers the integrations declare find it, each by a method of its own, as SymPy's
stationary points or SciPy's linear programming, and say what they found in a
:class:`Found`: the value, the point, how it was found, and whether it is proven global::

    @optimizer(
        "linprog",
        summary="linear programs, numerically, by SciPy's HiGHS",
        requires="scipy",
        numerical=True,
        examples=[Example(r"\max_{x \geq 0} \; 2 - x", 2, "scipy.optimize.linprog([1], ...)"), ...],
        problems=[Problem("max", 3*x + 5*y, (x, y), inequalities=(4 - x, x, y))],
    )
    class LinearProgram:
        def takes(self, problem): return problem.linear()
        def optimum(self, problem): ...

They are consulted in order as the program runs: those that compute exactly first, then,
where the problem has no parameters, those that compute numerically. The first that takes
the problem and finds its optimum gives the answer, and a note says which, how, and whether
it is global. Where the answer depends on the signs of parameters the program says nothing
of, it is found under the assumption books make, that they are real, and the fewest of
them positive, which the note says too (``design/notation.md``, decision 1).

The conformance suite checks, at the problems each optimizer declares, that what it reports
is feasible, that the conditions of optimality hold there, stationarity or the KKT
conditions, and that it agrees with every optimizer of the other kind, exact or numeric,
that takes the same problem.
"""

from __future__ import annotations

import contextlib
import itertools
import re
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Literal

import sympy
from sympy import Symbol

from relativisticpy import assumptions
from relativisticpy.errors import EvaluationError, RelativisticPyError
from relativisticpy.integrations._declarations import Capability, Example, near
from relativisticpy.integrations._discovery import catalogue
from relativisticpy.tensors.equation import no_number

#: How long an exact optimizer may look for a closed form, in seconds, where a numeric
#: one could answer instead.
EXACT_BUDGET = 10.0
#: How far a constraint may miss at a point found numerically, and still hold there.
TOLERANCE = 1e-7


# -- the problem ------------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    """The greatest (``sense`` "max") or least ("min") value of ``objective``, a SymPy
    expression of ``variables``, each real unless its symbol is an integer, or ``finite``
    gives the values it takes, subject to ``equalities``, each 0 where it holds,
    ``inequalities``, each at least 0 where it holds, ``strict`` ones, each above 0, and
    ``distinct`` ones, each other than 0; under the ``bounds`` the program assumes of its
    parameters, as a < 1."""

    sense: Literal["max", "min"]
    objective: Any
    variables: tuple[Symbol, ...]
    equalities: tuple[Any, ...] = ()
    inequalities: tuple[Any, ...] = ()
    strict: tuple[Any, ...] = ()
    distinct: tuple[Any, ...] = ()
    finite: tuple[tuple[Symbol, tuple[Any, ...]], ...] = ()
    bounds: tuple[Any, ...] = ()

    def __str__(self) -> str:
        from relativisticpy import display

        what = "the greatest" if self.sense == "max" else "the least"
        over = ", ".join(display.unicode(v) for v in self.variables)
        text = f"{what} {display.unicode(self.objective)} over {over}"
        constraints = [*self.relations()]
        if constraints:
            text += " with " + ", ".join(display.unicode(c) for c in constraints)
        return text

    def relations(self) -> Iterator[Any]:
        """The constraints, as relations: h = 0, g ≥ 0, g > 0 and g ≠ 0."""
        yield from (sympy.Eq(h, 0, evaluate=False) for h in self.equalities)
        yield from (sympy.Ge(g, 0, evaluate=False) for g in self.inequalities)
        yield from (sympy.Gt(g, 0, evaluate=False) for g in self.strict)
        yield from (sympy.Ne(g, 0, evaluate=False) for g in self.distinct)

    @property
    def expressions(self) -> tuple[Any, ...]:
        """The objective and every constraint, as expressions."""
        return (self.objective, *self.equalities, *self.inequalities, *self.strict, *self.distinct)

    @property
    def parameters(self) -> tuple[Symbol, ...]:
        """The symbols it is written with other than its variables, by name."""
        found: set[Symbol] = set()
        for expr in self.expressions:
            found |= {s for s in sympy.sympify(expr).free_symbols if isinstance(s, Symbol)}
        return tuple(sorted(found - set(self.variables), key=str))

    @property
    def values(self) -> dict[Symbol, tuple[Any, ...]]:
        """The values each variable that takes finitely many takes."""
        return dict(self.finite)

    @property
    def integers(self) -> tuple[Symbol, ...]:
        """Its variables that are whole numbers."""
        return tuple(v for v in self.variables if v.is_integer)

    def inexact(self) -> bool:
        """Whether it is written with decimals, as data is: numbers known to a precision,
        which a numeric optimizer takes first."""
        values = [each for _, given in self.finite for each in given]
        return any(sympy.sympify(e).atoms(sympy.Float) for e in (*self.expressions, *values))

    def continuous(self) -> bool:
        """Whether each of its variables takes every real value its constraints allow."""
        return not self.finite and not self.integers

    def degree(self) -> int | None:
        """The greatest degree, in the variables, of the objective and the constraints, as
        polynomials in them; ``None`` where one is not a polynomial in them."""
        degrees = []
        for expr in self.expressions:
            expr = sympy.sympify(expr)
            if not expr.has(*self.variables):
                degrees.append(0)
                continue
            try:
                polynomial = sympy.Poly(expr, *self.variables)
            except sympy.PolynomialError:
                return None
            if any(c.has(*self.variables) for c in polynomial.coeffs()):
                return None
            degrees.append(polynomial.total_degree())
        return max(degrees, default=0)

    def linear(self) -> bool:
        """Whether it is a linear program: the objective and every constraint linear in the
        variables, none strict, with numbers for coefficients."""
        degree = self.degree()
        return (
            degree is not None
            and degree <= 1
            and not self.strict
            and not self.distinct
            and self.continuous()
            and not self.parameters
        )

    def smooth(self) -> bool:
        """Whether the objective and the constraints have derivatives in the variables
        wherever they are written: no modulus, extreme, step or cases of them."""
        rough = (sympy.Abs, sympy.Min, sympy.Max, sympy.Piecewise, sympy.sign, sympy.floor,
                 sympy.ceiling, sympy.Heaviside)  # fmt: skip
        return not any(
            each.has(*self.variables)
            for expr in self.expressions
            for each in sympy.sympify(expr).atoms(*rough)
        )

    def ranges(self) -> dict[Symbol, tuple[Any, Any]]:
        """The least and greatest value its constraints allow of each variable, where its
        symbol's sign or a constraint on it alone, linear, says: -∞ and ∞ where none does."""
        found = {v: [-sympy.oo, sympy.oo] for v in self.variables}
        for v in self.variables:
            if v.is_nonnegative:
                found[v][0] = sympy.Integer(1 if v.is_integer and v.is_positive else 0)
            if v.is_nonpositive:
                found[v][1] = sympy.Integer(-1 if v.is_integer and v.is_negative else 0)
        for g in (*self.inequalities, *self.strict):
            g = sympy.expand(g)
            alone = [v for v in self.variables if g.has(v)]
            if len(alone) != 1:
                continue
            v = alone[0]
            slope = g.coeff(v)
            if slope.has(v) or sympy.expand(g - slope * v).has(v) or not slope.is_number:
                continue
            end = -sympy.expand(g - slope * v) / slope
            if slope.is_positive:
                found[v][0] = end if found[v][0] == -sympy.oo else sympy.Max(found[v][0], end)
            elif slope.is_negative:
                found[v][1] = end if found[v][1] == sympy.oo else sympy.Min(found[v][1], end)
        for v, values in self.finite:
            found[v] = [min(values), max(values)]
        return {v: (low, high) for v, (low, high) in found.items()}

    def bounded(self) -> bool:
        """Whether every point its constraints allow lies in a bounded region: each
        variable between finite ends, or a constraint that keeps a positive definite
        quadratic form of all of them at most a constant, as x² + y² = 2 does."""
        if all(low.is_finite and high.is_finite for low, high in self.ranges().values()):
            return True
        kept = [*self.equalities, *(-g for g in self.inequalities), *(-g for g in self.strict)]
        return any(_definite_quadratic(each, self.variables) for each in kept) or any(
            _definite_quadratic(-each, self.variables) for each in self.equalities
        )

    def convex(self) -> bool:
        """Whether it is a convex problem, whose stationary points are its global optimum:
        a convex objective to make least, or a concave one greatest, linear equalities and
        concave inequalities, g ≥ 0, each of degree at most two."""
        if not self.continuous() or self.degree() not in (0, 1, 2):
            return False
        sign = 1 if self.sense == "min" else -1
        if not _semidefinite(sign * self.objective, self.variables):
            return False
        if any(_degree(h, self.variables) > 1 for h in self.equalities):
            return False
        return all(_semidefinite(-g, self.variables) for g in (*self.inequalities, *self.strict))

    def violations(
        self, point: Sequence[Any], tolerance: float = TOLERANCE, *, whole: bool = True
    ) -> list[str]:
        """The constraints that do not hold at ``point``, a value of each variable, as text;
        none where it is feasible. Numbers decide to ``tolerance``, relative to their size.
        Without ``whole``, a variable that is a whole number may be any real one, as where
        the problem is solved over the reals first."""
        from relativisticpy import display

        at = dict(zip(self.variables, point, strict=True))
        broken = []
        for relation in self.relations():
            value = sympy.sympify(relation.lhs).xreplace(at)
            if _breaks(type(relation), value, self.bounds, tolerance):
                broken.append(display.unicode(relation))
        for v, values in self.finite:
            if not any(_near(at[v], each, tolerance) for each in values):
                broken.append(f"{display.unicode(v)} is one of {list(values)}")
        for v in self.integers if whole else ():
            value = sympy.sympify(at[v])
            if value.is_number and not _near(value, sympy.Integer(round(float(value))), tolerance):
                broken.append(f"{display.unicode(v)} is a whole number")
        return broken

    def at(self, point: Sequence[Any]) -> Any:
        """The objective at ``point``."""
        return sympy.sympify(self.objective).xreplace(dict(zip(self.variables, point, strict=True)))

    def renamed(self, symbols: dict[Symbol, Any]) -> Problem:
        """The same problem with ``symbols`` put in place of its parameters."""
        return replace(
            self,
            objective=sympy.sympify(self.objective).xreplace(symbols),
            equalities=tuple(sympy.sympify(e).xreplace(symbols) for e in self.equalities),
            inequalities=tuple(sympy.sympify(e).xreplace(symbols) for e in self.inequalities),
            strict=tuple(sympy.sympify(e).xreplace(symbols) for e in self.strict),
            distinct=tuple(sympy.sympify(e).xreplace(symbols) for e in self.distinct),
            bounds=tuple(sympy.sympify(b).xreplace(symbols) for b in self.bounds),
        )


def _degree(expr: Any, variables: Sequence[Symbol]) -> int:
    try:
        return int(sympy.Poly(expr, *variables).total_degree())
    except sympy.PolynomialError:
        return 99


def _hessian(expr: Any, variables: Sequence[Symbol]) -> Any:
    return sympy.hessian(sympy.sympify(expr), list(variables))


def _semidefinite(expr: Any, variables: Sequence[Symbol]) -> bool:
    """Whether an expression of degree at most two is convex: its Hessian, a matrix of
    numbers, positive semidefinite."""
    if _degree(expr, variables) > 2:
        return False
    hessian = _hessian(expr, variables)
    if hessian.free_symbols:
        return False
    return bool(hessian.is_positive_semidefinite)


def _definite_quadratic(expr: Any, variables: Sequence[Symbol]) -> bool:
    """Whether an expression is a positive definite quadratic form of all the variables,
    plus what is linear in them and a constant: bounded below, and above only where it is
    kept at most a constant."""
    if _degree(expr, variables) != 2:
        return False
    hessian = _hessian(expr, variables)
    return not hessian.free_symbols and bool(hessian.is_positive_definite)


def _breaks(relation: type, value: Any, bounds: Sequence[Any], tolerance: float) -> bool:
    """Whether a constraint, as a relation of ``value`` to 0, does not hold: decided as a
    number to ``tolerance``, or by what is assumed; not where that is not decided."""
    value = sympy.sympify(value)
    if value.is_number:
        number = complex(sympy.N(value, 30))
        if abs(number.imag) > tolerance * max(1.0, abs(number)):
            return True
        real = number.real
        if relation is sympy.Eq:
            return abs(real) > tolerance * max(1.0, _scale(value))
        if relation is sympy.Ne:
            return abs(real) <= tolerance
        if relation is sympy.Gt:
            return real <= 0
        return real < -tolerance * max(1.0, _scale(value))
    verdict = assumptions.holds(relation(value, 0), bounds)
    return verdict is False


def _scale(value: Any) -> float:
    return max((abs(float(a)) for a in sympy.Add.make_args(value) if a.is_number), default=1.0)


def _near(value: Any, other: Any, tolerance: float) -> bool:
    difference = sympy.sympify(value) - sympy.sympify(other)
    if difference.is_number:
        expected = complex(sympy.sympify(other))
        return near(expected + complex(difference), expected, tolerance)
    return bool(sympy.simplify(difference) == 0)


# -- what an optimizer found ------------------------------------------------------------


@dataclass(frozen=True)
class Found:
    """What an optimizer found: the optimum ``value``, and the ``point`` where it is, a
    value of each variable; ``attained`` is false where no point reaches it, as no x
    reaches the infimum 0 of e^{-x}, or it is infinite, and the point is then ``None``, as
    it is where the value is known as a closed form, as min(1 - 2p, 5p - 2), and the point
    is not. ``how`` it was found, in words, as a note gives it, and whether it is
    ``proven`` the global optimum; ``numeric`` where its numbers were found numerically;
    ``tied``, where more points than the one given reach it and that is known, the set of
    them, of points of one variable or of tuples of several, as [-1, 1] or {(0, 1), (1, 0)},
    or the condition every point of a family of them keeps, as x² + y² = 1; and ``by``,
    the optimizer that found it, which the framework sets."""

    value: Any
    point: tuple[Any, ...] | None
    how: str
    proven: bool
    attained: bool = True
    numeric: bool = False
    tied: Any = None
    by: str = ""

    def note(self, sense: str) -> str:
        """What a note says of it: ``the global minimum, found exactly by SymPy, from the
        stationary points and the ends of the domain (extrema)``."""
        exactness = "numerically" if self.numeric else "exactly"
        if self.attained:
            what = "maximum" if sense == "max" else "minimum"
            scope = f"the global {what}" if self.proven else f"a {what}, not proven global"
        else:
            what = "supremum" if sense == "max" else "infimum"
            scope = f"the {what}, which no point reaches"
            scope += "" if self.proven else ", not proven"
        text = f"{scope}, found {exactness} {self.how} ({self.by})"
        if self.tied is not None and self.point is not None:
            text += f"; {_reached(self.tied, self.point)}"
        return text


#: The most points that reach an optimum a note lists; of more, it says how many.
_MOST_LISTED = 6


def _reached(tied: Any, point: tuple[Any, ...]) -> str:
    """Where an optimum is reached, as a note says it: ``it is reached at -1 and 1, and
    -1 is given``, or ``every point of [-1, 1] reaches it, and -1 is given``."""
    given = _shown(point[0] if len(point) == 1 else sympy.Tuple(*point))
    if isinstance(tied, sympy.FiniteSet) and len(tied.args) > _MOST_LISTED:
        return f"{len(tied.args)} points reach it, and {given} is given"
    if isinstance(tied, sympy.FiniteSet):
        points = [_shown(each) for each in tied.args]
        return f"it is reached at {assumptions.listed(points)}, and {given} is given"
    if isinstance(tied, sympy.Interval):
        return f"every point of {_set_shown(tied)} reaches it, and {given} is given"
    if isinstance(tied, sympy.Set):
        return f"it is reached at {_set_shown(tied)}, and {given} is given"
    return f"every point where {_shown(tied)} reaches it, and {given} is given"


def _set_shown(found: Any) -> str:
    """A set of points as a note says it: ``[-1, 1]``, or ``π/2 + 2πn, of every integer
    n``, or several, listed."""
    if isinstance(found, sympy.Interval):
        left, right = "(" if found.left_open else "[", ")" if found.right_open else "]"
        return f"{left}{_shown(found.start)}, {_shown(found.end)}{right}"
    if isinstance(found, sympy.FiniteSet):
        return assumptions.listed([_shown(each) for each in found.args])
    if isinstance(found, sympy.ImageSet) and found.base_sets == (sympy.S.Integers,):
        (label,) = found.lamda.variables
        return f"{_shown(found.lamda.expr)}, of every integer {_shown(label)}"
    if isinstance(found, sympy.Intersection | sympy.Union):
        parts = [_set_shown(each) for each in found.args]
        return (" within " if isinstance(found, sympy.Intersection) else " and ").join(parts)
    return _shown(found)


class Undecided(Exception):
    """An optimum an optimizer cannot tell for not knowing the signs of parameters, or the
    bounds on them: which point is best, or whether it is a point of the optimum at all."""


# -- the kind -------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Optimizer(Capability):
    """What finds the optimum of a problem: its implementation ``takes`` a :class:`Problem`
    it can solve, and finds its ``optimum``, a :class:`Found`, or ``None`` where it finds
    none. ``numerical`` says whether its answers are numbers found numerically, as SciPy's
    are, rather than closed forms, as SymPy's are. The problems it is checked at are its
    ``points``, each a problem alone."""

    kind: ClassVar[str] = "optimizer"
    takes: ClassVar[str] = "a problem of optimisation"

    numerical: bool = False

    @property
    def exact(self) -> Callable[..., Any]:
        """What finds the optimum: its implementation's ``optimum``, given a problem."""
        method: Callable[..., Any] = self.implementation.optimum
        return method

    @property
    def problems(self) -> tuple[Problem, ...]:
        """The problems it is checked at."""
        return tuple(point[0] for point in self.points)

    def solves(self, problem: Problem) -> bool:
        """Whether it takes ``problem``."""
        return bool(self.implementation.takes(problem))


def optimizer(
    name: str,
    *,
    summary: str,
    examples: Sequence[Example],
    problems: Sequence[Problem],
    requires: str | None = None,
    numerical: bool = False,
) -> Callable[[type], Optimizer]:
    """Declare an optimizer, checked at ``problems``. Its class has a ``takes`` method,
    which says whether it solves a problem, and an ``optimum`` method, which solves it;
    ``numerical`` where it finds numbers numerically, as SciPy does."""

    def decorate(cls: type) -> Optimizer:
        for method, does in (("takes", "says whether it solves a problem"),
                             ("optimum", "finds the optimum of a problem")):  # fmt: skip
            if not callable(getattr(cls, method, None)):
                raise TypeError(
                    f"{name}: {cls.__name__} has no {method} method, which {does}; declare "
                    f"def {method}(self, problem): ..."
                )
        return Optimizer(
            cls(),
            name,
            summary,
            tuple(examples),
            requires=requires,
            points=tuple((problem,) for problem in problems),
            numerical=numerical,
        )

    return decorate


def optimizers() -> list[Optimizer]:
    """Every optimizer, in the order they are consulted: those that compute exactly
    first, then those that compute numerically, each in the order declared."""
    found = [c for c in catalogue().capabilities if isinstance(c, Optimizer)]
    return [c for c in found if not c.numerical] + [c for c in found if c.numerical]


# -- finding an optimum ---------------------------------------------------------------------


_ANSWERS: list[list[Found]] = []


@contextlib.contextmanager
def answering() -> Iterator[list[Found]]:
    """The answers the optimizers give while the block runs, in order: which optimizer
    answered an example, for the conformance suite."""
    found: list[Found] = []
    _ANSWERS.append(found)
    try:
        yield found
    finally:
        _ANSWERS.remove(found)


def find(problem: Problem) -> Found:
    """The optimum of ``problem``, as :func:`_found` finds it, of what is under a root where
    what it is of is a root of what is never negative, as the length of a vector is: the
    same points make both greatest or least, and the problem under the root is simpler,
    quadratic where the length is of a linear expression."""
    objective = sympy.sympify(problem.objective)
    if (
        isinstance(objective, sympy.Pow)
        and objective.exp.is_positive
        and objective.exp.is_number
        and objective.base.has(*problem.variables)
        and objective.base.is_nonnegative
    ):
        found = _found(replace(problem, objective=objective.base))
        value = sympy.sympify(found.value) ** objective.exp
        if found.point is not None:
            value = problem.at(found.point)
        return replace(found, value=_tidy(value))
    return _found(problem)


def _tidy(value: Any) -> Any:
    """A value an optimizer found, written with parameters, in its shortest form, expanded
    where that is as short as any, as -ν²/2 - ν is, or simplified, as c√(c²m² + p²) is."""
    value = sympy.sympify(value)
    if not value.free_symbols:
        return value
    forms = [sympy.expand(value)]
    with contextlib.suppress(TypeError, ValueError, NotImplementedError):
        forms.append(sympy.simplify(value))
    return min([*forms, value], key=sympy.count_ops)  # the first of the shortest


def _found(problem: Problem) -> Found:
    """The optimum of ``problem``, from the first optimizer that takes it and finds it:
    an exact one, with the problem as it is, then with the parameters nothing is said of
    real, then with the fewest of them positive, as books take scales to be; else a numeric
    one, where the problem has no parameters. A problem written with decimals, as data is,
    goes to the numeric ones first. An error that says why where none does."""
    exact = [c for c in optimizers() if not c.numerical]
    numeric = [c for c in optimizers() if c.numerical]
    taken: list[str] = []
    if problem.inexact() and not problem.parameters:
        for capability in numeric:
            if capability.solves(problem):
                taken.append(capability.name)
                found = _solved(capability, problem, None)
                if found is not None:
                    return _answered(replace(found, by=capability.name))
        numeric = []
    deadline = time.monotonic() + 2 * EXACT_BUDGET
    undecided: Undecided | None = None
    for assumed, positive in _signs(problem):
        if assumed is not problem and time.monotonic() > deadline:
            break  # what is left of the time goes to the numeric optimizers
        for capability in exact:
            if not capability.solves(assumed):
                continue
            taken.append(capability.name)
            try:
                found = _solved(capability, assumed, EXACT_BUDGET)
            except Undecided as why:
                undecided = undecided or why
                continue
            if found is not None:
                if assumed is not problem:
                    found = _given(found, problem, assumed, positive)
                found = replace(found, value=_tidy(found.value), by=capability.name)
                return _answered(found)
    if not problem.parameters:
        for capability in numeric:
            if capability.solves(problem):
                taken.append(capability.name)
                found = _solved(capability, problem, None)
                if found is not None:
                    return _answered(replace(found, by=capability.name))
    raise _not_found(problem, list(dict.fromkeys(taken)), undecided)


def _solved(capability: Optimizer, problem: Problem, budget: float | None) -> Found | None:
    """What an optimizer finds, within ``budget`` seconds where there is one: ``None``
    where it finds nothing, fails, or runs out of time. A missing library is said."""
    from relativisticpy.integrations._discovery import missing
    from relativisticpy.runtime.core import _budget, _OverBudget

    limit = _budget(budget) if budget is not None else contextlib.nullcontext()
    try:
        with limit:
            found: Found | None = capability.exact(problem)
    except (RelativisticPyError, Undecided):
        raise
    except ImportError as error:
        raise missing(capability.name, capability, error) from None
    except _OverBudget:
        return None
    except Exception:  # SymPy or SciPy could not: the next optimizer may
        return None
    return found


def _answered(found: Found) -> Found:
    for answers in _ANSWERS:
        answers.append(found)
    return found


def _signs(problem: Problem) -> Iterator[tuple[Problem, dict[Symbol, Symbol]]]:
    """The problem as it is, then with its parameters nothing is said of real, then with
    the fewest of them positive, one or two at a time, then all: each with the positive
    symbols put for the parameters taken positive."""
    yield problem, {}
    general = [s for s in problem.parameters if assumptions._general(s)]
    if not general:
        return
    real = {s: Symbol(s.name, real=True) for s in general}
    yield problem.renamed(real), {}
    tried = [list(c) for size in (1, 2) for c in itertools.combinations(general, size)]
    if len(general) > 2:
        tried.append(general)
    for chosen in tried:
        positive = {s: Symbol(s.name, positive=True) for s in chosen}
        yield problem.renamed({**real, **positive}), positive


def _given(found: Found, problem: Problem, assumed: Problem, positive: dict[Any, Any]) -> Found:
    """What was found under assumptions about the parameters, written with the program's
    own symbols again, and the remark that says what was assumed."""
    back = {
        s: original
        for original in problem.parameters
        for s in assumed.parameters
        if s.name == original.name
    }
    value = _tidy(found.value).xreplace(back)  # simplified where the assumption holds
    point = None
    if found.point is not None:
        point = tuple(sympy.sympify(p).xreplace(back) for p in found.point)
    if positive:  # said as an integral under the same assumption says it
        value = assumptions._assumed(value, positive)
    return replace(found, value=value, point=point)


def _not_found(problem: Problem, taken: list[str], undecided: Undecided | None) -> EvaluationError:
    """The error for an optimum no optimizer finds, which says why."""
    if undecided is not None:
        return EvaluationError(
            f"the optimum of {problem} is not decided: {undecided}",
            hint="say what the parameters are, as a > 0, before the optimum",
        )
    if not taken:
        return EvaluationError(
            f"no optimizer takes {problem}",
            notes=[
                "an optimum is found exactly over real or whole numbers, with constraints "
                "of equations and inequalities, and numerically where it has no parameters"
            ],
        )
    tried = assumptions.listed(taken)
    if problem.parameters:
        from relativisticpy import display

        named = assumptions.listed([display.unicode(s) for s in problem.parameters])
        return EvaluationError(
            f"no closed form was found for {problem} ({tried} tried), and, with the "
            f"parameters {named}, it cannot be found numerically",
            hint="give the parameters values, to find it numerically",
        )
    return EvaluationError(f"no optimum was found for {problem}: {tried} tried, and found none")


# -- what the program writes ---------------------------------------------------------------


def optimum(
    objective: Any,
    *rest: Any,
    sense: str,
    goal: str,
    unknowns: int,
    relations: tuple[str, ...],
    domains: tuple[tuple[Any, ...], ...],
    shape: tuple[Any, ...],
    written: str,
    assuming: tuple[Any, ...] = (),
) -> Any:
    """The optimum a program writes, as the IR computes it: ``objective`` is what it is of,
    the first ``unknowns`` of ``rest`` its variables and the others the two sides of each
    constraint, each of ``relations``, componentwise where a side is a vector. ``sense`` is
    "max", "min", "sup" or "inf"; ``goal`` "value", or "point" for where it is, in the
    ``shape`` of its variables. ``domains`` says what each variable takes beyond what its
    symbol says: () for every value, or ("values", (0, 1)). A remark says which optimizer
    found it, how, and whether it is global."""
    variables = tuple(rest[:unknowns])
    sides = rest[unknowns:]
    refused = no_number(objective, f"{written} is of a number")
    if refused is not None:
        raise refused
    problem = _problem(objective, variables, sides, sense, relations, domains, written)
    problem = replace(problem, bounds=tuple(assuming))
    _refuse_unknown_functions(problem, written)
    found = find(problem)
    bound = sense in ("sup", "inf")
    extreme = "maximum" if problem.sense == "max" else "minimum"
    if not bound and not found.attained:
        infinite = sympy.sympify(found.value).is_infinite
        if infinite:
            why = f"what it is of grows without bound, to {_shown(found.value)}"
        else:
            near = _shown(found.value)
            why = f"its values come as near as you like to {near}, which no point reaches"
        word = "sup" if problem.sense == "max" else "inf"
        raise EvaluationError(
            f"{written} has no {extreme}: {why}",
            hint=f"the bound it never reaches is \\{word}, as in \\{word}_{{x}} f(x)",
        )
    assumptions.remark(found.note(problem.sense))
    if goal == "value":
        return found.value
    if found.point is None:
        raise EvaluationError(
            f"where the {extreme} of {written} is depends on its parameters, which do not "
            f"decide it: its value is {_shown(found.value)}",
            hint="say what the parameters are, as a > 0, or give them values",
        )
    return _shaped(found.point, shape)


def _shown(value: Any) -> str:
    from relativisticpy import display

    return display.unicode(value)


#: Why an optimum of an unknown function is not found.
_BELLMAN = "an equation for an unknown function with an optimum in it, as Bellman's, is not solved"


def _refuse_unknown_functions(problem: Problem, written: str) -> None:
    """Refuse an optimum of a function nothing defines, of its variables, as Bellman's
    equation writes one, V(k - c): what it is greatest at is not known."""
    from sympy.core.function import AppliedUndef

    for expr in problem.expressions:
        for call in sympy.sympify(expr).atoms(AppliedUndef):
            if call.has(*problem.variables):
                raise EvaluationError(
                    f"{written} is of {_shown(call)}, a function the program does not define, "
                    "so its optimum cannot be found",
                    notes=[_BELLMAN],
                    hint=f"define {_shown(call.func)} before the optimum",
                )


def _problem(
    objective: Any,
    variables: tuple[Symbol, ...],
    sides: Sequence[Any],
    sense: str,
    relations: Sequence[str],
    domains: Sequence[tuple[Any, ...]],
    written: str,
) -> Problem:
    """The problem a program writes, from the values the IR computed: what it is of, a
    number, and each constraint, componentwise where its sides are vectors, as ``y \\geq 0``
    of a vector y is a constraint on each component."""
    value = _number(objective, written)
    by_name = {v.name: v for v in variables}
    same = {
        s: by_name[s.name]
        for s in sympy.sympify(value).free_symbols
        if isinstance(s, Symbol) and s.name in by_name and s not in variables
    }
    lists: dict[str, list[Any]] = {"=": [], ">=": [], ">": [], "!=": []}
    for relation, lhs, rhs in zip(relations, sides[0::2], sides[1::2], strict=True):
        kept = relation.replace("<", ">")  # a <= b is b - a >= 0
        for left, right in _componentwise(lhs, rhs, written):
            difference = sympy.sympify(left - right).xreplace(same)
            lists[kept].append(-difference if relation in ("<=", "<") else difference)
    for v in variables:  # what a variable's symbol says of its sign, as a constraint too
        for fact, kept, expr in (("positive", ">", v), ("nonnegative", ">=", v),
                                 ("negative", ">", -v), ("nonpositive", ">=", -v)):  # fmt: skip
            if getattr(v, f"is_{fact}") and not any(expr == each for each in lists[kept]):
                lists[kept].append(expr)
                break
    finite = tuple(
        (v, tuple(sympy.sympify(each) for each in domain[1]))
        for v, domain in zip(variables, domains, strict=True)
        if domain and domain[0] == "values"
    )
    return Problem(
        "max" if sense in ("max", "sup") else "min",
        sympy.sympify(value).xreplace(same),
        variables,
        equalities=tuple(lists["="]),
        inequalities=tuple(lists[">="]),
        strict=tuple(lists[">"]),
        distinct=tuple(lists["!="]),
        finite=finite,
    )


def _number(value: Any, written: str) -> Any:
    """What an optimum is of, as one number: a 1 × 1 matrix, as w^T Σ w gives, is its
    entry."""
    entries = _entries(value)
    if entries is None:
        return sympy.sympify(value)
    if len(entries) == 1:
        return entries[0]
    raise EvaluationError(
        f"what {written} is of has {len(entries)} values, not one: an optimum is of a number",
        hint="write what it is of as one number, as the length of a vector, \\|x\\|",
    )


def _entries(value: Any) -> list[Any] | None:
    """The components of a vector or a matrix, in order; ``None`` for a number."""
    if isinstance(value, sympy.NDimArray | sympy.MatrixBase):
        return [sympy.sympify(each) for each in sympy.flatten(value.tolist())]
    if isinstance(value, list | tuple):
        return [sympy.sympify(each) for each in sympy.flatten(value)]
    if hasattr(value, "shape") and getattr(value, "shape", ()) != ():
        import numpy as np

        return [sympy.sympify(each) for each in np.asarray(value).ravel().tolist()]
    if hasattr(value, "components"):  # a tensor
        return _entries(value.components)
    return None


def _componentwise(lhs: Any, rhs: Any, written: str) -> list[tuple[Any, Any]]:
    """The two sides of a constraint, a pair of numbers for each component where a side is
    a vector, the other side, if a number, the same for each."""
    left, right = _entries(lhs), _entries(rhs)
    if left is None and right is None:
        return [(sympy.sympify(lhs), sympy.sympify(rhs))]
    if left is None:
        left = [sympy.sympify(lhs)] * len(right or [])
    if right is None:
        right = [sympy.sympify(rhs)] * len(left)
    if len(left) != len(right):
        raise EvaluationError(
            f"a constraint of {written} relates {len(left)} values to {len(right)}: its two "
            "sides are vectors of different sizes"
        )
    return list(zip(left, right, strict=True))


def _shaped(point: tuple[Any, ...], shape: tuple[Any, ...]) -> Any:
    """Where an optimum is, in the shape of its variables: a number, a list of numbers,
    a column vector, or the components of a tensor."""
    if shape[0] == "scalar":
        return point[0]
    if shape[0] == "column":
        return sympy.ImmutableDenseNDimArray(list(point), (len(point), 1))
    if shape[0] == "tensor":
        return sympy.ImmutableDenseNDimArray(list(point), (len(point),))
    return list(point)


# -- generalized inequalities -------------------------------------------------------------


def order(lhs: Any, rhs: Any, *, relation: str, written: str) -> Any:
    """A generalized inequality, as Boyd and Vandenberghe write it: of a symmetric matrix
    and 0, or two such matrices, that their difference is positive definite (≻), positive
    semidefinite (⪰), or negative so (≺, ⪯); of vectors, that it holds of each component."""
    left, right = _entries(lhs), _entries(rhs)
    matrix = _square(lhs) or _square(rhs)
    if matrix:
        size = _square(lhs) or _square(rhs)
        first = sympy.Matrix(size, size, left) if left is not None else sympy.zeros(size, size)
        second = sympy.Matrix(size, size, right) if right is not None else sympy.zeros(size, size)
        if (left is None and lhs != 0) or (right is None and rhs != 0):
            raise EvaluationError(
                f"{written} compares a matrix with a number other than 0",
                hint="compare a matrix with 0, as A \\succ 0, or with a matrix of its size",
            )
        difference = first - second if relation in (">", ">=") else second - first
        if difference != difference.T:
            raise EvaluationError(
                f"{written} says what a matrix that is not symmetric is: definiteness is of "
                "symmetric matrices"
            )
        verdict = (
            difference.is_positive_definite
            if relation in (">", "<")
            else difference.is_positive_semidefinite
        )
        if verdict is None:
            raise EvaluationError(f"whether {written} holds is not decided")
        return sympy.true if verdict else sympy.false
    pairs = _componentwise(lhs, rhs, written)
    kind = {">": sympy.Gt, ">=": sympy.Ge, "<": sympy.Lt, "<=": sympy.Le}[relation]
    return sympy.And(*(kind(a, b) for a, b in pairs))


def _square(value: Any) -> int:
    """The size of a square matrix of more than one entry; 0 for anything else."""
    shape: tuple[int, ...] = tuple(getattr(value, "shape", ()))
    if len(shape) == 2 and shape[0] == shape[1] and shape[0] > 1:
        return int(shape[0])
    return 0


# -- the optima of data over an index -------------------------------------------------------


def extreme(values: Any, *, op: str, axis: int, labels: tuple[Any, ...], written: str) -> Any:
    """The greatest or least value of data, or of the components of a tensor, along
    ``axis``, as ``\\max_t x_t`` writes it; for ``argmax`` and ``argmin``, the label of the
    element where it is, as ``labels`` names each, the first where several share it, which
    a remark says."""
    import numpy as np

    from relativisticpy.tensors import numerical

    greatest = op in ("max", "argmax", "sup")
    if numerical.is_numerical(values):
        array = np.asarray(numerical.dense(numerical.as_array(values)))
        if np.iscomplexobj(array):
            raise EvaluationError(
                f"{written} compares complex numbers, which have no order",
                hint="compare their moduli, as |x_t|, or their real parts",
            )
        moved = np.moveaxis(array, axis, -1)
        if op in ("max", "min", "sup", "inf"):
            return (np.max if greatest else np.min)(moved, axis=-1)
        best = (np.max if greatest else np.min)(moved, axis=-1, keepdims=True)
        tied = np.isclose(moved, best, rtol=1e-12, atol=0.0)
        positions = np.argmax(tied, axis=-1)
        _remark_ties(int(np.max(tied.sum(axis=-1))), written, labels, tied)
        found = [_label(labels, int(p)) for p in np.ravel(positions)]
        return (
            found[0]
            if positions.ndim == 0
            else sympy.ImmutableDenseNDimArray(found, positions.shape)
        )
    array = (
        sympy.ImmutableDenseNDimArray(values) if not isinstance(values, sympy.NDimArray) else values
    )
    items = sympy.permutedims(array, _last(array.rank(), axis)) if array.rank() > 1 else array
    shape = items.shape[:-1]
    size = items.shape[-1]
    flat = list(sympy.flatten(items.tolist())) if items.rank() > 0 else [items]
    rows = [flat[k * size : (k + 1) * size] for k in range(len(flat) // size)]
    found = []
    for row in rows:
        if op in ("max", "min", "sup", "inf"):
            found.append(sympy.Max(*row) if greatest else sympy.Min(*row))
            continue
        position = _best_position(row, greatest, written)
        found.append(_label(labels, position))
        if not shape:
            sharing = [k for k, v in enumerate(row) if sympy.simplify(v - row[position]) == 0]
            _remark_ties(len(sharing), written, labels, sharing)
    if not shape:
        return found[0]
    return sympy.ImmutableDenseNDimArray(found, shape)


def _last(rank: int, axis: int) -> list[int]:
    order = [k for k in range(rank) if k != axis]
    return [*order, axis]


def _best_position(row: Sequence[Any], greatest: bool, written: str) -> int:
    """The position of the greatest or least of values, the first of those that share it."""
    best = 0
    for position, value in enumerate(row[1:], start=1):
        difference = sympy.sympify(value - row[best])
        verdict = difference.is_positive if greatest else difference.is_negative
        if verdict is None and difference.is_number:
            number = float(sympy.N(difference))
            verdict = number > 0 if greatest else number < 0
        if verdict is None:
            raise EvaluationError(
                f"{written} cannot tell which of its values is "
                f"{'greatest' if greatest else 'least'}: they are not numbers",
            )
        if verdict:
            best = position
    return best


def _label(labels: Sequence[Any], position: int) -> Any:
    """The label of the element at ``position``: its name in the basis of its space, a
    number where it is written as one, as 5 or 1/8, or else its position."""
    if not labels:
        return sympy.Integer(position)
    name = str(labels[position])
    return sympy.sympify(name) if _A_NUMBER.fullmatch(name) else name


#: A label written as a number: 5, -0.25, 1/8 or 2.5e-3.
_A_NUMBER = re.compile(r"-?(?:\d+(?:/\d+)?|\d*\.\d+(?:e[+-]?\d+)?)")


def _remark_ties(count: int, written: str, labels: Sequence[Any], tied: Any) -> None:
    """The remark that the optimum of data is reached at several of its elements, of which
    the first is given: ``tied`` marks them, or lists their positions."""
    if count <= 1:
        return
    import numpy as np

    if isinstance(tied, list) or tied.ndim == 1:
        positions = tied if isinstance(tied, list) else np.flatnonzero(tied)
        named = [str(_label(labels, int(p))) for p in positions]
        assumptions.remark(
            f"{written}: the optimum is reached at {assumptions.listed(named)}, and the first "
            "is given"
        )
    else:
        assumptions.remark(
            f"{written}: the optimum is reached at several labels, and the first is given"
        )


# -- the checks of the conformance suite ---------------------------------------------------

#: How far the values of two optimizers may differ, relative to their size.
AGREEMENT = 1e-6
#: The values put for the parameters of a problem an optimizer is checked at, in turn.
_SAMPLES = (sympy.Rational(13, 10), sympy.Rational(21, 10), sympy.Rational(37, 10))


def _sampled(problem: Problem) -> dict[Symbol, Any]:
    return {s: _SAMPLES[k % len(_SAMPLES)] for k, s in enumerate(problem.parameters)}


def _reported(capability: Optimizer, problem: Problem) -> tuple[Found | None, list[str]]:
    """What an optimizer reports at a problem it is checked at, or why there is nothing."""
    name = capability.name
    if not capability.solves(problem):
        return None, [
            f"{name} does not take the problem it is checked at, {problem}: declare problems "
            "it solves, or fix takes"
        ]
    try:
        found = capability.exact(problem)
    except RelativisticPyError as error:
        return None, [f"{name} fails at {problem}: {error.kind}: {error.message}"]
    except Exception as error:
        return None, [f"{name} fails at {problem}: {type(error).__name__}: {error}"]
    if not isinstance(found, Found):
        return None, [f"{name} finds no optimum of {problem}, which it takes: fix optimum"]
    return found, []


def feasibility(capability: Capability) -> list[str]:
    """At each problem an optimizer is checked at, the point it reports keeps every
    constraint, and the value it reports is the objective there."""
    assert isinstance(capability, Optimizer)
    failures: list[str] = []
    if not capability.problems:
        return [
            f"{capability.name}: declare problems=[Problem(...), ...], problems it solves, at "
            "which it is checked"
        ]
    for problem in capability.problems:
        found, failed = _reported(capability, problem)
        failures += failed
        if found is None or found.point is None:
            continue
        sample = _sampled(problem)
        point = [sympy.sympify(p).xreplace(sample) for p in found.point]
        checked = problem.renamed(sample)
        broken = checked.violations(point, TOLERANCE if found.numeric else 1e-10)
        if broken:
            failures.append(
                f"{capability.name}: at {problem}, the point it reports, {_shown(found.point)}, "
                f"breaks {', '.join(broken)}: fix optimum"
            )
        value = sympy.sympify(found.value).xreplace(sample)
        if not _near(checked.at(point), value, AGREEMENT if found.numeric else 1e-10):
            failures.append(
                f"{capability.name}: at {problem}, the value it reports, {_shown(found.value)}, "
                f"is not the objective at the point it reports, {_shown(problem.at(found.point))}"
            )
    return failures


def optimality(capability: Capability) -> list[str]:
    """At each problem an optimizer is checked at, the conditions of optimality hold at
    the point it reports: of a smooth problem, the KKT conditions, the gradient of the
    objective a combination of the gradients of the constraints that hold with equality
    there, with multipliers of the right signs, and the second-order condition, that the
    objective curves the right way along the directions that keep them; of one over
    finitely many points, no feasible point is better."""
    assert isinstance(capability, Optimizer)
    failures: list[str] = []
    for problem in capability.problems:
        found, _ = _reported(capability, problem)
        if found is None or found.point is None or not found.attained:
            continue
        sample = _sampled(problem)
        checked = problem.renamed(sample)
        point = [sympy.sympify(p).xreplace(sample) for p in found.point]
        if checked.finite or checked.integers:
            failures += _no_better(capability.name, checked, point, found)
        elif checked.smooth():
            failures += _kkt(capability.name, checked, point, found)
    return failures


def _no_better(name: str, problem: Problem, point: list[Any], found: Found) -> list[str]:
    """Of a problem over finitely many points, that no feasible point is better than the
    one reported."""
    values = problem.values
    if set(values) != set(problem.variables):
        return []
    best = problem.at(point)
    for candidate in itertools.product(*(values[v] for v in problem.variables)):
        if problem.violations(candidate):
            continue
        better = problem.at(candidate) - best
        if (better > 0) if problem.sense == "max" else (better < 0):
            return [
                f"{name}: at {problem}, {_shown(candidate)} is better than the point it reports, "
                f"{_shown(tuple(point))}: fix optimum"
            ]
    return []


def _kkt(name: str, problem: Problem, point: list[Any], found: Found) -> list[str]:
    """The KKT conditions of a smooth problem at a point: ∇f = Σ λ ∇h + Σ μ ∇g, with
    μ ≥ 0 for each g ≥ 0 that holds with equality, to make f least; of the greatest, the
    same of -f. And the second-order condition, which tells a minimum from a saddle or a
    maximum that keeps them too: along no direction that keeps those constraints does the
    Lagrangian curve downward."""
    import numpy as np

    sign = 1 if problem.sense == "min" else -1
    at = dict(zip(problem.variables, point, strict=True))
    tolerance = 1e-6 if found.numeric else 1e-9

    def gradient(expr: Any) -> Any:
        return np.array(
            [complex(sympy.N(sympy.diff(expr, v).xreplace(at), 30)).real for v in problem.variables]
        )

    active = [
        g
        for g in problem.inequalities
        if abs(complex(sympy.N(sympy.sympify(g).xreplace(at), 30))) <= 1e-7 * max(1.0, _scale(g))
    ]
    target = gradient(sign * problem.objective)
    columns = [gradient(h) for h in problem.equalities] + [gradient(g) for g in active]
    if columns:
        matrix = np.column_stack(columns)
        multipliers, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        residual = target - matrix @ multipliers
    else:
        multipliers, residual = np.zeros(0), target
    scale = max(1.0, float(np.max(np.abs(target))))
    if float(np.max(np.abs(residual), initial=0.0)) > tolerance * scale:
        return [
            f"{name}: at {problem}, the point it reports, {_shown(found.point)}, is not "
            f"stationary: the gradient of the objective, {np.round(sign * target, 9).tolist()}, "
            "is not a combination of the gradients of the constraints that hold there with "
            "equality: fix optimum"
        ]
    signs = multipliers[len(problem.equalities) :]
    if np.any(signs < -tolerance * scale):
        return [
            f"{name}: at {problem}, the point it reports, {_shown(found.point)}, keeps the "
            "first-order conditions with a multiplier of the wrong sign, so that moving into "
            "the feasible region improves the objective: fix optimum"
        ]
    constraints = [*problem.equalities, *active]
    lagrangian = sign * sympy.sympify(problem.objective) - sum(
        (float(m) * sympy.sympify(c) for m, c in zip(multipliers, constraints, strict=True)),
        sympy.Integer(0),
    )
    curvature = _curvature(lagrangian, problem.variables, at, columns)
    if curvature is not None and curvature[0] < -max(tolerance, 1e-7) * curvature[1]:
        kind = "minimum" if problem.sense == "min" else "maximum"
        return [
            f"{name}: at {problem}, the point it reports, {_shown(found.point)}, keeps the "
            f"first-order conditions but is no {kind}: along a direction that keeps the "
            "constraints that hold there with equality, the objective curves the other way: "
            "fix optimum"
        ]
    return []


def _curvature(
    lagrangian: Any, variables: Sequence[Symbol], at: dict[Symbol, Any], gradients: Sequence[Any]
) -> tuple[float, float] | None:
    """The least curvature of a Lagrangian at a point along the directions in which every
    constraint that holds there with equality, of the ``gradients`` given, stays so, with
    the size of its Hessian: the least eigenvalue of the Hessian on those directions, which
    is not negative at a minimum. ``None`` where no direction is left, as at a vertex, or
    the Hessian is no matrix of numbers."""
    import numpy as np

    try:
        hessian = np.array(
            [
                [complex(sympy.N(sympy.diff(lagrangian, u, v).xreplace(at), 30)).real
                 for v in variables]
                for u in variables
            ]
        )  # fmt: skip
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(hessian)):
        return None
    if gradients:
        held = np.vstack(gradients)
        _, sizes, rows = np.linalg.svd(held)
        rank = int(np.sum(sizes > 1e-9 * max(1.0, float(sizes.max(initial=0.0)))))
        along = rows[rank:].T  # the directions every gradient is at right angles to
        if along.shape[1] == 0:
            return None
        hessian = along.T @ hessian @ along
    least = float(np.linalg.eigvalsh((hessian + hessian.T) / 2).min())
    return least, max(1.0, float(np.max(np.abs(hessian))))


def agreement(capability: Capability) -> list[str]:
    """At each problem an optimizer is checked at, the optimizers of the other kind, exact
    or numeric, that take it find the same optimum: the same value, where both prove it
    global; never a better one than an optimum proven global; and never one strictly better
    than its own, at a point that keeps the constraints, proven or not."""
    assert isinstance(capability, Optimizer)
    failures: list[str] = []
    others = [c for c in optimizers() if c.numerical is not capability.numerical]
    for problem in capability.problems:
        if problem.parameters:
            continue
        found, _ = _reported(capability, problem)
        if found is None:
            continue
        for other in others:
            if not other.solves(problem):
                continue
            theirs, failed = _reported(other, problem)
            if theirs is None:
                failures += failed
                continue
            failures += _compared(capability, found, other, theirs, problem)
    return failures


def _compared(
    one: Optimizer, found: Found, other: Optimizer, theirs: Found, problem: Problem
) -> list[str]:
    """What is wrong where two optimizers find different optima of a problem: the other's is
    strictly better than this one's and is the objective at a point that keeps the
    constraints, so that this one's is no optimum, whatever it claims; or one is proven
    global, and the other is better, or proven too."""
    a, b = sympy.sympify(found.value), sympy.sympify(theirs.value)
    x, y = float(sympy.N(a)), float(sympy.N(b))  # an infinite one as inf
    if x == y or abs(x - y) <= AGREEMENT * max(1.0, abs(x), abs(y)):
        return []
    theirs_better = (y > x) if problem.sense == "max" else (y < x)
    if theirs_better and _at_a_point_that_keeps(theirs, problem):
        return [
            f"{one.name} gives {_shown(found.value)} at {problem}, and {other.name} finds "
            f"better, {_shown(theirs.value)}, at {_shown(theirs.point)}, which keeps the "
            f"constraints: fix {one.name}"
        ]
    if (found.proven and theirs.proven) or (found.proven and theirs_better):
        return [
            f"{one.name} and {other.name} disagree at {problem}: {one.name} gives "
            f"{_shown(found.value)}, {other.name} {_shown(theirs.value)}: fix the one that is "
            "wrong"
        ]
    if theirs.proven and not theirs_better:
        return [
            f"{one.name} finds {_shown(found.value)} at {problem}, better than the optimum "
            f"{other.name} proves global, {_shown(theirs.value)}: fix the one that is wrong"
        ]
    return []


def _at_a_point_that_keeps(found: Found, problem: Problem) -> bool:
    """Whether what an optimizer found is the objective at a point it gives, which keeps
    the constraints of the problem."""
    if found.point is None or not found.attained:
        return False
    tolerance = TOLERANCE if found.numeric else 1e-10
    if problem.violations(found.point, tolerance):
        return False
    return _near(problem.at(found.point), found.value, AGREEMENT if found.numeric else 1e-10)
