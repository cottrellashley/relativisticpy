r"""The core operations: what the compiler emits for notation that is no function of the
language, such as a sum, an evaluation bar or a time derivative, computed by the engines.

Each is registered by :mod:`relativisticpy.runtime.registry` under a name of its own, as
``core.sum`` or ``core.subs``, and is a function of the values of its arguments, then of
its attributes, by name: module-level, so that a worker process can run it.
"""

from __future__ import annotations

import contextlib
import gc
import inspect
import itertools
import math
import re
import signal
import threading
import time
import weakref
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from types import CodeType, FrameType
from typing import Any

import sympy
from sympy import Symbol
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational

from relativisticpy import assumptions, display, integrations
from relativisticpy.diffgeom.metric import differential
from relativisticpy.errors import EvaluationError, MetricError
from relativisticpy.tensors import Indices, Tensor, arrays, linalg, numerical
from relativisticpy.tensors.equation import TensorEquation
from relativisticpy.tensors.indices import latex_label
from relativisticpy.tensors.simplification import LARGE, is_zero, normal_trig

# Spaces up to this size get SymPy's identity and ones, which stay exact with symbols.
_SMALL = 16


# -- collections ---------------------------------------------------------------------


def make_list(*items: Any) -> list[Any]:
    return list(items)


def make_tuple(*items: Any) -> tuple[Any, ...]:
    return tuple(items)


def limits(variable: Any, lower: Any, upper: Any) -> tuple[Any, Any, Any]:
    """The limits of an integral, each a value, or written as the variable equal to one,
    as ``\\int_{x=0}^{x=1}`` writes them (Riley, Hobson and Bence (2006), section 6.1)."""
    return (variable, _end(variable, lower), _end(variable, upper))


def _end(variable: Any, end: Any) -> Any:
    """A limit of an integral: the value of ``x = a``, which must name its variable."""
    if not isinstance(end, sympy.Equality):
        return end
    for named, value in ((end.lhs, end.rhs), (end.rhs, end.lhs)):
        if named == variable:
            return value
    shown = display.unicode(variable)
    raise EvaluationError(
        f"a limit of the integral over {shown} is written {display.unicode(end)}, which does "
        f"not say what {shown} is",
        hint=f"write the limit as {shown} = a, or as a alone",
    )


def piecewise(*args: Any) -> Any:
    return sympy.Piecewise(*zip(args[::2], args[1::2], strict=True))


# -- calculus --------------------------------------------------------------------------


#: How long finding the closed form of a sum or a product may take, in seconds: past it,
#: it is kept as written (design/notation.md, decision 4).
_CLOSED_FORM_BUDGET = 5.0
#: How long finding the closed form of an integral may take, in seconds, past which it is
#: kept as written: longer than a sum, as SymPy takes a dozen seconds over integrals it
#: does find, as the area of a Kerr horizon, and over another before one it can do; and
#: several times that, as a machine half as fast must find them too.
_INTEGRAL_BUDGET = 60.0


def summation(
    body: Any, variable: Any, lower: Any, upper: Any, *, assuming: tuple[Any, ...] = ()
) -> Any:
    """A sum: carried out where it has a whole number of terms, as \\sum_{n=0}^{2} has;
    otherwise in closed form, where SymPy finds one within a time budget, with the index an
    integer and, where it converges only under conditions, under the bounds the program
    assumes or else the assumption books make (see
    :func:`relativisticpy.assumptions.summed`); else kept as it is written
    (design/notation.md, decision 4). A closed form of a series of numbers is checked by
    summing it numerically (see :mod:`relativisticpy.runtime.crosscheck`)."""
    from relativisticpy.runtime import crosscheck

    written = sympy.Sum(body, (variable, lower, upper))
    finite = _finite(written, lower, upper)
    if finite is not written:
        return finite
    try:
        with _budget(_CLOSED_FORM_BUDGET):
            found = assumptions.summed(body, [(variable, lower, upper)], assuming)
    except _OverBudget:
        _past_the_budget("sum")
        return written
    if isinstance(found, sympy.Basic) and found.has(sympy.Sum):
        return _kept(written)
    crosscheck.series(found, sympy.sympify(body), variable, lower, upper)
    return found


def _kept(written: Any) -> Any:
    """A series SymPy finds no closed form of, kept as written; or ∞, or -∞, where its terms
    do not tend to 0 and keep one sign, so that it diverges, as that of e^{-an} does where
    a < 0, which a remark says. Where its terms tend to no limit, as (-1)^n do, or where
    SymPy's tests of convergence show it to diverge otherwise, as Σ 1/(n ln n) does, a
    remark says it diverges: no closed form was missed for want of time."""
    body, (index, lower, upper) = written.function, written.limits[0]
    if upper != sympy.oo or not sympy.sympify(lower).is_finite:
        return written
    tests = (lambda: sympy.limit(body, index, sympy.oo), written.is_convergent)
    found = []
    for test in tests:
        try:
            with _budget(1.0):
                found.append(test())
        except (Exception, _OverBudget):  # what SymPy cannot test says nothing
            found.append(None)
    tends, converges = found
    for sign, infinite in ((1, sympy.oo), (-1, -sympy.oo)):
        if tends is not None and (tends == infinite or (sign * tends).is_positive):
            shown = display.unicode(tends)
            assumptions.remark(f"the sum diverges: its terms tend to {shown}, not to 0")
            return infinite
    if _no_limit(tends) and _not_small(body, index):
        assumptions.remark("the sum diverges: its terms do not tend to 0")
    elif converges in (False, sympy.false):
        assumptions.remark("the sum diverges, as SymPy's tests of convergence show")
    return written


def _no_limit(tends: Any) -> bool:
    """Whether SymPy finds that a sequence has no limit: nan, or bounds it keeps between, as
    AccumBounds(-1, 1) of cos n."""
    if isinstance(tends, sympy.AccumBounds):
        return not (tends.min == 0 and tends.max == 0)
    return tends is sympy.nan


#: Where the terms of a series are looked at, far out, to see that they do not tend to 0.
_FAR = (10**6, 10**6 + 1, 10**7 + 2, 10**7 + 3)


def _not_small(body: Any, index: Any) -> bool:
    """Whether the terms of a series are far from 0 far out, at two neighbouring indices
    or more of :data:`_FAR`, as ±1 of (-1)^n are: with no limit, they do not tend to 0."""
    try:
        sizes = [abs(complex(sympy.N(body.subs(index, n), 15))) for n in _FAR]
    except (TypeError, ValueError):  # not a number there: nothing is said
        return False
    return sum(size > 1e-3 for size in sizes) >= 2


def product(body: Any, variable: Any, lower: Any, upper: Any) -> Any:
    """A product, carried out as a sum is, where SymPy finds its closed form within a time
    budget: see :func:`summation`. An infinite one SymPy leaves undone is the limit of its
    partial products, where they have a closed form (see :func:`_partial_products`)."""
    written = sympy.Product(body, (variable, lower, upper))
    finite = _finite(written, lower, upper)
    if finite is not written:
        return finite
    try:
        with _budget(_CLOSED_FORM_BUDGET):
            found = written.doit()
            if isinstance(found, sympy.Basic) and found.has(sympy.Product):
                limit = _partial_products(written)
                found = found if limit is None else limit
    except _OverBudget:
        _past_the_budget("product")
        return written
    return written if isinstance(found, sympy.Basic) and found.has(sympy.Product) else found


def _partial_products(written: Any) -> Any:
    """An infinite product as the limit of its partial products, where SymPy finds them
    in closed form, as Wallis's product of 4n²/(4n² - 1), whose partial products are of
    factorials, tends to π/2. Where they tend to 0 or to ∞, the product diverges, as books
    say, which a remark says. ``None`` where SymPy finds no closed form or no limit."""
    body, (index, lower, upper) = written.function, written.limits[0]
    if upper != sympy.oo or not sympy.sympify(lower).is_integer:
        return None
    last = sympy.Dummy("N", integer=True, positive=True)
    try:
        partial = sympy.Product(body, (index, lower, lower + last)).doit()
        if partial.has(sympy.Product):
            return None
        value = sympy.limit(partial, last, sympy.oo)
    except (ValueError, TypeError, NotImplementedError):  # SymPy fails: nothing is found
        return None
    if value.has(sympy.Limit, sympy.nan, sympy.AccumBounds) or value.free_symbols - {
        s for s in body.free_symbols if s != index
    }:
        return None
    if value == 0 or value.is_infinite:
        assumptions.remark(
            f"the product diverges: its partial products tend to {display.unicode(value)}"
        )
    return sympy.gammasimp(value) if value.has(sympy.gamma) else value


def _past_the_budget(what: str) -> None:
    """Say that a closed form was looked for as long as it is given, and not found."""
    assumptions.remark(
        f"SymPy found no closed form of the {what} in the {_CLOSED_FORM_BUDGET:g} s it is given"
    )


#: How soon, in seconds, the exception of a budget, or of a time limit, is raised again,
#: where it ran out in a callback of the garbage collector, as JAX gives it one, in which
#: Python would lose the exception, or where the computation lost it.
_AGAIN = 0.5


class _OverBudget(BaseException):
    """A computation that took longer than it was given, and is kept as written. Not an
    Exception, so that SymPy, which catches those, does not carry on past it. Its ``token``
    is that of the budget that ran out, whose block alone takes it as its own."""

    def __init__(self, token: object = None) -> None:
        super().__init__()
        self.token = token


class _Passing(BaseException):
    """The ``alarm`` of a budget, on its way out through a budget inside it: the computation
    around that one ran out of time, not it, so that nothing that waits for the inner
    budget's alarm takes it."""

    def __init__(self, alarm: _OverBudget) -> None:
        super().__init__()
        self.alarm = alarm


@dataclass(eq=False)
class _Limit:
    """A budget in force: when it ends, by :func:`time.monotonic`, or, once it has, when
    its alarm is raised again; and its token."""

    end: float
    token: object


@dataclass(eq=False)
class _Around:
    """What was in force before the first budget: the handler of ``SIGALRM``, when a time
    limit set with it ends, if one was running, and how often that one rings ``again``
    once it has, 0 for never."""

    handler: Any
    end: float | None
    again: float


#: The budgets in force, the outermost first, and what was in force around them.
_LIMITS: list[_Limit] = []
_AROUND: list[_Around] = []


@contextlib.contextmanager
def _budget(seconds: float) -> Iterator[None]:
    """Raise :class:`_OverBudget` if the block runs longer than ``seconds``, and again every
    :data:`_AGAIN` seconds until it ends, as the exception cannot be raised in a callback
    of the garbage collector and a computation may lose one. A budget inside another has a
    token of its own: where the outer one runs out first, its alarm passes through the inner
    one's block, which never takes it for its own. A time limit in force around them is left
    as it was, with what remains of it; one that ends first decides. Only with ``SIGALRM``,
    in the main thread; elsewhere the block runs as long as it takes."""
    usable = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    if not usable:
        yield
        return
    now = time.monotonic()
    if not _LIMITS:
        outer, again = signal.getitimer(signal.ITIMER_REAL)
        if outer and outer <= seconds:  # a time limit around it ends first, and decides
            yield
            return
        handler = signal.signal(signal.SIGALRM, _ring)
        _AROUND.append(_Around(handler, now + outer if outer else None, again))
    limit = _Limit(now + seconds, object())
    _LIMITS.append(limit)
    _arm()
    try:
        yield
    except _OverBudget as alarm:
        if alarm.token is not limit.token:
            raise _Passing(alarm) from None
        raise
    except _Passing as passing:
        if passing.alarm.token is limit.token:
            raise passing.alarm from None
        raise
    finally:
        _LIMITS.remove(limit)
        if _LIMITS:
            _arm()
        else:
            _leave()


def _arm() -> None:
    """Set the alarm for when the first budget in force ends, and every :data:`_AGAIN`
    seconds after it."""
    ends = [limit.end for limit in _LIMITS]
    around = _AROUND[-1].end if _AROUND else None
    if around is not None:
        ends.append(around)
    soonest = min(ends) - time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, max(soonest, 1e-6), _AGAIN)


def _ring(signum: int, frame: FrameType | None) -> None:
    """The alarm: of the outermost budget that has run out, whose block it stops; of the
    time limit around them all, where that has ended, as its own handler raises it."""
    now = time.monotonic()
    around = _AROUND[-1] if _AROUND else None
    if around is not None and around.end is not None and now >= around.end - 1e-3:
        # it rings as it would have alone: once, or again as often as it was set to
        around.end = now + around.again if around.again else None
        if callable(around.handler):
            around.handler(signum, frame)
        return
    due = next((limit for limit in _LIMITS if now >= limit.end - 1e-3), None)
    if due is None:
        if _LIMITS:
            _arm()
        return
    # Raised now, and again then: never at once, as the alarm set again while a block
    # inside ends would ring in its leaving, for that block's ``except`` to take as its own.
    due.end = now + _AGAIN
    if not (_collecting(frame) or _entering(frame)):  # there the next alarm raises it
        raise _OverBudget(due.token)


def _leave() -> None:
    """Put back what was in force before the first budget: its handler, and the time limit
    it set, with what remains of it."""
    signal.setitimer(signal.ITIMER_REAL, 0)
    around = _AROUND.pop()
    signal.signal(signal.SIGALRM, around.handler)
    if around.end is not None:
        remaining = max(around.end - time.monotonic(), 1e-6)
        signal.setitimer(signal.ITIMER_REAL, remaining, around.again)


def _entering(frame: FrameType | None) -> bool:
    """Whether ``frame`` is in a budget's own entering or leaving. An alarm raised there
    would stop it half way, and leave its limit in force for ever."""
    while frame is not None:
        if frame.f_code in _OWN:
            return True
        frame = frame.f_back
    return False


#: The code of a budget's entering and leaving.
_OWN = frozenset({inspect.unwrap(_budget).__code__, _arm.__code__, _leave.__code__})


def _weak_callbacks() -> frozenset[CodeType]:
    """The callbacks the standard library gives the weak references of its sets and
    dictionaries, which run when what they refer to is freed."""
    found = set()
    for kind in (weakref.WeakSet, weakref.WeakValueDictionary, weakref.WeakKeyDictionary):
        made_in = getattr(kind.__init__, "__code__", None)
        for constant in getattr(made_in, "co_consts", ()):
            if isinstance(constant, CodeType) and constant.co_name in ("_remove", "remove"):
                found.add(constant)
    return frozenset(found)


_WEAK_CALLBACKS = _weak_callbacks()


def _collecting(frame: FrameType | None) -> bool:
    """Whether ``frame`` is where an exception raised is lost: in a callback the garbage
    collector runs, in a finalizer, or in a callback of a weak reference of the standard
    library's. Python reports it as unraisable, and goes on."""
    callbacks = {getattr(callback, "__code__", None) for callback in gc.callbacks}
    while frame is not None:
        code = frame.f_code
        if code in callbacks or code in _WEAK_CALLBACKS or code.co_name == "__del__":
            return True
        frame = frame.f_back
    return False


def _finite(written: Any, lower: Any, upper: Any) -> Any:
    bounds = [sympy.sympify(bound) for bound in (lower, upper)]
    if all(bound.is_Integer for bound in bounds) and bounds[1] - bounds[0] < 10_000:
        return written.doit()
    return written


def limit(body: Any, variable: Any, point: Any, direction: str) -> Any:
    """A limit (see :func:`_limit`); where SymPy cannot take it for not knowing the sign of
    a parameter, the limit under the assumption books make, which a note says."""
    try:
        return _limit(body, variable, point, direction)
    except NotImplementedError:  # "Result depends on the sign of sign(k)"
        found = assumptions.limited(_limit, body, variable, point, direction)
        if found is None:
            raise
        return found


def _limit(body: Any, variable: Any, point: Any, direction: str) -> Any:
    """A limit, never taken in a case that holds only at points of the variable, as
    ``Eq(k, 0)`` does of a scattering amplitude: a limit is of the values near the point,
    and SymPy would take that case's value at the point itself. One that does not exist,
    as of a step from both sides or of sin(1/h), says so, where SymPy would give a value."""
    if direction == "+-" and point in (sympy.oo, -sympy.oo):
        direction = "-" if point == sympy.oo else "+"
    near = _near(sympy.sympify(body), variable)
    if not near.has(sympy.Piecewise) or point in (sympy.oo, -sympy.oo):
        try:
            found = sympy.limit(near, variable, point, direction)
        except ValueError as error:  # |x|/x at 0: said in RelativisticPy's words
            if "does not exist" not in str(error):
                raise
            said = re.search(r"left hand limit = (.+) and right hand limit = (.+)", str(error))
            why = f": it is {said[1]} from below and {said[2]} from above" if said else ""
            raise ValueError(f"the limit does not exist{why}") from None
        if found is sympy.zoo and direction == "+-" and not _complex(variable):
            found = _from_both_sides(near, variable, point)
        return _existing(found)
    # SymPy takes a case as it holds at the point: each side is approached by a step
    step = sympy.Dummy("h", positive=True)
    values = {
        side: _existing(sympy.limit(near.subs(variable, point + sign * step), step, 0, "+"))
        for side, sign in (("+", 1), ("-", -1))
        if side in direction
    }
    if len(values) == 2 and sympy.simplify(values["-"] - values["+"]) != 0:
        raise ValueError(
            f"the limit does not exist: it is {values['-']} from below and {values['+']} from above"
        )
    return next(iter(values.values()))


def _from_both_sides(body: Any, variable: Any, point: Any) -> Any:
    """A limit from both sides that SymPy gives as complex infinity, zoo, as of 1/x at 0:
    what both sides tend to, where they agree; where they do not, as 1/x tends to -∞ from
    below and ∞ from above, a ValueError that says so: there is no limit."""
    below, above = (sympy.limit(body, variable, point, side) for side in ("-", "+"))
    if below != above:
        raise ValueError(
            f"the limit does not exist: it is {below} from below and {above} from above"
        )
    return above


def _complex(variable: Any) -> bool:
    """Whether a variable is complex, as z is, whose limit at a pole is ∞ from every way,
    the complex infinity: one assumed not real, or named as books name a complex one."""
    return getattr(variable, "is_real", None) is False or str(variable) in (
        assumptions._COMPLEX_VARIABLES
    )


def _existing(value: Any) -> Any:
    """A limit SymPy found, unless it only bounds the values, as AccumBounds(-1, 1) does
    of sin(1/h), whose values never settle."""
    if isinstance(value, sympy.AccumBounds):
        raise ValueError(
            f"the limit does not exist: the values keep between {value.min} and {value.max}"
        )
    return value


def _near(expr: Any, variable: Any) -> Any:
    """``expr`` with each piecewise part as it is near any point of ``variable``: without
    the cases that hold only at points, and the first case that holds at every point but
    some taken for all the rest, which no limit sees."""

    def generic(piecewise: Any) -> Any:
        kept = []
        for value, condition in piecewise.args:
            if variable not in condition.free_symbols:
                kept.append((value, condition))
            elif _at_points(condition):
                continue
            elif _but_at_points(condition):
                kept.append((value, sympy.true))
                break
            else:
                kept.append((value, condition))
        if len(kept) == 1 and kept[0][1] is sympy.true:
            return kept[0][0]
        return sympy.Piecewise(*kept) if kept else piecewise

    if not isinstance(expr, sympy.Basic) or not expr.has(sympy.Piecewise):
        return expr
    return expr.replace(lambda node: isinstance(node, sympy.Piecewise), generic)


def _at_points(condition: Any) -> bool:
    """Whether a condition holds only at isolated values, as ``Eq(k, 0)``, or either of
    several such, as ``Eq(k, 0) | Eq(\\theta, 0)`` does."""
    if isinstance(condition, sympy.Eq):
        return True
    if isinstance(condition, sympy.Or):
        return all(_at_points(part) for part in condition.args)
    if isinstance(condition, sympy.And):
        return any(_at_points(part) for part in condition.args)
    return False


def _but_at_points(condition: Any) -> bool:
    """Whether a condition holds everywhere but at isolated values, as ``Ne(x, 0)``."""
    if isinstance(condition, sympy.Ne):
        return True
    if isinstance(condition, sympy.And):
        return all(_but_at_points(part) for part in condition.args)
    if isinstance(condition, sympy.Or):
        return any(_but_at_points(part) for part in condition.args)
    return False


def differentiate(expr: Any, orders: tuple[int, ...], *variables: Any) -> Any:
    return sympy.diff(expr, *((v, n) for v, n in zip(variables, orders, strict=True)))


def derivative_at(body: Any, dummy: Any, order: int, point: Any) -> Any:
    """``f'(p)``. Where the cases of a piecewise function meet at p, as x sin(1/x) and 0 do
    at 0, it is taken from its definition, the limit of the difference quotient, which may
    not exist, rather than case by case, which would give the derivative of the case that
    holds at p alone."""
    body = sympy.sympify(body)
    if not _on_a_seam(body, dummy, point):
        return sympy.diff(body, dummy, order).subs(dummy, point)
    if order != 1:
        raise ValueError(
            f"a derivative of order {order} where the cases of a function meet is not taken yet"
        )
    step = sympy.Dummy("h")
    quotient = (body.subs(dummy, point + step) - body.subs(dummy, point)) / step
    try:
        return limit(quotient, step, 0, "+-")
    except ValueError as error:
        why = str(error).removeprefix("the limit does not exist").lstrip(": ")
        raise ValueError(
            f"the derivative at {point} does not exist: its difference quotient has no limit"
            + (f", as {why}" if why else "")
        ) from None


def _on_a_seam(body: Any, dummy: Any, point: Any) -> bool:
    """Whether the cases of a piecewise part of ``body`` meet at ``point``: a condition on
    ``dummy`` whose two sides are equal there, as x \\neq 0 at 0."""
    for piecewise in body.atoms(sympy.Piecewise):
        for _, condition in piecewise.args:
            for relation in condition.atoms(sympy.core.relational.Relational):
                if dummy in relation.free_symbols:
                    gap = (relation.lhs - relation.rhs).subs(dummy, point)
                    if sympy.simplify(gap) == 0:
                        return True
    return False


def gradient(expr: Any, vector: Any) -> Any:
    return sympy.ImmutableDenseNDimArray([sympy.diff(expr, component) for component in vector])


def time_derivative(expr: Any, order: int) -> Any:
    r"""``\dot{x}``: the derivative with respect to the single variable that the
    functions in ``x`` depend on; a plain symbol ``a`` gives the symbol ``adot``."""
    expr = sympy.sympify(expr)
    applied = expr.atoms(sympy.core.function.AppliedUndef)
    parameters = {argument for function in applied for argument in function.args}
    if len(parameters) == 1:
        return sympy.diff(expr, parameters.pop(), order)
    if isinstance(expr, Symbol):
        return sympy.Symbol(expr.name + ("dot" if order == 1 else "ddot"))
    raise ValueError(
        f"cannot tell what {expr} changes with: write it as a function of one variable, e.g. a(t)"
    )


def substitute(expr: Any, *pairs: Any) -> Any:
    """An evaluation bar, ``x - y |_{x=y, y=x}``: every substitution at once, as a
    change of variables is, rather than one after another. An expression may be 0/0 at
    the point as it stands, as (x^2 - 1)/(x - 1) is at x = 1, and as a scalar of the
    Schwarzschild metric is at r = 2M in the form it was computed in, which no one wrote:
    its value there is that of the form with what is common cancelled, as it is shown."""
    rules = list(zip(pairs[::2], pairs[1::2], strict=True))
    expr = sympy.sympify(expr)
    value = expr.subs(rules, simultaneous=True)
    undefined = _is_no_number(value) and not _is_no_number(expr)
    if undefined and isinstance(expr, sympy.Expr) and sympy.count_ops(expr) <= LARGE:
        cancelled = sympy.cancel(normal_trig(expr)).subs(rules, simultaneous=True)
        return value if _is_no_number(cancelled) else cancelled
    return value


def _is_no_number(value: Any) -> bool:
    return isinstance(value, sympy.Basic) and value.has(sympy.nan)


# -- functions and relations -------------------------------------------------------------


def function(*args: Any, name: str) -> Any:
    """``f(x, y)`` of a function no one defined, called ``name``: SymPy's undefined
    function, applied."""
    return sympy.Function(name)(*args)


def less(lhs: Any, rhs: Any, *, assuming: tuple[Any, ...] = ()) -> Any:
    return _decided(sympy.Lt(lhs, rhs), assuming)


def greater(lhs: Any, rhs: Any, *, assuming: tuple[Any, ...] = ()) -> Any:
    return _decided(sympy.Gt(lhs, rhs), assuming)


def less_equal(lhs: Any, rhs: Any, *, assuming: tuple[Any, ...] = ()) -> Any:
    return _decided(sympy.Le(lhs, rhs), assuming)


def greater_equal(lhs: Any, rhs: Any, *, assuming: tuple[Any, ...] = ()) -> Any:
    return _decided(sympy.Ge(lhs, rhs), assuming)


def _decided(relation: Any, bounds: tuple[Any, ...]) -> Any:
    """An inequality, decided where the bounds the program assumes decide it, as x < 2 is
    where 0 < x < 1 is assumed."""
    if not bounds or not isinstance(relation, Relational):
        return relation
    verdict = assumptions.holds(relation, bounds)
    return relation if verdict is None else sympy.true if verdict else sympy.false


def equal(lhs: Any, rhs: Any) -> Any:
    return sympy.Eq(lhs, rhs)


def unequal(lhs: Any, rhs: Any) -> Any:
    return sympy.Ne(lhs, rhs)


def logical_not(value: Any) -> Any:
    return sympy.Not(value)


def logical_and(*values: Any) -> Any:
    return sympy.And(*values)


def logical_or(*values: Any) -> Any:
    return sympy.Or(*values)


def item(whole: Any, index: Any) -> Any:
    """The item of a list or a tuple at ``index``, as of what ``solve`` or ``eig`` give."""
    return whole[index]


# -- the metric -------------------------------------------------------------------------


def where(components: Any, name: str, value: Any) -> Any:
    """A metric's components with a name it is written with given its value, as where
    \\Delta = r^2 - 2Mr + a^2 follows a line element: the symbol, or a function of the
    coordinates the name stands for, as \\alpha does for \\alpha(t, r)."""
    value = sympy.sympify(value)

    def put(component: Any) -> Any:
        component = sympy.sympify(component).replace(
            lambda e: isinstance(e, AppliedUndef) and e.func.__name__ == name, lambda e: value
        )
        return component.xreplace({s: value for s in component.free_symbols if s.name == name})

    return arrays.apply(arrays.to_array(components), put)


def where_function(components: Any, function: tuple[str, tuple[Any, ...]], body: Any) -> Any:
    """A metric's components with a function it is written with, as f is in
    -f(r) dt^2, given its definition: ``body`` in its ``parameters``. Where the function
    is written with other arguments than it is defined with, the definition says what it
    is in terms of the parameters' names, as \\alpha(r) := ... does of \\alpha(t, r)."""
    name, parameters = function
    body = sympy.sympify(body)

    def at(applied: Any) -> Any:
        if len(applied.args) == len(parameters):
            return body.xreplace(dict(zip(parameters, applied.args, strict=True)))
        return body

    def put(component: Any) -> Any:
        return sympy.sympify(component).replace(
            lambda e: isinstance(e, AppliedUndef) and e.func.__name__ == name, at
        )

    return arrays.apply(arrays.to_array(components), put)


def components(metric: Any, *, coordinates: tuple[Symbol, ...]) -> Any:
    """The components of a spacetime's metric in ``coordinates``, the symbols declared for
    the names of its own: a coordinate declared with assumptions, as by Positive := [r],
    takes the place of its namesake."""
    declared = {coordinate.name: coordinate for coordinate in coordinates}
    own = {c: declared.get(c.name, c) for c in metric.coordinates}
    return metric.components.xreplace(own)


def metric_diagnosis(given: Any, *, coordinates: tuple[Symbol, ...]) -> EvaluationError | None:
    """Why components given for a metric in ``coordinates`` are no metric, in the terms of
    what was written: of the wrong size, not symmetric, or degenerate, with the
    differentials no term of it has; ``None`` where none of these is why. The error is the
    definition's, which places it."""
    try:
        components = arrays.to_array(given)
    except Exception:  # what is no array says so itself
        return None
    names = [latex_label(c.name) for c in coordinates]
    listed = _listed(names)
    shape = tuple(components.shape)
    count = len(coordinates)
    if shape != (count, count):
        found = "×".join(map(str, shape)) if shape else "no"
        return EvaluationError(
            f"the metric has {found} components, but there are {count} coordinates",
            notes=[f"the coordinates are {listed}"],
            hint=f"give {count}×{count} components, a row and a column for each coordinate",
        )
    matrix = sympy.Matrix(components.tolist())
    for i, j in itertools.combinations(range(count), 2):
        if sympy.simplify(matrix[i, j] - matrix[j, i]) != 0:
            one, other = names[i], names[j]
            return EvaluationError(
                f"the metric is not symmetric: g_{{{one}{other}}} = "
                f"{display.text(matrix[i, j])} but g_{{{other}{one}}} = "
                f"{display.text(matrix[j, i])}",
                hint="a metric has g_{\\mu\\nu} = g_{\\nu\\mu}",
            )
    if sympy.simplify(matrix.det()) != 0:
        return None
    missing = [
        names[i] for i in range(count) if all(sympy.simplify(entry) == 0 for entry in matrix.row(i))
    ]
    if missing:
        differentials = _listed([f"d{name}" for name in missing], "or")
        return MetricError(
            "the metric is degenerate: its determinant is zero",
            notes=[f"no term of it has {differentials}"],
            hint=f"give it a term with {_listed([f'd{name}' for name in missing])}",
        )
    return MetricError(
        "the metric is degenerate: its determinant is zero",
        notes=["so it has no inverse, and no curvature can be computed from it"],
    )


def _listed(items: Sequence[str], conjunction: str = "and") -> str:
    """``t, r, \\theta and \\phi``."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


def line_element(metric: Any, coordinates: Sequence[Symbol]) -> Any:
    """:math:`g_{\\mu\\nu} dx^\\mu dx^\\nu`, in the differentials of the coordinates."""
    differentials = [differential(c) for c in coordinates]
    count = len(coordinates)
    return sympy.Add(
        *(
            metric[i, j] * differentials[i] * differentials[j]
            for i in range(count)
            for j in range(count)
        )
    )


# -- values for Python ---------------------------------------------------------------------


def tensor(components: Any, *, indices: Indices, coordinates: tuple[Symbol, ...]) -> Tensor:
    """Components wrapped in a :class:`~relativisticpy.tensors.Tensor`, for a function
    that wants one."""
    return Tensor(components, indices, coordinates)


def tensor_equation(
    equation: TensorEquation, *, indices: Indices, coordinates: tuple[Symbol, ...]
) -> TensorEquation:
    """A tensor equation with its indices and coordinates, for a function that wants one."""
    return TensorEquation(equation.lhs, equation.rhs, indices, coordinates)


# -- spaces ---------------------------------------------------------------------------------


def identity(dim: int) -> Any:
    """The identity matrix: SymPy's for a few dimensions, NumPy's for many."""
    if dim <= _SMALL:
        return sympy.ImmutableDenseNDimArray(sympy.eye(dim).tolist())
    return numerical.numpy().eye(dim)


def ones(dim: int) -> Any:
    if dim <= _SMALL:
        return sympy.ImmutableDenseNDimArray([1] * dim)
    return numerical.numpy().ones(dim)


def uniform(dim: int) -> Any:
    """The uniform measure: weight 1/n on each of n elements."""
    if dim <= _SMALL:
        return sympy.ImmutableDenseNDimArray([sympy.Rational(1, dim)] * dim)
    return numerical.numpy().full(dim, 1.0 / dim)


def positions(first: int, dim: int) -> Any:
    """The values an index takes over a space of ``dim`` elements, counted from ``first``,
    as n takes 0, 1, ..., N - 1 in ``x_n := n h``: SymPy's for a few, NumPy's for many."""
    if dim <= _SMALL:
        return sympy.ImmutableDenseNDimArray(list(range(first, first + dim)))
    return numerical.numpy().arange(first, first + dim)


def power(base: Any, exponent: Any) -> Any:
    """A power, taken for each component of a grid, as n^2 is for each n, or
    e^{-2πikn/N} for each k and n."""
    return base**exponent


def space_metric(components: Any, complex_space: bool) -> Any:
    """The components of a space's metric, once found to be one: symmetric (Hermitian, over
    a complex space) and, where they are numbers, positive definite."""
    asymmetric = EvaluationError(
        f"a metric is {'Hermitian' if complex_space else 'symmetric'}, but this one is not"
    )
    indefinite = EvaluationError(
        "a metric of a space is positive definite, but this one is not",
        hint="a covariance is, unless some combination of the data never varies",
    )
    if numerical.is_numerical(components):
        np = numerical.numpy()
        try:
            matrix = np.asarray(numerical.dense(components))
        except TypeError:  # numbers JAX is differentiating, not known yet
            return components
        if not linalg.hermitian(matrix):
            raise asymmetric
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            raise indefinite from None
        return components
    entries = sympy.Matrix(components.tolist())
    if not all(is_zero(entry) for entry in entries - (entries.H if complex_space else entries.T)):
        raise asymmetric
    if all(entry.is_number for entry in entries) and entries.is_positive_definite is False:
        raise indefinite
    return components


def elementwise_product(left: Any, right: Any) -> Any:
    """The elementwise product of lists or matrices of one shape, ``[1, 2] \\odot [3, 4]``,
    which is ``[3, 8]``; a number multiplies every element."""

    def array(value: Any) -> Any:
        listed = isinstance(value, list | tuple | sympy.MatrixBase)
        return arrays.to_array(value) if listed else value

    left, right = array(left), array(right)
    shapes = [tuple(getattr(value, "shape", ())) for value in (left, right)]
    if () in shapes:
        return left * right
    if shapes[0] != shapes[1]:
        raise EvaluationError(
            f"an elementwise product is of arrays of one shape, not {shapes[0]} and {shapes[1]}"
        )
    return linalg.hadamard(left, right)


def conjugated(components: Any, complex_space: bool) -> Any:
    """The complex conjugate of components: of numbers, always (real ones are their own),
    and of SymPy's where they are over a complex space."""
    if numerical.is_numerical(components):
        complex_data = numerical.xp(components).iscomplexobj(components)
        return components.conj() if complex_data else components
    return arrays.apply(components, sympy.conjugate) if complex_space else components


def squared_modulus(components: Any, complex_space: bool) -> Any:
    """:math:`|x|^2` of each component, which is real: :math:`x \\bar x` of complex numbers,
    and of SymPy's over a complex space; :math:`x^2` otherwise."""
    if numerical.is_numerical(components):
        if not numerical.xp(components).iscomplexobj(components):
            return linalg.hadamard(components, components)
        return linalg.hadamard(components, components.conj()).real
    if complex_space:
        return arrays.apply(components, lambda c: sympy.expand(c * sympy.conjugate(c)))
    return linalg.hadamard(components, components)


def real(value: Any) -> Any:
    """The real part of numbers that are real but for rounding, such as the square of a
    complex vector's length; SymPy values as they are."""
    if numerical.is_numerical(value) and numerical.xp(value).iscomplexobj(value):
        return value.real
    return value


# -- linear algebra and statistics ------------------------------------------------------------


def matrix_function(name: str, components: Any, rows: int) -> Any:
    """``exp``, ``sqrt`` or ``log``, called ``name``, of a map."""
    return linalg.function(name, components, rows)


def of_a_map(components: Any, *rest: Any, name: str, position: int, rows: int) -> Any:
    """The function of a matrix called ``name`` of the components of a map, as a matrix whose
    rows are its first ``rows`` axes, given at ``position`` among the ``rest`` of its
    arguments: computed with arrays where they are numbers, and with SymPy otherwise."""
    capability = integrations.find(name)

    def compute(matrix: Any) -> Any:
        arguments = [*rest[:position], matrix, *rest[position:]]
        if numerical.is_numerical(matrix) and capability.numeric is not None:
            return integrations.Numerically(name)(numerical.xp(matrix), *arguments)
        return integrations.Exactly(name)(*arguments)

    return linalg.of_a_map(components, rows, compute, scalar=capability.scalar)


def identity_beside(matrix: Any, *, ones: bool) -> Any:
    """The identity of the size of a square matrix, as I is in A - \\lambda I; beside a
    vector, where it is written \\mathbf{1}, the vector of ones of its shape
    (design/notation.md, 12 and 23)."""
    array = arrays.to_array(matrix)
    shape = tuple(array.shape)
    if len(shape) == 2 and shape[0] == shape[1]:
        if numerical.is_numerical(array):
            return numerical.xp(array).eye(shape[0])
        return sympy.ImmutableDenseNDimArray(sympy.eye(shape[0]).tolist())
    if ones and len(shape) == 2 and 1 in shape:
        return _ones_of(array, shape)
    what = "the identity" if not ones else "the identity, or the vector of ones,"
    raise EvaluationError(
        f"{what} is of the size of the matrix beside it, but a {_size(shape)} matrix is not square",
        hint="write it beside a square matrix, as in A - \\lambda I",
    )


def ones_beside(matrix: Any, *, side: str, row: bool) -> Any:
    """The vector of ones as long as a matrix has ``side``, rows or columns: a row, where
    it is ``\\mathbf{1}^T``, or a column."""
    array = arrays.to_array(matrix)
    shape = tuple(array.shape)
    if len(shape) != 2:
        raise EvaluationError(f"the vector of ones meets a {_size(shape)} array, no matrix")
    length = shape[0] if side == "rows" else shape[1]
    return _ones_of(array, (1, length) if row else (length, 1))


def _ones_of(like: Any, shape: tuple[int, ...]) -> Any:
    """Ones in an array of ``shape``: of numbers where ``like`` is, and exact otherwise."""
    if numerical.is_numerical(like):
        return numerical.xp(like).ones(shape)
    return sympy.ImmutableDenseNDimArray([1] * math.prod(shape), shape)


def entry(matrix: Any, row: int, column: int) -> Any:
    """The entry of a matrix in a row and a column, counted from 0."""
    array = arrays.to_array(matrix)
    shape = tuple(array.shape)
    if len(shape) != 2 or row >= shape[0] or column >= shape[1]:
        raise EvaluationError(
            f"a {_size(shape)} matrix has no entry in row {row + 1} and column {column + 1}"
        )
    value = array[row, column]
    return numerical.number(value) if numerical.is_numerical(array) else value


def _size(shape: tuple[int, ...]) -> str:
    """A shape as it is said, as 2×3."""
    return "×".join(map(str, shape)) or "scalar"


def ledoit_wolf(samples: Any) -> Any:
    """The Ledoit–Wolf covariance of samples in rows, from scikit-learn."""
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            'the Ledoit–Wolf covariance needs scikit-learn: pip install "relativisticpy[data]"'
        ) from error
    data = numerical.dense(numerical.as_array(samples))
    return LedoitWolf(assume_centered=False).fit(data).covariance_
