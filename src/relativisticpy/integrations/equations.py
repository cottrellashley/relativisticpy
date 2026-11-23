r"""Equations.

Solving equations, algebraic and differential, with SymPy; taking them apart; making an
equation, and symbols; and the LaTeX of a value.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import sympy
from sympy.core.function import AppliedUndef

from relativisticpy import assumptions, display
from relativisticpy.errors import EvaluationError, RelativisticPyError
from relativisticpy.integrations import Example, constructor, function, solver
from relativisticpy.tensors.equation import TensorEquation

x, y = sympy.symbols("x y")
f = sympy.Function("f")


def _equations(values: Iterable[Any]) -> list[Any]:
    """Tensor equations expanded into their component equations."""
    result: list[Any] = []
    for value in values:
        if isinstance(value, TensorEquation):
            result.extend(value.equations())
        elif isinstance(value, list | tuple):
            result.extend(_equations(value))
        else:
            result.append(value)
    return result


@solver(
    "solve",
    summary=r"`solve(x^2 = 4, x)`, the solutions; `solve(E, T_{\mu\nu})` solves an equation "
    "between tensors for one",
    examples=[
        Example("solve(x^2 = 4, x)", [-2, 2], "sorted(numpy.roots([1, 0, -4]))"),
        Example("solve(2x + 1 = 0, x)", [sympy.Rational(-1, 2)], "standard"),
    ],
)
class Solve:
    def exact(self, equations: Any, *unknowns: Any, assuming: tuple[Any, ...] = ()) -> Any:
        if isinstance(equations, TensorEquation | list | tuple):
            equations = _equations([equations])
        return _solved(equations, unknowns, assuming)


def _solved(equations: Any, unknowns: tuple[Any, ...], bounds: tuple[Any, ...]) -> Any:
    """The solutions of ``equations`` for ``unknowns``, in the form SymPy gives them, but
    those an assumption rules out, which a note names (see
    :func:`relativisticpy.assumptions.ruled_out`).

    Each equation is one fraction with its common factors cancelled, so that a root that
    also zeroes a factor the fraction cancels is kept, as the horizons of g^{rr} are; but
    not where a side, as written, is not defined (see :func:`_defined`). It is solved for
    unknowns that assume nothing, so that what an assumption rules out is known and said;
    where that leaves a family of solutions, as one complex equation does of two real
    unknowns, for unknowns that keep what numbers they are, as real ones; and by
    :func:`_robustly`, so that a failure inside SymPy that another path avoids does not end
    it."""
    listed = equations if isinstance(equations, list) else [equations]
    named = _flat(unknowns)
    sides = [_side(equation) for equation in listed]
    decided = [side for side in sides if side in (sympy.true, sympy.false)]
    if named and decided and len(decided) == len(sides):  # x - x = 0, as SymPy reads it
        if sympy.false in decided:
            return []
        raise EvaluationError(
            f"the equation holds whatever {display.unicode(named[0])} is, so there is "
            "nothing to solve for"
        )
    absent = [u for u in named if not any(_mentions(side, u) for side in sides)]
    if absent and named:
        shown = ", ".join(display.unicode(equation) for equation in listed)
        missing = display.unicode(absent[0])
        raise EvaluationError(
            f"{shown} does not contain {missing}, so it cannot be solved for {missing}",
            hint=f"solve for what the equation is written with, or write it with {missing}",
        )
    assumed = [u for u in named if isinstance(u, sympy.Symbol) and _assumed(u)]
    for keep in (False, True):
        if keep and not any(_domain(u) for u in assumed):
            break
        general = {u: sympy.Dummy(u.name, **(_domain(u) if keep else {})) for u in assumed}
        back = {dummy: u for u, dummy in general.items()}
        plain = [general.get(u, u) for u in named]
        prepared = [
            _cancelled(_without_vanishing_terms(_put(side, general), plain), plain)
            for side in sides
        ]
        given = [_put(each, general) for each in unknowns]
        found = _robustly(prepared if isinstance(equations, list) else prepared[0], given, plain)
        found = _put(found, back)
        if not _family(found, named):
            break
    return _allowed(_defined(found, listed, named), named, bounds)


def _defined(found: Any, equations: list[Any], unknowns: list[Any]) -> Any:
    """The solutions found, but those where a side of an equation, as written, is not
    defined, as (x² - 1)/(x - 1) is not at 1, which its common factor cancelled would give:
    the denominator vanishes there, and the side, its common factors cancelled, is not 0. A
    side that is, as r²f²/(r² - 2Mr + Q²) is at a horizon, has a zero there written as 0/0,
    which is kept. A note names the solutions dropped."""
    sides = [
        side
        for equation in equations
        for side in (
            (equation.lhs, equation.rhs) if isinstance(equation, sympy.Equality) else (equation,)
        )
    ]
    kept: list[tuple[dict[Any, Any], Any]] = []
    dropped: list[str] = []
    for mapping, item in _items(found, unknowns):
        undefined = next((side for side in sides if _undefined_at(side, mapping)), None)
        if undefined is None:
            kept.append((mapping, item))
        else:
            dropped.append(f"{_shown(mapping)}, where {display.unicode(undefined)} is not defined")
    for each in dropped:
        assumptions.remark(f"solve dropped {each}")
    return _rebuilt(found, kept) if dropped else found


def _undefined_at(side: Any, mapping: dict[Any, Any]) -> bool:
    """Whether a side of an equation is not defined at a solution: its denominator vanishes
    there, and it is no zero written as 0/0."""
    if not isinstance(side, sympy.Expr) or not all(isinstance(u, sympy.Symbol) for u in mapping):
        return False
    if not side.has(*mapping):
        return False
    _, denominator = sympy.fraction(sympy.together(side))
    if _vanishes(denominator.xreplace(mapping)) is not True:
        return False
    if not side.is_rational_function(*mapping):
        return True
    numerator, rest = sympy.fraction(sympy.cancel(side))
    if _vanishes(rest.xreplace(mapping)) is not False:
        return True  # still divided by 0
    return _vanishes(numerator.xreplace(mapping)) is not True


def _domain(unknown: sympy.Symbol) -> dict[str, bool]:
    """What numbers an unknown is, of what is assumed of it: an integer, or real."""
    if unknown.is_integer:
        return {"integer": True}
    if unknown.is_rational:
        return {"rational": True}
    return {"real": True} if unknown.is_extended_real else {}


def _family(found: Any, unknowns: list[Any]) -> bool:
    """Whether a solution leaves an unknown free: of SymPy's, (iω³ + 3ω² − 3iω − 1, ω)."""
    for mapping, _ in _items(found, unknowns):
        if len(mapping) < len(unknowns) or any(
            sympy.sympify(value).has(*unknowns) for value in mapping.values()
        ):
            return True
    return False


def _flat(unknowns: Iterable[Any]) -> list[Any]:
    """The unknowns of a solve, in order, whether written alone or in a list: ``x``, or
    ``[K, \\omega]``."""
    named: list[Any] = []
    for each in unknowns:
        named.extend(_flat(each) if isinstance(each, list | tuple) else [each])
    return named


def _side(equation: Any) -> Any:
    """An equation as the expression its sides differ by; anything else as it is."""
    if isinstance(equation, sympy.Equality):
        return equation.lhs - equation.rhs
    return equation


def _mentions(expr: Any, unknown: Any) -> bool:
    return isinstance(expr, sympy.Basic) and expr.has(unknown)


def _assumed(symbol: sympy.Symbol) -> bool:
    """Whether anything is assumed of a symbol, as that it is positive."""
    return bool(getattr(symbol, "_assumptions_orig", None))


def _put(value: Any, symbols: dict[Any, Any]) -> Any:
    return assumptions.mapped(value, lambda expr: expr.xreplace(symbols))


def _without_vanishing_terms(expr: Any, unknowns: list[Any]) -> Any:
    """A polynomial in the unknowns without the terms whose coefficients vanish only by an
    identity SymPy does not apply, as (\\cos^2\\theta + \\sin^2\\theta - 1) E'^2 of Compton's
    equation: SymPy would divide by it, and find no solution. Anything else as it is."""
    if not isinstance(expr, sympy.Expr) or not all(isinstance(u, sympy.Symbol) for u in unknowns):
        return expr
    try:
        polynomial = sympy.Poly(sympy.expand(expr), *unknowns)
    except sympy.PolynomialError:
        return expr
    terms = polynomial.terms()
    kept = [(powers, c) for powers, c in terms if c.is_number or sympy.trigsimp(c) != 0]
    if len(kept) == len(terms):
        return expr
    monomials = [sympy.Mul(*(u**k for u, k in zip(unknowns, p, strict=True))) for p, _ in kept]
    return sympy.Add(*(c * monomial for (_, c), monomial in zip(kept, monomials, strict=True)))


def _cancelled(expr: Any, unknowns: list[Any]) -> Any:
    """An expression with a denominator in the unknowns as one fraction, with the factors
    its numerator and denominator share cancelled, so that a root of both is a root."""
    if not isinstance(expr, sympy.Expr) or not all(isinstance(u, sympy.Symbol) for u in unknowns):
        return expr
    _, denominator = sympy.fraction(sympy.together(expr))
    if denominator.has(*unknowns) and expr.is_rational_function(*unknowns):
        return sympy.cancel(expr)
    return expr


def _robustly(equations: Any, unknowns: list[Any], named: list[Any]) -> Any:
    """The solutions SymPy finds, by the first of several paths that does not fail inside
    SymPy, as its own check of a solution does under some orders of a set of symbols
    (design/interface.md, the same answer every time): the numerators alone, without the
    factors that cannot vanish, each solution checked here; then the equations as they are;
    then without SymPy's check, each solution checked here; then with the unknowns in
    another order."""
    reduced = _essential(equations, named)
    attempts: list[Callable[[], Any]] = []
    if reduced is not None:
        attempts.append(
            lambda: _checked(sympy.solve(reduced, *unknowns, check=False), equations, named)
        )
    attempts.append(lambda: sympy.solve(equations, *unknowns))
    attempts.append(
        lambda: _checked(sympy.solve(equations, *unknowns, check=False), equations, named)
    )
    if len(named) > 1:
        attempts.append(
            lambda: _checked(sympy.solve(equations, *named[::-1], dict=True), equations, named)
        )
    failure: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except (TimeoutError, RelativisticPyError):  # a time limit, or ours: never passed by
            raise
        except Exception as error:  # SymPy failed inside itself: another path may not
            failure = failure or error
    assert failure is not None
    raise failure


def _essential(equations: Any, unknowns: list[Any]) -> Any:
    """Each equation as the part of its numerator that can vanish: without the factors
    that never do, as e^{...} or a positive symbol, and those the unknowns are not in;
    ``None`` where an unknown is not a symbol, or an equation is no expression."""
    listed = equations if isinstance(equations, list) else [equations]
    if not unknowns or not all(isinstance(u, sympy.Symbol) for u in unknowns):
        return None
    if not all(isinstance(each, sympy.Expr) for each in listed):
        return None
    reduced = []
    for each in listed:
        numerator, _ = sympy.fraction(sympy.together(each))
        factors = sympy.Mul.make_args(sympy.factor_terms(numerator))
        reduced.append(sympy.Mul(*(f for f in factors if _may_vanish(f, unknowns))))
    return reduced if isinstance(equations, list) else reduced[0]


def _may_vanish(factor: Any, unknowns: list[Any]) -> bool:
    """Whether a factor of an equation's numerator is zero for some values of the
    unknowns: not one they are not in, an exponential, nor one known to be nonzero."""
    base = factor.base if factor.is_Pow and factor.exp.is_positive else factor
    if not base.has(*unknowns) or isinstance(base, sympy.exp):
        return False
    return base.is_nonzero is not True


def _checked(found: Any, equations: Any, unknowns: list[Any]) -> Any:
    """The solutions SymPy found, in the form it gave them, but those that do not solve the
    equations: where an equation is not defined, as where its denominator vanishes, or does
    not vanish itself."""
    listed = equations if isinstance(equations, list) else [equations]
    kept = [
        (mapping, item) for mapping, item in _items(found, unknowns) if _solves(mapping, listed)
    ]
    return _rebuilt(found, kept)


def _solves(mapping: dict[Any, Any], equations: list[Any]) -> bool:
    for equation in equations:
        if not isinstance(equation, sympy.Expr):
            continue
        numerator, denominator = sympy.fraction(sympy.together(equation))
        if _vanishes(denominator.xreplace(mapping)) is not False:
            return False
        value = equation.xreplace(mapping)
        if value.has(sympy.nan, sympy.zoo, sympy.oo, -sympy.oo):
            return False
        polynomial = all(numerator.is_polynomial(u) for u in mapping)
        if not polynomial and _vanishes(value) is not True:
            return False
    return True


def _vanishes(value: Any) -> bool | None:
    """Whether a value is zero: by SymPy's facts, by simplification, and else by numbers
    at a few points of its symbols; ``None`` where that is not decided."""
    if value.is_zero is not None:
        return bool(value.is_zero)
    if sympy.simplify(value) == 0:
        return True
    if not value.free_symbols:
        return None
    for seed in (0.7, 1.3, 2.9):
        point = {
            s: sympy.Float(seed + 0.11 * k)
            for k, s in enumerate(sorted(value.free_symbols, key=str))
        }
        number = complex(value.xreplace(point).evalf())
        if abs(number) > 1e-9 * max(1.0, abs(number)):
            return False
    return True


def _items(found: Any, unknowns: list[Any]) -> list[tuple[dict[Any, Any], Any]]:
    """Each solution SymPy gives, in whichever form it gives it, as the values of the
    unknowns, with the item it is: a dictionary, a tuple of the values in the order of the
    unknowns, or the value of the one unknown."""
    if isinstance(found, dict):
        return [(found, found)]
    if not isinstance(found, list):
        return []
    items: list[tuple[dict[Any, Any], Any]] = []
    for item in found:
        if isinstance(item, dict):
            items.append((item, item))
        elif isinstance(item, tuple):
            items.append((dict(zip(unknowns, item, strict=False)), item))
        else:
            items.append(({unknowns[0]: item} if unknowns else {}, item))
    return items


def _rebuilt(found: Any, kept: list[tuple[dict[Any, Any], Any]]) -> Any:
    """What SymPy found, in its form, with only the solutions ``kept``."""
    if isinstance(found, dict):
        return found if kept else []
    if not isinstance(found, list):
        return found
    return [item for _, item in kept]


def _allowed(found: Any, unknowns: list[Any], bounds: tuple[Any, ...]) -> Any:
    """The solutions the program's assumptions allow; a note says which others were
    dropped, and why."""
    items = _items(found, unknowns)
    kept: list[tuple[dict[Any, Any], Any]] = []
    dropped: dict[str, list[str]] = {}
    for mapping, item in items:
        why = assumptions.ruled_out(mapping, bounds)
        if why is None:
            kept.append((mapping, item))
        else:
            dropped.setdefault(why, []).append(_shown(mapping))
    for why, shown in dropped.items():
        assumptions.remark(f"solve dropped {assumptions.listed(shown)}, as {why}")
    return _rebuilt(found, kept) if dropped else found


def _shown(mapping: dict[Any, Any]) -> str:
    """A solution as a note shows it: ``v = −√2·√(g h)``, or ``(K, ω) = (8, −√3)``."""
    names = [display.unicode(unknown) for unknown in mapping]
    values = [display.unicode(value) for value in mapping.values()]
    if len(names) == 1:
        return f"{names[0]} = {values[0]}"
    return f"({', '.join(names)}) = ({', '.join(values)})"


@solver(
    "dsolve",
    summary="`dsolve(f'(x) = f(x), f(x))`, the solution of a differential equation; "
    "`dsolve(\\frac{dy}{dt} = r y, y(0) = y_0)` with conditions",
    examples=[
        Example("dsolve(f'(x) = f(x), f(x))",
                sympy.Eq(f(x), sympy.Symbol("C1") * sympy.exp(x)), "standard"),
        Example("dsolve(f''(x) + f(x) = 0, f(x))",
                sympy.Eq(f(x), sympy.Symbol("C1") * sympy.sin(x)
                         + sympy.Symbol("C2") * sympy.cos(x)), "standard"),
        Example("dsolve(f'(x) = f(x), f(0) = 2)", sympy.Eq(f(x), 2 * sympy.exp(x)),
                "sympy.dsolve(f(x).diff(x) - f(x), ics={f(0): 2})"),
    ],
)  # fmt: skip
class DifferentialSolve:
    def exact(self, equations: Any, *unknowns: Any) -> Any:
        listed = _equations([equations])
        given = _flat(unknowns)
        if given and isinstance(given[0], sympy.Symbol):  # y'' + y = 0, for y
            raise EvaluationError(
                f"dsolve solves for a function written with its variable, as in "
                f"dsolve(f''(x) + f(x) = 0, f(x)), not for {display.text(given[0])}"
            )
        differential = [each for each in listed if _differential(each)]
        variables = {
            v for each in differential for d in each.atoms(sympy.Derivative) for v in d.variables
        }
        conditions = [each for each in [*listed, *given] if _a_condition(each, variables)]
        functions = [each for each in given if each not in conditions]
        stray = [
            each
            for each in [*listed, *functions]
            if each not in differential and not _a_function(each)
        ]
        if not differential:
            shown = ", ".join(display.unicode(each) for each in listed)
            raise EvaluationError(
                f"{shown} is no differential equation: it has no derivative of a function "
                "written with its variable",
                notes=[
                    "a name nothing defines is a symbol, whose derivative is 0, unless it is "
                    "written with its variable, as y(t)"
                ],
                hint="write the unknown with its variable, as in "
                "dsolve(\\frac{d}{dt} y(t) = r y(t), y(0) = y_0)",
            )
        if stray:
            what = display.unicode(stray[0])
            if stray[0] in (sympy.true, sympy.false):  # as SymPy reads |u(0)| < \infty
                what = f"a condition that SymPy reads as {what}"
            raise EvaluationError(
                "dsolve takes conditions on the values of the solution, as y(0) = y_0 or "
                f"y'(0) = 0, and not {what}",
                hint="a condition that the solution stays finite, as |u(0)| < \\infty, is not "
                "supported yet: leave it out, and choose the constants it fixes",
            )
        (variable,) = variables if len(variables) == 1 else (None,)
        ics = {_at(condition.lhs, variable): condition.rhs for condition in conditions}
        try:
            return sympy.dsolve(
                differential[0] if len(differential) == 1 else differential,
                *functions,
                ics=ics or None,
            )
        except (NotImplementedError, ValueError) as error:
            raise EvaluationError(
                "SymPy could not solve this differential equation"
                + (" with these conditions" if ics else ""),
                notes=[f"SymPy says: {error}"],
            ) from None


def _differential(equation: Any) -> bool:
    """Whether an equation is a differential one: it has a derivative of an unknown
    function, written with its variable."""
    return isinstance(equation, sympy.Basic) and any(
        isinstance(d.expr, AppliedUndef) for d in equation.atoms(sympy.Derivative)
    )


def _a_condition(equation: Any, variables: set[Any]) -> bool:
    """Whether an equation is a condition on the solution of a differential equation, the
    value of the function or of a derivative at a point: ``y(0) = y_0``, ``y'(0) = 0``."""
    if not isinstance(equation, sympy.Equality):
        return False
    at = equation.lhs
    if isinstance(at, sympy.Subs) and isinstance(at.expr, sympy.Derivative):
        return True
    return isinstance(at, AppliedUndef) and not any(arg.has(*variables) for arg in at.args)


def _at(condition: Any, variable: Any) -> Any:
    """A derivative at a point, y'(0), written with the variable of the equation, as SymPy
    takes a condition on it."""
    if isinstance(condition, sympy.Subs) and variable is not None and len(condition.variables) == 1:
        return condition.xreplace({condition.variables[0]: variable})
    return condition


def _a_function(value: Any) -> bool:
    """Whether a value is an unknown function written with its variable, as y(t)."""
    return isinstance(value, AppliedUndef)


def _equation(value: Any, name: str) -> Any:
    """``value``, which ``name`` takes apart, if it is an equation; an error otherwise."""
    if not hasattr(value, "lhs") or not hasattr(value, "rhs"):
        raise EvaluationError(
            f"{name} takes an equation, as in {name}(x^2 = 4), but {display.text(value)} is not one"
        )
    return value


@function(
    "RHS",
    aliases=["rhs"],
    summary="the right side of an equation",
    examples=[
        Example("RHS(x^2 = 4)", 4, "sympy.Eq(x**2, 4).rhs"),
        Example("rhs(y = 2x + 1)", 2 * x + 1, "standard"),
    ],
    lifts=False,
    scalar=False,
)
class RightSide:
    def exact(self, equation: Any) -> Any:
        if isinstance(equation, list | tuple):
            sides = [self.exact(item) for item in equation]
            return sides[0] if len(sides) == 1 else sides
        return _equation(equation, "RHS").rhs


@function(
    "LHS",
    aliases=["lhs"],
    summary="the left side of an equation",
    examples=[
        Example("LHS(x^2 = 4)", x**2, "sympy.Eq(x**2, 4).lhs"),
        Example("lhs(y = 2x + 1)", y, "standard"),
    ],
    lifts=False,
    scalar=False,
)
class LeftSide:
    def exact(self, equation: Any) -> Any:
        if isinstance(equation, list | tuple):
            sides = [self.exact(item) for item in equation]
            return sides[0] if len(sides) == 1 else sides
        return _equation(equation, "LHS").lhs


@constructor(
    "Eq",
    summary="`Eq(a, b)`, the equation a = b",
    examples=[
        Example("Eq(x, 2)", sympy.Eq(x, 2), "standard"),
        Example("Eq(1, 1)", True, "sympy.Eq(1, 1)"),
    ],
)
class Equation:
    def exact(self, lhs: Any, rhs: Any) -> Any:
        return sympy.Eq(lhs, rhs)


@constructor(
    "symbols",
    summary="`symbols('x y')`, the symbols named in quotes",
    examples=[
        Example("symbols('x y')", [x, y], "sympy.symbols('x y')"),
        Example("symbols('t')", sympy.Symbol("t"), "standard"),
    ],
    takes_names=True,
)
class Symbols:
    def exact(self, names: Any) -> Any:
        """The symbols named in a string; a name is its own symbol."""
        if isinstance(names, sympy.Symbol):
            return names
        if not isinstance(names, str):
            raise EvaluationError(
                f"symbols takes names in quotes, as in symbols('x y'), not {display.text(names)}"
            )
        return sympy.symbols(names)


@function(
    "latex",
    summary="the LaTeX of a value, as text",
    examples=[
        Example("latex(x^2)", "x^{2}", "sympy.latex(x**2)"),
        Example("latex(1/2)", r"\frac{1}{2}", "standard"),
    ],
    lifts=False,
    scalar=False,
)
class Latex:
    def exact(self, value: Any) -> Any:
        if hasattr(value, "latex") and callable(value.latex):
            return str(value.latex())
        return str(sympy.latex(value))
