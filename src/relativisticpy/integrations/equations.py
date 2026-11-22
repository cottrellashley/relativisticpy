r"""Equations.

Solving equations, algebraic and differential, with SymPy; taking them apart; making an
equation, and symbols; and the LaTeX of a value.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import sympy

from relativisticpy import display
from relativisticpy.errors import EvaluationError
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
        Example("solve(x^2 = 4, x)", [-2, 2], "standard"),
        Example("solve(2x + 1 = 0, x)", [sympy.Rational(-1, 2)], "standard"),
    ],
)
class Solve:
    def exact(self, equations: Any, *unknowns: Any) -> Any:
        if isinstance(equations, TensorEquation | list | tuple):
            equations = _equations([equations])
        return sympy.solve(equations, *unknowns)


@solver(
    "dsolve",
    summary="`dsolve(f'(x) = f(x), f(x))`, the solution of a differential equation",
    examples=[
        Example("dsolve(f'(x) = f(x), f(x))",
                sympy.Eq(f(x), sympy.Symbol("C1") * sympy.exp(x)), "standard"),
        Example("dsolve(f''(x) + f(x) = 0, f(x))",
                sympy.Eq(f(x), sympy.Symbol("C1") * sympy.sin(x)
                         + sympy.Symbol("C2") * sympy.cos(x)), "standard"),
    ],
)  # fmt: skip
class DifferentialSolve:
    def exact(self, equations: Any, *unknowns: Any) -> Any:
        if isinstance(equations, TensorEquation | list | tuple):
            found = _equations([equations])
            equations = found[0] if len(found) == 1 else found
        if unknowns and isinstance(unknowns[0], sympy.Symbol):  # y'' + y = 0, for y
            raise EvaluationError(
                f"dsolve solves for a function written with its variable, as in "
                f"dsolve(f''(x) + f(x) = 0, f(x)), not for {display.text(unknowns[0])}"
            )
        return sympy.dsolve(equations, *unknowns)


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
        Example("RHS(x^2 = 4)", 4, "standard"),
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
        Example("LHS(x^2 = 4)", x**2, "standard"),
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
        Example("Eq(1, 1)", True, "standard"),
    ],
)
class Equation:
    def exact(self, lhs: Any, rhs: Any) -> Any:
        return sympy.Eq(lhs, rhs)


@constructor(
    "symbols",
    summary="`symbols('x y')`, the symbols named in quotes",
    examples=[
        Example("symbols('x y')", [x, y], "standard"),
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
        Example("latex(x^2)", "x^{2}", "standard"),
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
