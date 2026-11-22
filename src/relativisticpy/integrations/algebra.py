r"""Algebra and calculus.

SymPy's simplification, expansion and factorisation, derivatives, integrals, limits,
series and substitutions, and numerical evaluation. Each applies to every component of a
tensor, as ``simplify(R_{abcd})`` simplifies each.
"""

from __future__ import annotations

from typing import Any

import sympy

from relativisticpy import assumptions
from relativisticpy.errors import EvaluationError
from relativisticpy.integrations import Example, function

x, y, a, b, n, N = sympy.symbols("x y a b n N")


@function(
    "simplify",
    aliases=["tsimplify"],
    summary="the simplest form; of an equation between tensors without components, its "
    "canonical form",
    examples=[
        Example("simplify(sin(x)^2 + cos(x)^2)", 1, "standard"),
        Example("simplify((x^2 - 1)/(x - 1))", x + 1, "standard"),
    ],
    points=[(sympy.sin(x) ** 2 + sympy.cos(x) ** 2,), ((x**2 - 1) / (x - 1),)],
)
class Simplify:
    def exact(self, expr: Any) -> Any:
        return sympy.simplify(expr)


@function(
    "expand",
    summary="multiplied out",
    examples=[
        Example("expand((x + 1)^2)", x**2 + 2 * x + 1, "standard"),
        Example("expand((a + b)(a - b))", a**2 - b**2, "standard"),
    ],
    points=[((x + 1) ** 2,), ((x - 1) * (x + 1),)],
)
class Expand:
    def exact(self, expr: Any) -> Any:
        return sympy.expand(expr)


@function(
    "factor",
    summary="factorised",
    examples=[
        Example("factor(x^2 - 1)", (x - 1) * (x + 1), "standard"),
        Example("factor(x^2 + 2x + 1)", (x + 1) ** 2, "standard"),
    ],
    points=[(x**2 - 1,), (x**2 + 2 * x + 1,)],
)
class Factor:
    def exact(self, expr: Any) -> Any:
        return sympy.factor(expr)


@function(
    "collect",
    summary="`collect(f, x)`, the terms of f gathered by powers of x",
    examples=[
        Example("collect(a x + b x, x)", (a + b) * x, "standard"),
        Example("collect(a x^2 + x^2 + b x, x)", (a + 1) * x**2 + b * x, "standard"),
    ],
    points=[(a * x + b * x, x), (a * x**2 + x**2, x)],
)
class Collect:
    def exact(self, expr: Any, syms: Any, *options: Any) -> Any:
        return sympy.collect(expr, syms, *options)


@function(
    "cancel",
    summary="a fraction, with common factors cancelled",
    examples=[
        Example("cancel((x^2 - 1)/(x - 1))", x + 1, "standard"),
        Example("cancel((x^2 + 2x + 1)/(x + 1))", x + 1, "standard"),
    ],
    points=[((x**2 - 1) / (x - 1),), ((x**2 + 2 * x + 1) / (x + 1),)],
)
class Cancel:
    def exact(self, expr: Any) -> Any:
        return sympy.cancel(expr)


@function(
    "apart",
    summary="partial fractions",
    examples=[
        Example("apart(1/(x (x + 1)))", 1 / x - 1 / (x + 1), "standard"),
        Example("apart(1/(x^2 - 1))", 1 / (2 * (x - 1)) - 1 / (2 * (x + 1)), "standard"),
    ],
    points=[(1 / (x * (x + 1)),), (1 / (x**2 - 1),)],
)
class Apart:
    def exact(self, expr: Any) -> Any:
        return sympy.apart(expr)


@function(
    "together",
    summary="one fraction",
    examples=[
        Example("together(1/x + 1/y)", (x + y) / (x * y), "standard"),
        Example("together(1/x + 1)", (x + 1) / x, "standard"),
    ],
    points=[(1 / x + 1 / y,), (1 / x + 1,)],
)
class Together:
    def exact(self, expr: Any) -> Any:
        return sympy.together(expr)


@function(
    "trigsimp",
    summary="trigonometric functions simplified",
    examples=[
        Example("trigsimp(2 sin(x) cos(x))", sympy.sin(2 * x), "standard"),
        Example("trigsimp(sin(x)^2 + cos(x)^2)", 1, "standard"),
    ],
    points=[(2 * sympy.sin(x) * sympy.cos(x),), (sympy.sin(x) ** 2 + sympy.cos(x) ** 2,)],
)
class TrigonometricSimplify:
    def exact(self, expr: Any) -> Any:
        return sympy.trigsimp(expr)


@function(
    "powsimp",
    summary="powers combined: `x^a x^b` is `x^{a+b}`",
    examples=[
        Example("powsimp(x^a x^b)", x ** (a + b), "standard"),
        Example("powsimp(x^2 x^3)", x**5, "standard"),
    ],
    points=[(x**a * x**b,), (y**2 * y**a,)],
)
class PowerSimplify:
    def exact(self, expr: Any) -> Any:
        return sympy.powsimp(expr)


@function(
    "expand_trig",
    summary="functions of sums and multiples of angles, expanded",
    examples=[
        Example("expand_trig(sin(2x))", 2 * sympy.sin(x) * sympy.cos(x), "standard"),
        Example(
            "expand_trig(cos(x + y))",
            sympy.cos(x) * sympy.cos(y) - sympy.sin(x) * sympy.sin(y),
            "standard",
        ),
    ],
    points=[(sympy.sin(2 * x),), (sympy.cos(x + y),)],
)
class ExpandTrigonometric:
    def exact(self, expr: Any) -> Any:
        return sympy.expand_trig(expr)


@function(
    "nsimplify",
    summary="the exact number a decimal stands for: `0.5` is `1/2`",
    examples=[
        Example("nsimplify(0.5)", sympy.Rational(1, 2), "standard"),
        Example("nsimplify(0.25)", sympy.Rational(1, 4), "standard"),
    ],
    points=[(0.5,), (0.25,)],
)
class NumberSimplify:
    def exact(self, expr: Any) -> Any:
        return sympy.nsimplify(expr)


@function(
    "diff",
    summary=r"`diff(f, x)`, `diff(f, x, 2)`, `diff(f, x, y)`: derivatives, as `\frac{d}{dx}`",
    examples=[
        Example("diff(x^3, x)", 3 * x**2, "standard"),
        Example("diff(sin(x), x, 2)", -sympy.sin(x), "standard"),
    ],
    points=[(x**3, x), (sympy.sin(x), x)],
)
class Differentiate:
    def exact(self, f: Any, *symbols: Any) -> Any:
        return sympy.diff(f, *symbols)


@function(
    "integrate",
    summary=r"`integrate(f, x)`, `integrate(f, (x, 0, 1))`: integrals, as `\int`",
    examples=[
        Example("integrate(x^2, x)", x**3 / 3, "standard"),
        Example("integrate(x, (x, 0, 2))", 2, "standard"),
    ],
    points=[(x**2, x), (sympy.cos(x), x)],
)
class Integrate:
    def exact(self, f: Any, *limits: Any, assuming: tuple[Any, ...] = ()) -> Any:
        return assumptions.integrated(f, limits, assuming)


@function(
    "limit",
    aliases=["lim"],
    summary=r"`limit(f, x, a)`, as x tends to a; `\lim_{x \to a^+}` for one side",
    examples=[
        Example("limit(sin(x)/x, x, 0)", 1, "standard"),
        Example("limit(1/x, x, oo)", 0, "standard"),
    ],
    points=[(sympy.sin(x) / x, x, 0), ((1 + 1 / x) ** x, x, sympy.oo)],
)
class Limit:
    def exact(self, f: Any, x: Any, a: Any, dir: Any = "+") -> Any:
        return sympy.limit(f, x, a, dir)


@function(
    "series",
    summary="`series(f, x, a, n)`, the Taylor series about a, to order n",
    examples=[
        Example("series(exp(x), x, 0, 3)", 1 + x + x**2 / 2 + sympy.O(x**3), "standard"),
        Example("series(sin(x), x, 0, 4)", x - x**3 / 6 + sympy.O(x**4), "standard"),
    ],
    points=[(sympy.exp(x), x, 0, 3), (sympy.cos(x), x, 0, 3)],
)
class Series:
    def exact(self, expr: Any, x: Any = None, x0: Any = 0, n: Any = 6) -> Any:
        return sympy.series(expr, x, x0, n)


_ABSENT = object()


@function(
    "subs",
    summary="`subs(f, x, a)`, f with a put for x; `subs(f, x = a)` too",
    examples=[
        Example("subs(x^2, x, 3)", 9, "standard"),
        Example("subs(x + y, x = 1)", y + 1, "standard"),
    ],
    points=[(x**2, x, 3), (x + y, x, 1)],
)
class Substitute:
    def exact(self, expr: Any, old: Any, new: Any = _ABSENT) -> Any:
        """``subs(expr, old, new)``, or ``subs(expr, old = new, ...)`` with one rule or
        more, each an equation."""
        rules = (old,) if new is _ABSENT else (old, new)
        if all(isinstance(rule, sympy.Equality) for rule in rules):
            return sympy.sympify(expr).subs([(rule.lhs, rule.rhs) for rule in rules])
        if new is _ABSENT:
            raise EvaluationError(
                "subs takes an expression and what to put for what, as in subs(x^2, x, 3), "
                "or rules, as in subs(x^2, x = 3)"
            )
        return sympy.sympify(expr).subs(old, new)


@function(
    "N",
    aliases=["numerical", "evalf"],
    summary="`N(x)`, `N(x, digits)`: a decimal number",
    examples=[
        Example("N(pi, 5)", 3.1416, "standard: pi to five digits", tolerance=1e-5),
        Example("numerical(sqrt(2))", 1.4142135623730951, "math.sqrt(2)"),
    ],
    points=[(sympy.pi,), (sympy.sqrt(2),)],
)
class Numerical:
    def exact(self, value: Any, digits: Any = 15) -> Any:
        return sympy.N(value, digits)


@function(
    "doit",
    summary="sums, products and integrals, kept as written, carried out",
    examples=[
        Example(r"doit(\sum_{n=1}^{N} n)", N * (N + 1) / 2, "standard"),
        Example(r"doit(\prod_{n=1}^{N} 2)", 2**N, "standard"),
    ],
    points=[(sympy.Sum(n, (n, 1, N)),), (sympy.Product(2, (n, 1, N)),)],
)
class CarryOut:
    def exact(self, value: Any) -> Any:
        return value.doit() if hasattr(value, "doit") else value
