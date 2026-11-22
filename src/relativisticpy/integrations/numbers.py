r"""Numbers.

Logarithms to a base and roots, binomial coefficients, the angle of a point, the least and
the greatest of several values, and the greatest common divisor and least common multiple,
with SymPy. Each takes its arguments whole.
"""

from __future__ import annotations

from typing import Any

import sympy

from relativisticpy.integrations import Example, function

x, n = sympy.symbols("x n")


@function(
    "log",
    summary=r"the logarithm; `log(x, b)` to the base b, `\log_b x` in LaTeX",
    examples=[
        Example("log(e^3)", 3, "standard"),
        Example("log(8, 2)", 3, "standard"),
    ],
    lifts=False,
)
class Logarithm:
    def exact(self, x: Any, base: Any = None) -> Any:
        return sympy.log(x) if base is None else sympy.log(x, base)

    def numeric(self, xp: Any, x: Any) -> Any:
        return xp.log(x)


@function(
    "root",
    summary="`root(x, n)`, the nth root",
    examples=[
        Example("root(8, 3)", 2, "standard"),
        Example("root(16, 4)", 2, "standard"),
    ],
    lifts=False,
)
class Root:
    def exact(self, x: Any, n: Any) -> Any:
        return sympy.root(x, n)


@function(
    "binomial",
    summary=r"`binomial(n, k)`; `\binom{n}{k}` too",
    examples=[
        Example("binomial(5, 2)", 10, "standard"),
        Example("binomial(n, 1)", n, "standard"),
    ],
    lifts=False,
)
class Binomial:
    def exact(self, n: Any, k: Any) -> Any:
        return sympy.binomial(n, k)


@function(
    "atan2",
    summary="`atan2(y, x)`, the angle of the point (x, y)",
    examples=[
        Example("atan2(1, 1)", sympy.pi / 4, "standard"),
        Example("atan2(1, 0)", sympy.pi / 2, "standard"),
    ],
    lifts=False,
)
class Angle:
    def exact(self, y: Any, x: Any) -> Any:
        return sympy.atan2(y, x)


@function(
    "min",
    aliases=["Min"],
    summary="the least of the arguments",
    examples=[
        Example("min(3, 1, 2)", 1, "standard"),
        Example("min(x, x + 1)", x, "standard"),
    ],
    lifts=False,
)
class Least:
    def exact(self, first: Any, *rest: Any) -> Any:
        return sympy.Min(first, *rest)


@function(
    "max",
    aliases=["Max"],
    summary="the greatest of the arguments",
    examples=[
        Example("max(3, 1, 2)", 3, "standard"),
        Example("max(x, x + 1)", x + 1, "standard"),
    ],
    lifts=False,
)
class Greatest:
    def exact(self, first: Any, *rest: Any) -> Any:
        return sympy.Max(first, *rest)


@function(
    "gcd",
    summary="the greatest common divisor",
    examples=[
        Example("gcd(12, 18)", 6, "standard"),
        Example("gcd(x^2 - 1, x - 1)", x - 1, "standard"),
    ],
    lifts=False,
)
class GreatestCommonDivisor:
    def exact(self, a: Any, b: Any, *rest: Any) -> Any:
        return sympy.gcd(a, b, *rest)


@function(
    "lcm",
    summary="the least common multiple",
    examples=[
        Example("lcm(4, 6)", 12, "standard"),
        Example("lcm(x, x^2)", x**2, "standard"),
    ],
    lifts=False,
)
class LeastCommonMultiple:
    def exact(self, a: Any, b: Any, *rest: Any) -> Any:
        return sympy.lcm(a, b, *rest)
