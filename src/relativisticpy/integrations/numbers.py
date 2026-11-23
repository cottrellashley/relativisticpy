r"""Numbers.

Logarithms to a base and roots, binomial coefficients, the angle of a point, the least and
the greatest of several values, and the greatest common divisor and least common multiple,
with SymPy. Each takes its arguments whole. The greatest common divisor of a whole number
and a symbol is of integers, and kept as it is until the symbol has a value, as the index
of a sum gets one; of polynomials alone, it is theirs.
"""

from __future__ import annotations

import math
from typing import Any

import sympy

from relativisticpy.integrations import Example, function

x, n = sympy.symbols("x n")


@function(
    "log",
    summary=r"the logarithm; `log(x, b)` to the base b, `\log_b x` in LaTeX",
    examples=[
        Example("log(e^3)", 3, "standard"),
        Example("log(8, 2)", 3, "math.log(8, 2)"),
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
        Example("root(8, 3)", 2, "mpmath.root(8, 3)"),
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
        Example("binomial(5, 2)", 10, "math.comb(5, 2)"),
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
        Example("atan2(1, 1)", sympy.pi / 4, "math.atan2(1, 1)"),
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
        Example("min(3, 1, 2)", 1, "numpy.min([3, 1, 2])"),
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
        Example("max(3, 1, 2)", 3, "numpy.max([3, 1, 2])"),
        Example("max(x, x + 1)", x + 1, "standard"),
    ],
    lifts=False,
)
class Greatest:
    def exact(self, first: Any, *rest: Any) -> Any:
        return sympy.Max(first, *rest)


class gcd(sympy.Function):  # type: ignore[misc]  # named as its calls print
    """The greatest common divisor of integers, kept as it is where one is no number yet,
    as the index of a sum is before the sum gives it its values."""

    @classmethod
    def eval(cls, *args: Any) -> Any:
        if all(arg.is_Integer for arg in args):
            return sympy.Integer(math.gcd(*(int(arg) for arg in args)))
        return None

    def _eval_is_integer(self) -> bool | None:
        return True if all(arg.is_integer for arg in self.args) else None


class lcm(sympy.Function):  # type: ignore[misc]  # named as its calls print
    """The least common multiple of integers, kept as it is where one is no number yet."""

    @classmethod
    def eval(cls, *args: Any) -> Any:
        if all(arg.is_Integer for arg in args):
            return sympy.Integer(math.lcm(*(int(arg) for arg in args)))
        return None

    def _eval_is_integer(self) -> bool | None:
        return True if all(arg.is_integer for arg in self.args) else None


def _of_integers(values: tuple[Any, ...]) -> bool:
    """Whether a greatest common divisor or a least common multiple is of integers, not of
    polynomials: given a whole number, as gcd(k, 6) is, or integers alone, as the indices of
    sums are, but not numbers alone, which it computes."""
    given = [sympy.sympify(value) for value in values]
    if all(value.is_Integer for value in given):
        return False
    return any(value.is_Integer for value in given) or all(value.is_integer for value in given)


@function(
    "gcd",
    summary="the greatest common divisor",
    examples=[
        Example("gcd(12, 18)", 6, "standard"),
        Example("gcd(x^2 - 1, x - 1)", x - 1, "standard"),
        Example(r"\frac{1}{6} \sum_{k=0}^{5} 2^{\gcd(k, 6)}", 14,
                "OEIS A000029: the necklaces of six beads in two colours"),
    ],
    lifts=False,
)  # fmt: skip
class GreatestCommonDivisor:
    def exact(self, a: Any, b: Any, *rest: Any) -> Any:
        values = (a, b, *rest)
        return gcd(*values) if _of_integers(values) else sympy.gcd(*values)


@function(
    "lcm",
    summary="the least common multiple",
    examples=[
        Example("lcm(4, 6)", 12, "math.lcm(4, 6)"),
        Example("lcm(x, x^2)", x**2, "standard"),
    ],
    lifts=False,
)
class LeastCommonMultiple:
    def exact(self, a: Any, b: Any, *rest: Any) -> Any:
        values = (a, b, *rest)
        return lcm(*values) if _of_integers(values) else sympy.lcm(*values)
