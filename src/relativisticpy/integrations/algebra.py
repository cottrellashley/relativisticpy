r"""Algebra and calculus.

SymPy's simplification, expansion and factorisation, derivatives, integrals, limits,
series and substitutions, and numerical evaluation. Each applies to every component of a
tensor, as ``simplify(R_{abcd})`` simplifies each.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sympy

from relativisticpy import assumptions, display
from relativisticpy.errors import EvaluationError
from relativisticpy.integrations import Example, function
from relativisticpy.tensors.numerical import whole_number

x, y, a, b, n, N = sympy.symbols("x y a b n N")
HALF = sympy.Rational(1, 2)


@function(
    "simplify",
    aliases=["tsimplify"],
    summary="the simplest form; of an equation between tensors without components, its "
    "canonical form",
    gives_form=True,
    examples=[
        Example(
            "simplify(sin(x)^2 + cos(x)^2)", 1, "sympy.simplify(sympy.sin(x)**2 + sympy.cos(x)**2)"
        ),
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
    gives_form=True,
    examples=[
        Example("expand((x + 1)^2)", x**2 + 2 * x + 1, "sympy.expand((x + 1)**2)"),
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
    gives_form=True,
    examples=[
        Example("factor(x^2 - 1)", (x - 1) * (x + 1), "sympy.factor(x**2 - 1)"),
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
    gives_form=True,
    examples=[
        Example("collect(a x + b x, x)", (a + b) * x, "sympy.collect(a*x + b*x, x)"),
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
    gives_form=True,
    examples=[
        Example("cancel((x^2 - 1)/(x - 1))", x + 1, "sympy.cancel((x**2 - 1)/(x - 1))"),
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
    gives_form=True,
    examples=[
        Example("apart(1/(x (x + 1)))", 1 / x - 1 / (x + 1), "sympy.apart(1/(x*(x + 1)))"),
        Example("apart(1/(x^2 - 1))", HALF / (x - 1) - HALF / (x + 1), "standard"),
    ],
    points=[(1 / (x * (x + 1)),), (1 / (x**2 - 1),)],
)
class Apart:
    def exact(self, expr: Any) -> Any:
        return sympy.apart(expr)


@function(
    "together",
    summary="one fraction",
    gives_form=True,
    examples=[
        Example("together(1/x + 1/y)", (x + y) / (x * y), "sympy.together(1/x + 1/y)"),
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
    gives_form=True,
    examples=[
        Example(
            "trigsimp(2 sin(x) cos(x))",
            sympy.sin(2 * x),
            "sympy.trigsimp(2*sympy.sin(x)*sympy.cos(x))",
        ),
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
    gives_form=True,
    examples=[
        Example("powsimp(x^a x^b)", x ** (a + b), "sympy.powsimp(x**a*x**b)"),
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
    gives_form=True,
    examples=[
        Example(
            "expand_trig(sin(2x))",
            2 * sympy.sin(x) * sympy.cos(x),
            "sympy.expand_trig(sympy.sin(2*x))",
        ),
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
        Example("nsimplify(0.5)", sympy.Rational(1, 2), "fractions.Fraction(0.5)"),
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
        Example("diff(x^3, x)", 3 * x**2, "sympy.diff(x**3, x)"),
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
        Example("integrate(x, (x, 0, 2))", 2, "scipy.integrate.quad(lambda x: x, 0, 2)[0]"),
    ],
    points=[(x**2, x), (sympy.cos(x), x)],
)
class Integrate:
    def exact(self, f: Any, *limits: Any, assuming: tuple[Any, ...] = ()) -> Any:
        try:
            return assumptions.integrated(f, limits, assuming)
        except TypeError:  # SymPy compares the limits, and fails where one is not real
            off = _off_the_real_line(limits)
            if off is None:
                raise
            raise EvaluationError(
                f"SymPy could not compute this integral, whose limit {display.unicode(off)} "
                "is not real",
                notes=[
                    "SymPy computes an integral along a path off the real line only in "
                    "simple cases, and fails where it compares a limit that is not real"
                ],
                hint="change its variable so that it runs along the real line",
            ) from None


def _off_the_real_line(limits: Sequence[Any]) -> Any:
    """A limit of an integral that is not real, as -i∞ of \\int_z^{-i\\infty}; ``None``
    where SymPy knows of none."""
    for limit in limits:
        if isinstance(limit, tuple | sympy.Tuple) and len(limit) == 3:
            for end in limit[1:]:
                if sympy.sympify(end).is_extended_real is False:
                    return end
    return None


#: The sides a limit is taken from: above, below, or both.
_SIDES = ("+", "-", "+-")


@function(
    "limit",
    aliases=["lim"],
    summary=r"`limit(f, x, a)`, as x tends to a from both sides, as `\lim_{x \to a}` is; "
    r"`limit(f, x, a, '-')` from below and `'+'` from above, as `\lim_{x \to a^-}`",
    examples=[
        Example("limit(sin(x)/x, x, 0)", 1, "mpmath.limit(lambda x: mpmath.sin(x)/x, 0)"),
        Example("limit(1/x, x, oo)", 0, "standard"),
    ],
    points=[(sympy.sin(x) / x, x, 0), ((1 + 1 / x) ** x, x, sympy.oo)],
    takes_names=True,
)
class Limit:
    def exact(self, f: Any, x: Any, a: Any, direction: Any = "+-") -> Any:
        """The limit of ``f`` as ``x`` tends to ``a``, as the notation takes it: from both
        sides, where it exists, unless ``direction`` gives one, '+' from above or '-' from
        below. The side is the only text it takes."""
        from relativisticpy.runtime import core

        if any(_text_in(each) for each in (f, x, a)):
            raise EvaluationError(
                "text in quotes is not a value to compute with",
                hint="write the expression without quotes: only the side of a limit is "
                "given in quotes, as in limit(1/x, x, 0, '-')",
            )
        if not isinstance(direction, str) or direction not in _SIDES:
            raise EvaluationError(
                "a limit is taken from above, '+', from below, '-', or from both sides, "
                f"'+-', not {display.text(direction)}",
                hint="for example limit(1/x, x, 0, '-')",
            )
        return core.limit(f, x, a, direction)


def _text_in(value: Any) -> bool:
    """Whether a value is text, or a list or a tuple that holds some."""
    if isinstance(value, str):
        return True
    return isinstance(value, list | tuple) and any(_text_in(item) for item in value)


@function(
    "series",
    summary="`series(f, x, a, n)`, the Taylor series about a, to order n",
    examples=[
        Example(
            "series(exp(x), x, 0, 3)",
            1 + x + x**2 / 2 + sympy.O(x**3),
            "sympy.series(sympy.exp(x), x, 0, 3)",
        ),
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
        Example("subs(x^2, x, 3)", 9, "numpy.polyval([1, 0, 0], 3)"),
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
        return sympy.N(value, _digits(digits))


def _digits(digits: Any) -> int:
    """How many significant digits ``N`` gives: a whole number from 1, given as a decimal as
    data gives it; an error for any other, which SymPy would read as some number of digits."""
    count = whole_number(digits)
    if not (count.is_Integer and count >= 1):
        shown = format(float(count), "g") if count.is_Float else count
        raise EvaluationError(
            f"N gives a number to a whole number of significant digits from 1, not {shown}",
            hint="as in N(pi, 5), pi to five digits",
        )
    return int(count)


@function(
    "doit",
    summary="sums, products and integrals, kept as written, carried out",
    examples=[
        Example(r"doit(\sum_{n=1}^{N} n)", N * (N + 1) / 2, "sympy.Sum(n, (n, 1, N)).doit()"),
        Example(r"doit(\prod_{n=1}^{N} 2)", 2**N, "standard"),
    ],
    points=[(sympy.Sum(n, (n, 1, N)),), (sympy.Product(2, (n, 1, N)),)],
)
class CarryOut:
    def exact(self, value: Any) -> Any:
        return value.doit() if hasattr(value, "doit") else value
