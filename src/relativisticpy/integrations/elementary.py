r"""Elementary functions.

The trigonometric and hyperbolic functions and their inverses, the exponential, logarithm
and square root, the parts of a complex number, rounding, the factorials, and Euler's gamma
and error functions: exactly with SymPy and, where NumPy has the function itself, with
NumPy or JAX for numbers. Each applies to every component of a tensor.
"""

from __future__ import annotations

from typing import Any

import sympy

from relativisticpy.integrations import Example, function
from relativisticpy.tensors.numerical import continued, whole_number

_UNIT = [(0.3,), (0.7,)]  # inside (-1, 1), where the inverse sine, cosine and others are real
_BEYOND = [(1.6,), (2.5,)]  # beyond 1, where the inverse hyperbolic cosine and others are real


@function(
    "sin",
    summary=r"the sine; `\sin^2\theta` is the square of `\sin\theta`",
    examples=[
        Example(r"\sin(\pi/6)", sympy.Rational(1, 2), "math.sin(math.pi / 6)"),
        Example("sin(0)", 0, "standard"),
    ],
)
class Sine:
    def exact(self, x: Any) -> Any:
        return sympy.sin(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        return xp.sin(x)


@function(
    "cos",
    summary="the cosine",
    examples=[
        Example(r"\cos(\pi/3)", sympy.Rational(1, 2), "math.cos(math.pi / 3)"),
        Example("cos(0)", 1, "standard"),
    ],
)
class Cosine:
    def exact(self, x: Any) -> Any:
        return sympy.cos(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        return xp.cos(x)


@function(
    "tan",
    summary="the tangent",
    examples=[
        Example(r"\tan(\pi/4)", 1, "math.tan(math.pi / 4)"),
        Example("tan(0)", 0, "standard"),
    ],
)
class Tangent:
    def exact(self, x: Any) -> Any:
        return sympy.tan(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        return xp.tan(x)


@function(
    "cot",
    summary="the cotangent",
    examples=[
        Example(r"\cot(\pi/4)", 1, "mpmath.cot(mpmath.pi / 4)"),
        Example(r"\cot(\pi/2)", 0, "standard"),
    ],
)
class Cotangent:
    def exact(self, x: Any) -> Any:
        return sympy.cot(x)


@function(
    "sec",
    summary="the secant",
    examples=[
        Example("sec(0)", 1, "standard"),
        Example(r"\sec(\pi/3)", 2, "mpmath.sec(mpmath.pi / 3)"),
    ],
)
class Secant:
    def exact(self, x: Any) -> Any:
        return sympy.sec(x)


@function(
    "csc",
    summary="the cosecant",
    examples=[
        Example(r"\csc(\pi/2)", 1, "standard"),
        Example(r"\csc(\pi/6)", 2, "mpmath.csc(mpmath.pi / 6)"),
    ],
)
class Cosecant:
    def exact(self, x: Any) -> Any:
        return sympy.csc(x)


@function(
    "asin",
    aliases=["arcsin"],
    summary=r"the inverse sine; `\sin^{-1} x` too",
    examples=[
        Example("asin(1)", sympy.pi / 2, "math.asin(1)"),
        Example(r"\arcsin(1/2)", sympy.pi / 6, "standard"),
    ],
    points=_UNIT,
)
class InverseSine:
    def exact(self, x: Any) -> Any:
        return sympy.asin(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        return xp.arcsin(x)


@function(
    "acos",
    aliases=["arccos"],
    summary="the inverse cosine",
    examples=[
        Example("acos(0)", sympy.pi / 2, "math.acos(0)"),
        Example(r"\arccos(1/2)", sympy.pi / 3, "standard"),
    ],
    points=_UNIT,
)
class InverseCosine:
    def exact(self, x: Any) -> Any:
        return sympy.acos(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        return xp.arccos(x)


@function(
    "atan",
    aliases=["arctan"],
    summary="the inverse tangent",
    examples=[
        Example("atan(1)", sympy.pi / 4, "math.atan(1)"),
        Example(r"\arctan(\sqrt{3})", sympy.pi / 3, "standard"),
    ],
)
class InverseTangent:
    def exact(self, x: Any) -> Any:
        return sympy.atan(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        return xp.arctan(x)


@function(
    "acot",
    summary="the inverse cotangent",
    examples=[
        Example("acot(1)", sympy.pi / 4, "mpmath.acot(1)"),
        Example("acot(0)", sympy.pi / 2, "standard"),
    ],
)
class InverseCotangent:
    def exact(self, x: Any) -> Any:
        return sympy.acot(x)


@function(
    "asec",
    summary="the inverse secant",
    examples=[
        Example("asec(1)", 0, "standard"),
        Example("asec(2)", sympy.pi / 3, "mpmath.asec(2)"),
    ],
    points=_BEYOND,
)
class InverseSecant:
    def exact(self, x: Any) -> Any:
        return sympy.asec(x)


@function(
    "acsc",
    summary="the inverse cosecant",
    examples=[
        Example("acsc(1)", sympy.pi / 2, "standard"),
        Example("acsc(2)", sympy.pi / 6, "mpmath.acsc(2)"),
    ],
    points=_BEYOND,
)
class InverseCosecant:
    def exact(self, x: Any) -> Any:
        return sympy.acsc(x)


@function(
    "sinh",
    summary="the hyperbolic sine",
    examples=[
        Example("sinh(0)", 0, "standard"),
        Example(r"\sinh(\ln 2)", sympy.Rational(3, 4), "math.sinh(math.log(2))"),
    ],
)
class HyperbolicSine:
    def exact(self, x: Any) -> Any:
        return sympy.sinh(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        return xp.sinh(x)


@function(
    "cosh",
    summary="the hyperbolic cosine",
    examples=[
        Example("cosh(0)", 1, "standard"),
        Example(r"\cosh(\ln 2)", sympy.Rational(5, 4), "math.cosh(math.log(2))"),
    ],
)
class HyperbolicCosine:
    def exact(self, x: Any) -> Any:
        return sympy.cosh(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        return xp.cosh(x)


@function(
    "tanh",
    summary="the hyperbolic tangent",
    examples=[
        Example("tanh(0)", 0, "standard"),
        Example(r"\tanh(\ln 2)", sympy.Rational(3, 5), "math.tanh(math.log(2))"),
    ],
)
class HyperbolicTangent:
    def exact(self, x: Any) -> Any:
        return sympy.tanh(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        return xp.tanh(x)


@function(
    "coth",
    summary="the hyperbolic cotangent",
    examples=[
        Example(r"\coth(\ln 2)", sympy.Rational(5, 3), "mpmath.coth(mpmath.log(2))"),
        Example(r"\coth(\ln 3)", sympy.Rational(5, 4), "standard"),
    ],
)
class HyperbolicCotangent:
    def exact(self, x: Any) -> Any:
        return sympy.coth(x)


@function(
    "sech",
    summary="the hyperbolic secant",
    examples=[
        Example("sech(0)", 1, "standard"),
        Example(r"\operatorname{sech}(\ln 2)", sympy.Rational(4, 5), "mpmath.sech(mpmath.log(2))"),
    ],
)
class HyperbolicSecant:
    def exact(self, x: Any) -> Any:
        return sympy.sech(x)


@function(
    "csch",
    summary="the hyperbolic cosecant",
    examples=[
        Example(r"\operatorname{csch}(\ln 2)", sympy.Rational(4, 3), "mpmath.csch(mpmath.log(2))"),
        Example(r"\operatorname{csch}(\ln 3)", sympy.Rational(3, 4), "standard"),
    ],
)
class HyperbolicCosecant:
    def exact(self, x: Any) -> Any:
        return sympy.csch(x)


@function(
    "asinh",
    aliases=["arsinh"],
    summary="the inverse hyperbolic sine",
    examples=[
        Example("asinh(0)", 0, "standard"),
        Example("asinh(3/4)", sympy.log(2), "math.asinh(3 / 4)"),
    ],
)
class InverseHyperbolicSine:
    def exact(self, x: Any) -> Any:
        return sympy.asinh(x)


@function(
    "acosh",
    aliases=["arcosh"],
    summary="the inverse hyperbolic cosine",
    examples=[
        Example("acosh(1)", 0, "standard"),
        Example("acosh(5/4)", sympy.log(2), "math.acosh(5 / 4)"),
    ],
    points=_BEYOND,
)
class InverseHyperbolicCosine:
    def exact(self, x: Any) -> Any:
        return sympy.acosh(x)


@function(
    "atanh",
    aliases=["artanh"],
    summary="the inverse hyperbolic tangent",
    examples=[
        Example("atanh(0)", 0, "standard"),
        Example("atanh(3/5)", sympy.log(2), "math.atanh(3 / 5)"),
    ],
    points=_UNIT,
)
class InverseHyperbolicTangent:
    def exact(self, x: Any) -> Any:
        return sympy.atanh(x)


@function(
    "acoth",
    aliases=["arcoth"],
    summary="the inverse hyperbolic cotangent",
    examples=[
        Example("acoth(5/3)", sympy.log(2), "mpmath.acoth(mpmath.mpf(5) / 3)"),
        Example("acoth(5/4)", sympy.log(3), "standard"),
    ],
    points=_BEYOND,
)
class InverseHyperbolicCotangent:
    def exact(self, x: Any) -> Any:
        return sympy.acoth(x)


@function(
    "asech",
    aliases=["arsech"],
    summary="the inverse hyperbolic secant",
    examples=[
        Example("asech(1)", 0, "standard"),
        Example("asech(4/5)", sympy.log(2), "mpmath.asech(mpmath.mpf(4) / 5)"),
    ],
    points=_UNIT,
)
class InverseHyperbolicSecant:
    def exact(self, x: Any) -> Any:
        return sympy.asech(x)


@function(
    "acsch",
    aliases=["arcsch"],
    summary="the inverse hyperbolic cosecant",
    examples=[
        Example("acsch(4/3)", sympy.log(2), "mpmath.acsch(mpmath.mpf(4) / 3)"),
        Example("acsch(3/4)", sympy.log(3), "standard"),
    ],
)
class InverseHyperbolicCosecant:
    def exact(self, x: Any) -> Any:
        return sympy.acsch(x)


@function(
    "exp",
    summary="the exponential; `e^{x}` too",
    examples=[
        Example("exp(0)", 1, "standard"),
        Example(r"\exp(\ln 3)", 3, "math.exp(math.log(3))"),
    ],
)
class Exponential:
    def exact(self, x: Any) -> Any:
        return sympy.exp(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        return xp.exp(x)


@function(
    "ln",
    summary="the natural logarithm",
    examples=[
        Example("ln(1)", 0, "standard"),
        Example(r"\ln(e^2)", 2, "math.log(math.e ** 2)"),
    ],
)
class NaturalLogarithm:
    def exact(self, x: Any) -> Any:
        return sympy.log(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        return continued(xp, xp.log, x)


@function(
    "sqrt",
    summary=r"the square root; `\sqrt[n]{x}` is the nth root",
    examples=[
        Example("sqrt(16)", 4, "math.sqrt(16)"),
        Example(r"\sqrt{1/4}", sympy.Rational(1, 2), "standard"),
    ],
)
class SquareRoot:
    def exact(self, x: Any) -> Any:
        return sympy.sqrt(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        return continued(xp, xp.sqrt, x)


@function(
    "abs",
    aliases=["Abs"],
    summary=r"the absolute value; `\lvert x \rvert` too",
    examples=[
        Example("abs(-3)", 3, "standard"),
        Example("abs(3 - 4i)", 5, "numpy.abs(3 - 4j)"),
    ],
)
class AbsoluteValue:
    def exact(self, x: Any) -> Any:
        return sympy.Abs(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        return xp.abs(x)


@function(
    "sign",
    summary="the sign: −1, 0 or 1",
    examples=[
        Example("sign(-2)", -1, "numpy.sign(-2)"),
        Example("sign(0)", 0, "standard"),
    ],
)
class Sign:
    def exact(self, x: Any) -> Any:
        return sympy.sign(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        return xp.sign(x)


@function(
    "floor",
    summary=r"rounded down; `\lfloor x \rfloor` too",
    examples=[
        Example("floor(5/2)", 2, "math.floor(5 / 2)"),
        Example("floor(-1/2)", -1, "standard"),
    ],
)
class Floor:
    def exact(self, x: Any) -> Any:
        return sympy.floor(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        return xp.floor(x)


@function(
    "ceiling",
    summary=r"rounded up; `\lceil x \rceil` too",
    examples=[
        Example("ceiling(5/2)", 3, "math.ceil(5 / 2)"),
        Example("ceiling(-1/2)", 0, "standard"),
    ],
)
class Ceiling:
    def exact(self, x: Any) -> Any:
        return sympy.ceiling(x)


@function(
    "factorial",
    summary="the factorial; `n!` too",
    requires="scipy",
    examples=[
        Example("factorial(5)", 120, "math.factorial(5)"),
        Example("factorial(0)", 1, "standard"),
    ],
)
class Factorial:
    def exact(self, x: Any) -> Any:
        return sympy.factorial(whole_number(x))

    def numeric(self, xp: Any, x: Any) -> Any:
        from scipy.special import gamma

        return gamma(x + 1)  # x! = Γ(x + 1), as SymPy has it of any x (DLMF 5.4.1)


@function(
    "factorial2",
    summary="the double factorial; `n!!` too",
    examples=[
        Example("factorial2(7)", 105, "scipy.special.factorial2(7, exact=True)"),
        Example("factorial2(8)", 384, "standard: 8·6·4·2"),
    ],
    points=[(3,), (4,), (7,)],
)
class DoubleFactorial:
    def exact(self, x: Any) -> Any:
        return sympy.factorial2(whole_number(x))


@function(
    "gamma",
    latex=r"\Gamma(x)",
    unicode="Γ(x)",
    summary=r"Euler's gamma function; `\gamma` is the letter",
    examples=[
        Example("gamma(5)", 24, "DLMF 5.4.1"),
        Example("gamma(1/2)", sympy.sqrt(sympy.pi), "DLMF 5.4.6"),
    ],
)
class Gamma:
    def exact(self, x: Any) -> Any:
        return sympy.gamma(x)


@function(
    "re",
    aliases=["Re"],
    summary=r"the real part; `\Re` too",
    examples=[
        Example("re(3 + 4i)", 3, "numpy.real(3 + 4j)"),
        Example("re(i)", 0, "standard"),
    ],
)
class RealPart:
    def exact(self, x: Any) -> Any:
        return sympy.re(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        return xp.real(x)


@function(
    "im",
    aliases=["Im"],
    summary=r"the imaginary part; `\Im` too",
    examples=[
        Example("im(3 + 4i)", 4, "numpy.imag(3 + 4j)"),
        Example("im(2)", 0, "standard"),
    ],
)
class ImaginaryPart:
    def exact(self, x: Any) -> Any:
        return sympy.im(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        return xp.imag(x)


@function(
    "conjugate",
    aliases=["conj"],
    summary="the complex conjugate",
    examples=[
        Example("conjugate(3 + 4i)", 3 - 4 * sympy.I, "numpy.conj(3 + 4j)"),
        Example("conj(2)", 2, "standard"),
    ],
)
class Conjugate:
    def exact(self, x: Any) -> Any:
        return sympy.conjugate(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        return xp.conj(x)


@function(
    "arg",
    summary="the argument of a complex number",
    examples=[
        Example("arg(i)", sympy.pi / 2, "cmath.phase(1j)"),
        Example("arg(-1)", sympy.pi, "standard"),
    ],
)
class Argument:
    def exact(self, x: Any) -> Any:
        return sympy.arg(x)


@function(
    "erf",
    summary="the error function",
    examples=[
        Example("erf(0)", 0, "DLMF 7.2.1"),
        Example("erf(1)", 0.8427007929497149, "scipy.special.erf(1.0)", tolerance=1e-15),
    ],
)
class ErrorFunction:
    def exact(self, x: Any) -> Any:
        return sympy.erf(x)
