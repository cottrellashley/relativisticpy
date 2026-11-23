r"""Bessel and Airy functions.

The Bessel functions of the first and second kinds and their modified forms, by their
book names, ``J_{\nu}(x)``, ``Y_{\nu}(x)``, ``I_{\nu}(x)`` and ``K_{\nu}(x)``; the spherical
Bessel functions ``j_n(x)`` and ``y_n(x)``; and the Airy functions. Exact with SymPy, which
writes those of half an odd integer as the elementary functions they are, and numeric with
SciPy, for data, in complex numbers past 0, where they are complex. DLMF is the NIST
Digital Library of Mathematical Functions, chapters 9 and 10.
"""

from __future__ import annotations

from typing import Any

import sympy

from relativisticpy.integrations import Example, function
from relativisticpy.tensors.numerical import continued

_ORDERS = [(0, 1.5), (1, 2.5), (0.5, 3.0), (2.5, 0.7)]  # orders and arguments
_DEGREES = [(0, 1.5), (1, 2.5), (2, 3.0), (3, 0.7)]  # whole orders, as SciPy takes them


def _elementary(value: Any, order: Any, spherical: Any) -> Any:
    """A Bessel function of an order of half an odd integer, written as the elementary
    function it is through the spherical one (DLMF 10.47.3 and 10.49.3)."""
    order = sympy.sympify(order)
    if order.is_Rational and not order.is_integer and (2 * order).is_integer:
        return sympy.expand_func(value.rewrite(spherical))
    return value


@function(
    "besselj",
    latex=r"J_{\nu}(x)",
    summary=r"the Bessel function of the first kind, of order `\nu`",
    requires="scipy",
    examples=[
        Example("J_0(1)", 0.7651976865579666, "scipy.special.jv(0, 1.0)"),
        Example(r"J_{1/2}(\pi/2)", 2 / sympy.pi, "DLMF 10.16.1"),
    ],
    points=_ORDERS,
)
class BesselJ:
    def exact(self, nu: Any, x: Any) -> Any:
        return _elementary(sympy.besselj(nu, x), nu, sympy.jn)

    def numeric(self, xp: Any, nu: Any, x: Any) -> Any:
        from scipy.special import jv

        return continued(xp, lambda values: jv(nu, values), x)


@function(
    "bessely",
    latex=r"Y_{\nu}(x)",
    summary=r"the Bessel function of the second kind, of order `\nu`",
    requires="scipy",
    examples=[
        Example("Y_0(1)", 0.088256964215677, "scipy.special.yv(0, 1.0)"),
        Example(r"Y_{1/2}(\pi)", sympy.sqrt(2) / sympy.pi, "DLMF 10.16.1"),
    ],
    points=_ORDERS,
)
class BesselY:
    def exact(self, nu: Any, x: Any) -> Any:
        return _elementary(sympy.bessely(nu, x), nu, sympy.yn)

    def numeric(self, xp: Any, nu: Any, x: Any) -> Any:
        from scipy.special import yv

        return continued(xp, lambda values: yv(nu, values), x)


@function(
    "besseli",
    latex=r"I_{\nu}(x)",
    summary=r"the modified Bessel function of the first kind, of order `\nu`",
    requires="scipy",
    examples=[
        Example("I_0(1)", 1.2660658777520084, "scipy.special.iv(0, 1.0)"),
        Example("I_1(0)", 0, "DLMF 10.30.1"),
    ],
    points=_ORDERS,
)
class BesselI:
    def exact(self, nu: Any, x: Any) -> Any:
        return sympy.besseli(nu, x)

    def numeric(self, xp: Any, nu: Any, x: Any) -> Any:
        from scipy.special import iv

        return continued(xp, lambda values: iv(nu, values), x)


@function(
    "besselk",
    latex=r"K_{\nu}(x)",
    summary=r"the modified Bessel function of the second kind, of order `\nu`",
    requires="scipy",
    examples=[
        Example("K_0(1)", 0.42102443824070834, "scipy.special.kv(0, 1.0)"),
        Example(r"K_{1/2}(1)", sympy.sqrt(sympy.pi / 2) / sympy.E, "DLMF 10.39.2"),
    ],
    points=_ORDERS,
)
class BesselK:
    def exact(self, nu: Any, x: Any) -> Any:
        return sympy.besselk(nu, x)

    def numeric(self, xp: Any, nu: Any, x: Any) -> Any:
        from scipy.special import kv

        return continued(xp, lambda values: kv(nu, values), x)


@function(
    "jn",
    latex="j_n(x)",
    summary="the spherical Bessel function of the first kind, of order n",
    requires="scipy",
    examples=[
        Example(r"j_0(\pi)", 0, "DLMF 10.49.3: j_0(z) = sin z / z"),
        Example(r"j_1(\pi/2)", 4 / sympy.pi**2, "DLMF 10.49.3"),
    ],
    points=_DEGREES,
)
class SphericalBesselJ:
    def exact(self, n: Any, x: Any) -> Any:
        value = sympy.jn(n, x)
        return sympy.expand_func(value) if sympy.sympify(n).is_Integer else value

    def numeric(self, xp: Any, n: Any, x: Any) -> Any:
        from scipy.special import spherical_jn

        return spherical_jn(xp.asarray(n).astype(int), x)


@function(
    "yn",
    latex="y_n(x)",
    summary="the spherical Bessel function of the second kind, of order n",
    requires="scipy",
    examples=[
        Example(r"y_0(\pi)", 1 / sympy.pi, "DLMF 10.49.4: y_0(z) = -cos z / z"),
        Example(r"y_0(\pi/2)", 0, "DLMF 10.49.4"),
    ],
    points=_DEGREES,
)
class SphericalBesselY:
    def exact(self, n: Any, x: Any) -> Any:
        value = sympy.yn(n, x)
        return sympy.expand_func(value) if sympy.sympify(n).is_Integer else value

    def numeric(self, xp: Any, n: Any, x: Any) -> Any:
        from scipy.special import spherical_yn

        return spherical_yn(xp.asarray(n).astype(int), x)


@function(
    "airyai",
    latex=r"\operatorname{Ai}(x)",
    summary="the Airy function Ai",
    requires="scipy",
    examples=[
        Example(r"\operatorname{Ai}(0)",
                1 / (3 ** sympy.Rational(2, 3) * sympy.gamma(sympy.Rational(2, 3))),
                "DLMF 9.2.3"),
        Example(r"\operatorname{Ai}(1)", 0.13529241631288147, "scipy.special.airy(1.0)[0]"),
    ],
)  # fmt: skip
class AiryAi:
    def exact(self, x: Any) -> Any:
        return sympy.airyai(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        from scipy.special import airy

        return airy(x)[0]


@function(
    "airybi",
    latex=r"\operatorname{Bi}(x)",
    summary="the Airy function Bi",
    requires="scipy",
    examples=[
        Example(r"\operatorname{Bi}(0)",
                1 / (3 ** sympy.Rational(1, 6) * sympy.gamma(sympy.Rational(2, 3))),
                "DLMF 9.2.4"),
        Example(r"\operatorname{Bi}(1)", 1.2074235949528715, "scipy.special.airy(1.0)[2]"),
    ],
)  # fmt: skip
class AiryBi:
    def exact(self, x: Any) -> Any:
        return sympy.airybi(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        from scipy.special import airy

        return airy(x)[2]
