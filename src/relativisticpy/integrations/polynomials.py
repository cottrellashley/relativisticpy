r"""Orthogonal polynomials and spherical harmonics.

The Legendre polynomials ``P_n(x)`` and associated Legendre functions ``P_l^m(x)``, the
physicists' Hermite polynomials ``H_n(x)`` (probabilists write He_n), the Laguerre
polynomials ``L_n(x)`` and ``L_n^{(\alpha)}(x)``, the Chebyshev polynomials ``T_n(x)`` and
``U_n(x)``, and the spherical harmonics ``Y_l^m(\theta, \phi)``, by their book names. Exact
with SymPy, which writes those of a given degree out, and numeric with SciPy. DLMF is the
NIST Digital Library of Mathematical Functions, chapters 14 and 18.
"""

from __future__ import annotations

from typing import Any

import sympy

from relativisticpy.integrations import Example, function

x, alpha, theta, phi = sympy.symbols("x alpha theta phi")
_DEGREES = [(0, 0.3), (1, 0.7), (2, -0.4), (5, 0.9)]  # degrees and arguments


@function(
    "legendre",
    latex="P_n(x)",
    summary="the Legendre polynomial of degree n",
    requires="scipy",
    examples=[
        Example("P_3(x)", (5 * x**3 - 3 * x) / 2, "DLMF 18.3, Table 18.3.1"),
        Example("P_2(1/2)", sympy.Rational(-1, 8), "DLMF 18.3: P_2(x) = (3x^2 - 1)/2"),
    ],
    points=_DEGREES,
)
class Legendre:
    def exact(self, n: Any, x: Any) -> Any:
        return sympy.legendre(n, x)

    def numeric(self, xp: Any, n: Any, x: Any) -> Any:
        from scipy.special import eval_legendre

        return eval_legendre(n, x)


@function(
    "assoc_legendre",
    latex=r"P_{\ell}^m(x)",
    summary=r"the associated Legendre function of degree `\ell` and order m, with the phase "
    "(-1)^m",
    requires="scipy",
    examples=[
        Example("P_1^1(x)", -sympy.sqrt(1 - x**2), "DLMF 14.5.11, 14.7.10"),
        Example("P_2^1(1/2)", -3 * sympy.sqrt(3) / 4, "scipy.special.lpmv(1, 2, 0.5)"),
    ],
    points=[(3, 1, 0.3), (2, 1, 0.5), (3, 2, -0.4), (2, 0, 0.7)],  # |m| <= l, any two
)
class AssociatedLegendre:
    def exact(self, ell: Any, m: Any, x: Any) -> Any:
        return sympy.assoc_legendre(ell, m, x)

    def numeric(self, xp: Any, ell: Any, m: Any, x: Any) -> Any:
        from scipy.special import lpmv

        return lpmv(m, ell, x)


@function(
    "hermite",
    latex="H_n(x)",
    summary="the Hermite polynomial of degree n, the physicists' (probabilists write He_n)",
    requires="scipy",
    examples=[
        Example("H_3(x)", 8 * x**3 - 12 * x, "DLMF 18.3, Table 18.3.1"),
        Example("H_2(1)", 2, "DLMF 18.3: H_2(x) = 4x^2 - 2"),
    ],
    points=_DEGREES,
)
class Hermite:
    def exact(self, n: Any, x: Any) -> Any:
        return sympy.hermite(n, x)

    def numeric(self, xp: Any, n: Any, x: Any) -> Any:
        from scipy.special import eval_hermite

        return eval_hermite(xp.asarray(n).astype(int), x)


@function(
    "laguerre",
    latex="L_n(x)",
    summary="the Laguerre polynomial of degree n",
    requires="scipy",
    examples=[
        Example("L_2(x)", x**2 / 2 - 2 * x + 1, "DLMF 18.3, Table 18.3.1"),
        Example("L_1(1)", 0, "DLMF 18.3: L_1(x) = 1 - x"),
    ],
    points=_DEGREES,
)
class Laguerre:
    def exact(self, n: Any, x: Any) -> Any:
        return sympy.laguerre(n, x)

    def numeric(self, xp: Any, n: Any, x: Any) -> Any:
        from scipy.special import eval_laguerre

        return eval_laguerre(n, x)


@function(
    "assoc_laguerre",
    latex=r"L_n^{(\alpha)}(x)",
    summary=r"the generalised Laguerre polynomial of degree n and parameter `\alpha`",
    requires="scipy",
    examples=[
        Example(r"L_1^{(\alpha)}(x)", -x + alpha + 1, "DLMF 18.5.12"),
        Example(r"L_2^{(1)}(0)", 3, "DLMF 18.6.1: L_n^{(a)}(0) = binomial(n + a, n)"),
    ],
    points=[(0, 0.5, 0.3), (1, 1.5, 0.7), (2, 2.0, 1.4), (3, 0.5, 2.5)],
)
class GeneralisedLaguerre:
    def exact(self, n: Any, alpha: Any, x: Any) -> Any:
        return sympy.assoc_laguerre(n, alpha, x)

    def numeric(self, xp: Any, n: Any, alpha: Any, x: Any) -> Any:
        from scipy.special import eval_genlaguerre

        return eval_genlaguerre(n, alpha, x)


@function(
    "chebyshevt",
    latex="T_n(x)",
    summary="the Chebyshev polynomial of the first kind, of degree n",
    requires="scipy",
    examples=[
        Example("T_3(x)", 4 * x**3 - 3 * x, "DLMF 18.3, Table 18.3.1"),
        Example("T_4(1/2)", sympy.Rational(-1, 2), "DLMF 18.5.1: T_n(cos t) = cos(n t)"),
    ],
    points=_DEGREES,
)
class ChebyshevT:
    def exact(self, n: Any, x: Any) -> Any:
        return sympy.chebyshevt(n, x)

    def numeric(self, xp: Any, n: Any, x: Any) -> Any:
        from scipy.special import eval_chebyt

        return eval_chebyt(n, x)


@function(
    "chebyshevu",
    latex="U_n(x)",
    summary="the Chebyshev polynomial of the second kind, of degree n",
    requires="scipy",
    examples=[
        Example("U_2(x)", 4 * x**2 - 1, "DLMF 18.3, Table 18.3.1"),
        Example("U_3(1/2)", -1, "DLMF 18.5.2: U_n(cos t) = sin((n + 1)t)/sin t"),
    ],
    points=_DEGREES,
)
class ChebyshevU:
    def exact(self, n: Any, x: Any) -> Any:
        return sympy.chebyshevu(n, x)

    def numeric(self, xp: Any, n: Any, x: Any) -> Any:
        from scipy.special import eval_chebyu

        return eval_chebyu(n, x)


@function(
    "Ynm",
    latex=r"Y_{\ell}^m(\theta, \phi)",
    summary=r"the spherical harmonic of degree `\ell` and order m, at polar angle `\theta` and "
    r"azimuth `\phi`",
    requires="scipy",
    examples=[
        Example(r"Y_1^0(\theta, \phi)", sympy.sqrt(3 / (4 * sympy.pi)) * sympy.cos(theta),
                "Griffiths (2005), Table 4.3"),
        Example(r"Y_0^0(\theta, \phi)", 1 / (2 * sympy.sqrt(sympy.pi)),
                "Griffiths (2005), Table 4.3"),
    ],
    points=[(2, -1, 1.2, 2.5), (2, 0, 0.5, 1.0), (3, 1, 0.3, 0.7), (2, 2, 0.3, 0.7)],
)  # fmt: skip
class SphericalHarmonic:
    def exact(self, ell: Any, m: Any, theta: Any, phi: Any) -> Any:
        value = sympy.Ynm(ell, m, theta, phi)
        known = sympy.sympify(ell).is_Integer and sympy.sympify(m).is_Integer
        return value.expand(func=True) if known else value

    def numeric(self, xp: Any, ell: Any, m: Any, theta: Any, phi: Any) -> Any:
        from scipy.special import sph_harm_y

        return sph_harm_y(xp.asarray(ell).astype(int), xp.asarray(m).astype(int), theta, phi)
