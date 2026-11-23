r"""Orthogonal polynomials and spherical harmonics.

The Legendre polynomials ``P_n(x)`` and associated Legendre functions ``P_l^m(x)``, the
physicists' Hermite polynomials ``H_n(x)`` (probabilists write He_n), the Laguerre
polynomials ``L_n(x)`` and ``L_n^{(\alpha)}(x)``, the Chebyshev polynomials ``T_n(x)`` and
``U_n(x)``, and the spherical harmonics ``Y_l^m(\theta, \phi)``, by their book names. Exact
with SymPy, which writes those of a given degree out, and of decimals at the fractions
they denote; numeric with SciPy, which evaluates a polynomial of a whole degree at real
numbers by its recurrence, and with mpmath for any other, which SciPy sums as a series
in double precision and loses. A degree a polynomial is not of, as a Laguerre polynomial
is of no degree 1/2, is refused by both. DLMF is the NIST Digital Library of
Mathematical Functions, chapters 14 and 18.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import sympy

from relativisticpy.errors import EvaluationError
from relativisticpy.integrations import Example, function
from relativisticpy.tensors.numerical import integers, realised, through_fractions, whole_number

x, alpha, theta, phi = sympy.symbols("x alpha theta phi")
_DEGREES = [(0, 0.3), (1, 0.7), (2, -0.4), (5, 0.9)]  # degrees and arguments


def _degree(n: Any, name: str, *, negative: bool = False, fractional: bool = False) -> Any:
    """``n`` as the polynomial called ``name`` takes its degree: a whole number given as a
    decimal, as data gives it, made an integer; a symbol or an expression kept as it is,
    for a value later; and a number refused that is below 0, unless the polynomial has
    degrees below 0 (``negative``), or that is no whole number, unless it is a function of
    any degree (``fractional``)."""
    n = whole_number(n)
    if n.is_number and not _allowed(n, negative=negative, fractional=fractional):
        raise _no_degree(name, n, negative=negative, fractional=fractional)
    return n


def _degrees(
    xp: Any, n: Any, name: str, *, negative: bool = False, fractional: bool = False
) -> Any:
    """Degrees given as numbers: as integers where every one is whole, and as they are
    where the polynomial is a function of any degree (``fractional``) and one is not;
    refused as :func:`_degree` refuses them."""
    whole = integers(xp, n)
    values = whole if whole is not None else xp.asarray(n)
    complex_ = xp.iscomplexobj(values)
    below = not complex_ and bool(xp.any(values < 0))
    if (whole is None and (complex_ or not fractional)) or (below and not negative):
        for each in xp.ravel(values).tolist():  # the first it is not of, to say so
            value = sympy.nsimplify(each)
            if not _allowed(value, negative=negative, fractional=fractional):
                raise _no_degree(name, value, negative=negative, fractional=fractional)
    return values


def _allowed(n: Any, *, negative: bool, fractional: bool) -> bool:
    """Whether a number is a degree of a polynomial that has degrees below 0 or not, and
    real degrees that are no whole numbers or not. No degree is complex: SymPy reflects
    P_n(-x) into (-1)^n P_n(x), which holds of whole numbers alone."""
    whole = n.is_Integer or (fractional and n.is_extended_real)
    return bool(whole and (negative or not n.is_extended_negative))


def _no_degree(name: str, n: Any, *, negative: bool, fractional: bool) -> EvaluationError:
    number = "a real number" if fractional else "a whole number"
    wanted = number if negative else f"{number} from 0"
    return EvaluationError(
        f"{name} takes a degree that is {wanted}, not {n}",
        hint="each polynomial is of the degrees books define it of, as n = 0, 1, 2, ...",
    )


def _recurrent(xp: Any, degrees: Any, x: Any) -> bool:
    """Whether SciPy evaluates a polynomial by its recurrence, which keeps its accuracy at
    any degree: of whole degrees at real numbers. Of a complex number, or of a degree that
    is no whole number, it sums a series in double precision and loses it as the degree
    grows: T_40(0.3 + 0i) would be 356.8, and is 0.929; U_20.5(-0.9) would be -979037, and
    is 2.21."""
    return bool(degrees.dtype.kind == "i") and not xp.iscomplexobj(x)


#: How many binary digits of a sum mpmath may lose to cancellation before it is taken as 0.
_NO_DIGITS = 5000


def _by_series(compute: Callable[..., Any], *arrays: Any) -> Any:
    """A function of mpmath's at each of the numbers of arrays: mpmath sums a series with
    the digits it needs, where SciPy's are lost. A sum that cancels beyond any digits, as
    a polynomial's does at a root of it, is 0."""
    import numpy as np

    each = np.vectorize(
        lambda *values: complex(compute(*values, zeroprec=_NO_DIGITS, maxprec=4 * _NO_DIGITS)),
        otypes=[complex],
    )
    return realised(np, each(*(np.asarray(array) for array in arrays)))


def _order(ell: Any, m: Any, name: str) -> None:
    """Refuse a degree and an order, whole numbers, that the function called ``name`` is
    not of: a degree below 0, or an order larger than it."""
    if ell < 0 or abs(m) > ell:
        raise EvaluationError(
            f"{name} is of a degree l from 0 and an order m from -l to l, not l = {ell}, m = {m}",
            hint="P_l^m(x) and Y_l^m are of l = 0, 1, 2, ... and m = -l, ..., l",
        )


def _orders(xp: Any, degrees: Any, orders: Any, name: str) -> None:
    """Refuse degrees and orders given as whole numbers as :func:`_order` refuses them,
    saying the first pair the function is not of."""
    wrong = (degrees < 0) | (xp.abs(orders) > degrees)
    if bool(xp.any(wrong)):
        degrees, orders, wrong = xp.broadcast_arrays(degrees, orders, wrong)
        first = int(xp.argmax(wrong))
        _order(int(xp.ravel(degrees)[first]), int(xp.ravel(orders)[first]), name)


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
        n = _degree(n, "legendre", negative=True, fractional=True)
        return through_fractions(sympy.legendre, n, x) if n.is_Integer else sympy.legendre(n, x)

    def numeric(self, xp: Any, n: Any, x: Any) -> Any:
        import mpmath
        from scipy.special import eval_legendre

        degrees = _degrees(xp, n, "legendre", negative=True, fractional=True)
        if _recurrent(xp, degrees, x):
            return eval_legendre(degrees, x)
        return _by_series(mpmath.legendre, degrees, x)


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
        ell = _degree(ell, "assoc_legendre", negative=True, fractional=True)
        m = _degree(m, "assoc_legendre", negative=True, fractional=True)
        if ell.is_Integer and m.is_Integer:
            _order(ell, m, "assoc_legendre")
            return through_fractions(sympy.assoc_legendre, ell, m, x)
        return sympy.assoc_legendre(ell, m, x)

    def numeric(self, xp: Any, ell: Any, m: Any, x: Any) -> Any:
        from scipy.special import lpmv

        degrees, orders = integers(xp, ell), integers(xp, m)
        if degrees is not None and orders is not None:
            _orders(xp, degrees, orders, "assoc_legendre")
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
        n = _degree(n, "hermite", fractional=True)
        return through_fractions(sympy.hermite, n, x) if n.is_Integer else sympy.hermite(n, x)

    def numeric(self, xp: Any, n: Any, x: Any) -> Any:
        import mpmath
        from scipy.special import eval_hermite

        degrees = _degrees(xp, n, "hermite", fractional=True)
        if _recurrent(xp, degrees, x):
            return eval_hermite(degrees, x)
        return _by_series(mpmath.hermite, degrees, x)  # the Hermite function, as SymPy has it


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
        n = _degree(n, "laguerre", negative=True)
        return through_fractions(sympy.laguerre, n, x) if n.is_Integer else sympy.laguerre(n, x)

    def numeric(self, xp: Any, n: Any, x: Any) -> Any:
        import mpmath
        from scipy.special import eval_laguerre

        degrees = _degrees(xp, n, "laguerre", negative=True)
        if not _recurrent(xp, degrees, x):
            return _by_series(lambda n, x, **few: mpmath.laguerre(n, 0, x, **few), degrees, x)
        below = degrees < 0
        if not bool(xp.any(below)):
            return eval_laguerre(degrees, x)
        whole = xp.where(below, -degrees - 1, degrees)  # L_{-n}(x) = e^x L_{n-1}(-x)
        return xp.where(below, xp.exp(x) * eval_laguerre(whole, -x), eval_laguerre(whole, x))


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
        n = _degree(n, "assoc_laguerre")
        if n.is_Integer:
            return through_fractions(sympy.assoc_laguerre, n, alpha, x)
        return sympy.assoc_laguerre(n, alpha, x)

    def numeric(self, xp: Any, n: Any, alpha: Any, x: Any) -> Any:
        import mpmath
        from scipy.special import eval_genlaguerre

        degrees = _degrees(xp, n, "assoc_laguerre")
        if _recurrent(xp, degrees, x) and not xp.iscomplexobj(alpha):
            return eval_genlaguerre(degrees, alpha, x)
        return _by_series(mpmath.laguerre, degrees, alpha, x)


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
        n = _degree(n, "chebyshevt", negative=True, fractional=True)
        if n.is_Integer:
            return through_fractions(sympy.chebyshevt, n, x)
        return sympy.chebyshevt(n, x)

    def numeric(self, xp: Any, n: Any, x: Any) -> Any:
        import mpmath
        from scipy.special import eval_chebyt

        degrees = _degrees(xp, n, "chebyshevt", negative=True, fractional=True)
        if _recurrent(xp, degrees, x):
            return eval_chebyt(degrees, x)
        return _by_series(mpmath.chebyt, degrees, x)


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
        n = _degree(n, "chebyshevu", negative=True, fractional=True)
        if n.is_number and n.is_extended_negative and not n.is_Integer:
            # through the hypergeometric function, as SciPy computes it: SymPy reflects
            # U_n into -U_{-n-2}, which is below 0 too, and fails
            half = sympy.Rational(3, 2)
            return sympy.hyperexpand((n + 1) * sympy.hyper([-n, n + 2], [half], (1 - x) / 2))
        if n.is_Integer:
            return through_fractions(sympy.chebyshevu, n, x)
        return sympy.chebyshevu(n, x)

    def numeric(self, xp: Any, n: Any, x: Any) -> Any:
        import mpmath
        from scipy.special import eval_chebyu

        degrees = _degrees(xp, n, "chebyshevu", negative=True, fractional=True)
        if _recurrent(xp, degrees, x):
            return eval_chebyu(degrees, x)
        return _by_series(mpmath.chebyu, degrees, x)


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
        ell, m = _degree(ell, "Ynm", negative=True), _degree(m, "Ynm", negative=True)
        if not (ell.is_Integer and m.is_Integer):
            return sympy.Ynm(ell, m, theta, phi)
        _order(ell, m, "Ynm")
        return through_fractions(
            lambda *given: sympy.Ynm(*given).expand(func=True), ell, m, theta, phi
        )

    def numeric(self, xp: Any, ell: Any, m: Any, theta: Any, phi: Any) -> Any:
        degrees = _degrees(xp, ell, "Ynm", negative=True)
        orders = _degrees(xp, m, "Ynm", negative=True)
        _orders(xp, degrees, orders, "Ynm")
        try:
            from scipy.special import sph_harm_y
        except ImportError:  # SciPy before 1.15 names it sph_harm, and takes the azimuth first
            from scipy.special import sph_harm

            value = sph_harm(orders, degrees, phi, theta)
        else:
            value = sph_harm_y(degrees, orders, theta, phi)
        # exact writes sin(theta)^|m|, of an angle past pi too, where SciPy has |sin(theta)|^|m|
        return value * xp.sign(xp.sin(xp.abs(theta))) ** xp.abs(orders)
