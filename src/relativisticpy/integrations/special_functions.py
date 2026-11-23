r"""Special functions.

Euler's Beta function and Riemann's zeta function, the principal logarithm, the
complementary and imaginary error functions, the exponential, sine and cosine integrals,
Lambert's W, the complete elliptic integrals, the generalised hypergeometric function
``{}_pF_q(a; b; z)`` and the Pochhammer symbol ``(a)_n``, the standard normal distribution
function, and the unit step and the Dirac delta. Exact with SymPy, which gives the closed
form where there is one, and numeric with SciPy, for data. DLMF is the NIST Digital
Library of Mathematical Functions.
"""

from __future__ import annotations

from typing import Any

import sympy

from relativisticpy.errors import EvaluationError
from relativisticpy.integrations import Example, function
from relativisticpy.tensors.numerical import continued, integers, realised, whole_number

x, z = sympy.symbols("x z")


@function(
    "beta",
    latex=r"\mathrm{B}(a, b)",
    summary="Euler's Beta function",
    requires="scipy",
    examples=[
        Example(r"\mathrm{B}\left(\tfrac{1}{2}, \tfrac{1}{2}\right)", sympy.pi,
                "DLMF 5.12.1, with DLMF 5.4.6"),
        Example("beta(2, 3)", sympy.Rational(1, 12), "DLMF 5.12.1"),
    ],
    points=[(0.5, 0.5), (2.0, 3.0), (1.5, 2.5)],
)  # fmt: skip
class Beta:
    def exact(self, a: Any, b: Any) -> Any:
        value = sympy.beta(a, b)
        numbers = sympy.sympify(a).is_number and sympy.sympify(b).is_number
        return sympy.expand_func(value) if numbers else value

    def numeric(self, xp: Any, a: Any, b: Any) -> Any:
        from scipy.special import beta

        return beta(a, b)


@function(
    "zeta",
    latex=r"\zeta(s)",
    unicode="ζ(s)",
    summary="Riemann's zeta function",
    requires="scipy",
    examples=[
        Example(r"\zeta(2)", sympy.pi**2 / 6, "DLMF 25.6.1"),
        Example(r"\zeta(4)", sympy.pi**4 / 90, "DLMF 25.6.1"),
    ],
    points=[(2.0,), (3.5,), (1.5,), (0.5,)],
)
class Zeta:
    def exact(self, s: Any) -> Any:
        return sympy.zeta(whole_number(s))

    def numeric(self, xp: Any, s: Any) -> Any:
        from scipy.special import zeta

        return zeta(s)


@function(
    "Log",
    summary=r"the principal branch of the logarithm, `\operatorname{Log}`",
    examples=[
        Example(r"\operatorname{Log}(-1 - i)",
                sympy.log(sympy.sqrt(2)) - 3 * sympy.pi * sympy.I / 4,
                "cmath.log(-1 - 1j)"),
        Example(r"\operatorname{Log}(i)", sympy.pi * sympy.I / 2, "standard"),
    ],
)  # fmt: skip
class PrincipalLogarithm:
    def exact(self, z: Any) -> Any:
        return sympy.log(z)

    def numeric(self, xp: Any, z: Any) -> Any:
        return continued(xp, xp.log, z)


@function(
    "erfc",
    summary="the complementary error function, 1 - erf",
    requires="scipy",
    examples=[
        Example(r"\operatorname{erfc}(0)", 1, "DLMF 7.2.2"),
        Example("erfc(1)", 0.15729920705028516, "scipy.special.erfc(1.0)"),
    ],
)
class ComplementaryErrorFunction:
    def exact(self, x: Any) -> Any:
        return sympy.erfc(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        from scipy.special import erfc

        return erfc(x)


@function(
    "erfi",
    summary="the imaginary error function, -i erf(ix)",
    requires="scipy",
    examples=[
        Example(r"\operatorname{erfi}(0)", 0, "DLMF 7.2.4"),
        Example("erfi(1)", 1.6504257587975428, "scipy.special.erfi(1.0)"),
    ],
)
class ImaginaryErrorFunction:
    def exact(self, x: Any) -> Any:
        return sympy.erfi(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        from scipy.special import erfi

        return erfi(x)


@function(
    "Ei",
    summary="the exponential integral",
    requires="scipy",
    examples=[
        Example(r"\operatorname{Ei}(1)", 1.895117816355937, "scipy.special.expi(1.0)"),
        Example("Ei(2)", 4.954234356001891, "scipy.special.expi(2.0)"),
    ],
)
class ExponentialIntegral:
    def exact(self, x: Any) -> Any:
        return sympy.Ei(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        from scipy.special import expi

        return expi(x)


@function(
    "Si",
    summary="the sine integral",
    requires="scipy",
    examples=[
        Example(r"\operatorname{Si}(0)", 0, "DLMF 6.2.9"),
        Example("Si(1)", 0.9460830703671831, "scipy.special.sici(1.0)[0]"),
    ],
)
class SineIntegral:
    def exact(self, x: Any) -> Any:
        return sympy.Si(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        from scipy.special import sici

        return sici(x)[0]


@function(
    "Ci",
    summary="the cosine integral",
    requires="scipy",
    examples=[
        Example(r"\operatorname{Ci}(1)", 0.33740392290096816, "scipy.special.sici(1.0)[1]"),
        Example("Ci(2)", 0.422980828774865, "scipy.special.sici(2.0)[1]"),
    ],
)
class CosineIntegral:
    def exact(self, x: Any) -> Any:
        return sympy.Ci(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        from scipy.special import sici

        return continued(xp, lambda values: sici(values)[1], x, below=0)


@function(
    "LambertW",
    latex="W_k(x)",
    summary="Lambert's W, the inverse of w e^w, on branch k: 0, the principal, or -1",
    requires="scipy",
    examples=[
        Example("W_0(e)", 1, "DLMF 4.13.1: W e^W = x"),
        Example("W_{-1}(-1/e)", -1, "DLMF 4.13.1"),
    ],
    points=[(0.5,), (1.0,), (2.7,)],
)
class LambertW:
    def exact(self, x: Any, k: Any = 0) -> Any:
        k = whole_number(k)
        if k.is_number and not k.is_Integer:
            raise _no_branch(k)
        return sympy.LambertW(x, k)

    def numeric(self, xp: Any, x: Any, k: Any = 0) -> Any:
        from scipy.special import lambertw

        branches = integers(xp, k)
        if branches is None:
            raise _no_branch(k)
        return realised(xp, lambertw(x, branches))


def _no_branch(k: Any) -> EvaluationError:
    return EvaluationError(
        f"Lambert's W has a branch k for each whole number, and none for {k}",
        hint="W_0 is the principal branch, and W_{-1} the other that is real, on [-1/e, 0)",
    )


@function(
    "elliptic_k",
    latex=r"\operatorname{K}(m)",
    summary="the complete elliptic integral of the first kind, of parameter m = k^2",
    requires="scipy",
    examples=[
        Example(r"\operatorname{K}(0)", sympy.pi / 2, "DLMF 19.6.1"),
        Example("elliptic_k(1/2)", 1.8540746773013719, "scipy.special.ellipk(0.5)"),
    ],
    points=[(0.3,), (0.7,), (0.9,)],  # below 1, where they are real
)
class EllipticK:
    def exact(self, m: Any) -> Any:
        return sympy.elliptic_k(m)

    def numeric(self, xp: Any, m: Any) -> Any:
        from scipy.special import ellipk

        return ellipk(m)


@function(
    "elliptic_e",
    latex=r"\operatorname{E}(m)",
    summary="the complete elliptic integral of the second kind, of parameter m = k^2",
    requires="scipy",
    examples=[
        Example(r"\operatorname{E}(0)", sympy.pi / 2, "DLMF 19.6.1"),
        Example("elliptic_e(1)", 1, "DLMF 19.6.1"),
    ],
    points=[(0.3,), (0.7,), (0.9,)],  # below 1, where they are real
)
class EllipticE:
    def exact(self, m: Any) -> Any:
        return sympy.elliptic_e(m)

    def numeric(self, xp: Any, m: Any) -> Any:
        from scipy.special import ellipe

        return ellipe(m)


def _parameters(values: Any) -> list[Any]:
    """The parameters of a hypergeometric function before or after the first semicolon,
    as a list: one of them is a list of one."""
    listed = isinstance(values, list | tuple | sympy.Tuple) or getattr(values, "ndim", 0) > 0
    return list(values) if listed else [values]


def _counted(a: Any, b: Any, p: Any, q: Any) -> tuple[list[Any], list[Any]]:
    """The parameters of {}_pF_q before and after its first semicolon, as many as p and q
    count where they are written; an error otherwise."""
    a, b = _parameters(a), _parameters(b)
    for count, parameters, where in ((p, a, "before"), (q, b, "after")):
        if count is not None and count != len(parameters):
            raise EvaluationError(
                f"{{}}_{p}F_{q} takes {count} parameters {where} its first semicolon, "
                f"not {len(parameters)}",
                hint="the subscripts count the parameters of each list, as in {}_2F_1(a, b; c; z)",
            )
    return a, b


def _number(value: Any) -> float | complex:
    """A parameter as mpmath takes it: a real number, or a complex one where it is."""
    found = complex(value)
    return found.real if found.imag == 0 else found


@function(
    "hyper",
    latex="{}_pF_q(a; b; z)",
    summary="the generalised hypergeometric function; `hyper([a_1, ...], [b_1, ...], z)`",
    examples=[
        Example("{}_2F_1(1, 1; 2; z)", -sympy.log(1 - z) / z,
                "Abramowitz and Stegun (1964), 15.1.3"),
        Example(r"{}_2F_1\left(\tfrac{1}{2}, \tfrac{1}{2}; 2; 1\right)", 4 / sympy.pi,
                "Abramowitz and Stegun (1964), 15.1.20"),
    ],
    points=[([1, 1], [2], 0.3), ([0.5, 0.5], [1.5], 0.5), ([1], [2], 0.7)],
)  # fmt: skip
class Hypergeometric:
    def exact(self, a: Any, b: Any, z: Any, *, p: Any = None, q: Any = None) -> Any:
        a, b = _counted(a, b, p, q)
        return sympy.hyperexpand(sympy.hyper(a, b, z))

    def numeric(self, xp: Any, a: Any, b: Any, z: Any, *, p: Any = None, q: Any = None) -> Any:
        import mpmath
        import numpy as np

        a, b = ([_number(v) for v in values] for values in _counted(a, b, p, q))
        each = np.vectorize(lambda value: complex(mpmath.hyper(a, b, value)), otypes=[complex])
        return realised(np, each(np.asarray(z)))

    def arguments(self, call: Any) -> dict[str, Any]:
        """The arguments that print a call of the hypergeometric function in its form."""
        return {
            "a": list(call.ap),
            "b": list(call.bq),
            "z": call.argument,
            "p": len(call.ap),
            "q": len(call.bq),
        }


@function(
    "rf",
    aliases=["RisingFactorial"],
    latex="(a)_n",
    summary="the Pochhammer symbol, the rising factorial a(a + 1)...(a + n - 1)",
    requires="scipy",
    examples=[
        Example(r"\left(\tfrac{1}{2}\right)_3", sympy.Rational(15, 8), "DLMF 5.2.4"),
        Example("rf(3, 2)", 12, "DLMF 5.2.4"),
    ],
    points=[(0.5, 3.0), (3.0, 2.0), (1.5, 0.5)],
)
class Pochhammer:
    def exact(self, a: Any, n: Any) -> Any:
        return sympy.rf(a, n)

    def numeric(self, xp: Any, a: Any, n: Any) -> Any:
        from scipy.special import poch

        return poch(a, n)


@function(
    "normcdf",
    latex=r"\Phi(x)",
    unicode="Φ(x)",
    summary="the standard normal distribution function, (1 + erf(x/√2))/2",
    requires="scipy",
    examples=[
        Example(r"\Phi(0)", sympy.Rational(1, 2), "standard"),
        Example(r"\Phi(1.1)", 0.8643339390536173, "scipy.stats.norm.cdf(1.1)"),
    ],
)
class NormalDistributionFunction:
    def exact(self, x: Any) -> Any:
        return (1 + sympy.erf(x / sympy.sqrt(2))) / 2

    def numeric(self, xp: Any, x: Any) -> Any:
        from scipy.special import ndtr

        return ndtr(x)


@function(
    "Heaviside",
    latex="u_c(t)",
    summary="the unit step, 0 before c and 1 after it, and 1/2 at c",
    examples=[
        Example("Heaviside(2)", 1, "standard"),
        Example(r"u_{\pi/4}(\pi/8)", 0, "Boyce and DiPrima (2012), section 6.3"),
    ],
    points=[(-0.5,), (0.7,), (1.6,)],
)
class UnitStep:
    def exact(self, t: Any, c: Any = 0) -> Any:
        return sympy.Heaviside(t - c)

    def numeric(self, xp: Any, t: Any, c: Any = 0) -> Any:
        return xp.heaviside(t - c, 0.5)


@function(
    "DiracDelta",
    latex=r"\operatorname{DiracDelta}(x)",
    summary="the Dirac delta, a distribution; `\\delta(x)` is not read as it, as `\\delta` is "
    "a variation in physics",
    examples=[
        Example("DiracDelta(1)", 0, "standard"),
        Example(
            "integrate(DiracDelta(x), (x, -1, 1))",
            1,
            "sympy.integrate(sympy.DiracDelta(x), (x, -1, 1))",
        ),
    ],
    points=[(1.0,), (-2.0,), (0.5,)],
)
class DiracDelta:
    def exact(self, x: Any) -> Any:
        return sympy.DiracDelta(x)
