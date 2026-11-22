r"""Integral transforms.

The Laplace, Fourier and z-transforms and their inverses, by their book names,
``\mathcal{L}\{f(t)\}(s)``, ``\mathcal{F}\{f\}(\omega)`` and ``\mathcal{Z}\{x\}(z)``, exact
with SymPy; the z-transform, for which SymPy has no routine, by summation, and its inverse
by partial fractions. Their variables are named by convention, ℒ taking t to s, ℱ x to s or
t to ω, and 𝒵 n to z, and the Fourier transform is computed in the convention a program
chooses (``design/notation.md``, 21).
"""

from __future__ import annotations

from typing import Any

import sympy

from relativisticpy.errors import EvaluationError
from relativisticpy.integrations import Example, sequences, transform
from relativisticpy.integrations.sequences import impulse, step

t, s, x, p, a, n, z = sympy.symbols("t s x p a n z")
omega = sympy.Symbol("omega")


def _variables(f: Any, given: Any, taken: Any, letters: tuple[str, str]) -> tuple[Any, Any]:
    """The variables of a transform called without them, as ``LaplaceTransform(t^2)``: the
    function's, the conventional one where it is written with it, or else its one
    variable; and the transform's, the conventional one."""
    if given is None:
        written = sorted(
            (v for v in sympy.sympify(f).free_symbols if isinstance(v, sympy.Symbol)), key=str
        )
        named = [v for v in written if v.name == letters[0]]
        if named:
            given = named[0]
        elif len(written) == 1:
            given = written[0]
        else:
            given = sympy.Symbol(letters[0])
    return given, taken if taken is not None else sympy.Symbol(letters[1])


@transform(
    "LaplaceTransform",
    aliases=["laplace_transform"],
    latex=[r"\mathcal{L}\{f\}(s)", r"\mathcal{L}\{f\}"],
    summary=r"the Laplace transform; of f(t), `\int_0^\infty e^{-st} f(t)\, dt`, a function of s",
    variables=[("t", "s")],
    inverse="InverseLaplaceTransform",
    examples=[
        Example(r"\mathcal{L}\{t^2\}", 2 / s**3, "Boyce and DiPrima (2012), table 6.2.1"),
        Example(r"\mathcal{L}\{e^{a t}\}(p)", 1 / (p - a), "Boyce and DiPrima (2012), table 6.2.1"),
        Example("LaplaceTransform(sin(t))", 1 / (s**2 + 1),
                "Boyce and DiPrima (2012), table 6.2.1"),
    ],
    points=[(t**2,), (sympy.exp(-2 * t),), (sympy.sin(3 * t),), (t * sympy.exp(-t),)],
)  # fmt: skip
class Laplace:
    samples = (sympy.Rational(13, 10), sympy.Rational(21, 10), sympy.Rational(37, 10))

    def exact(self, f: Any, t: Any = None, s: Any = None) -> Any:
        t, s = _variables(f, t, s, ("t", "s"))
        point = sympy.Dummy("s")
        return sympy.laplace_transform(f, t, point, noconds=True).subs(point, s)

    def integral(self, f: Any, t: Any, s: Any) -> Any:
        return sympy.Integral(sympy.exp(-s * t) * f, (t, 0, sympy.oo))


@transform(
    "InverseLaplaceTransform",
    aliases=["inverse_laplace_transform"],
    latex=[r"\mathcal{L}^{-1}\{F\}(t)", r"\mathcal{L}^{-1}\{F\}"],
    summary="the inverse Laplace transform; of F(s), the function of t it is the transform of, "
    "for t > 0",
    variables=[("s", "t")],
    inverse="LaplaceTransform",
    examples=[
        Example(r"\mathcal{L}^{-1}\left\{\frac{1}{s^2}\right\}", t,
                "Boyce and DiPrima (2012), table 6.2.1"),
        Example(r"\mathcal{L}^{-1}\left\{\frac{e^{-\pi s/4}}{s^2 + 1}\right\}(t)",
                sympy.Heaviside(t - sympy.pi / 4) * sympy.sin(t - sympy.pi / 4),
                "Boyce and DiPrima (2012), section 6.3"),
    ],
    points=[(1 / s**2,), (1 / (s + 2),), (3 / (s**2 + 9),)],
)  # fmt: skip
class InverseLaplace:
    samples = (sympy.Rational(7, 10), sympy.Rational(13, 10), sympy.Rational(29, 10))

    def exact(self, F: Any, s: Any = None, t: Any = None) -> Any:
        s, t = _variables(F, s, t, ("s", "t"))
        point = sympy.Dummy("t", positive=True)  # a step at t = 0 is 1 where it is defined
        return sympy.inverse_laplace_transform(F, s, point).subs(point, t)


def _fourier(f: Any, x: Any, k: Any, factor: Any, exponent: Any) -> Any:
    r""":math:`a \int f(x) e^{i b x k}\, dx`, with SymPy's transform of ordinary frequency at
    ν = -bk/2π; for every real k, where SymPy gives it for k > 0 alone, the rest from f's
    mirror image."""
    nu = sympy.Dummy("nu", real=True)
    found = _ordinary(sympy.sympify(f), x, nu)
    return sympy.sympify(factor) * found.subs(nu, -exponent * k / (2 * sympy.pi))


def _ordinary(f: Any, x: Any, nu: Any) -> Any:
    r""":math:`\int f(x) e^{-2\pi i x \nu}\, dx` for every real ν."""
    found, condition = _with_condition(f, x, nu)
    if condition in (True, sympy.true) or not condition.has(nu):
        return found
    if condition not in (nu > 0, nu >= 0):
        return sympy.FourierTransform(f, x, nu)
    mirrored, other = _with_condition(f.subs(x, -x), x, nu)
    if other not in (nu > 0, nu >= 0, True, sympy.true):
        return sympy.FourierTransform(f, x, nu)
    if sympy.simplify(found - mirrored) == 0:  # an even function's transform
        return found.subs(nu, sympy.Abs(nu))
    at_zero = sympy.limit(found, nu, 0, "+")
    return sympy.Piecewise((found, nu > 0), (mirrored.subs(nu, -nu), nu < 0), (at_zero, True))


def _with_condition(f: Any, x: Any, nu: Any) -> tuple[Any, Any]:
    """SymPy's transform of ordinary frequency and where it holds; by integration over the
    real line where SymPy's routine gives none."""
    result = sympy.fourier_transform(f, x, nu, noconds=False)
    if isinstance(result, tuple):
        return result[0], result[1]
    if not isinstance(result, sympy.FourierTransform):
        return result, sympy.true
    real = sympy.Dummy("x", real=True)
    kernel = sympy.exp(-2 * sympy.pi * sympy.I * real * nu)
    integrated = sympy.integrate(f.subs(x, real) * kernel, (real, -sympy.oo, sympy.oo))
    if integrated.has(sympy.Integral):
        return result, sympy.true
    return integrated, sympy.true


@transform(
    "FourierTransform",
    aliases=["fourier_transform"],
    latex=[r"\mathcal{F}\{f\}(s)", r"\mathcal{F}\{f\}"],
    summary=r"the Fourier transform; of f(x), `\int f(x) e^{-2\pi i x s}\, dx`, Bracewell's, of "
    r"f(t), `\int f(t) e^{-i\omega t}\, dt`, or as a program defines it by its integral",
    variables=[("x", "s"), ("t", "omega")],
    inverse="InverseFourierTransform",
    examples=[
        Example(r"\mathcal{F}\{e^{-\pi x^2}\}", sympy.exp(-sympy.pi * s**2),
                "Bracewell (2000), chapter 2: the Gaussian is its own transform"),
        Example(r"\mathcal{F}\{e^{-t^2}\}(\omega)",
                sympy.sqrt(sympy.pi) * sympy.exp(-(omega**2) / 4),
                "standard: the integral of e^{-t^2} e^{-i omega t}"),
        Example(r"\mathcal{F}\{\frac{1}{1 + x^2}\}(s)",
                sympy.pi * sympy.exp(-2 * sympy.pi * sympy.Abs(s)),
                "Bracewell (2000), chapter 2"),
    ],
    points=[(sympy.exp(-sympy.pi * x**2),), (x * sympy.exp(-(x**2)),),
            (sympy.exp(-sympy.Abs(x)),)],
)  # fmt: skip
class Fourier:
    fourier = "forward"
    samples = (sympy.Rational(3, 10), sympy.Rational(-7, 10), sympy.Rational(11, 10))

    def exact(
        self, f: Any, x: Any = None, s: Any = None, *, factor: Any = 1, exponent: Any = None
    ) -> Any:
        x, s = _variables(f, x, s, ("x", "s"))
        b = -2 * sympy.pi if exponent is None else exponent
        return _fourier(f, x, s, factor, b)

    def integral(self, f: Any, x: Any, s: Any, *, factor: Any = 1, exponent: Any = None) -> Any:
        b = -2 * sympy.pi if exponent is None else exponent
        kernel = sympy.exp(sympy.I * b * x * s)
        return factor * sympy.Integral(f * kernel, (x, -sympy.oo, sympy.oo))


@transform(
    "InverseFourierTransform",
    aliases=["inverse_fourier_transform"],
    latex=[r"\mathcal{F}^{-1}\{F\}(x)", r"\mathcal{F}^{-1}\{F\}"],
    summary="the inverse Fourier transform; of s to x or of ω to t, in the convention of the "
    "Fourier transform",
    variables=[("s", "x"), ("omega", "t")],
    inverse="FourierTransform",
    examples=[
        Example(r"\mathcal{F}^{-1}\{e^{-\pi s^2}\}", sympy.exp(-sympy.pi * x**2),
                "Bracewell (2000), chapter 2"),
        Example(r"\mathcal{F}^{-1}\{\sqrt{\pi} e^{-\omega^2/4}\}(t)", sympy.exp(-(t**2)),
                "standard: the inverse of the integral of e^{-t^2} e^{-i omega t}"),
    ],
    points=[(sympy.exp(-sympy.pi * s**2),), (s * sympy.exp(-(s**2)),)],
)  # fmt: skip
class InverseFourier:
    fourier = "inverse"
    samples = (sympy.Rational(3, 10), sympy.Rational(-7, 10), sympy.Rational(11, 10))

    def exact(
        self, F: Any, s: Any = None, x: Any = None, *, factor: Any = 1, exponent: Any = None
    ) -> Any:
        s, x = _variables(F, s, x, ("s", "x"))
        b = 2 * sympy.pi if exponent is None else exponent
        return _fourier(F, s, x, factor, b)

    def integral(self, F: Any, s: Any, x: Any, *, factor: Any = 1, exponent: Any = None) -> Any:
        b = 2 * sympy.pi if exponent is None else exponent
        kernel = sympy.exp(sympy.I * b * s * x)
        return factor * sympy.Integral(F * kernel, (s, -sympy.oo, sympy.oo))


def _z(sequence: Any, n: Any, z: Any) -> Any:
    r""":math:`\sum_{n=0}^{\infty} x_n z^{-n}`, a term at a time: a geometric series
    :math:`c r^n` summed to c/(1 - r), where it converges, and any other by SymPy, which
    gives each where it converges; unevaluated where it does not find one. A sequence
    written with impulses and steps, as Oppenheim and Schafer write one, is summed over
    every n, where they are not 0 (:func:`_two_sided`)."""
    if sequences.of_sequences(sequence, n):
        return _two_sided(sequence, n, z)
    step = sympy.Dummy("n", integer=True, nonnegative=True)
    written = sympy.sympify(sequence).subs(n, step)
    trigonometric = written.has(sympy.sin, sympy.cos)
    exponential = written.rewrite([sympy.sin, sympy.cos, sympy.sinh, sympy.cosh], sympy.exp)
    terms = sympy.Add.make_args(sympy.expand(exponential * z ** (-step)))
    total = sympy.Integer(0)
    for term in terms:
        summed = _geometric(term, step)
        if summed is None:
            combined = sympy.powsimp(term, force=True)  # n 3^{-n} z^{-n} is n (3z)^{-n}
            summed = _converged(sympy.summation(combined, (step, 0, sympy.oo)))
        if summed is None:
            return sympy.Sum(sympy.sympify(sequence) * z ** (-n), (n, 0, sympy.oo))
        total += summed
    if trigonometric:
        return sympy.simplify(total.rewrite(sympy.cos))
    return sympy.simplify(total)


def _two_sided(sequence: Any, n: Any, z: Any) -> Any:
    r""":math:`\sum_{n=-\infty}^{\infty} x[n] z^{-n}` of a sequence written with impulses and
    steps: each impulse a power of z, and each other part summed over the range its steps
    leave it, as a series from its first n, or up to its last, which :func:`_z` sums from 0,
    shifted there."""
    index = sympy.Dummy("n", integer=True)
    written = sympy.sympify(sequence).xreplace({n: index})
    found = sequences.pieces(written, index, -sympy.oo, sympy.oo)
    if found is None:
        return sympy.Sum(sympy.sympify(sequence) * z ** (-n), (n, -sympy.oo, sympy.oo))
    m = sympy.Dummy("m", integer=True, nonnegative=True)
    total = sympy.Integer(0)
    for term, first, last in found:
        if first == last:  # an impulse's
            total += term * z ** (-first)
        elif first.is_infinite and last.is_infinite:
            later = _z(term.xreplace({index: m}), m, z)
            earlier = z * _z(term.xreplace({index: -1 - m}), m, 1 / z)
            total += later + earlier
        elif last.is_infinite:
            total += z ** (-first) * _z(term.xreplace({index: first + m}), m, z)
        elif first.is_infinite:
            total += z ** (-last) * _z(term.xreplace({index: last - m}), m, 1 / z)
        else:
            summed = sympy.summation(term * z ** (-index), (index, first, last))
            total += sequences.emptied(summed, first, last)
    return sympy.simplify(total)


def _geometric(term: Any, step: Any) -> Any:
    """The sum over n ≥ 0 of a term c r^n, c/(1 - r); ``None`` for any other term."""
    first = term.subs(step, 0)
    if first == 0:
        return None
    ratio = sympy.simplify(term.subs(step, 1) / first)
    if ratio.has(step) or sympy.simplify(term - first * ratio**step) != 0:
        return None
    return first / (1 - ratio)


def _converged(summed: Any) -> Any:
    """A sum SymPy found, without the sum it gives where it does not converge."""
    if isinstance(summed, sympy.Piecewise):
        for value, _ in summed.args:
            if not value.has(sympy.Sum):
                return value
        return None
    return None if summed.has(sympy.Sum) else summed


@transform(
    "ZTransform",
    aliases=["z_transform"],
    latex=[r"\mathcal{Z}\{x\}(z)", r"\mathcal{Z}\{x\}"],
    summary=r"the z-transform; of a sequence from n = 0, `\sum_{n=0}^{\infty} x_n z^{-n}`, "
    "where it converges, and of one written with impulses and steps over every n",
    variables=[("n", "z")],
    inverse="InverseZTransform",
    examples=[
        Example(r"\mathcal{Z}\{(1/2)^n\}(z)", z / (z - sympy.Rational(1, 2)),
                "Oppenheim and Schafer (2010), table 3.1"),
        Example(r"\mathcal{Z}\{n\}", z / (z - 1) ** 2, "Oppenheim and Schafer (2010), table 3.1"),
    ],
    points=[(sympy.Rational(1, 2) ** n,), (n,), (sympy.Integer(1),),
            (n * sympy.Rational(1, 3) ** n,)],
)  # fmt: skip
class ZTransform:
    samples = (sympy.Integer(2), sympy.Integer(3), sympy.Rational(5, 2))

    def exact(self, x: Any, n: Any = None, z: Any = None) -> Any:
        n, z = _variables(x, n, z, ("n", "z"))
        return _z(x, n, z)

    def integral(self, x: Any, n: Any, z: Any) -> Any:
        return sympy.Sum(x * z ** (-n), (n, 0, sympy.oo))


def _inverse_z(transformed: Any, z: Any, n: Any) -> Any:
    """The sequence from n = 0 whose z-transform a rational function of z is, by partial
    fractions of X(z)/z: c/(z - p)^m is c C(n, m-1) p^{n-m+1} u[n], and c/z^m a delayed
    impulse, c δ[n - m + 1]."""
    rational = sympy.together(sympy.sympify(transformed))
    if not rational.is_rational_function(z):
        raise EvaluationError(
            "the inverse z-transform is computed of a rational function of z, by partial "
            "fractions, and this is not one"
        )
    total, impulses = sympy.Integer(0), sympy.Integer(0)
    for term in sympy.Add.make_args(sympy.apart(rational / z, z, full=True).doit()):
        numerator, denominator = term.as_numer_denom()
        if not denominator.has(z):
            raise EvaluationError(
                "this is the z-transform of no sequence from n = 0: a power of z above the "
                "first is in it, which shifts the sequence before n = 0"
            )
        poles = sympy.roots(sympy.Poly(denominator, z))
        if len(poles) != 1:
            raise EvaluationError("the partial fractions of this rational function are not found")
        ((pole, order),) = poles.items()
        coefficient = sympy.simplify(numerator / sympy.Poly(denominator, z).LC())
        if pole == 0:
            impulses += coefficient * impulse(n - order + 1)
        else:
            total += coefficient * sympy.binomial(n, order - 1) * pole ** (n - order + 1)
    return sympy.simplify(total) * step(n) + impulses


@transform(
    "InverseZTransform",
    aliases=["inverse_z_transform"],
    latex=[r"\mathcal{Z}^{-1}\{X\}(n)", r"\mathcal{Z}^{-1}\{X\}"],
    summary="the inverse z-transform; of a rational function of z, by partial fractions, the "
    "sequence from n = 0, with its step and impulses",
    variables=[("z", "n")],
    inverse="ZTransform",
    examples=[
        Example(r"\mathcal{Z}^{-1}\left\{\frac{z}{z - 1/2}\right\}",
                sympy.Rational(1, 2) ** n * step(n), "Oppenheim and Schafer (2010), table 3.1"),
        Example(r"\mathcal{Z}^{-1}\left\{\frac{1}{\left(1 - \frac{1}{4}z^{-1}\right)"
                r"\left(1 - \frac{1}{2}z^{-1}\right)}\right\}",
                (2 * sympy.Rational(1, 2) ** n - sympy.Rational(1, 4) ** n) * step(n),
                "Oppenheim and Schafer (2010), example 3.9"),
    ],
    points=[(z / (z - sympy.Rational(1, 2)),), (z / (z - 1) ** 2,), (z / (z - 1),)],
)  # fmt: skip
class InverseZTransform:
    samples = (sympy.Integer(0), sympy.Integer(1), sympy.Integer(2), sympy.Integer(5))

    def exact(self, X: Any, z: Any = None, n: Any = None) -> Any:
        z, n = _variables(X, z, n, ("z", "n"))
        return _inverse_z(X, z, n)
