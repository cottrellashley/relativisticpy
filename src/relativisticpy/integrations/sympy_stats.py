r"""Distributions, from SymPy's sympy.stats and SciPy's scipy.stats.

Each is written as books write it, after ``\sim``, and its forms say what each argument is:
``\mathcal{N}(\mu, \sigma^2)`` takes the variance, as most books write it, and the plain
text ``Normal(mu, sigma)`` the standard deviation, as SymPy and SciPy do. ``\Gamma(k,
\theta)`` takes the shape and the scale, ``\operatorname{Exp}(\lambda)`` the rate, and
``\operatorname{Geom}(p)`` counts the trials up to the first success, from 1. SymPy gives
each law exactly, and SciPy numerically, where SymPy finds no closed form within the
budget and there are no parameters.
"""

from __future__ import annotations

from typing import Any

import sympy
from sympy.stats.crv_types import (
    BetaDistribution,
    ChiSquaredDistribution,
    ExponentialDistribution,
    NormalDistribution,
    StudentTDistribution,
    UniformDistribution,
)
from sympy.stats.crv_types import (
    GammaDistribution as SymPyGamma,
)
from sympy.stats.drv_types import GeometricDistribution, PoissonDistribution
from sympy.stats.frv_types import BernoulliDistribution, BinomialDistribution

from relativisticpy.errors import EvaluationError
from relativisticpy.integrations import Example
from relativisticpy.integrations._distributions import distribution

R = sympy.Rational
lam, n, p, t, k, theta = sympy.symbols("lambda n p t k theta")
a, b, mu, sigma = sympy.symbols("a b mu sigma")


def _law(name: str, made: Any, *parameters: Any) -> Any:
    """SymPy's law ``made`` of ``parameters``, which it checks first: parameters a law does
    not allow are refused, saying which."""
    values = [sympy.sympify(each) for each in parameters]
    shown = ", ".join(str(each) for each in parameters[: len(values)])
    for each in values:
        if isinstance(each, list | tuple | sympy.Tuple | sympy.NDimArray | sympy.MatrixBase):
            raise EvaluationError(
                f"{name}({shown}) is no distribution: a parameter of a law is one value, "
                f"and {each} is several",
                hint="declare a random variable for each value, each with a law of its own",
            )
    try:
        made.check(*values)
    except (ValueError, TypeError) as error:  # TypeError: a parameter that is not real
        raise EvaluationError(f"{name}({shown}) is no distribution: {error}") from None
    return made(*values)


@distribution(
    "Normal",
    latex=[r"\mathcal{N}(\mu, \sigma^2)", r"N(\mu, \sigma^2)"],
    summary="the normal distribution, of mean μ and standard deviation σ",
    requires="scipy",
    moments="DeGroot and Schervish (2012), section 5.6",
    examples=[
        Example(
            "X \\sim \\mathcal{N}(0, 1)\n\\Pr(X > 1)", 0.15865525393145707, "scipy.stats.norm.sf(1)"
        ),
        Example(
            "X \\sim \\mathcal{N}(2, 9)\n\\operatorname{Var}(3X + 1)",
            81,
            "DeGroot and Schervish (2012), section 4.3: Var(aX + b) = a^2 Var(X)",
        ),
    ],
    points=[(0, 1), (R(3, 2), 2)],
)
class Normal:
    def exact(self, mu: Any, sigma: Any) -> Any:
        return _law("Normal", NormalDistribution, mu, sigma)

    def numeric(self, xp: Any, mu: Any, sigma: Any) -> Any:
        from scipy.stats import norm

        return norm(mu, sigma)

    def mean(self, mu: Any, sigma: Any) -> Any:
        return mu

    def variance(self, mu: Any, sigma: Any) -> Any:
        return sigma**2

    def mgf(self, t: Any, mu: Any, sigma: Any) -> Any:
        return sympy.exp(mu * t + sigma**2 * t**2 / 2)


@distribution(
    "Poisson",
    latex=[r"\operatorname{Poisson}(\lambda)", r"\operatorname{Pois}(\lambda)"],
    summary="the Poisson distribution, of mean λ",
    requires="scipy",
    discrete=True,
    moments="DeGroot and Schervish (2012), section 5.4",
    examples=[
        Example(
            "X \\sim \\operatorname{Poisson}(\\lambda)\nP(X = 2)",
            lam**2 * sympy.exp(-lam) / 2,
            "DeGroot and Schervish (2012), section 5.4",
        ),
        Example(
            "N \\sim \\operatorname{Poisson}(3)\n\\Pr(N \\le 2)",
            0.42319008112684353,
            "scipy.stats.poisson.cdf(2, 3)",
        ),
    ],
    points=[(3,), (R(1, 2),)],
)
class Poisson:
    def exact(self, rate: Any) -> Any:
        return _law("Poisson", PoissonDistribution, rate)

    def numeric(self, xp: Any, rate: Any) -> Any:
        from scipy.stats import poisson

        return poisson(rate)

    def mean(self, rate: Any) -> Any:
        return rate

    def variance(self, rate: Any) -> Any:
        return rate

    def mgf(self, t: Any, rate: Any) -> Any:
        return sympy.exp(rate * (sympy.exp(t) - 1))


@distribution(
    "Binomial",
    latex=[r"\operatorname{Bin}(n, p)", r"\operatorname{Binom}(n, p)"],
    summary="the binomial distribution, of the successes in n trials of probability p",
    requires="scipy",
    discrete=True,
    moments="DeGroot and Schervish (2012), section 5.2",
    examples=[
        Example(
            "X \\sim \\operatorname{Bin}(10, 1/2)\nP(X = 3)",
            R(15, 128),
            "Feller (1968), section VI.2",
        ),
        Example(
            "X \\sim \\operatorname{Bin}(n, p)\n\\operatorname{Var}(X)",
            n * p * (1 - p),
            "DeGroot and Schervish (2012), section 5.2",
        ),
    ],
    points=[(10, R(3, 10)), (4, R(1, 2))],
)
class Binomial:
    def exact(self, n: Any, p: Any) -> Any:
        return _law("Binomial", BinomialDistribution, n, p, 1, 0)

    def numeric(self, xp: Any, n: Any, p: Any) -> Any:
        from scipy.stats import binom

        return binom(n, p)

    def mean(self, n: Any, p: Any) -> Any:
        return n * p

    def variance(self, n: Any, p: Any) -> Any:
        return n * p * (1 - p)

    def mgf(self, t: Any, n: Any, p: Any) -> Any:
        return (1 - p + p * sympy.exp(t)) ** n


@distribution(
    "Bernoulli",
    latex=[r"\operatorname{Bernoulli}(p)", r"\operatorname{Ber}(p)", r"\operatorname{Bern}(p)"],
    summary="the Bernoulli distribution, 1 with probability p and 0 otherwise",
    requires="scipy",
    discrete=True,
    moments="DeGroot and Schervish (2012), section 5.2",
    examples=[
        Example(
            "Y \\sim \\operatorname{Bernoulli}(p)\n\\mathbb{E}[Y]",
            p,
            "DeGroot and Schervish (2012), section 5.2",
        ),
        Example(
            "Y \\sim \\operatorname{Bernoulli}(1/3)\n\\operatorname{Var}(Y)",
            R(2, 9),
            "scipy.stats.bernoulli.var(1/3)",
        ),
    ],
    points=[(R(3, 10),), (R(1, 2),)],
)
class Bernoulli:
    def exact(self, p: Any) -> Any:
        return _law("Bernoulli", BernoulliDistribution, p, 1, 0)

    def numeric(self, xp: Any, p: Any) -> Any:
        from scipy.stats import bernoulli

        return bernoulli(p)

    def mean(self, p: Any) -> Any:
        return p

    def variance(self, p: Any) -> Any:
        return p * (1 - p)

    def mgf(self, t: Any, p: Any) -> Any:
        return 1 - p + p * sympy.exp(t)


@distribution(
    "Uniform",
    latex=[r"U(a, b)", r"\mathcal{U}(a, b)", r"\operatorname{Unif}(a, b)"],
    summary="the uniform distribution on the interval from a to b",
    requires="scipy",
    moments="DeGroot and Schervish (2012), section 4.3",
    examples=[
        Example("U \\sim U(0, 1)\n\\mathbb{E}[U^2]", R(1, 3), "scipy.stats.uniform.moment(2)"),
        Example(
            "U \\sim \\mathcal{U}(a, b)\n\\operatorname{Var}(U)",
            (b - a) ** 2 / 12,
            "DeGroot and Schervish (2012), section 4.3",
        ),
    ],
    points=[(0, 1), (-1, 3)],
)
class Uniform:
    def exact(self, a: Any, b: Any) -> Any:
        return _law("Uniform", UniformDistribution, a, b)

    def numeric(self, xp: Any, a: Any, b: Any) -> Any:
        from scipy.stats import uniform

        return uniform(a, b - a)

    def mean(self, a: Any, b: Any) -> Any:
        return (a + b) / 2

    def variance(self, a: Any, b: Any) -> Any:
        return (b - a) ** 2 / 12


@distribution(
    "Exponential",
    latex=[r"\operatorname{Exp}(\lambda)"],
    summary="the exponential distribution, of rate λ and mean 1/λ",
    requires="scipy",
    moments="DeGroot and Schervish (2012), section 5.7",
    examples=[
        Example(
            "T \\sim \\operatorname{Exp}(\\lambda)\n\\Pr(T > t)",
            sympy.exp(-lam * t),
            "DeGroot and Schervish (2012), section 5.7",
        ),
        Example(
            "T \\sim \\operatorname{Exp}(2)\n\\mathbb{E}[T]",
            R(1, 2),
            "scipy.stats.expon(scale=1/2).mean()",
        ),
    ],
    points=[(2,), (R(1, 2),)],
)
class Exponential:
    def exact(self, rate: Any) -> Any:
        return _law("Exponential", ExponentialDistribution, rate)

    def numeric(self, xp: Any, rate: Any) -> Any:
        from scipy.stats import expon

        return expon(scale=1 / rate)

    def mean(self, rate: Any) -> Any:
        return 1 / rate

    def variance(self, rate: Any) -> Any:
        return 1 / rate**2

    def mgf(self, t: Any, rate: Any) -> Any:
        return rate / (rate - t)

    def mgf_domain(self, t: Any, rate: Any) -> Any:
        return sympy.Lt(t, rate)


@distribution(
    "GammaDistribution",
    latex=[r"\Gamma(k, \theta)"],
    summary="the gamma distribution, of shape k and scale θ, and mean kθ",
    requires="scipy",
    moments="Wasserman (2004), section 2.4",
    examples=[
        Example("G \\sim \\Gamma(2, 3)\n\\mathbb{E}[G]", 6, "scipy.stats.gamma(2, scale=3).mean()"),
        Example(
            "G \\sim \\Gamma(k, \\theta)\n\\operatorname{Var}(G)",
            k * theta**2,
            "Wasserman (2004), section 2.4",
        ),
    ],
    points=[(2, 3), (R(1, 2), R(3, 2))],
)
class Gamma:
    def exact(self, k: Any, theta: Any) -> Any:
        return _law("GammaDistribution", SymPyGamma, k, theta)

    def numeric(self, xp: Any, k: Any, theta: Any) -> Any:
        from scipy.stats import gamma

        return gamma(k, scale=theta)

    def mean(self, k: Any, theta: Any) -> Any:
        return k * theta

    def variance(self, k: Any, theta: Any) -> Any:
        return k * theta**2

    def mgf(self, t: Any, k: Any, theta: Any) -> Any:
        return (1 - theta * t) ** -k

    def mgf_domain(self, t: Any, k: Any, theta: Any) -> Any:
        return sympy.Lt(t, 1 / sympy.sympify(theta))


@distribution(
    "ChiSquared",
    latex=[r"\chi^2_k", r"\chi_k^2", r"\chi^2(k)"],
    summary="the chi-squared distribution, of k degrees of freedom",
    requires="scipy",
    moments="DeGroot and Schervish (2012), section 8.2",
    examples=[
        Example("C \\sim \\chi^2_3\n\\operatorname{Var}(C)", 6, "scipy.stats.chi2.var(3)"),
        Example("C \\sim \\chi^2_2\n\\Pr(C > 2)", 0.36787944117144233, "scipy.stats.chi2.sf(2, 2)"),
    ],
    points=[(3,), (4,)],
)
class ChiSquared:
    def exact(self, k: Any) -> Any:
        return _law("ChiSquared", ChiSquaredDistribution, k)

    def numeric(self, xp: Any, k: Any) -> Any:
        from scipy.stats import chi2

        return chi2(k)

    def mean(self, k: Any) -> Any:
        return k

    def variance(self, k: Any) -> Any:
        return 2 * k

    def mgf(self, t: Any, k: Any) -> Any:
        return (1 - 2 * t) ** (-k / 2)

    def mgf_domain(self, t: Any, k: Any) -> Any:
        return sympy.Lt(t, R(1, 2))


@distribution(
    "StudentT",
    latex=[r"t_\nu", r"t(\nu)"],
    summary="Student's t distribution, of ν degrees of freedom",
    requires="scipy",
    moments="DeGroot and Schervish (2012), section 8.4",
    examples=[
        Example("T \\sim t_5\n\\operatorname{Var}(T)", R(5, 3), "scipy.stats.t.var(5)"),
        Example("T \\sim t_{10}\n\\Pr(T > 0)", R(1, 2), "scipy.stats.t.sf(0, 10)"),
    ],
    points=[(5,), (10,)],
)
class StudentT:
    def exact(self, nu: Any) -> Any:
        return _law("StudentT", StudentTDistribution, nu)

    def numeric(self, xp: Any, nu: Any) -> Any:
        from scipy.stats import t

        return t(nu)

    def mean(self, nu: Any) -> Any:
        return sympy.Piecewise((0, sympy.sympify(nu) > 1), (sympy.nan, True))

    def variance(self, nu: Any) -> Any:
        nu = sympy.sympify(nu)
        return sympy.Piecewise((nu / (nu - 2), nu > 2), (sympy.oo, nu > 1), (sympy.nan, True))


@distribution(
    "BetaDistribution",
    latex=[r"\operatorname{Beta}(\alpha, \beta)"],
    summary="the beta distribution on the interval from 0 to 1, of shapes α and β",
    requires="scipy",
    moments="DeGroot and Schervish (2012), section 5.8",
    examples=[
        Example(
            "B \\sim \\operatorname{Beta}(2, 3)\n\\mathbb{E}[B]",
            R(2, 5),
            "scipy.stats.beta(2, 3).mean()",
        ),
        Example(
            "B \\sim \\operatorname{Beta}(2, 2)\n\\Pr(B < 1/2)",
            R(1, 2),
            "scipy.stats.beta(2, 2).cdf(0.5)",
        ),
    ],
    points=[(2, 3), (R(1, 2), R(1, 2))],
)
class Beta:
    def exact(self, alpha: Any, beta: Any) -> Any:
        return _law("BetaDistribution", BetaDistribution, alpha, beta)

    def numeric(self, xp: Any, alpha: Any, beta: Any) -> Any:
        from scipy.stats import beta as beta_distribution

        return beta_distribution(alpha, beta)

    def mean(self, alpha: Any, beta: Any) -> Any:
        return alpha / (alpha + beta)

    def variance(self, alpha: Any, beta: Any) -> Any:
        return alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))


@distribution(
    "Geometric",
    latex=[r"\operatorname{Geom}(p)"],
    summary="the geometric distribution, of the trials up to the first success, from 1",
    requires="scipy",
    discrete=True,
    moments="DeGroot and Schervish (2012), section 5.5",
    examples=[
        Example(
            "G \\sim \\operatorname{Geom}(1/3)\n\\mathbb{E}[G]", 3, "scipy.stats.geom.mean(1/3)"
        ),
        Example(
            "G \\sim \\operatorname{Geom}(p)\nP(G = 2)",
            (1 - p) * p,
            "DeGroot and Schervish (2012), section 5.5",
        ),
    ],
    points=[(R(3, 10),), (R(1, 2),)],
)
class Geometric:
    def exact(self, p: Any) -> Any:
        return _law("Geometric", GeometricDistribution, p)

    def numeric(self, xp: Any, p: Any) -> Any:
        from scipy.stats import geom

        return geom(p)

    def mean(self, p: Any) -> Any:
        return 1 / p

    def variance(self, p: Any) -> Any:
        return (1 - p) / p**2

    def mgf(self, t: Any, p: Any) -> Any:
        return p * sympy.exp(t) / (1 - (1 - p) * sympy.exp(t))

    def mgf_domain(self, t: Any, p: Any) -> Any:
        return sympy.Lt(t, -sympy.log(1 - sympy.sympify(p)))
