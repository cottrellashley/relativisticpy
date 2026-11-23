r"""Distributions: a kind of capability, the laws random variables are declared with, and
what the language computes of random variables.

A distribution is written as books write it, after ``\sim``: ``X \sim \mathcal{N}(\mu,
\sigma^2)``, ``N \sim \operatorname{Poisson}(\lambda)``, ``T \sim t_\nu``. Its forms say what
each argument is, in terms of the parameters of ``exact``, taken in order: the form
``\mathcal{N}(\mu, \sigma^2)`` says that its second argument is the square of the second
parameter, the variance, as most books write it::

    @distribution(
        "Normal",
        latex=[r"\mathcal{N}(\mu, \sigma^2)", r"N(\mu, \sigma^2)"],
        summary="the normal distribution, of mean μ and standard deviation σ",
        requires="scipy",
        moments="DeGroot and Schervish (2012), section 5.6",
        examples=[...],
        points=[(0, 1), (1.5, 2)],
    )
    class Normal:
        def exact(self, mu, sigma):
            return NormalDistribution(mu, sigma)          # SymPy's law

        def numeric(self, xp, mu, sigma):
            from scipy.stats import norm

            return norm(mu, sigma)                        # SciPy's frozen law

        def mean(self, mu, sigma):
            return mu

        def variance(self, mu, sigma):
            return sigma**2

A random variable is a SymPy random symbol, of a law, given others, or of a law the program
does not give: ``X \mid N \sim \operatorname{Bin}(N, p)`` gives X's law for each value of
N, and ``\operatorname{Var}(aX + b) = a^2 \operatorname{Var}(X)`` holds whatever X's law is.
Probabilities, expectations, variances, covariances, densities, distribution functions,
quantiles and the moment-generating and characteristic functions are computed exactly by
SymPy, by the laws of total expectation and probability where a law is given conditionally,
and from the mean and variance each distribution declares where SymPy would sum or integrate;
where SymPy finds no closed form within the budget, and the problem has no parameters, SciPy
computes it numerically, and a note says so.

The conformance suite checks, at the parameters each distribution declares, that its density
integrates, or sums, to 1, that its mean and variance are those it declares, by SymPy and by
SciPy, and that SymPy's density and distribution function agree with SciPy's.
"""

from __future__ import annotations

import functools
import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import sympy
import sympy.stats as stats
from sympy import Basic, Dummy, Symbol, Tuple
from sympy.logic.boolalg import Boolean
from sympy.stats.crv import SingleContinuousDistribution, SingleContinuousPSpace
from sympy.stats.drv import SingleDiscreteDistribution, SingleDiscretePSpace
from sympy.stats.frv import SingleFiniteDistribution, SingleFinitePSpace
from sympy.stats.rv import PSpace, RandomSymbol

from relativisticpy import assumptions
from relativisticpy.errors import EvaluationError, RelativisticPyError
from relativisticpy.integrations._declarations import Capability, Example, declares, near
from relativisticpy.integrations._discovery import catalogue, on_change

#: How long SymPy may look for a closed form, in seconds, before SciPy computes the value
#: numerically, where the problem has no parameters.
BUDGET = 5.0
#: How far SymPy's value and SciPy's may differ, relative to their size.
AGREEMENT = 1e-7


# -- the kind -------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Distribution(Capability):
    """The law of a random variable, as the normal distribution and the Poisson distribution
    are: ``exact`` gives SymPy's law of its parameters, ``numeric`` SciPy's frozen law, and
    the implementation's ``mean`` and ``variance`` the closed forms ``moments`` names the
    source of. A ``discrete`` one takes whole numbers. It is written after ``\\sim``, in its
    forms, whose variables are its parameters in order; its ``points`` are parameters it is
    checked at."""

    kind: ClassVar[str] = "distribution"
    takes: ClassVar[str] = "its parameters"

    discrete: bool = False
    moments: str = ""

    def mean(self, *parameters: Any) -> Any:
        """Its mean, as the closed form it declares gives it."""
        return self.implementation.mean(*parameters)

    def variance(self, *parameters: Any) -> Any:
        """Its variance, as the closed form it declares gives it."""
        return self.implementation.variance(*parameters)

    def generating(self, t: Any, *parameters: Any) -> Any | None:
        """Its moment-generating function at ``t``, where it declares one; ``None`` where
        SymPy's law gives it."""
        method = getattr(self.implementation, "mgf", None)
        return None if method is None else method(t, *parameters)

    def generating_domain(self, t: Any, *parameters: Any) -> Any:
        """Where the moment-generating function it declares is finite, as a condition on a
        real ``t``, as t < λ of the exponential distribution: every t, where it declares
        none."""
        method = getattr(self.implementation, "mgf_domain", None)
        return sympy.true if method is None else sympy.sympify(method(t, *parameters))

    def frozen(self, *parameters: Any) -> Any:
        """SciPy's law of its parameters, as numbers."""
        import numpy as np

        numeric = self.numeric
        assert numeric is not None
        return numeric(np, *(float(each) for each in parameters))


def distribution(
    name: str,
    *,
    summary: str,
    examples: Sequence[Example],
    points: Sequence[Sequence[Any]],
    moments: str,
    latex: str | Sequence[str] = (),
    unicode: str | Sequence[str] = (),
    aliases: Sequence[str] = (),
    requires: str | None = None,
    discrete: bool = False,
) -> Callable[[type], Distribution]:
    """Declare a distribution, checked at ``points``, each its parameters. Its class has
    ``exact``, which gives SymPy's law, ``numeric``, which gives SciPy's frozen law, and
    ``mean`` and ``variance``, the closed forms of its moments, from ``moments``; it may have
    ``mgf(t, ...)``, its moment-generating function, where SymPy's law has none, and then
    ``mgf_domain(t, ...)``, the condition on t it is finite under, where that is not every
    t: outside it the closed form is no expectation, which is infinite there."""

    def decorate(cls: type) -> Distribution:
        for method, does in (("mean", "gives its mean"), ("variance", "gives its variance")):
            if not callable(getattr(cls, method, None)):
                raise TypeError(
                    f"{name}: {cls.__name__} has no {method} method, which {does}; declare "
                    f"def {method}(self, ...): ... with the parameters of exact"
                )
        declare = declares(
            Distribution,
            name,
            summary=summary,
            examples=examples,
            aliases=aliases,
            latex=latex,
            unicode=unicode,
            requires=requires,
            points=points,
            discrete=discrete,
            moments=moments,
        )
        return declare(cls)

    return decorate


def distributions() -> list[Distribution]:
    """Every distribution the integrations declare, in their order."""
    return [c for c in catalogue().capabilities if isinstance(c, Distribution)]


_BY_LAW: dict[type, Distribution] | None = None


def _forget() -> None:
    global _BY_LAW
    _BY_LAW = None


on_change(_forget)


def declared(law: Any) -> tuple[Distribution, tuple[Any, ...]] | None:
    """The distribution SymPy's ``law`` is of, and its parameters; ``None`` for a law no
    integration declares."""
    global _BY_LAW
    if _BY_LAW is None:
        found: dict[type, Distribution] = {}
        for capability in distributions():
            if capability.points:
                found.setdefault(type(capability.exact(*capability.points[0])), capability)
        _BY_LAW = found
    known = _BY_LAW.get(type(law))
    if known is None:
        return None
    return known, tuple(law.args[: len(known.parameters)])


# -- random variables -----------------------------------------------------------------------


def _randoms(expr: Any) -> set[RandomSymbol]:
    """The random variables an expression, or a law, is written with."""
    return set(sympy.sympify(expr).atoms(RandomSymbol))


class Given(PSpace):  # type: ignore[misc]
    """The probability space of a random variable whose law is given by others: a law for
    each of the conditions on them, as X | Y = 0 ~ N(-1, 1) and X | Y = 1 ~ N(1, 1) give,
    or one law whose parameters are random, as X | N ~ Bin(N, p) gives."""

    is_real = True

    def __new__(cls, symbol: Any, pieces: Any) -> Given:
        pieces = Tuple(*(Tuple(condition, law) for condition, law in pieces))
        made: Given = Basic.__new__(cls, symbol, pieces)
        return made

    @property
    def symbol(self) -> Any:
        return self.args[0]

    @property
    def pieces(self) -> tuple[tuple[Any, Any], ...]:
        """Each condition, with the law under it."""
        return tuple((piece[0], piece[1]) for piece in self.args[1].args)

    @property
    def value(self) -> RandomSymbol:
        return RandomSymbol(self.symbol, self)

    @property
    def parents(self) -> list[RandomSymbol]:
        """The random variables its law is given by, by name."""
        found: set[RandomSymbol] = set()
        for condition, law in self.pieces:
            found |= _randoms(condition) | _randoms(law)
        return sorted(found, key=str)


class Unknown(PSpace):  # type: ignore[misc]
    """The probability space of a random variable whose law is not given, as X's in
    Var(aX + b) = a² Var(X), which holds whatever its law is."""

    is_real = True

    def __new__(cls, symbol: Any) -> Unknown:
        made: Unknown = Basic.__new__(cls, symbol)
        return made

    @property
    def value(self) -> RandomSymbol:
        return RandomSymbol(self.args[0], self)


def unknown(name: str) -> RandomSymbol:
    """A random variable called ``name``, of a law the program does not give."""
    return Unknown(Symbol(name)).value


def variable(symbol: Any, *pieces: Any) -> RandomSymbol:
    """The random variable ``symbol``: of a law, or of a law under each of the conditions,
    alternating a condition and the law under it, as ``X \\mid Y = 0 \\sim ...`` gives."""
    symbol = Symbol(str(symbol)) if not isinstance(symbol, Symbol) else symbol
    if len(pieces) == 1:  # a law, maybe of random parameters, as X | N ~ Bin(N, p)
        return _of_law(symbol, pieces[0])
    pairs = list(zip(pieces[::2], pieces[1::2], strict=True))
    return Given(symbol, pairs).value


def _space(symbol: Any, law: Any) -> PSpace:
    """SymPy's probability space of one random variable of ``law``."""
    if isinstance(law, SingleContinuousDistribution):
        return SingleContinuousPSpace(symbol, law)
    if isinstance(law, SingleDiscreteDistribution):
        return SingleDiscretePSpace(symbol, law)
    if isinstance(law, SingleFiniteDistribution):
        return SingleFinitePSpace(symbol, law)
    raise EvaluationError(f"{law} is no distribution of one random variable")


def _fresh(variable: RandomSymbol, law: Any) -> RandomSymbol:
    """A random variable of ``law``, called as ``variable`` is."""
    return _of_law(variable.symbol, law)


def _of_law(symbol: Any, law: Any) -> RandomSymbol:
    """A random variable of ``law``, given others where its parameters are random."""
    if _randoms(law):
        return Given(symbol, [(sympy.true, law)]).value
    return _space(symbol, law).value


def _unknown(variable: RandomSymbol) -> bool:
    return isinstance(variable.pspace, Unknown)


def _given(variable: RandomSymbol) -> bool:
    return isinstance(variable.pspace, Given)


def _law(variable: RandomSymbol) -> Any:
    """The law of a random variable of one: SymPy's distribution."""
    return variable.pspace.distribution


def _discrete(variable: RandomSymbol) -> bool:
    return not isinstance(_law(variable), SingleContinuousDistribution)


def _at(expr: Any, parent: RandomSymbol, value: Any) -> Any:
    """``expr`` where the random variable ``parent`` takes ``value``: it, and each variable
    given it, of the law it has there."""
    replaced: dict[Any, Any] = {parent: value}
    for each in _randoms(expr):
        if _given(each) and parent in each.pspace.parents:
            replaced[each] = _variable_at(each, parent, value)
    return sympy.sympify(expr).xreplace(replaced)


def _variable_at(variable: RandomSymbol, parent: RandomSymbol, value: Any) -> RandomSymbol:
    """A random variable given ``parent``, of the law it has where ``parent`` is
    ``value``: the law under the condition that holds there, or each under its own."""
    pieces = []
    for condition, law in variable.pspace.pieces:
        condition = condition.xreplace({parent: value})
        if condition is sympy.false:
            continue
        pieces.append((condition, law.xreplace({parent: value})))
    if not pieces:
        raise EvaluationError(
            f"{variable} has no law given where {parent} is {value}",
            hint=f"give its law there, as {variable} \\mid {parent} = {value} \\sim ...",
        )
    if len(pieces) == 1 and pieces[0][0] is sympy.true:
        return _of_law(variable.symbol, pieces[0][1])
    return Given(variable.symbol, pieces).value


def _value_of(variable: RandomSymbol) -> Dummy:
    """A value a random variable takes, as a symbol: an integer, of a discrete one."""
    if not (_given(variable) or _unknown(variable)) and _discrete(variable):
        return Dummy(str(variable.symbol), integer=True)
    return Dummy(str(variable.symbol), real=True)


# -- samples ----------------------------------------------------------------------------------


class Sample(Basic):  # type: ignore[misc]
    """A sample of independent random variables of one law, as X_1, ..., X_n
    iid ~ N(μ, σ²) declares: the name its members are written with, the law, how many
    there are, which may be a symbol, and the number of the first, 1 unless it says."""

    def __new__(cls, symbol: Any, law: Any, size: Any, first: Any = 1) -> Sample:
        made: Sample = Basic.__new__(cls, symbol, law, sympy.sympify(size), sympy.sympify(first))
        return made

    @property
    def name(self) -> str:
        return str(self.args[0])

    @property
    def law(self) -> Any:
        return self.args[1]

    @property
    def first(self) -> Any:
        """The number of its first member."""
        return self.args[3]

    @property
    def last(self) -> Any:
        """The number of its last member."""
        return self.args[3] + self.args[2] - 1


class Member(sympy.Function):  # type: ignore[misc]
    """X_i, the member of a sample at i: the random variable X_k where i is a whole
    number k, and, of a symbol, the member a sum over i takes in turn."""

    is_real = True

    @classmethod
    def eval(cls, sample: Any, index: Any) -> Any:
        if isinstance(sample, Sample) and index.is_Integer:
            _refuse_beyond(sample, index, index)
            symbol = Symbol(f"{sample.name}_{int(index)}")
            return _space(symbol, sample.law).value
        return None

    def _sympystr(self, printer: Any) -> str:
        return str(printer._print(self._as_symbol(printer)))

    def _relativisticpy_plain(self, printer: Any) -> str:
        return str(printer._print(self._as_symbol(printer)))

    def _latex(self, printer: Any) -> str:
        return str(printer._print(self._as_symbol(printer)))

    def _as_symbol(self, printer: Any) -> Symbol:
        sample, index = self.args
        return Symbol(f"{sample.name}_{index}")


def sample(symbol: Any, law: Any, size: Any, first: Any = 1) -> Sample:
    """The sample ``symbol``, of ``size`` independent random variables of ``law``, numbered
    from ``first``."""
    return Sample(Symbol(str(symbol)), law, size, first)


def member(of: Any, index: Any) -> Any:
    """The member of a sample at ``index``."""
    return Member(of, index)


def _members(expr: Any) -> set[Any]:
    return set(sympy.sympify(expr).atoms(Member))


def _expect_of_sample(expr: Any) -> Any:
    """The expectation of an expression of the members of samples: by linearity, a sum over
    the members, ``\\sum_{i=1}^{n} f(X_i)``, is n times the expectation of f(X) of one; a
    member at a symbol is a random variable of the law."""
    total = sympy.S.Zero
    for term in sympy.Add.make_args(sympy.expand(expr)):
        sums = [factor for factor in sympy.Mul.make_args(term) if isinstance(factor, sympy.Sum)]
        rest = sympy.Mul(*(f for f in sympy.Mul.make_args(term) if not isinstance(f, sympy.Sum)))
        if len(sums) > 1 or (sums and _members(rest)):
            raise EvaluationError(
                f"the expectation of {_shown(term)}, a product of sums over a sample, is not "
                "computed",
                hint="write it as sums of single terms, as variances and means give them",
            )
        if not sums:
            total += _expect(_at_members(rest))
            continue
        (summed,) = sums
        body, *limits = summed.args
        if len(limits) != 1:
            raise EvaluationError(f"the expectation of {_shown(summed)} is not computed")
        _within_the_sample(body, limits[0])
        expected = _expect(_at_members(body))
        total += rest * sympy.Sum(expected, *limits).doit()
    return _tidy(total)


def _at_members(expr: Any) -> Any:
    """``expr`` with each member of a sample at a symbol a random variable of its law: two
    at indices that differ by a number are two members, independent, as X_{k+1} and X_k
    are, and one at the index a numbered member has is that member. An error that says so
    where it is not decided whether two are the same one, as of X_k and X_1, or where one
    is beyond its sample, as X_{n+1} is of n."""
    expr = sympy.sympify(expr)
    found = sorted(_members(expr), key=str)
    replaced: dict[Any, Any] = {}
    for position, each in enumerate(found):
        _within_the_sample(each)
        of, index = each.args
        earlier = [(replaced[m], m, m.args[1]) for m in found[:position] if m.args[0] == of]
        numbered = [(v, v, number) for v, number in _numbered(expr, of).items()]
        for variable, written, at in [*earlier, *numbered]:
            apart = sympy.simplify(index - at)
            if apart == 0:
                replaced[each] = variable
            elif not apart.is_nonzero:
                raise EvaluationError(
                    f"{_shown(each)} and {_shown(written)} are members of a sample that may be "
                    "the same one, which is not decided",
                    hint="say which member it is, or write a sum over the sample, as "
                    f"\\sum_{{i=1}}^{{n}} {of.name}_i",
                )
        replaced.setdefault(each, _space(Symbol(f"{of.name}_{index}"), of.law).value)
    return expr.xreplace(replaced)


def _numbered(expr: Any, of: Sample) -> dict[RandomSymbol, Any]:
    """The members of a sample at whole numbers that an expression is written with, each
    with its number: X_1 is the random variable its sample makes at 1, known by its name
    and its law."""
    found = {}
    for each in sorted(_randoms(expr), key=str):
        name, _, number = str(each.symbol).rpartition("_")
        declared_so = not _given(each) and not _unknown(each) and _law(each) == of.law
        if name == of.name and number.isdigit() and declared_so:
            found[each] = sympy.Integer(int(number))
    return found


def _within_the_sample(written: Any, limits: Any = None) -> None:
    """Refuse a member beyond its sample, as X_{n+1} is of X_1, ..., X_n, or a sum over
    members that runs beyond it, where that is decided."""
    for each in [written] if isinstance(written, Member) else sorted(_members(written), key=str):
        of, index = each.args
        ends = [index, index]
        if limits is not None and index.has(limits[0]):
            ends = [index.subs(limits[0], limits[1]), index.subs(limits[0], limits[2])]
        _refuse_beyond(of, *ends)


def _refuse_beyond(of: Sample, lowest: Any, highest: Any) -> None:
    """Refuse members of a sample numbered from ``lowest`` to ``highest`` where one is
    surely before its first member or after its last."""
    before = sympy.simplify(of.first - lowest).is_positive
    after = sympy.simplify(highest - of.last).is_positive
    if before or after:
        beyond = highest if after else lowest
        raise EvaluationError(
            f"{_member_written(of, beyond)} is beyond the sample, whose members are "
            f"{_member_written(of, of.first)} to {_member_written(of, of.last)}",
        )


def _member_written(of: Sample, index: Any) -> str:
    """A member of a sample as a message writes it: X_1, X_n, or X_{n + 1}."""
    shown = _shown(index)
    return f"{of.name}_{shown}" if shown.isalnum() else f"{of.name}_{{{shown}}}"


def _variance_of_sample(expr: Any) -> Any:
    """The variance of an expression of the members of a sample: of a sum over the members,
    ``\\sum_{i=1}^{n} f(X_i)``, times a constant, n times the variance of f(X) of one, as
    its terms are independent, each of a member of its own; of anything written without a
    sum, as X_{k+1} - X_k, of the members as random variables. Refused, saying why, where
    the terms of a sum share a member, as those of ``\\sum (X_{i+1} - X_i)`` do, whose
    variance is not the sum of theirs."""
    expr = sympy.sympify(expr)
    sums = sorted(expr.atoms(sympy.Sum), key=str)
    if not sums:
        return _variance(_at_members(expr))
    stand = Dummy("S")
    try:
        linear = sympy.Poly(expr.xreplace({sums[0]: stand}), stand)
    except sympy.PolynomialError:
        linear = None
    alone = (
        len(sums) == 1
        and linear is not None
        and linear.degree() == 1
        and not any(c.has(Member) or _randoms(c) for c in linear.all_coeffs())
        and len(sums[0].limits) == 1
    )
    if not alone:
        raise EvaluationError(
            f"the variance of {_shown(expr)} is not computed: of a sample, the variance of "
            "one sum over its members, times a constant, is",
            hint="write it as one sum over the sample, as \\operatorname{Var}(\\sum_{i=1}^{n} X_i)",
        )
    assert linear is not None
    body, limits = sums[0].function, sums[0].limits[0]
    _independent_terms(body, limits)
    _within_the_sample(body, limits)
    spread = _variance(_at_members(body))
    return _tidy(linear.all_coeffs()[0] ** 2 * sympy.Sum(spread, limits).doit())


def _independent_terms(body: Any, limits: Any) -> None:
    """Refuse the variance of a sum whose terms are not independent: each is to be of one
    member of a sample, at the index of the sum, and of nothing else that is random, which
    every term would share."""
    index = limits[0]
    at = {each.args[1] for each in _members(body)}
    shared = [each for each in _members(body) if not each.args[1].has(index)]
    if len(at) > 1 or shared or _randoms(body):
        raise EvaluationError(
            f"the variance of the sum of {_shown(body)} over {_shown(index)} is not the sum "
            "of the variances of its terms: they share members of the sample, and so are "
            "not independent",
            hint="write the sum out, or as a sum whose terms are each of one member, as a "
            "sum that telescopes is of its first and last",
        )


# -- what a program computes of random variables -----------------------------------------------


def expectation(value: Any, *given: Any) -> Any:
    """``\\mathbb{E}[value]``, or ``\\mathbb{E}[value \\mid given]``: given random variables, a
    function of them; given that one takes a value, or an event, a number."""
    return _conditioned(_expect, value, given, "expectation")


def probability(event: Any, *given: Any) -> Any:
    """``\\Pr(event)``, or ``\\Pr(event \\mid given)``."""
    event = sympy.sympify(event)
    if not isinstance(event, Boolean):
        raise EvaluationError(
            f"the probability of {_shown(event)}, which is no event",
            hint="an event is a relation of random variables, as X > 1 or X = 2",
        )
    return _conditioned(_probability, event, given, "probability")


def variance(value: Any, *given: Any) -> Any:
    """``\\operatorname{Var}(value)``, or the conditional variance given ``given``."""
    return _conditioned(_variance, value, given, "variance")


def covariance(first: Any, second: Any, *given: Any) -> Any:
    """``\\operatorname{Cov}(first, second)``."""
    pair = Tuple(first, second)
    return _conditioned(lambda both: _covariance(*both.args), pair, given, "covariance")


def density(value: Any, at: Any) -> Any:
    """The density of a random variable at ``at``, ``f_X(x)``: of a discrete one, the
    probability that it takes the value."""
    variable = _one(value, "density")
    if _given(variable):
        return _total(variable, variable, lambda each: density(each, at))
    law = _law(variable)
    parameters = _parameters(variable, at)

    def exactly() -> Any:
        return _exact_density(variable, at)

    def numerically() -> Any:
        frozen = _frozen(variable)
        number = float(at)
        if isinstance(law, SingleContinuousDistribution):
            return sympy.Float(frozen.pdf(number))
        return sympy.Float(frozen.pmf(number))

    return _found(exactly, numerically, parameters, f"the density of {variable}")


def cdf(value: Any, at: Any) -> Any:
    """The distribution function of a random variable at ``at``, ``F_X(x)``, the
    probability that it is at most ``at``."""
    variable = _one(value, "distribution function")
    if sympy.sympify(at).is_number:
        return _probability(sympy.Le(variable, at))
    if _given(variable):
        return _total(variable, variable, lambda each: cdf(each, at))
    parameters = _parameters(variable, at)

    def exactly() -> Any:
        return stats.cdf(variable)(at)

    return _found(exactly, None, parameters, f"the distribution function of {variable}")


def quantile(value: Any, at: Any) -> Any:
    """The quantile of a random variable at ``at``, ``F_X^{-1}(q)``: the least x at which its
    distribution function reaches q."""
    variable = _one(value, "quantile")
    if _given(variable):
        raise EvaluationError(
            f"the quantile of {variable}, whose law is given by others, is not computed",
            hint="compute its distribution function, F_X(x), and solve it for x",
        )
    parameters = _parameters(variable, at)

    def exactly() -> Any:
        return stats.quantile(variable)(at)

    def numerically() -> Any:
        return sympy.Float(_frozen(variable).ppf(float(at)))

    return _found(exactly, numerically, parameters, f"the quantile of {variable}")


def generating(value: Any, at: Any) -> Any:
    """The moment-generating function of a random variable at ``at``, ``M_X(t)``,
    ``\\mathbb{E}[e^{tX}]``."""
    variable = _one(value, "moment-generating function")
    return _expect(sympy.exp(at * variable))


def characteristic(value: Any, at: Any) -> Any:
    """The characteristic function of a random variable at ``at``, ``\\varphi_X(t)``,
    ``\\mathbb{E}[e^{itX}]``."""
    variable = _one(value, "characteristic function")
    return _expect(sympy.exp(sympy.I * at * variable))


def _one(value: Any, what: str) -> RandomSymbol:
    value = sympy.sympify(value)
    if not isinstance(value, RandomSymbol):
        raise EvaluationError(
            f"the {what} is of one random variable, and {_shown(value)} is not one",
            hint="write it of a random variable, as f_X(x) of X \\sim \\mathcal{N}(0, 1)",
        )
    if _unknown(value):
        raise EvaluationError(
            f"the {what} of {value} is not known: its law is not given",
            hint=f"declare it, as {value} \\sim \\mathcal{{N}}(0, 1)",
        )
    return value


# -- conditioning ---------------------------------------------------------------------------------


def _conditioned(compute: Callable[[Any], Any], value: Any, given: Sequence[Any], what: str) -> Any:
    """What ``compute`` gives of ``value`` given ``given``, each a random variable, of which
    the value is a function; a random variable at a value, as Y = 0; or an event, by the
    ratio of probabilities."""
    value = sympy.sympify(value)
    restored: dict[Any, Any] = {}
    events: list[Any] = []
    for each in given:
        each = sympy.sympify(each)
        if isinstance(each, sympy.Eq) and _a_variable(each.lhs) and not _randoms(each.rhs):
            value = _at(value, each.lhs, each.rhs)  # given Y = 0
        elif isinstance(each, Boolean):
            events.append(each)
        elif _a_variable(each):
            known = _value_of(each)
            value = _at(value, each, known)
            restored[known] = each
        else:
            raise EvaluationError(
                f"the {what} given {_shown(each)}: a condition is an event, as X > 1, or a "
                "random variable, as N",
            )
    if events:
        condition = sympy.And(*events)
        found = _given_event(compute, value, condition, what)
    else:
        found = compute(value)
    return _defined(sympy.sympify(found), value, what).xreplace(restored)


def _defined(found: Any, value: Any, what: str) -> Any:
    """What was computed of random variables, where it is a value: not where it is
    undefined, as the mean of a law without one, which is refused, saying so; nor where
    it still holds a random variable whose law is given, which SymPy left in it."""
    shown = _shown(value.args[0] if isinstance(value, Tuple) and len(value) == 1 else value)
    if found is sympy.nan:
        raise EvaluationError(
            f"the {what} of {shown} is not defined: an expectation it is made of does not "
            "exist, or is infinite",
            notes=["a law with tails as heavy as Cauchy's has no mean, and so no variance"],
        )
    symbolic = (stats.Expectation, stats.Variance, stats.Covariance, stats.Probability)
    bare = found.replace(lambda each: isinstance(each, symbolic), lambda each: sympy.S.Zero)
    names = {str(each.symbol) for each in _randoms(value) if not _unknown(each)}
    left = {str(each) for each in _randoms(bare) if not _unknown(each)}
    left |= {s.name for s in bare.free_symbols if type(s) is Symbol and s.name in names}
    if left:
        raise EvaluationError(
            f"the {what} of {shown} is not computed: what SymPy gives of it still holds "
            f"{assumptions.listed(sorted(left))}, which the {what} is no function of",
        )
    return found


def _a_variable(value: Any) -> bool:
    return isinstance(value, RandomSymbol)


def _given_event(compute: Callable[[Any], Any], value: Any, condition: Any, what: str) -> Any:
    """What ``compute`` gives of ``value`` given an event: P(A | B) = P(A and B)/P(B), and
    E[X | B] = E[X 1_B]/P(B)."""
    weight = _probability(condition)
    if sympy.sympify(weight).is_zero:
        raise EvaluationError(
            f"the {what} given {_shown(condition)}, an event of probability 0, is not defined",
            hint="give a value a random variable takes, as Y = 0, where its law is given so",
        )
    if what == "probability":
        return _tidy(_probability(sympy.And(value, condition)) / weight)
    if what == "expectation":
        return _tidy(_where(value, condition) / weight)
    raise EvaluationError(
        f"the {what} given an event is not computed",
        hint="write it with expectations, as E[X^2 | B] - E[X | B]^2",
    )


def _where(value: Any, condition: Any) -> Any:
    """``\\mathbb{E}[value 1_B]``, the expectation of a value where an event B holds and of
    0 elsewhere: of one random variable, by :func:`_partial`; of several, independent, the
    sum, over the values a discrete one of the event's can take where it holds, of what it
    is at each, where they are finitely many, as of X where X + Y = 3; else as SymPy
    finds the expectation of the value in cases."""
    condition = sympy.sympify(condition)
    if condition is sympy.true:
        return _expect(value)
    if condition is sympy.false:
        return sympy.S.Zero
    randoms = sorted(_randoms(value) | _randoms(condition), key=str)
    declared_so = not any(_given(each) or _unknown(each) for each in randoms)
    if len(randoms) == 1 and declared_so:
        partial = _partial(value, randoms[0], condition)
        if partial is not None:
            return partial
    split = _split(condition, sorted(_randoms(condition), key=str)) if declared_so else None
    if split is not None:
        variable, values = split
        held = sympy.sympify(value)
        return _over_values(
            variable,
            values,
            lambda v: _where(held.xreplace({variable: v}), condition.xreplace({variable: v})),
        )
    return _expect(sympy.Piecewise((value, condition), (0, True)))


def _over_values(variable: RandomSymbol, values: Sequence[Any], of: Callable[[Any], Any]) -> Any:
    """The sum, over values a discrete random variable takes, of the probability of each
    times what ``of`` gives there: the law of total probability, within four times the
    time a closed form is given, past which it is refused, saying so."""
    from relativisticpy.runtime import core

    try:
        with core._budget(4 * BUDGET):
            return sum((_mass(variable, each) * of(each) for each in values), sympy.S.Zero)
    except core._OverBudget:
        raise EvaluationError(
            f"a sum over the {len(values)} values of {variable} an event leaves it is not "
            f"computed within {4 * BUDGET:g} s",
            hint="give the event fewer values to hold at",
        ) from None


def _partial(value: Any, variable: RandomSymbol, condition: Any) -> Any | None:
    """``\\mathbb{E}[g(X) 1_B]``, of an event B of one random variable: the integral of g
    times its density over the values B holds at, or the sum of g times its masses, by
    SymPy, or, numerically, by SciPy; ``None`` where those values are not found."""
    solved = _solved(condition, variable)
    if solved is None:
        return None
    within, values, taken, inside = solved
    discrete = _discrete(variable)
    law = _law(variable)
    point = Dummy("k", integer=True) if discrete else Dummy("x", real=True)
    weighted = sympy.sympify(value).xreplace({variable: point, **taken})

    def exactly() -> Any:
        total = sympy.S.Zero
        for low, high in _ranges(within, discrete):
            if discrete:
                masses = (
                    law.pmf(point) if isinstance(law, SingleFiniteDistribution) else law.pdf(point)
                )
                total += sympy.summation(weighted * masses, (point, low, high))
            else:
                total += sympy.integrate(weighted * law.pdf(point), (point, low, high))
        return _taken_back(total, taken, inside, variable)

    def numerically() -> Any:
        frozen = _frozen(variable)
        function = sympy.lambdify(point, weighted, modules=["scipy", "numpy"])
        total = 0.0
        for low, high in _ranges(values, discrete):
            total += float(frozen.expect(function, lb=float(low), ub=float(high)))
        return sympy.Float(total)

    parameters = _parameters(variable, value, condition)
    what = f"the expectation of {_shown(value)} given {_shown(condition)}"
    return _found(exactly, numerically, parameters, what)


def _ranges(values: Any, discrete: bool) -> list[tuple[Any, Any]]:
    """The ends of each interval of ``values``: of a discrete variable, the least and the
    greatest whole number in it."""
    parts = values.args if isinstance(values, sympy.Union) else (values,)
    found = []
    for part in parts:
        if isinstance(part, sympy.FiniteSet):
            found += [(point, point) for point in part]
            continue
        if part is sympy.S.EmptySet:
            continue
        low, high = part.start, part.end
        if discrete:
            low = sympy.floor(low) + 1 if part.left_open else sympy.ceiling(low)
            high = sympy.ceiling(high) - 1 if part.right_open else sympy.floor(high)
        found.append((low, high))
    return found


# -- expectations -----------------------------------------------------------------------------


def _expect(expr: Any) -> Any:
    """``\\mathbb{E}[expr]``."""
    expr = sympy.sympify(expr)
    if expr.has(Member):
        return _expect_of_sample(expr)
    randoms = _randoms(expr)
    if not randoms:
        return expr
    if any(_unknown(each) for each in randoms):
        return _symbolic(stats.Expectation(expr))
    given = sorted((each for each in randoms if _given(each)), key=str)
    if given:
        return _total(expr, given[0], _expect)
    return _declared_expectation(expr)


def _total(expr: Any, variable: RandomSymbol, compute: Callable[[Any], Any]) -> Any:
    """What ``compute`` gives of ``expr``, in which ``variable``'s law is given by others,
    by the laws of total expectation and probability: weighed by the probability of each
    condition its laws are given under, or as the expectation, over the variables its law
    is given by, of what it gives where they take values."""
    pieces = variable.pspace.pieces
    if any(_randoms(condition) for condition, _ in pieces):
        return _over_conditions(expr, variable, compute)
    if len(pieces) == 1 and pieces[0][0] is sympy.true:
        parent = variable.pspace.parents[0]
        known = _value_of(parent)
        inner = compute(_at(expr, parent, known))
        return _expect(sympy.sympify(inner).xreplace({known: parent}))
    return sympy.Piecewise(
        *((compute(sympy.sympify(expr).xreplace({variable: _fresh(variable, law)})), condition)
          for condition, law in pieces)
    )  # fmt: skip


def _over_conditions(expr: Any, variable: RandomSymbol, compute: Callable[[Any], Any]) -> Any:
    """``compute`` of ``expr``, of a variable given for each value of another, as X given
    Y = 0 and Y = 1: what it gives under each, weighed by the probability of each; refused
    where those probabilities leave some out."""
    total, covered = sympy.S.Zero, sympy.S.Zero
    conditions = []
    for condition, _ in variable.pspace.pieces:
        if not (isinstance(condition, sympy.Eq) and _a_variable(condition.lhs)):
            raise EvaluationError(
                f"{variable} is given under {_shown(condition)}, which is no value of a random "
                "variable",
                hint=f"give its law for each value of one, as {variable} \\mid Y = 0 \\sim ...",
            )
        parent, value = condition.lhs, condition.rhs
        weight = _probability(condition)
        total += weight * compute(_at(expr, parent, value))
        covered += weight
        conditions.append(_shown(condition))
    if not _is_one(covered):
        raise EvaluationError(
            f"the law of {variable} is given where {', '.join(conditions)}, which together have "
            f"probability {_shown(_tidy(covered))}, not 1",
            hint="give its law for every value the variable it is given by takes",
        )
    return _tidy(total)


def _is_one(value: Any) -> bool:
    value = sympy.sympify(value)
    if value.is_number:
        return abs(complex(sympy.N(value)) - 1) < 1e-12
    return bool(sympy.simplify(value - 1) == 0)


def _symbolic(written: Any) -> Any:
    """An expectation, a variance or a covariance of a random variable whose law is not given:
    by the rules they keep, linearity and the rest, with what is of variables whose law is
    given computed."""
    expanded = written.expand()

    def known(term: Any) -> Any:
        if not any(_unknown(each) for each in _randoms(term)):
            return term.doit()
        return term

    return expanded.replace(
        lambda term: isinstance(term, stats.Expectation | stats.Variance | stats.Covariance),
        known,
    )


def _declared_expectation(expr: Any) -> Any:
    """The expectation of an expression of random variables of laws the integrations
    declare, independent: from their moments, of a polynomial in them; from the
    moment-generating function, of an exponential; else by SymPy, or by SciPy."""
    randoms = sorted(_randoms(expr), key=str)
    found = _by_moments(expr, randoms)
    if found is not None:
        return _tidy(found)
    found = _by_generating(expr, randoms)
    if found is not None:
        return _tidy(found)
    parameters = _parameters(*randoms, expr)

    def exactly() -> Any:
        return stats.E(expr)

    numerically = None
    if len(randoms) == 1:
        (variable,) = randoms

        def numerically() -> Any:
            return _numeric_expectation(expr, variable)

    what = f"the expectation of {_shown(expr)}"
    found = _found(exactly, numerically, parameters, what)
    return _borne_out(found, expr, what) if len(randoms) > 1 else found


#: How many times each random variable is drawn to bear out a value SymPy gives of several.
_DRAWS = 200_000


def _borne_out(value: Any, expr: Any, what: str) -> Any:
    """A value SymPy gives of several random variables together, as the expectation of
    ``expr``, where a simulation bears it out. SymPy's sums and integrals over several
    variables are not all right: it gives the probability that the product of two uniform
    variables is below 1/2 as 0. A value a simulation contradicts is refused, saying so;
    one written with parameters, or left undone, which no simulation checks, is as it is."""
    randoms = sorted(_randoms(expr), key=str)
    value = sympy.sympify(value)
    if _parameters(*randoms, expr) or _unfinished(value) or not value.is_finite:
        return value
    try:
        estimate, error = _simulated(expr, randoms)
    except Exception:  # what cannot be simulated is not checked
        return value
    number = complex(sympy.N(value))
    if not math.isfinite(estimate + error) or abs(number - estimate) <= 6 * error + 10 / _DRAWS:
        return value
    raise EvaluationError(
        f"{what} is not computed: SymPy gives {_shown(value)}, and a simulation of it, of "
        f"{_DRAWS} draws, {estimate:.4g}, with a standard error of {error:.2g}",
        notes=["SymPy's sums and integrals over several random variables are not all right"],
    )


def _simulated(expr: Any, randoms: Sequence[RandomSymbol]) -> tuple[float, float]:
    """The mean of an expression of independent random variables over draws of them with a
    fixed seed, by SciPy, and its standard error."""
    import numpy as np

    generator = np.random.default_rng(0)
    points = [Dummy(str(each.symbol), real=True) for each in randoms]
    function = sympy.lambdify(
        points, expr.xreplace(dict(zip(randoms, points, strict=True))), modules=["scipy", "numpy"]
    )
    draws = [_frozen(each).rvs(size=_DRAWS, random_state=generator) for each in randoms]
    with np.errstate(all="ignore"):
        values = np.broadcast_to(np.asarray(function(*draws), dtype=float), (_DRAWS,))
    return float(np.mean(values)), float(np.std(values) / np.sqrt(_DRAWS))


def _by_moments(expr: Any, randoms: Sequence[RandomSymbol]) -> Any | None:
    """The expectation of a polynomial in independent random variables, from the moments of
    each; ``None`` for any other expression."""
    try:
        polynomial = sympy.Poly(sympy.expand(expr), *randoms)
    except sympy.PolynomialError:
        return None
    if any(_randoms(c) for c in polynomial.coeffs()):
        return None
    total = sympy.S.Zero
    for powers, coefficient in polynomial.terms():
        term = coefficient
        for variable, power in zip(randoms, powers, strict=True):
            term *= _moment(variable, power)
        total += term
    return total


def _moment(variable: RandomSymbol, power: int) -> Any:
    """``\\mathbb{E}[X^k]``: 1, the mean, and the variance plus the square of the mean, as the
    distribution declares them; a higher one from its moment-generating function, or by
    SymPy. The second moment of a law is never undefined: it is ∞ where the law has no
    mean, as of Cauchy's, and every moment above it is then ∞ too, or, of an odd power,
    undefined."""
    if power == 0:
        return sympy.S.One
    known = declared(_law(variable))
    if known is not None:
        capability, parameters = known
        mean = sympy.sympify(capability.mean(*parameters))
        if power == 1:
            return mean
        second = sympy.piecewise_fold(capability.variance(*parameters) + mean**2)
        second = second.xreplace({sympy.nan: sympy.oo})  # no mean: E|X| is ∞, and so E[X²]
        if power == 2:
            return second
        if second == sympy.oo:
            return sympy.oo if power % 2 == 0 else sympy.nan
        t = Dummy("t")
        generating_function = capability.generating(t, *parameters)
        if generating_function is not None:
            return sympy.diff(generating_function, t, power).subs(t, 0)
    return _declared_expectation_by_sympy(variable**power, variable)


def _declared_expectation_by_sympy(expr: Any, variable: RandomSymbol) -> Any:
    parameters = _parameters(variable, expr)

    def exactly() -> Any:
        return stats.E(expr)

    def numerically() -> Any:
        return _numeric_expectation(expr, variable)

    return _found(exactly, numerically, parameters, f"the expectation of {_shown(expr)}")


def _by_generating(expr: Any, randoms: Sequence[RandomSymbol]) -> Any | None:
    """``\\mathbb{E}[e^{a + bX}]``, of one random variable: e^a times its moment-generating
    function at b, or its characteristic function at b/i; ``None`` for anything else. The
    moment-generating function a distribution declares is the expectation only where it is
    finite (see :func:`_finite_at`): elsewhere the expectation of a real exponential is ∞."""
    if len(randoms) != 1 or not isinstance(expr, sympy.exp):
        return None
    (variable,) = randoms
    exponent = sympy.expand(expr.args[0])
    slope = exponent.coeff(variable)
    rest = sympy.expand(exponent - slope * variable)
    if _randoms(slope) or _randoms(rest) or slope == 0:
        return None
    law = _law(variable)
    known = declared(law)
    generated = None
    if known is not None:
        capability, parameters = known
        generated = capability.generating(slope, *parameters)
        if generated is not None and not _finite_at(capability, slope, parameters):
            real = slope.is_extended_real and rest.is_extended_real
            return sympy.oo if real else None
    if generated is None and slope.has(sympy.I):
        found = law._characteristic_function(sympy.expand(slope / sympy.I))
        generated = found
    elif generated is None:
        generated = law._moment_generating_function(slope)
    if generated is None:
        return None
    return sympy.exp(rest) * generated


def _finite_at(capability: Distribution, slope: Any, parameters: Sequence[Any]) -> bool:
    """Whether the moment-generating function a distribution declares is finite at
    ``slope``, as that of the exponential distribution is where its real part is below λ;
    always where it is imaginary, as the characteristic function's is. Where what is
    written with symbols does not decide it, it is taken to be, which a remark says, the
    symbols nothing is said of real and the parameters of the law those it allows."""
    general = [s for s in slope.free_symbols if isinstance(s, Symbol) and assumptions._general(s)]
    real = {s: Dummy(s.name, real=True) for s in general}
    part = sympy.re(slope.xreplace(real))
    if part == 0:
        return True
    condition = capability.generating_domain(part, *parameters)
    allowed = {
        s: Dummy(s.name, positive=True)
        for each in parameters
        for s in sympy.sympify(each).free_symbols
        if isinstance(s, Symbol) and assumptions._general(s)
    }
    verdict = assumptions.holds(condition.xreplace(allowed), ())
    if verdict is None:
        said = _shown(condition.xreplace({dummy: s for s, dummy in real.items()}))
        assumptions.remark(
            f"assuming {said}, where the moment-generating function is finite", assuming=[said]
        )
        return True
    return verdict


# -- probabilities --------------------------------------------------------------------------


def _probability(event: Any) -> Any:
    """``\\Pr(event)``."""
    event = sympy.sympify(event)
    if event is sympy.true:
        return sympy.S.One
    if event is sympy.false:
        return sympy.S.Zero
    randoms = _randoms(event)
    if not randoms:
        raise EvaluationError(
            f"{_shown(event)} is no event of a random variable, and it is not decided",
            hint="an event is a relation of random variables, as X > 1",
        )
    unknown = sorted((each for each in randoms if _unknown(each)), key=str)
    if unknown:
        raise EvaluationError(
            f"the probability of {_shown(event)} is not known: the law of "
            f"{', '.join(map(str, unknown))} is not given",
            hint=f"declare it, as {unknown[0]} \\sim \\mathcal{{N}}(0, 1)",
        )
    given = sorted((each for each in randoms if _given(each)), key=str)
    if given:
        return _total(event, given[0], _probability)
    if len(randoms) == 1:
        (variable,) = randoms
        return _of_one(event, variable)
    return _of_several(event, sorted(randoms, key=str))


#: The most values of a random variable an event of several is summed over, and the most
#: events joined by or whose probabilities are put together.
_MOST_VALUES = 1000
_MOST_EVENTS = 4


def _of_several(event: Any, randoms: Sequence[RandomSymbol]) -> Any:
    """The probability of an event of several independent random variables. Events that
    share no variable are independent, and their probabilities multiply; a variable one
    of them gives a value, as X = 1 does beside X + Y = 3, takes it in the others; and an
    event is else summed over the values a discrete variable of it can take where it
    holds, where they are finitely many. One relation none of these takes apart is as
    SymPy finds it; events joined together are never given to SymPy, which multiplies and
    adds their probabilities as if they shared no variable."""
    what = f"the probability of {_shown(event)}"
    if isinstance(event, sympy.Not):
        return _tidy(1 - _probability(event.args[0]))
    if isinstance(event, sympy.Or):
        return _either(event.args, what)
    conjuncts = list(sympy.And.make_args(event))
    groups = _sharing(conjuncts)
    if len(groups) > 1:
        return _tidy(sympy.Mul(*(_probability(sympy.And(*group)) for group in groups)))
    for conjunct in conjuncts if len(conjuncts) > 1 else ():
        fixed = _fixed(conjunct)
        if fixed is None:
            continue
        variable, value = fixed
        if not _discrete(variable):
            return sympy.S.Zero
        others = sympy.And(*(each for each in conjuncts if each is not conjunct))
        rest = _probability(others.xreplace({variable: value}))
        return _tidy(_mass(variable, value) * rest)
    split = _split(event, randoms)
    if split is not None:
        variable, values = split
        return _tidy(
            _over_values(variable, values, lambda v: _probability(event.xreplace({variable: v})))
        )
    if isinstance(event, sympy.Lt | sympy.Le | sympy.Gt | sympy.Ge):
        opposite = sympy.Not(event)  # X + Y ≥ 2 holds wherever X + Y < 2 does not
        if _split(opposite, randoms) is not None:
            return _tidy(1 - _of_several(opposite, randoms))
    parameters = _parameters(*randoms, event)
    holds = sympy.Piecewise((1, event), (0, True))
    if len(conjuncts) == 1:
        return _borne_out(_found(lambda: stats.P(event), None, parameters, what), holds, what)
    if any(_discrete(each) for each in randoms):
        raise EvaluationError(
            f"{what} is not computed: its events share random variables, and the values "
            "they hold at together are not found",
            hint="write the events as one, or give one of the variables a value, as X = 1",
        )
    found = _found(lambda: stats.E(holds), None, parameters, what)  # an integral over where
    return _borne_out(found, holds, what)


def _either(events: Sequence[Any], what: str) -> Any:
    """The probability that one at least of several events holds, by inclusion and
    exclusion: the sum of the probabilities of each, less those of each two together, and
    so on."""
    if len(events) > _MOST_EVENTS:
        raise EvaluationError(
            f"{what}, of {len(events)} events joined by or, is not computed: more than "
            f"{_MOST_EVENTS} are not put together",
        )
    total = sympy.S.Zero
    for size in range(1, len(events) + 1):
        for chosen in itertools.combinations(events, size):
            total += (-1) ** (size + 1) * _probability(sympy.And(*chosen))
    return _tidy(total)


def _sharing(events: Sequence[Any]) -> list[list[Any]]:
    """Events in groups, those of one group sharing random variables, from one to the
    next, and those of two groups none: the groups are independent of each other."""
    groups: list[tuple[set[RandomSymbol], list[Any]]] = []
    for event in events:
        variables = set(_randoms(event))
        joined = [group for group in groups if group[0] & variables]
        kept = [group for group in groups if not group[0] & variables]
        merged = (
            variables.union(*(group[0] for group in joined)),
            [each for group in joined for each in group[1]] + [event],
        )
        groups = [*kept, merged]
    return [group[1] for group in groups]


def _fixed(event: Any) -> tuple[RandomSymbol, Any] | None:
    """The random variable an event gives a value, and the value, as X and 1 of X = 1;
    ``None`` of any other event."""
    if not isinstance(event, sympy.Eq):
        return None
    for variable, value in ((event.lhs, event.rhs), (event.rhs, event.lhs)):
        if isinstance(variable, RandomSymbol) and not _randoms(value):
            return variable, value
    return None


def _split(event: Any, randoms: Sequence[RandomSymbol]) -> tuple[RandomSymbol, list[Any]] | None:
    """A discrete random variable of an event of several, with finitely many values that
    hold every one it can take where the event holds: every value it takes, where they are
    finitely many, or else those a relation of the event leaves it, as X + Y = 3 leaves X
    those from 0 to 3 where Y is at least 0. ``None`` where no variable has such."""
    for variable in randoms:
        if not _discrete(variable):
            continue
        others = [each for each in randoms if each is not variable]
        try:
            values = _taken(variable)
            if values is None:
                found = (_left_by(each, variable, others) for each in sympy.And.make_args(event))
                values = next((each for each in found if each is not None), None)
        except (TypeError, ValueError, AttributeError, NotImplementedError):
            continue  # a law whose values are not known, as of a symbol's trials
        if values is not None and len(values) <= _MOST_VALUES:
            return variable, values
    return None


def _taken(variable: RandomSymbol) -> list[Any] | None:
    """Every value a discrete random variable takes, where they are finitely many."""
    support = _support(_law(variable))
    if isinstance(support, sympy.FiniteSet):
        return sorted(support, key=sympy.default_sort_key)
    return None


def _left_by(
    relation: Any, variable: RandomSymbol, others: Sequence[RandomSymbol]
) -> list[Any] | None:
    """The whole numbers a relation that is linear in the random variables leaves one of
    them, whatever the others take: of a x + r, with r between its least and its greatest
    value over what the others take, compared with 0. More than it can take where the
    relation is strict, never fewer. ``None`` of any other relation, or where they are
    infinitely many."""
    if not isinstance(relation, sympy.Eq | sympy.Lt | sympy.Le | sympy.Gt | sympy.Ge):
        return None
    try:
        linear = sympy.Poly(relation.lhs - relation.rhs, variable, *others)
    except sympy.PolynomialError:
        return None
    if linear.total_degree() > 1 or not all(c.is_number for c in linear.coeffs()):
        return None
    slope = linear.coeff_monomial(variable)
    least = greatest = linear.coeff_monomial(1)
    for other in others:
        weight = linear.coeff_monomial(other)
        if weight == 0:
            continue
        support = _support(_law(other))
        ends = sorted((weight * support.inf, weight * support.sup))
        least, greatest = least + ends[0], greatest + ends[1]
    if slope == 0 or sympy.nan in (least, greatest):
        return None
    # slope x + r holds the relation with 0 for some r from least to greatest
    low, high = -sympy.oo, sympy.oo
    if isinstance(relation, sympy.Eq | sympy.Lt | sympy.Le):
        high = -least  # slope x + least <= 0
    if isinstance(relation, sympy.Eq | sympy.Gt | sympy.Ge):
        low = -greatest  # slope x + greatest >= 0
    low, high = sorted((low / slope, high / slope))
    support = _support(_law(variable))
    first, last = sympy.Max(low, support.inf), sympy.Min(high, support.sup)
    if not (first.is_finite and last.is_finite):
        return None
    return [sympy.Integer(k) for k in range(int(sympy.ceiling(first)), int(sympy.floor(last)) + 1)]


def _of_one(event: Any, variable: RandomSymbol) -> Any:
    """The probability of an event of one random variable: of a value, by its density; else
    from the values the event holds at, intervals of its distribution function, which SymPy
    gives exactly and SciPy numerically; else as SymPy finds it."""
    discrete = _discrete(variable)
    if isinstance(event, sympy.Eq) and variable in (event.lhs, event.rhs):
        value = event.rhs if event.lhs == variable else event.lhs
        if not _randoms(value):
            if not discrete:
                return sympy.S.Zero
            return _tidy(_mass(variable, value))
    parameters = _parameters(variable, event)
    what = f"the probability of {_shown(event)}"
    solved = _solved(event, variable)
    if solved is None:
        return _found(lambda: stats.P(event), None, parameters, what)
    within, values, taken, inside = solved

    def exactly() -> Any:
        found = _on(within, discrete, lambda b: _below(variable, b), _mass_of(variable))
        return _taken_back(found, taken, inside, variable)

    def numerically() -> Any:
        frozen = _frozen(variable)
        if discrete:
            mass: Callable[[Any], Any] = lambda point: float(frozen.pmf(float(point)))  # noqa: E731
        else:
            mass = lambda point: 0.0  # noqa: E731
        found = _on(values, discrete, lambda b: float(frozen.cdf(float(b))), mass)
        return sympy.Float(float(found))

    return _found(exactly, numerically, parameters, what)


def _mass(variable: RandomSymbol, value: Any) -> Any:
    """The probability that a discrete random variable takes ``value``."""
    law = _law(variable)
    value = sympy.sympify(value)
    inside = _support(law).contains(value)
    if inside is sympy.false:
        return sympy.S.Zero
    found = law.pmf(value) if isinstance(law, SingleFiniteDistribution) else law.pdf(value)
    if inside is sympy.true or not value.is_number:
        return found
    return sympy.Piecewise((found, inside), (0, True))


def _mass_of(variable: RandomSymbol) -> Callable[[Any], Any]:
    return lambda value: _mass(variable, value)


def _support(law: Any) -> Any:
    """The values a law takes, as a SymPy set."""
    found = law.set
    if isinstance(found, set | frozenset):  # a finite law's values
        return sympy.FiniteSet(*found)
    return found


def _below(variable: RandomSymbol, bound: Any) -> Any:
    """The probability that a random variable is at most ``bound``, a value it takes or an
    end of the values it takes: its distribution function, written for a symbol and then
    taken at the bound, as SymPy's own is wrong beyond the values a law takes; of a discrete
    one, the sum of its masses."""
    law = _law(variable)
    support = _support(law)
    low, high = support.inf, support.sup
    if _decided(bound - high, "nonnegative"):
        return sympy.S.One
    if isinstance(law, SingleFiniteDistribution):
        return sum(
            (sympy.Piecewise((mass, sympy.Le(value, bound)), (0, True))
             for value, mass in law.dict.items()),
            sympy.S.Zero,
        )  # fmt: skip
    if isinstance(law, SingleContinuousDistribution):
        if _decided(low - bound, "nonnegative"):
            return sympy.S.Zero
        point = Dummy("x", real=True)
        return law.cdf(point).subs(point, bound)
    k = Dummy("k", integer=True)
    return sympy.summation(law.pdf(k), (k, low, sympy.floor(bound)))


def _decided(value: Any, sign: str) -> bool:
    """Whether a difference, maybe infinite, is known to have ``sign``: ``nonnegative`` or
    ``positive``."""
    return bool(getattr(sympy.sympify(value), f"is_extended_{sign}"))


def _values(event: Any, variable: RandomSymbol) -> Any | None:
    """The set of values of ``variable`` at which ``event`` holds, as a union of intervals
    and points of the real line; ``None`` where SymPy cannot say."""
    point = Dummy("x", real=True)
    try:
        found = _set_of(event.xreplace({variable: point}), point)
    except (NotImplementedError, ValueError, TypeError):
        return None
    return found if _intervals(found) else None


def _set_of(event: Any, point: Dummy) -> Any:
    """The values of ``point`` at which ``event`` holds: of a relation, as SymPy solves it;
    of events joined by and, or and not, their intersection, union and complement."""
    if isinstance(event, sympy.And):
        parts = [_set_of(each, point) for each in event.args]
        met = parts[0]
        for part in parts[1:]:
            met = _meet(met, part)
        return met
    if isinstance(event, sympy.Or):
        return sympy.Union(*(_set_of(each, point) for each in event.args))
    if isinstance(event, sympy.Not):
        return sympy.Complement(sympy.S.Reals, _set_of(event.args[0], point))
    return sympy.solveset(event, point, sympy.S.Reals)


def _meet(one: Any, other: Any, inside: list[Any] | None = None) -> Any:
    """The intersection of two sets of intervals and points, whose ends what is assumed of
    their symbols orders, as (s, ∞) and (s + t, ∞) meet in (s + t, ∞) where t > 0; SymPy's
    own intersection where it does not. Given a list, ``inside``, an end of ``one`` that
    nothing orders against one of ``other`` is taken within ``other``, and the relation that
    says so added to it."""
    if isinstance(one, sympy.Union):
        return sympy.Union(*(_meet(part, other, inside) for part in one.args))
    if isinstance(other, sympy.Union):
        return sympy.Union(*(_meet(one, part, inside) for part in other.args))
    if isinstance(one, sympy.Interval) and isinstance(other, sympy.Interval):
        start, left_open = _end(
            one.start, one.left_open, other.start, other.left_open, "max", inside
        )
        end, right_open = _end(one.end, one.right_open, other.end, other.right_open, "min", inside)
        if start is not None and end is not None:
            if inside is not None and not (_decided(end - start, "nonnegative") or start == end):
                inside.append(sympy.Le(start, end))  # an interval taken so is not empty
            return sympy.Interval(start, end, left_open, right_open)
    return sympy.Intersection(one, other)


def _end(
    one: Any,
    one_open: bool,
    other: Any,
    other_open: bool,
    which: str,
    inside: list[Any] | None = None,
) -> tuple[Any | None, bool]:
    """The greater (``max``) or the lesser (``min``) of two ends of intervals, with whether
    it is open; ``None`` where nothing assumed orders them, unless ``inside`` is given, when
    ``one`` is taken, and the relation that takes it so added to ``inside``."""
    if one == other:
        return one, one_open or other_open
    later = _decided(one - other, "positive")
    earlier = _decided(other - one, "positive")
    if not (later or earlier):
        if inside is None:
            return None, False
        inside.append(sympy.Ge(one, other) if which == "max" else sympy.Le(one, other))
        return one, one_open
    first = later if which == "max" else earlier
    return (one, one_open) if first else (other, other_open)


def _solved(
    event: Any, variable: RandomSymbol
) -> tuple[Any, Any, dict[Any, Any], list[Any]] | None:
    """The values a random variable takes at which ``event`` holds, as intervals within those
    it takes, and as the event gives them. The parameters the program says nothing of are
    taken real where that orders the ends, as an inequality takes them, each as a real symbol;
    and where that does not either, each end is taken within the values the variable takes,
    as a book gives P(U < c) = c for 0 ≤ c ≤ 1, with the relations that take them so; ``None``
    where the values are not found."""
    general = sorted(
        (s for s in event.free_symbols if isinstance(s, Symbol) and assumptions._general(s)),
        key=str,
    )
    real = {s: Dummy(s.name, real=True) for s in general}
    for taken, inside in (({}, None), (real, None), (real, [])):
        if taken is real and not real and inside is None:
            continue
        values = _values(event.xreplace(taken), variable)
        within = _within(values, variable, inside) if values is not None else None
        if within is not None:
            return within, values, taken, inside or []
    return None


def _taken_back(value: Any, taken: dict[Any, Any], inside: list[Any], variable: Any) -> Any:
    """A value found with parameters taken real, written with the program's own symbols
    again; where ends were taken within the values a random variable takes, the remark that
    says so."""
    back = {each: name for name, each in taken.items()}
    if inside:
        said = [_shown(relation.xreplace(back)) for relation in inside]
        names = sorted({_shown(s) for r in inside for s in r.xreplace(back).free_symbols})
        verb = "is" if len(names) == 1 else "are"
        assumptions.remark(
            f"assuming {assumptions.listed(said)}, within the values {_shown(variable)} takes; "
            f"the program does not say where {assumptions.listed(names)} {verb}",
            assuming=said,
        )
        facts = [
            sympy.Q.ge(r.lhs, r.rhs) if isinstance(r, sympy.Ge) else sympy.Q.le(r.lhs, r.rhs)
            for r in inside
            if isinstance(r, sympy.Ge | sympy.Le)
        ]
        value = sympy.refine(value, sympy.And(*facts))
    return sympy.sympify(value).xreplace(back)


def _within(values: Any, variable: RandomSymbol, inside: list[Any] | None = None) -> Any | None:
    """The values in ``values`` a random variable takes, as intervals within the least and
    greatest of them; ``None`` where nothing assumed orders their ends, unless ``inside`` is
    given (see :func:`_meet`)."""
    support = _support(_law(variable))
    found = _meet(values, sympy.Interval(support.inf, support.sup), inside)
    return found if _intervals(found) else None


def _intervals(found: Any) -> bool:
    """Whether a set is intervals and points of the real line, which probabilities are
    found of, rather than what SymPy leaves undecided, as the intersection of (s, ∞) and
    (s + t, ∞) where nothing says which end comes first."""
    if isinstance(found, sympy.Union):
        return all(_intervals(part) for part in found.args)
    return isinstance(found, sympy.Interval | sympy.FiniteSet) or found is sympy.S.EmptySet


def _on(
    values: Any, discrete: bool, below: Callable[[Any], Any], mass: Callable[[Any], Any]
) -> Any:
    """The probability that a random variable is in ``values``, from the probability that it
    is at most each end, ``below``, and that it takes each point, ``mass``."""
    if isinstance(values, sympy.Union):
        return sum((_on(part, discrete, below, mass) for part in values.args), sympy.S.Zero)
    if isinstance(values, sympy.FiniteSet):
        return sum((mass(point) for point in values), sympy.S.Zero)
    if values is sympy.S.EmptySet:
        return sympy.S.Zero
    if not isinstance(values, sympy.Interval):
        raise EvaluationError(f"the probability of being in {values} is not computed")
    low, high = values.start, values.end
    upper = sympy.S.One if high is sympy.oo else below(high)
    lower = sympy.S.Zero if low is -sympy.oo else below(low)
    if discrete:
        if high is not sympy.oo and values.right_open and sympy.sympify(high).is_integer:
            upper -= mass(high)
        if low is not -sympy.oo and not values.left_open and sympy.sympify(low).is_integer:
            lower -= mass(low)
    return upper - lower


# -- variances and covariances -------------------------------------------------------------------


def _variance(expr: Any) -> Any:
    """``\\operatorname{Var}(expr)``."""
    expr = sympy.sympify(expr)
    if expr.has(Member):
        return _variance_of_sample(expr)
    randoms = _randoms(expr)
    if not randoms:
        return sympy.S.Zero
    if any(_unknown(each) for each in randoms):
        return _symbolic(stats.Variance(expr))
    if not any(_given(each) for each in randoms):
        linear = _linear(expr, randoms)
        if linear is not None:
            return _tidy(linear)
    return _tidy(_expect(sympy.expand(expr**2)) - _expect(expr) ** 2)


def _linear(expr: Any, randoms: set[RandomSymbol]) -> Any | None:
    """The variance of a sum of independent random variables, each times a constant, as
    their declared variances give it; ``None`` for anything else."""
    try:
        polynomial = sympy.Poly(sympy.expand(expr), *sorted(randoms, key=str))
    except sympy.PolynomialError:
        return None
    if polynomial.total_degree() != 1 or any(_randoms(c) for c in polynomial.coeffs()):
        return None
    total = sympy.S.Zero
    for variable in randoms:
        known = declared(_law(variable))
        if known is None:
            return None
        capability, parameters = known
        total += polynomial.coeff_monomial(variable) ** 2 * capability.variance(*parameters)
    return total


def _covariance(first: Any, second: Any) -> Any:
    """``\\operatorname{Cov}(first, second)``."""
    randoms = _randoms(first) | _randoms(second)
    if any(_unknown(each) for each in randoms):
        return _symbolic(stats.Covariance(first, second))
    product = _expect(sympy.expand(first * second))
    return _tidy(product - _expect(first) * _expect(second))


# -- exactly, or numerically -------------------------------------------------------------------


def _parameters(*exprs: Any) -> set[Any]:
    """The symbols other than random variables that ``exprs``, and the laws of the random
    variables in them, are written with."""
    found: set[Any] = set()
    for expr in exprs:
        expr = sympy.sympify(expr)
        for each in _randoms(expr):
            space = each.pspace
            if isinstance(space, Unknown):
                continue
            law = space.args[1] if isinstance(space, Given) else _law(each)
            found |= {s for s in law.free_symbols if not isinstance(s, RandomSymbol)}
        found |= {s for s in expr.free_symbols if not isinstance(s, RandomSymbol)}
    return found


def _found(
    exactly: Callable[[], Any],
    numerically: Callable[[], Any] | None,
    parameters: set[Any],
    what: str,
) -> Any:
    """What ``exactly`` gives, by SymPy, within the budget; else, where there are no
    parameters, what ``numerically`` gives, by SciPy, which a note says."""
    from relativisticpy.runtime import core

    failure: BaseException | None = None
    value: Any = None
    try:
        with core._budget(BUDGET):
            value = exactly()
        if not _unfinished(value):
            return _tidy(value)
        failure = None
    except core._OverBudget as error:
        failure = error
    except RelativisticPyError:
        raise
    except Exception as error:
        failure = error
    if numerically is not None and not parameters:
        value = numerically()
        if isinstance(value, sympy.Float) and not math.isfinite(float(value)):
            raise EvaluationError(f"{what} is not finite")
        assumptions.remark(
            f"{what} is found numerically, by SciPy, as SymPy finds no closed form of it "
            f"within {BUDGET:g} s"
        )
        return value
    if failure is None:
        return value  # what SymPy leaves as written, as an integral it cannot carry out
    why = "within the time a closed form is given" if _over(failure) else "in closed form"
    named = ", ".join(sorted(map(str, parameters)))
    raise EvaluationError(
        f"{what} is not found {why}",
        hint=f"give {named} values, to compute it numerically" if parameters else None,
    )


def _over(failure: BaseException) -> bool:
    from relativisticpy.runtime import core

    return isinstance(failure, core._OverBudget)


def _unfinished(value: Any) -> bool:
    """Whether SymPy left what it was asked to compute undone, in its answer."""
    value = sympy.sympify(value)
    undone = (sympy.Integral, stats.Expectation, stats.Probability)
    return bool(value.atoms(*undone)) or (
        bool(value.atoms(sympy.Sum)) and not value.free_symbols - _randoms(value)
    )


def _tidy(value: Any) -> Any:
    """A value, in its shortest form, simplified where that is shorter, within the budget: the
    complementary error function written with the error function, so that the two meet."""
    from relativisticpy.runtime import core

    value = sympy.sympify(value)
    if value.is_number and value.atoms(sympy.Float):
        return sympy.N(value)
    forms = [value]
    try:
        with core._budget(BUDGET):
            forms.append(sympy.simplify(value))
            if value.has(sympy.erfc):
                forms.append(sympy.simplify(value.rewrite(sympy.erf)))
    except (Exception, core._OverBudget):
        pass
    return min(forms, key=sympy.count_ops)


def _frozen(variable: RandomSymbol) -> Any:
    """SciPy's law of a random variable whose parameters are numbers."""
    known = declared(_law(variable))
    if known is None:
        raise EvaluationError(f"the law of {variable} has no counterpart in SciPy")
    capability, parameters = known
    return capability.frozen(*parameters)


def _numeric_expectation(expr: Any, variable: RandomSymbol) -> Any:
    """The expectation of a function of one random variable, by SciPy; refused, saying so,
    where SciPy warns that its integral does not converge, whose value is then no
    expectation."""
    import warnings

    from scipy.integrate import IntegrationWarning

    point = Dummy("x", real=True)
    function = sympy.lambdify(point, expr.xreplace({variable: point}), modules=["scipy", "numpy"])
    frozen = _frozen(variable)
    imaginary = sympy.sympify(expr).has(sympy.I)
    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        try:
            found = complex(frozen.expect(lambda x: complex(function(x)).real))
            if imaginary:
                found += 1j * complex(frozen.expect(lambda x: complex(function(x)).imag)).real
        except IntegrationWarning:
            raise EvaluationError(
                f"the expectation of {_shown(expr)} is not found: SymPy finds no closed form "
                "of it, and SciPy's integral of it does not converge",
            ) from None
    if imaginary:
        return sympy.Float(found.real) + sympy.I * sympy.Float(found.imag)
    return sympy.Float(found.real)


def _shown(value: Any) -> str:
    from relativisticpy import display

    try:
        return display.unicode(value)
    except Exception:
        return str(value)


# -- the conformance checks ---------------------------------------------------------------------


def form_failures(capability: Distribution) -> list[str]:
    """Each written form of a distribution parses, written with variables for its parameters
    in order, and reads, after ``\\sim``, as it and no other."""
    from relativisticpy.reading import probability

    failures = []
    count = len(capability.parameters)
    for form, written in (("latex", capability.latex), ("unicode", capability.unicode)):
        for each in written:
            found = probability.form(each)
            if found is None:
                failures.append(
                    f"{capability.name}: the {form} form {each} does not parse as a distribution "
                    r"is written, as \mathcal{N}(\mu, \sigma^2) or \chi^2_k is"
                )
                continue
            if len(found) != count:
                failures.append(
                    f"{capability.name}: the form {each} writes {len(found)} parameter(s), and "
                    f"exact takes {count}; write each, in order"
                )
                continue
            expr = probability.parsed(each)
            law = probability.law(expr) if expr is not None else None
            if law is None or law.capability is not capability:
                other = law.capability.name if law is not None else "nothing"
                failures.append(
                    f"{capability.name}: the form {each} reads, after \\sim, as {other}; write it "
                    "so that it is this distribution's alone"
                )
    return failures


#: How long SymPy may take over each value a check holds it to, in seconds: more than a
#: program gives it, as a check SymPy is too slow for is a check failed, not passed.
CHECK_BUDGET = 30.0


def _by_sympy(
    compute: Callable[[], Any], what: str, capability: Distribution, point: Sequence[Any]
) -> tuple[Any, list[str]]:
    """What SymPy computes of a distribution at a point it is checked at, as a number; or
    why there is none, as a failure of the check: SymPy fails, takes longer than checks give
    it, or gives no number. SciPy is not asked instead: the check is of SymPy's law."""
    from relativisticpy.runtime import core

    where = f"{capability.name}{_at_point(capability, point)}"
    try:
        with core._budget(CHECK_BUDGET):
            found = sympy.sympify(compute())
            number = complex(sympy.N(found, 30))
    except core._OverBudget:
        return None, [
            f"{where}: SymPy does not compute {what} within {CHECK_BUDGET:g} s; fix exact, or "
            "declare points it computes at"
        ]
    except Exception as error:
        return None, [
            f"{where}: SymPy does not compute {what} ({type(error).__name__}: {error}); fix exact"
        ]
    if number != number:  # nan
        return None, [f"{where}: SymPy gives {what} as {found}, no number; fix exact"]
    return found, []


def density_checks(capability: Capability) -> list[str]:
    """At each of its points, the density of SymPy's law integrates, or sums, to 1, by
    SymPy: where SymPy does not compute it, the check fails."""
    assert isinstance(capability, Distribution)
    failures = []
    for point in capability.points:
        x = Dummy("x", integer=True) if capability.discrete else Dummy("x", real=True)
        total, failed = _by_sympy(
            functools.partial(_whole_at, capability, point, x),
            "what its density " + ("sums" if capability.discrete else "integrates") + " to",
            capability,
            point,
        )
        failures += failed
        if total is not None and abs(complex(sympy.N(total)) - 1) > AGREEMENT:
            failures.append(
                f"{capability.name}{_at_point(capability, point)}: its density "
                f"{'sums' if capability.discrete else 'integrates'} to {sympy.N(total, 12)}, "
                "not 1; fix exact, or declare points where its parameters are allowed"
            )
    return failures


def _checked_variable(capability: Distribution, point: Sequence[Any]) -> RandomSymbol:
    """A random variable of SymPy's law of a distribution, at a point it is checked at."""
    return _space(Symbol("X"), capability.exact(*point)).value


def _whole_at(capability: Distribution, point: Sequence[Any], x: Any) -> Any:
    return _whole(capability.exact(*point), x)


def _whole(law: Any, x: Any) -> Any:
    """The density of ``law`` integrated, or summed, over its values, by SymPy."""
    support = law.set
    if isinstance(law, SingleContinuousDistribution):
        low, high = support.inf, support.sup
        return sympy.integrate(law.pdf(x), (x, low, high))
    if isinstance(law, SingleFiniteDistribution):
        return sum(law.dict.values(), sympy.S.Zero)
    low, high = support.inf, support.sup
    return sympy.summation(law.pdf(x), (x, low, high))


def moment_checks(capability: Capability) -> list[str]:
    """At each of its points, its mean and variance are those it declares: by SymPy, which
    fails the check where it does not compute them, and by SciPy."""
    assert isinstance(capability, Distribution)
    failures = []
    for point in capability.points:
        declared_mean = capability.mean(*point)
        declared_variance = capability.variance(*point)
        for what, moment, expected in (
            ("mean", _mean_of_the_law, declared_mean),
            ("variance", _variance_of_the_law, declared_variance),
        ):
            value, failed = _by_sympy(
                functools.partial(_of_the_law, moment, capability, point),
                f"its {what}",
                capability,
                point,
            )
            failures += failed
            if value is not None and not _agree(value, expected):
                failures.append(
                    f"{capability.name}{_at_point(capability, point)}: SymPy's {what} is "
                    f"{sympy.N(value, 12)}, and its declared {what} "
                    f"{sympy.N(expected, 12)} ({capability.moments}); fix {what}"
                )
        frozen = capability.frozen(*point)
        for what, value, expected in (
            ("mean", frozen.mean(), declared_mean),
            ("variance", frozen.var(), declared_variance),
        ):
            if not _agree(value, expected):
                failures.append(
                    f"{capability.name}{_at_point(capability, point)}: SciPy's {what} is "
                    f"{float(value):.12g}, and its declared {what} {sympy.N(expected, 12)} "
                    f"({capability.moments}); fix {what}, or numeric"
                )
    return failures


def _mean_of_the_law(variable: RandomSymbol) -> Any:
    """The mean of SymPy's law, by SymPy: as its statistics give it, or, where they ask for
    a moment-generating function the law has none of, as Student's t has none, from its
    density."""
    try:
        return stats.E(variable)
    except NotImplementedError:
        return _from_the_density(variable, 1)


def _variance_of_the_law(variable: RandomSymbol) -> Any:
    """The variance of SymPy's law, by SymPy (see :func:`_mean_of_the_law`)."""
    try:
        return stats.variance(variable)
    except NotImplementedError:
        return _from_the_density(variable, 2) - _from_the_density(variable, 1) ** 2


def _from_the_density(variable: RandomSymbol, power: int) -> Any:
    """A moment of a continuous law, the integral of x^k times its density."""
    law = _law(variable)
    if not isinstance(law, SingleContinuousDistribution):
        raise NotImplementedError(f"SymPy gives no moment of {law}")
    x = Dummy("x", real=True)
    return sympy.integrate(x**power * law.pdf(x), (x, law.set.inf, law.set.sup))


def agreement(capability: Capability) -> list[str]:
    """At each of its points, the density and the distribution function SymPy's law gives,
    as the language computes them exactly, agree with SciPy's, at the quartiles of the law.
    Where SymPy does not compute one, the check fails: SciPy's value is not put for it."""
    assert isinstance(capability, Distribution)
    failures = []
    for point in capability.points:
        frozen = capability.frozen(*point)
        for value in _quartiles(frozen):
            pdf = frozen.pmf(value) if capability.discrete else frozen.pdf(value)
            at = (
                sympy.Integer(round(value))
                if capability.discrete
                else sympy.Rational(value).limit_denominator(10**9)
            )

            computed: list[tuple[str, Any, Callable[[], Any]]] = [
                (
                    "density",
                    pdf,
                    functools.partial(_of_the_law, _exact_density, capability, point, at),
                ),
                (
                    "distribution function",
                    frozen.cdf(value),
                    functools.partial(_of_the_law, _below, capability, point, at),
                ),
            ]
            for what, numeric, compute in computed:
                found, failed = _by_sympy(compute, f"its {what} at {value:g}", capability, point)
                failures += failed
                if found is not None and not _agree(found, numeric):
                    failures.append(
                        f"{capability.name}{_at_point(capability, point)}: its {what} at "
                        f"{value:g} is {sympy.N(found, 12)} by SymPy and {float(numeric):.12g} "
                        "by SciPy; fix exact or numeric"
                    )
    return failures


def _of_the_law(
    compute: Callable[..., Any], capability: Distribution, point: Sequence[Any], *at: Any
) -> Any:
    """What ``compute`` gives of a random variable of SymPy's law of a distribution, at a
    point it is checked at."""
    return compute(_checked_variable(capability, point), *at)


def _exact_density(variable: RandomSymbol, at: Any) -> Any:
    """The density of a random variable at a value, or its mass there, by SymPy alone."""
    found = stats.density(variable)(at)
    return found.doit() if isinstance(found, Basic) else found


def _quartiles(frozen: Any) -> list[float]:
    return sorted({float(frozen.ppf(q)) for q in (0.25, 0.5, 0.75)})


def _agree(value: Any, expected: Any) -> bool:
    try:
        a = complex(sympy.N(sympy.sympify(value), 30))
        b = complex(sympy.N(sympy.sympify(expected), 30))
    except (TypeError, ValueError):
        return False
    return near(a, b, AGREEMENT)


def _at_point(capability: Distribution, point: Sequence[Any]) -> str:
    named = ", ".join(f"{n}={v}" for n, v in zip(capability.parameters, point, strict=False))
    return f" at ({named})"
