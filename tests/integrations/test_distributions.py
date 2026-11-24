r"""Distributions: the laws random variables are declared with, a kind of capability whose
forms are read after ``\sim``, held by the conformance suite to integrate to 1 and to have
the mean and variance they declare, by SymPy and by SciPy; and what is computed of random
variables, exactly where SymPy can and numerically where it cannot, with a note."""

from __future__ import annotations

import pytest
import sympy
from sympy.stats.crv_types import ExponentialDistribution, NormalDistribution
from sympy.stats.frv_types import BinomialDistribution

from relativisticpy import Workbook, integrations
from relativisticpy.errors import EvaluationError
from relativisticpy.integrations import Example, distribution
from relativisticpy.integrations import _distributions as distributions

lam, t = sympy.symbols("lambda t")


def laplace(*, variance: object = None, latex: str = r"\operatorname{Laplace}(\mu, b)"):
    """The Laplace distribution, declared as an integration would declare it; with a variance
    of its own, as a mistake would give it."""

    @distribution(
        "Laplace",
        latex=latex,
        summary="the Laplace distribution, of mean μ and scale b",
        requires="scipy",
        moments="Johnson, Kotz and Balakrishnan (1995), chapter 24",
        examples=[
            Example(
                "X \\sim \\operatorname{Laplace}(0, 1)\n\\operatorname{Var}(X)",
                2,
                "scipy.stats.laplace.var()",
            ),
            Example(
                "X \\sim \\operatorname{Laplace}(0, 1)\n\\Pr(X > 1)",
                0.18393972058572117,
                "scipy.stats.laplace.sf(1)",
            ),
        ],
        points=[(0, 1), (1, 2)],
    )
    class Laplace:
        def exact(self, mu, b):
            from sympy.stats.crv_types import LaplaceDistribution

            return LaplaceDistribution(mu, b)

        def numeric(self, xp, mu, b):
            from scipy.stats import laplace

            return laplace(mu, b)

        def mean(self, mu, b):
            return mu

        def variance(self, mu, b):
            return 2 * b**2 if variance is None else variance

    return Laplace


class TestTheKind:
    def test_a_distribution_written_out_passes_the_suite(self):
        found = integrations.check(laplace())
        assert found == {
            "declaration": [],
            "examples": [],
            "agreement": [],
            "errors": [],
            "reference": found["reference"],  # an integration's own is in the reference
            "density": [],
            "moments": [],
        }

    def test_a_variance_that_is_not_the_laws_is_said(self):
        (failure, *_) = integrations.check(laplace(variance=1), ["moments"])["moments"]
        assert failure.startswith("Laplace at (mu=0, b=1): SymPy's variance is 2.0")
        assert "fix variance" in failure

    def test_a_form_that_reads_as_another_distribution_is_said(self):
        failures = integrations.check(laplace(latex=r"\mathcal{N}(\mu, b^2)"), ["declaration"])
        assert failures["declaration"] == [
            r"Laplace: the form \mathcal{N}(\mu, b^2) reads, after \sim, as Normal; write it so "
            "that it is this distribution's alone"
        ]

    def test_a_law_is_known_by_its_distribution(self):
        capability, parameters = distributions.declared(NormalDistribution(1, 2))
        assert (capability.name, parameters) == ("Normal", (1, 2))
        capability, parameters = distributions.declared(BinomialDistribution(10, 0.5, 1, 0))
        assert (capability.name, parameters) == ("Binomial", (10, 0.5))

    def test_its_forms_are_read_after_sim_alone(self, wb):
        wb.run(r"X \sim \Gamma(2, 3)")
        assert wb.expr(r"\mathbb{E}[X]") == 6
        assert wb.expr(r"\Gamma(5)") == 24  # Euler's gamma, as before

    def test_a_law_sympy_computes_nothing_of_fails_each_check(self):
        # SciPy's law is not put for SymPy's: a check of the exact side that passes on the
        # numeric one alone checks nothing
        from sympy.stats.crv import SingleContinuousDistribution

        class Broken(SingleContinuousDistribution):
            _argnames = ("mu",)
            set = sympy.Interval(-sympy.oo, sympy.oo)

            def pdf(self, x):
                raise NotImplementedError("no density")

        @distribution(
            "Brokenlaw",
            latex=[r"\operatorname{Brk}(\mu)"],
            summary="a law SymPy computes nothing of",
            requires="scipy",
            moments="none",
            examples=[Example("1", 1, "standard"), Example("2", 2, "standard")],
            points=[(0,), (1,)],
        )
        class BrokenLaw:
            def exact(self, mu):
                return Broken(sympy.sympify(mu))

            def numeric(self, xp, mu):
                from scipy.stats import norm

                return norm(mu, 1)

            def mean(self, mu):
                return mu

            def variance(self, mu):
                return 1

        found = integrations.check(BrokenLaw, ["agreement", "density", "moments"])
        assert found["density"] == [
            f"Brokenlaw at (mu={mu}): SymPy does not compute what its density integrates to "
            "(NotImplementedError: no density); fix exact"
            for mu in (0, 1)
        ]
        assert len(found["agreement"]) == 12  # the density and the distribution function
        assert found["agreement"][0] == (
            "Brokenlaw at (mu=0): SymPy does not compute its density at -0.67449 "
            "(NotImplementedError: no density); fix exact"
        )
        assert [failure.split(": ")[1].split(" (")[0] for failure in found["moments"]] == [
            "SymPy does not compute its mean",
            "SymPy does not compute its variance",
        ] * 2

    def test_a_check_sympy_is_too_slow_for_fails(self, monkeypatch):
        import time

        monkeypatch.setattr(distributions, "CHECK_BUDGET", 0.05)
        monkeypatch.setattr(distributions, "_whole", lambda law, x: time.sleep(5))
        (failure, *_) = integrations.check(laplace(), ["density"])["density"]
        assert failure == (
            "Laplace at (mu=0, b=1): SymPy does not compute what its density integrates to "
            "within 0.05 s; fix exact, or declare points it computes at"
        )


class TestComputing:
    def test_a_probability_of_one_variable_is_from_its_distribution_function(self):
        variable = distributions.variable(sympy.Symbol("T"), ExponentialDistribution(lam))
        found = distributions.probability(sympy.And(variable > 1, variable < 3))
        assert sympy.simplify(found - (sympy.exp(-lam) - sympy.exp(-3 * lam))) == 0

    def test_where_sympy_finds_no_closed_form_scipy_computes_it(self, monkeypatch):
        from relativisticpy import assumptions

        def fails(*args, **kwargs):
            raise NotImplementedError

        monkeypatch.setattr(distributions.stats, "E", fails)
        variable = distributions.variable(sympy.Symbol("X"), NormalDistribution(0, 1))
        with assumptions.remarks() as remarks:
            found = distributions.expectation(sympy.cos(variable**3))
        assert float(found) == pytest.approx(0.5965649688262752)  # scipy.stats.norm.expect
        assert "numerically, by SciPy" in remarks[0].note

    def test_with_parameters_it_is_refused_saying_what_to_give(self, monkeypatch):
        def fails(*args, **kwargs):
            raise NotImplementedError

        monkeypatch.setattr(distributions.stats, "E", fails)
        variable = distributions.variable(sympy.Symbol("X"), NormalDistribution(t, 1))
        with pytest.raises(integrations._discovery.EvaluationError, match="not found") as refused:
            distributions.expectation(sympy.cos(variable**3))
        assert refused.value.hint == "give t values, to compute it numerically"

    def test_a_law_given_for_some_values_only_is_refused(self):
        with Workbook() as wb:
            wb.run("Y \\sim \\operatorname{Poisson}(1)\nX \\mid Y = 0 \\sim \\mathcal{N}(0, 1)")
            with pytest.raises(integrations._discovery.EvaluationError, match="not 1"):
                wb.expr(r"\mathbb{E}[X]")

    def test_a_statistic_of_a_list_of_numbers_is_refused_saying_how_data_writes_it(self):
        # it failed inside RelativisticPy: a list is data, whose statistics the language
        # writes over an index, and whose mean it does not give unasked
        with Workbook() as wb:
            wb.run("x := [1, 2, 3, 6]")
            for statistic in (
                r"\operatorname{Var}(x)",
                r"\operatorname{Cov}(x, x)",
                "\\mathbb{E}[2x]",
            ):
                with pytest.raises(EvaluationError, match="a list, which holds no random"):
                    wb.expr(statistic)
            with pytest.raises(EvaluationError, match="a list, which holds no random") as no:
                wb.expr(r"\mathbb{E}[x]")
        hint = no.value.hint
        assert "x^i := [1, 2, 3, 6]; then \\mathbb{E}_i[x^i] is the mean" in hint
        with Workbook() as wb:  # what the hint writes gives the mean, (1 + 2 + 3 + 6)/4
            wb.run("\\mathcal{N} := \\mathbb{R}^{4}\ni \\in \\mathcal{N}\nx^i := [1, 2, 3, 6]")
            assert wb.expr(r"\mathbb{E}_i[x^i]") == 3

    def test_a_statistic_of_a_vector_of_random_variables_is_taken_of_each(self):
        with Workbook() as wb:
            wb.run("X \\sim \\mathcal{N}(0, 1)\nY \\sim \\mathcal{N}(1, 1)")
            with pytest.raises(EvaluationError, match="a list of random variables") as no:
                wb.expr(r"\mathbb{E}[(X, Y)]")
            assert no.value.hint == "take it of each, as (\\mathbb{E}[X], \\mathbb{E}[Y])"
            assert wb.expr(r"(\mathbb{E}[X], \mathbb{E}[Y])") == (0, 1)
            with pytest.raises(EvaluationError, match="covariance here is between two"):
                wb.expr(r"\operatorname{Cov}(X [1, 2], Y)")

    def test_a_law_of_several_values_is_refused(self):
        # the expectation failed inside SymPy, which cannot hash the list of its law
        with Workbook() as wb:
            wb.run("X \\sim \\mathcal{N}([0, 1], 1)")
            with pytest.raises(EvaluationError, match=r"law is one value, and \[0, 1\] is several"):
                wb.expr(r"\mathbb{E}[X]")

    def test_a_function_of_a_random_variable_at_several_values_is_taken_at_each(self):
        # each failed inside RelativisticPy, given a list where it takes a number
        with Workbook() as wb:
            wb.run("X \\sim \\mathcal{N}(0, 1)")
            for written in ("f_X([1, 2])", "F_X([1, 2])", "M_X([1, 2])"):
                with pytest.raises(EvaluationError, match=r"at one value, and \[1, 2\] is") as no:
                    wb.expr(written)
            assert no.value.hint == "take it at each, as (M_{X}(1), M_{X}(2))"
            # as the hint writes it: M_X(t) = e^{t^2/2}, DeGroot and Schervish (2012), 5.6
            assert wb.expr("(M_{X}(1), M_{X}(2))") == (
                sympy.exp(sympy.Rational(1, 2)),
                sympy.exp(2),
            )

    def test_an_integral_scipy_says_diverges_is_no_expectation(self):
        # the mean of |T| of Cauchy's law is infinite: SciPy's quadrature gives a number
        # of it, with a warning
        from sympy.stats.crv_types import StudentTDistribution

        variable = distributions.variable(sympy.Symbol("T"), StudentTDistribution(1))
        with pytest.raises(EvaluationError, match="SciPy's integral of it does not converge"):
            distributions._numeric_expectation(sympy.Abs(variable), variable)

    def test_a_value_of_sympys_that_a_simulation_contradicts_is_refused(self):
        # SymPy gives the probability that the product of two uniform variables is below
        # 1/2 as 0; it is (1 + ln 2)/2
        from sympy.stats.crv_types import UniformDistribution

        u, v = (
            distributions.variable(sympy.Symbol(name), UniformDistribution(0, 1)) for name in "UV"
        )
        with pytest.raises(EvaluationError, match="SymPy gives 0, and a simulation of it") as no:
            distributions.probability(u * v < sympy.Rational(1, 2))
        assert "0.846" in no.value.message
        assert distributions.probability(sympy.And(u > 0, v > 0)) == 1  # events apart multiply

    def test_the_values_a_relation_leaves_a_variable(self):
        from sympy.stats.drv_types import PoissonDistribution

        x, y = (distributions.variable(sympy.Symbol(name), PoissonDistribution(2)) for name in "XY")
        left = distributions._left_by
        assert left(sympy.Eq(x + y, 3), x, [y]) == [0, 1, 2, 3]
        assert left(sympy.Le(2 * x + y, 4), x, [y]) == [0, 1, 2]
        assert left(sympy.Ge(x + y, 3), x, [y]) is None  # every value: not finitely many
        assert left(sympy.Eq(x - y, 1), x, [y]) is None
        assert left(sympy.Eq(x * y, 4), x, [y]) is None  # no linear relation


class TestSamples:
    """The members of a sample at symbols: which are the same one is decided, or said not
    to be; the language writes them ``X_k``."""

    n, k, i, j = sympy.symbols("n k i j", integer=True, positive=True)
    sample = distributions.sample("X", NormalDistribution(0, 1), n)

    def member(self, index):
        return distributions.member(self.sample, sympy.sympify(index))

    def test_two_at_indices_a_number_apart_are_two_members(self):
        k = self.k
        assert distributions.variance(self.member(k + 1) - self.member(k)) == 2
        assert distributions.covariance(self.member(k), self.member(k + 2)) == 0
        assert distributions.variance(self.member(k) + self.member(k)) == 4  # the same one

    def test_one_at_a_symbol_is_of_the_law(self):
        assert distributions.expectation(self.member(self.n - 1)) == 0
        assert distributions.expectation(self.member(self.k) ** 2) == 1

    def test_one_beyond_the_sample_is_refused(self):
        for index in (self.n + 1, 0):
            with pytest.raises(EvaluationError, match="is beyond the sample, whose members"):
                distributions.variance(self.member(index))
        beyond = sympy.Sum(self.member(self.i), (self.i, 1, self.n + 1))
        with pytest.raises(EvaluationError, match=r"X_\{n \+ 1\} is beyond the sample"):
            distributions.expectation(beyond)

    def test_two_that_may_be_the_same_one_are_refused(self):
        k, j = self.k, self.j
        for one, other in ((self.member(k), self.member(1)), (self.member(k), self.member(j))):
            with pytest.raises(EvaluationError, match="may be the same one, which is not decided"):
                distributions.expectation(one * other)
            with pytest.raises(EvaluationError, match="may be the same one"):
                distributions.covariance(one, other)
        assert distributions.expectation(self.member(1) * self.member(2)) == 0  # numbered

    def test_the_variance_of_a_sum_is_of_independent_terms_alone(self):
        i, n = self.i, self.n
        assert distributions.variance(sympy.Sum(self.member(i), (i, 1, n))) == n
        assert distributions.variance(sympy.Sum(self.member(i), (i, 1, n)) / n) == 1 / n
        # X_{i+1} - X_i and X_{i+2} - X_{i+1} share X_{i+1}: the sum telescopes to
        # X_n - X_1, of variance 2, and its terms' variances add up to 2(n - 1)
        steps = sympy.Sum(self.member(i + 1) - self.member(i), (i, 1, n - 1))
        with pytest.raises(EvaluationError, match="they share members of the sample"):
            distributions.variance(steps)
        squares = sympy.Sum((self.member(i + 1) - self.member(i)) ** 2, (i, 1, n - 1))
        assert distributions.expectation(squares) == 2 * n - 2  # an expectation is linear
