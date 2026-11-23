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

    def test_an_integral_scipy_says_diverges_is_no_expectation(self):
        # the mean of |T| of Cauchy's law is infinite: SciPy's quadrature gives a number
        # of it, with a warning
        from sympy.stats.crv_types import StudentTDistribution

        variable = distributions.variable(sympy.Symbol("T"), StudentTDistribution(1))
        with pytest.raises(EvaluationError, match="SciPy's integral of it does not converge"):
            distributions._numeric_expectation(sympy.Abs(variable), variable)
