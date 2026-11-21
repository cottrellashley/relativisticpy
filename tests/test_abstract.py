r"""Abstract index notation: canonical forms, the identities of General Relativity,
and solving tensor equations as tensor equations."""

import operator
from functools import reduce

import pytest
from sympy import Rational, Symbol, pi

from relativisticpy.abstract import (
    Algebra,
    IndexEquation,
    IndexExpression,
    canonical,
    latex,
    latex_labels,
    solve,
    substitute,
    trace,
)
from relativisticpy.errors import CompileError

space = Algebra(4)
mu, nu, rho, sigma, a = (space.index(name, False) for name in ("mu", "nu", "rho", "sigma", "a"))
g, delta = space.metric, space.delta
Riemann = space.head("R", 4, "riemann")
Ricci = space.head("R", 2, "ricci")
Einstein = space.head("G", 2, "einstein")
Weyl = space.head("C", 4, "weyl")
T = space.head("T", 2)
u = space.head("u", 1)
R, G, Lam, rho_, p = space.ricci_scalar, Symbol("G"), Symbol("Lambda"), Symbol("rho"), Symbol("p")


def up(index):
    return -index


class TestCanonicalForms:
    @pytest.mark.parametrize(
        "zero",
        [
            Riemann(rho, sigma, mu, nu) + Riemann(sigma, rho, mu, nu),  # antisymmetric pairs
            Riemann(rho, sigma, mu, nu) - Riemann(mu, nu, rho, sigma),  # pair exchange
            Riemann(rho, sigma, mu, nu) + Riemann(rho, mu, nu, sigma) + Riemann(rho, nu, sigma, mu),
            Ricci(mu, nu) - Ricci(nu, mu),
            T(up(a), mu) * u(a) - T(up(rho), mu) * u(rho),  # dummies renamed
        ],
    )
    def test_identities_vanish(self, zero):
        assert canonical(zero, space) == 0

    @pytest.mark.parametrize(
        ("contraction", "expected"),
        [
            (Riemann(up(a), sigma, a, nu), Ricci(sigma, nu)),  # R^a_{sad} = R_{sd}
            (Riemann(up(a), sigma, nu, a), -Ricci(sigma, nu)),
            (Riemann(sigma, up(a), a, nu), -Ricci(sigma, nu)),
            (Riemann(up(a), up(rho), a, rho), R),  # twice contracted
            (Ricci(up(a), a), R),
            (Einstein(up(a), a), -R),  # (1 - n/2) R in four dimensions
            (Weyl(up(a), sigma, a, nu), 0),
            (g(up(a), a), 4),
            (g(up(mu), up(a)) * g(a, nu), delta(up(mu), nu)),
            (T(up(a), a), Symbol("T")),  # the trace of T is T
        ],
    )
    def test_contractions(self, contraction, expected):
        assert canonical(contraction - expected, space) == 0

    def test_the_trace_of_the_metric_is_the_dimension(self):
        three = Algebra(3)
        b = three.index("b", True)
        assert canonical(three.metric(b, -b), three) == 3


class TestSolving:
    efe = (Einstein(mu, nu) + Lam * g(mu, nu), 8 * pi * G * T(mu, nu))

    def test_for_a_tensor(self):
        solution = solve(*self.efe, T(mu, nu), space)
        assert canonical(solution - (Einstein(mu, nu) + Lam * g(mu, nu)) / (8 * pi * G), space) == 0

    def test_for_a_tensor_whose_trace_appears_too(self):
        # The Ricci tensor of Einstein's equations: trace reversal.
        solution = solve(*self.efe, Ricci(mu, nu), space)
        expected = Lam * g(mu, nu) + 8 * pi * G * (
            T(mu, nu) - Rational(1, 2) * Symbol("T") * g(mu, nu)
        )
        assert canonical(solution - expected, space) == 0

    def test_with_the_indices_raised(self):
        solution = solve(*self.efe, T(up(mu), up(nu)), space)
        expected = (Einstein(up(mu), up(nu)) + Lam * g(up(mu), up(nu))) / (8 * pi * G)
        assert canonical(solution - expected, space) == 0

    @pytest.mark.parametrize(
        ("lhs", "message"),
        [
            (T(mu, a) * T(up(a), nu), "linear in it"),
            (T(mu, a) * u(up(a)) * u(nu), "multiplied by other tensors"),
            (Einstein(mu, nu), "does not contain T"),
            (Symbol("T") * T(mu, nu), "multiplied by its own trace"),  # T T_{ab}: not linear
        ],
    )
    def test_what_cannot_be_solved(self, lhs, message):
        with pytest.raises(CompileError, match=message):
            solve(lhs, 0 * g(mu, nu), T(mu, nu), space)


class TestRules:
    def test_a_definition_applies_whatever_the_indices(self):
        b, c = space.index("b", False), space.index("c", False)
        rule = (Einstein(b, c), Ricci(b, c) - R / 2 * g(b, c))
        swapped = substitute(Einstein(nu, mu), *rule, space)
        assert canonical(swapped - (Ricci(mu, nu) - R / 2 * g(mu, nu)), space) == 0
        mixed = substitute(Einstein(up(mu), nu) * u(up(nu)), *rule, space)
        assert (
            canonical(mixed - (Ricci(up(mu), nu) - R / 2 * delta(up(mu), nu)) * u(up(nu)), space)
            == 0
        )

    def test_a_rule_with_a_dummy_index(self):
        # The rule's dummy cannot meet the expression's, nor take a free index's name.
        A, B, C = (space.head(name, 2) for name in "ABC")
        b, c, d, e = (space.index(name, False) for name in "bcde")
        rule = (A(c, d), B(c, e) * C(up(e), d))
        traced = substitute(A(a, b) * u(up(a)) * u(up(b)), *rule, space)
        expected = B(a, rho) * C(up(rho), b) * u(up(a)) * u(up(b))
        assert canonical(traced - expected, space) == 0
        free = substitute(A(e, b) * u(up(b)), *rule, space)
        assert canonical(free - B(e, rho) * C(up(rho), b) * u(up(b)), space) == 0

    def test_a_rule_for_a_scalar(self):
        efe = Einstein(mu, nu) + Lam * g(mu, nu)
        assert canonical(substitute(efe, Lam, 0, space) - Einstein(mu, nu), space) == 0
        assert substitute(4 * Lam - R, R, 0, space) == 4 * Lam
        with pytest.raises(CompileError, match="a scalar value"):
            substitute(efe, Lam, Ricci(up(a), a) * Ricci(mu, nu), space)

    def test_a_contraction_rule(self):
        fluid = (rho_ + p) * u(up(mu)) * u(up(nu)) + p * g(up(mu), up(nu))
        result = substitute(fluid * u(nu), u(up(a)) * u(a), -1, space)
        assert canonical(result + rho_ * u(up(mu)), space) == 0

    def test_a_trace(self):
        traced = trace(Ricci(mu, nu) - R / 2 * g(mu, nu), space)
        assert canonical(traced + R, space) == 0
        assert trace(0, space) == 0  # the side of an equation that is zero


class TestEquality:
    def test_different_free_indices_are_not_equal(self):
        written = IndexEquation(T(mu, nu), Lam * g(mu, nu), space)
        assert written != IndexEquation(T(rho, sigma), Lam * g(rho, sigma), space)
        assert IndexExpression(T(mu, nu), space) != IndexExpression(u(mu), space)

    def test_coefficients_are_compared_with_common_factors_cancelled(self):
        x, y = Symbol("x"), Symbol("y")
        fraction = IndexExpression((x**2 - y**2) / (x - y) * T(mu, nu), space)
        assert fraction == IndexExpression((x + y) * T(mu, nu), space)
        assert latex(fraction.value) == r"\left(x + y\right) T_{\mu\nu}"


class TestDisplay:
    def test_a_solution(self):
        solution = solve(
            Einstein(mu, nu) + Lam * g(mu, nu), 8 * pi * G * T(mu, nu), T(mu, nu), space
        )
        assert latex(solution) == r"\frac{G_{\mu\nu} + \Lambda g_{\mu\nu}}{8 \pi G}"

    def test_the_simplest_terms_first(self):
        einstein = Ricci(mu, nu) - R / 2 * g(mu, nu)
        assert latex(einstein) == r"R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu}"

    def test_positions_and_dummies(self):
        assert latex(Riemann(up(rho), sigma, mu, nu)) == r"R^{\rho}{}_{\sigma\mu\nu}"
        assert latex(canonical(T(mu, a) * u(up(a)), space)) == r"T_{\mu}{}^{\alpha} u_{\alpha}"
        assert latex(canonical(g(up(mu), up(a)) * g(a, nu), space)) == r"\delta^{\mu}{}_{\nu}"

    def test_more_dummies_than_greek_letters(self):
        dummies = [space.index(f"q{k}", False) for k in range(20)]
        product = canonical(reduce(operator.mul, (u(up(i)) * u(i) for i in dummies)), space)
        assert r"\alpha_{1}" in latex(product)  # twenty dummies: the letters numbered

    def test_labels(self):
        assert latex_labels(["mu", "nu"]) == r"\mu\nu"
        assert latex_labels(["mu", "a", "b"]) == r"\mu ab"  # not \muab
        assert latex_labels(["mu'", "nu"]) == r"\mu'\nu"

    def test_coefficients_before_tensors(self):
        fluid = (rho_ + p) * u(mu) * u(nu) + p * g(mu, nu)
        assert latex(fluid) == r"\left(p + \rho\right) u_{\mu} u_{\nu} + p g_{\mu\nu}"

    def test_an_equation(self):
        equation = IndexEquation(T(mu, nu), Lam * g(mu, nu), space)
        assert str(equation) == r"T_{\mu\nu} = \Lambda g_{\mu\nu}"
        assert not equation.holds()
        assert IndexEquation(Ricci(mu, nu), Ricci(nu, mu), space).holds()
