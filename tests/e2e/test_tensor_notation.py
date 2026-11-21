r"""Solving tensor equations as tensor equations, then asking for components.

An equation with a tensor that has no components yet is kept as index notation,
and solve, trace, simplify and subs rearrange it as notation. Defining a tensor
from the result, with a metric in force, gives its components.
"""

import re

import pytest
from sympy import Eq, Function, Symbol, diff, pi

from helpers import same
from relativisticpy import CompileError, IndexNotationError
from relativisticpy.abstract import IndexEquation, IndexExpression

t, G, Lam, k = Symbol("t"), Symbol("G"), Symbol("Lambda"), Symbol("k")
EINSTEIN = r"EFE := G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}"


class TestNotation:
    def test_an_equation_with_a_new_tensor_is_kept_as_notation(self, wb):
        wb.exe(EINSTEIN)
        result = wb.expr("EFE")
        assert isinstance(result, IndexEquation)
        assert str(result) == r"G_{\mu\nu} + \Lambda g_{\mu\nu} = 8 \pi G T_{\mu\nu}"

    def test_solve_for_a_tensor(self, wb):
        wb.exe(EINSTEIN)
        assert str(wb.expr(r"solve(EFE, T_{\mu\nu})")) == (
            r"T_{\mu\nu} = \frac{G_{\mu\nu} + \Lambda g_{\mu\nu}}{8 \pi G}"
        )

    def test_solve_for_the_ricci_tensor_reverses_the_trace(self, wb):
        wb.exe(EINSTEIN)
        solution = wb.expr(r"solve(EFE, R_{\mu\nu})")
        expected = wb.expr(
            r"R_{\mu\nu} = \Lambda g_{\mu\nu}"
            r" + 8\pi G \left(T_{\mu\nu} - \frac{1}{2} T g_{\mu\nu}\right)"
        )
        assert solution == expected

    def test_trace_and_subs(self, wb):
        wb.exe(EINSTEIN)
        assert str(wb.expr("trace(EFE)")) == r"4 \Lambda - R = 8 \pi G T"
        expanded = wb.expr(r"subs(EFE, G_{ab} = R_{ab} - \frac{1}{2} R g_{ab})")
        assert expanded == wb.expr(
            r"R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}"
        )

    def test_the_trace_of_an_equation_with_a_side_zero(self, wb):
        assert str(wb.expr(r"trace(R_{\mu\nu} = 0)")) == "R = 0"
        assert str(wb.expr(r"trace(0 = T_{ab})")) == "0 = T"

    def test_equations_with_other_free_indices_differ(self, wb):
        assert wb.expr(r"A_{ab} = B_{ab}") != wb.expr(r"A_{cd} = B_{cd}")

    def test_the_bianchi_identity_with_a_contracted_index(self, wb):
        bianchi = (
            r"R_{\rho\sigma\mu\nu} u^\nu + R_{\rho\mu\nu\sigma} u^\nu + R_{\rho\nu\sigma\mu} u^\nu"
        )
        assert str(wb.expr(rf"simplify({bianchi})")) == "0"
        assert wb.expr(rf"{bianchi} = 0").holds()

    def test_subs_with_a_scalar_rule(self, wb):
        wb.exe(EINSTEIN)
        assert str(wb.expr(r"subs(EFE, \Lambda = 0)")) == r"G_{\mu\nu} = 8 \pi G T_{\mu\nu}"
        assert str(wb.expr("subs(trace(EFE), T = 0)")) == r"4 \Lambda - R = 0"

    def test_subs_with_a_rule_that_sums(self, wb):
        result = wb.expr(r"subs(A_{ab} u^a u^b, A_{cd} = B_{ce} C^{e}{}_{d})")
        assert result == wb.expr(r"simplify(u^a B_{ae} C^{e}{}_{b} u^b)")

    def test_a_perfect_fluid_contracted_with_its_velocity(self, wb):
        fluid = r"((\rho + p) u^\mu u^\nu + p g^{\mu\nu}) u_\nu"
        result = wb.expr(rf"subs({fluid}, u^\alpha u_\alpha = -1)")
        assert isinstance(result, IndexExpression)
        assert str(result) == r"- \rho u^{\mu}"

    def test_simplify_knows_the_identities(self, wb):
        bianchi = r"R_{\rho\sigma\mu\nu} + R_{\rho\mu\nu\sigma} + R_{\rho\nu\sigma\mu}"
        assert str(wb.expr(rf"simplify({bianchi})")) == "0"
        assert str(wb.expr(r"simplify(R^{\alpha}{}_{\mu\alpha\nu} - R_{\nu\mu})")) == "0"
        assert str(wb.expr(r"simplify(G^\mu{}_\mu)")) == "- R"
        # Simplifying keeps the Einstein tensor as written; equality sees through it.
        assert str(wb.expr(r"simplify(G_{\mu\nu} + g_{\mu\nu})")) == r"G_{\mu\nu} + g_{\mu\nu}"
        assert wb.expr(r"simplify(G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu})").holds()

    def test_what_cannot_be_solved_says_why(self, wb):
        wb.exe(EINSTEIN)
        with pytest.raises(CompileError, match="does not contain W"):
            wb.expr(r"solve(EFE, W_{\mu\nu})")
        with pytest.raises(CompileError, match="derivatives are not yet part of tensor notation"):
            wb.expr(r"solve(\nabla_\mu T^{\mu\nu} = 0, T^{\mu\nu})")


class TestWithoutAMetric:
    """Without a metric there are no components, so tensor equations are notation."""

    def test_identities_hold(self, wb):
        assert wb.expr(r"G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu}").holds()
        assert wb.expr(r"g^{\mu\nu} g_{\nu\sigma} = \delta^\mu_\sigma").holds()
        assert wb.expr(r"R_{\rho\sigma\mu\nu} = -R_{\sigma\rho\mu\nu}").holds()
        assert not wb.expr(r"R_{\mu\nu} = 0").holds()  # a field equation

    def test_a_name_for_notation(self, wb):
        wb.exe(r"E := R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu}")
        assert str(wb.expr("trace(E)")) == "- R"
        with pytest.raises(IndexNotationError, match="whose indices are its own"):
            wb.expr(r"simplify(E^\mu{}_\mu)")

    def test_written_as_a_physicist_writes(self, wb):
        fluid = wb.expr(r"T_{\mu\nu} = (\rho + p) u_\mu u_\nu + p g_{\mu\nu}")
        assert str(fluid) == r"T_{\mu\nu} = \left(p + \rho\right) u_{\mu} u_{\nu} + p g_{\mu\nu}"
        einstein = wb.expr(r"T_{\mu\nu} = \frac{1}{8\pi G}(R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu})")
        assert (
            str(einstein) == r"T_{\mu\nu} = \frac{R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu}}{8 \pi G}"
        )
        primed = wb.expr(r"x^{\mu'} = \Lambda^{\mu'}{}_\nu x^\nu")
        assert str(primed) == r"x^{\mu'} = \Lambda^{\mu'}{}_{\alpha} x^{\alpha}"
        assert str(wb.expr(r"\omega_{\mu ab} = -\omega_{\mu ba}")).startswith(r"\omega_{\mu ab}")

    def test_letters_run_together_as_in_latex(self, wb):
        # GT_{\mu\nu} is G T_{\mu\nu}, and \kappa(...) of tensors is a product.
        assert str(wb.expr(r"G_{\mu\nu} = 8\pi GT_{\mu\nu} - \Lambda g_{\mu\nu}")) == (
            r"G_{\mu\nu} = 8 \pi G T_{\mu\nu} - \Lambda g_{\mu\nu}"
        )
        assert str(wb.expr(r"R_{\mu\nu} = \kappa(T_{\mu\nu} - \frac{1}{2} Tg_{\mu\nu})")) == (
            r"R_{\mu\nu} = \kappa \left(T_{\mu\nu} - \frac{1}{2} T g_{\mu\nu}\right)"
        )

    def test_components_keep_their_meaning(self, wb):
        # In a paper, \Gamma^\theta_{r\theta} = 1/r is a component: notation would sum over
        # \theta, and its two sides would have different free indices.
        for source in (r"\Gamma^r_{rr} = 0", r"\Gamma^\theta_{r\theta} = \frac{1}{r}"):
            with pytest.raises(CompileError, match="needs coordinates"):
                wb.expr(source)
        wb.exe("Coordinates := [t, r]")
        with pytest.raises(CompileError, match="needs a metric"):
            wb.expr("R_{tt} = 0")

    def test_powers_keep_their_meaning(self, wb):
        x, y, n = Symbol("x"), Symbol("y"), Symbol("n")
        assert wb.expr("simplify(x^n y^n)") == x**n * y**n
        assert wb.expr("x^n y^n = (x y)^n") == Eq(x**n * y**n, (x * y) ** n)

    @pytest.mark.parametrize(
        ("source", "message"),
        [
            (r"T_{\mu\nu} u^\nu u_\nu = 0", r"the index \nu appears 3 times in one term"),
            (
                r"solve(G_{\mu\nu} = 8\pi T_{\mu\nu}, T_{\mu\mu})",
                r"the index \mu appears twice in one term, both times down",
            ),
            (
                r"G_{\mu\nu} = 8\pi T_{\mu\nu\rho}",
                r"the two sides have different free indices: _{\mu\nu} on the left, "
                r"_{\mu\nu\rho} on the right",
            ),
            (
                r"T_{\mu\nu} + u_\mu u^\nu = 0",
                r"the terms of a sum have different free indices: _{\mu\nu} and ^{\nu}_{\mu}",
            ),
            (r"T_{\mu\nu} u^\nu = \frac{u_\mu}{v^\alpha}", "cannot divide by a tensor"),
        ],
    )
    def test_what_breaks_the_rules_says_so(self, wb, source, message):
        with pytest.raises(IndexNotationError, match=re.escape(message)):
            wb.expr(source)

    def test_derivatives_are_not_scalars(self, wb):
        # Not even when, as scalars, they would vanish and make the equation hold.
        geodesic = r"""\frac{d^2 x^\mu}{d\tau^2}
            + \Gamma^\mu_{\rho\sigma} \frac{dx^\rho}{d\tau} \frac{dx^\sigma}{d\tau} = 0"""
        with pytest.raises(CompileError, match="derivatives are not yet part of tensor notation"):
            wb.expr(geodesic)


class TestComponents:
    @pytest.fixture
    def flrw(self, wb):
        wb.exe(r"ds^2 = -dt^2 + a(t)^2 \left(\frac{dr^2}{1 - k r^2} + r^2 d\Omega^2\right)")
        wb.exe(EINSTEIN)
        return wb

    def test_define_a_tensor_from_a_solution(self, flrw):
        flrw.exe(r"T_{\mu\nu} := RHS(solve(EFE, T_{\mu\nu}))")
        a = Function("a")(t)
        friedmann = (3 * (diff(a, t) ** 2 + k) - Lam * a**2) / (8 * pi * G * a**2)
        assert same(flrw.expr(r"T_{tt}"), friedmann)

    def test_a_solution_defines_with_an_equals_sign_too(self, flrw):
        # The T_{\mu\nu} that solve is asked for names the unknown: T is defined.
        assert flrw.exe(r"T_{\mu\nu} = RHS(solve(EFE, T_{\mu\nu}))") == []
        a = Function("a")(t)
        friedmann = (3 * (diff(a, t) ** 2 + k) - Lam * a**2) / (8 * pi * G * a**2)
        assert same(flrw.expr(r"T_{tt}"), friedmann)

    def test_notation_meets_components(self, flrw):
        flrw.exe(r"T_{\mu\nu} := RHS(solve(EFE, T_{\mu\nu}))")
        # The trace-reversed equations, solved as notation, hold component by component.
        assert flrw.expr(r"R_{\mu\nu} = RHS(solve(EFE, R_{\mu\nu}))")
        assert flrw.expr(r"G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}")

    def test_an_equation_has_no_components(self, flrw):
        with pytest.raises(CompileError, match="an equation in index notation has no components"):
            flrw.exe(r"X_{\mu\nu} := EFE")


class TestMeaningsKept:
    def test_powers_and_constants(self, wb):
        assert wb.expr(r"e^{i\pi}") == -1
        assert wb.expr("e^{i pi}") == -1

    def test_a_new_tensor_alone_is_still_an_error(self, wb):
        wb.exe("Coordinates := [x, y]")
        with pytest.raises(CompileError, match="T is not a tensor"):
            wb.expr("T_{ab}")

    def test_with_a_metric_built_in_tensors_are_components(self, wb):
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        assert wb.expr(r"R_{\mu\nu} = 0")
        assert wb.expr(r"simplify(R_{\mu\nu})").nonzero() == {}
