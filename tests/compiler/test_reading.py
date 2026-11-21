r"""Reading physics as it is written: the conventions of papers and textbooks.

Physicists leave a great deal implicit. A line element names its coordinates,
``\alpha`` written once as ``\alpha(t, r)`` is a function from then on, a
derivation is a chain of equalities, and ``r(\partial_r \beta - 1)`` is a
product because ``r`` is a coordinate. These tests pin each convention down.
"""

import pytest
from sympy import Function, Symbol, diff, exp, sympify, true

from helpers import same, zero
from relativisticpy import CompileError, IndexNotationError

t, r, theta, phi, M = (Symbol(name) for name in ("t", "r", "theta", "phi", "M"))
alpha, beta = Function("alpha")(t, r), Function("beta")(t, r)
SPHERICAL = r"ds^2 = -e^{2\alpha(t,r)}{\rm d} t^2 + e^{2\beta(t,r)}{\rm d} r^2 + r^2 d\Omega^2"


class TestImplicitContext:
    def test_a_line_element_names_its_coordinates(self, wb):
        wb.exe(r"ds^2 = -dt^2 + dx^2 + dy^2 + dz^2")
        assert [c.name for c in wb.coordinates] == ["t", "x", "y", "z"]
        wb.exe(SPHERICAL)  # a new line element, new coordinates
        assert [c.name for c in wb.coordinates] == ["t", "r", "theta", "phi"]

    def test_declared_coordinates_stay(self, wb):
        wb.exe("Coordinates := [t, x]")
        with pytest.raises(CompileError, match="d must be followed by a coordinate"):
            wb.exe("ds^2 = -d t^2 + d y^2")  # y is no coordinate: a mistake, not new ones

    def test_a_function_written_once_with_its_arguments(self, wb):
        wb.exe(SPHERICAL)
        assert wb.expr(r"\partial_0 \alpha") == diff(alpha, t)

    def test_a_dependence_declared_again_is_the_value(self, wb):
        wb.exe(SPHERICAL)
        wb.exe(r"\beta = \beta(r) + 1")
        assert wb.expr(r"\beta(r) - 1") == Function("beta")(r)

    def test_repeated_derivatives(self, wb):
        wb.exe(SPHERICAL)
        assert wb.expr(r"\partial_1^2 \beta") == diff(beta, r, 2)

    def test_a_spacetimes_coordinates_keep_declared_assumptions(self, wb):
        wb.exe("Positive := [r]")
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        positive = Symbol("r", positive=True)
        assert wb.coordinates[1] == positive
        assert wb.expr(r"\partial_r (r^2)") == 2 * positive
        assert wb.expr(r"\sqrt{r^2}") == positive

    def test_a_coordinate_called_once_as_a_function_stays_a_coordinate(self, wb):
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        assert wb.expr(r"\frac{d r(t)}{dt}") == diff(Function("r")(t), t)
        assert wb.expr("r") == r  # not r(t) from now on
        assert wb.expr(r"\partial_r (r^2)") == 2 * r


class TestLatexReading:
    def test_a_script_binds_to_the_last_letter(self, wb):
        assert wb.expr(r"re^{-2\beta}") == r * exp(-2 * Symbol("beta"))

    def test_only_the_last_letter_is_called(self, wb):
        assert wb.expr(r"2Gm(r) + \alpha") == 2 * Symbol("G") * Function("m")(r) + Symbol("alpha")

    def test_a_coordinate_with_an_expression_is_a_product(self, wb):
        wb.exe("Coordinates := [t, r]")
        assert wb.expr(r"r(\beta - 1)") == r * (Symbol("beta") - 1)
        assert wb.expr(r"r(\tau) + \alpha") == Function("r")(Symbol("tau")) + Symbol("alpha")

    def test_in_latex_a_defined_value_with_brackets_is_a_product(self, wb):
        wb.exe("y := 2")
        assert wb.expr(r"y(3) + \alpha") == 6 + Symbol("alpha")
        with pytest.raises(CompileError, match="y is not a function"):
            wb.expr("y(3)")  # in plain text, a mistake

    def test_symbols_and_decorations(self, wb):
        assert wb.expr(r"\hbar\omega") == Symbol("hbar") * Symbol("omega")
        assert wb.expr(r"r^* + r_+ + \alpha") == Symbol("r^*") + Symbol("r_+") + Symbol("alpha")


class TestRelations:
    def test_a_chain_is_one_result_per_link(self, wb):
        assert wb.exe("x = x = 2 - 2 + x") == [true, true]

    def test_an_identity_is_true_whatever_its_form(self, wb):
        assert wb.expr(r"e^{2a}(1 + b) = e^{2a} + b e^{2a}") is true

    def test_relations_are_read_but_not_computed(self, wb):
        with pytest.raises(CompileError, match="'≈' is read but not computed"):
            wb.expr(r"x \approx 1")

    def test_plus_or_minus_is_both(self, wb):
        assert wb.expr(r"1 \pm \sqrt{4}") == [3, -1]
        assert wb.expr(r"\mp 2") == [-2, 2]

    @pytest.mark.parametrize(
        ("source", "message"),
        [
            (r"a + \cdots + b", "⋯ stands for terms that are not written"),
            (r"{\rm d}t = 2\,{\rm d}x", "a relation between differentials"),
            (r"*F", "the Hodge dual is not supported yet"),
            (r"\int d^4x\, f", "integrals over regions of spacetime"),
            (r"\sum_b x^b", "needs its limits"),
            (r"f \circ g", "the composition of maps is not supported"),
        ],
    )
    def test_what_cannot_be_computed_says_so(self, wb, source, message):
        with pytest.raises(CompileError, match=message):
            wb.expr(source)


class TestOperators:
    def test_the_wave_operator(self, wb):
        wb.exe(r"ds^2 = -dt^2 + dr^2 + r^2 d\Omega^2")
        assert same(wb.expr(r"\Box (r^2 - t^2)"), 8)  # 6 from the Laplacian, 2 from time
        assert same(wb.expr(r"\Box \frac{1}{r}"), 0)  # outside the origin

    def test_the_lie_derivative_of_a_scalar(self, wb):
        wb.exe(r"Coordinates := [t, r]" "\n" r"V^{\mu} := [1, r]")
        f = Function("f")(t, r)
        assert same(wb.expr(r"\pounds_V f(t, r)"), diff(f, t) + r * diff(f, r))

    def test_killing_vectors_leave_the_metric_alone(self, wb):
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        wb.exe(
            r"K^{\mu} := [1, 0, 0, 0]"
            "\n"
            r"L^{\mu} := [0, 0, 0, 1]"
            "\n"
            r"V^{\mu} := [0, 1, 0, 0]"
        )
        assert zero(wb.expr(r"\pounds_K g_{\mu\nu}"))
        assert zero(wb.expr(r"\pounds_L g_{\mu\nu}"))
        assert not zero(wb.expr(r"\pounds_V g_{\mu\nu}"))

    def test_the_lie_derivative_of_the_metric_is_a_symmetrised_derivative(self, wb):
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)" "\n" r"V^{\mu} := [r, 0, \cos\theta, 0]")
        assert wb.expr(r"\pounds_V g_{\mu\nu} = \nabla_\mu V_\nu + \nabla_\nu V_\mu")

    def test_a_lie_derivative_needs_a_vector(self, wb):
        wb.exe(r"Coordinates := [t, r]" "\n" r"W_{\mu} := [1, 0]")
        with pytest.raises(IndexNotationError, match="along a vector, with its index up"):
            wb.expr(r"\pounds_W t")
        with pytest.raises(CompileError, match="needs U to be a vector field"):
            wb.expr(r"\pounds_U t")

    def test_the_wedge_product_of_one_forms(self, wb):
        wb.exe(r"Coordinates := [x, y]" "\n" r"A_{\mu} := [1, 0]" "\n" r"B_{\mu} := [0, 1]")
        assert wb.expr(r"A_{\mu} \wedge B_{\nu} = A_\mu B_\nu - A_\nu B_\mu")
        assert wb.expr(r"A_{\mu} \wedge B_{\nu}").components.tolist() == [[0, 1], [-1, 0]]
        assert wb.expr(r"A_{\mu} \wedge A_{\nu}").components.tolist() == [[0, 0], [0, 0]]


def test_carroll_style_spherical_symmetry(wb):
    # The connection and curvature of the general spherically symmetric metric, as a
    # textbook lists them, with the numbered indices 0 = t and 1 = r.
    wb.exe(SPHERICAL)
    claims = wb.exe(
        r"""
        \Gamma^0_{00} = \partial_0 \alpha \qquad \Gamma^1_{22} = -r e^{-2\beta}
        \Gamma^2_{33} = -\sin\theta\cos\theta \qquad \Gamma^3_{23} = {\cos\theta \over \sin\theta}
        R^2{}_{323} = (1 - e^{-2\beta}) \sin^2\theta
        R_{01} = {2 \over r} \partial_0 \beta
        """
    )
    assert claims == [true] * 6
    # With the opposite sign on the time derivatives, R_00 does not hold.
    wrong = wb.expr(
        r"""R_{00} = [\partial_0^2\beta + (\partial_0\beta)^2 - \partial_0\alpha \partial_0\beta]
        + e^{2(\alpha-\beta)}[\partial_1^2\alpha + (\partial_1\alpha)^2
        - \partial_1\alpha \partial_1\beta + {2 \over r}\partial_1\alpha]"""
    )
    assert wrong is not true
    difference = sympify(wrong.lhs - wrong.rhs)
    assert same(
        difference, -2 * (diff(beta, t, 2) + diff(beta, t) ** 2 - diff(alpha, t) * diff(beta, t))
    )
