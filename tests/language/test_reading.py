r"""Reading as a person means it: what the front end makes of notation, run end to end.

The parser's own tests state structure; these state what a user sees when they
write the notation, through a :class:`~relativisticpy.Workbook`.
"""

import numpy as np
import pytest
import sympy

from relativisticpy import Workbook, latex

SCHWARZSCHILD = "Coordinates := [t, r, \\theta, \\phi]\ng_{\\mu\\nu} := Schwarzschild(M)"


@pytest.fixture
def schwarzschild():
    with Workbook() as wb:
        wb.exe(SCHWARZSCHILD)
        yield wb


class TestScripts:
    @pytest.mark.parametrize(
        ("unbraced", "braced"),
        [
            ("R^a_b", "R^{a}_{b}"),
            ("G^t_t", "G^{t}_{t}"),
            ("R^t_r", "R^{t}_{r}"),
            ("g^a_b", "g^{a}_{b}"),
            (r"\delta^a_b", r"\delta^{a}_{b}"),
            (r"\Gamma^r_{tt}", r"\Gamma^{r}_{tt}"),
            ("x^a_b", "x^{a}_{b}"),
        ],
    )
    def test_a_script_without_braces_is_one_letter(self, schwarzschild, unbraced, braced):
        # R^a_b has two indices, a up and b down, as TeX sets it: not one, named a_b. The
        # values agree; each is shown as it was written.
        mine, theirs = schwarzschild.expr(unbraced), schwarzschild.expr(braced)
        assert getattr(mine, "indices", None) == getattr(theirs, "indices", None)
        assert getattr(mine, "components", mine) == getattr(theirs, "components", theirs)

    def test_data_bound_with_mixed_indices_reads_back(self):
        with Workbook() as wb:
            wb["A^i_j"] = np.array([[1.0, 2.0], [3.0, 4.0]])
            assert wb.expr("A^i_i") == 5.0
            assert wb.expr("A^i_j").to_numpy().tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_names_written_with_an_underscore_stay_names(self):
        with Workbook() as wb:
            assert str(wb.expr("r_s := 2 G M\nx_0 := 3\n1 - r_s/r + x_0^2")) == "-2*G*M/r + 10"

    def test_a_sign_as_a_script_labels_its_symbol(self):
        # h_+ is a polarisation, so a wave written with it is not zero, as h_{+\cos} made it
        with Workbook() as wb:
            wb.exe("ds^2 = -dt^2 + dx^2 + dy^2 + dz^2")
            wb.exe(r"h_{\mu\nu} := h_+ \cos[\omega (t - z)] \begin{pmatrix} 0 & 0 & 0 & 0 \\ "
                   r"0 & 1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}")  # fmt: skip
            omega, z, h = sympy.symbols("omega z h_+")
            assert sympy.simplify(wb.expr("h_{xx}") - h * sympy.cos(omega * (t - z))) == 0
            assert wb.expr("x^+ x^- + e^-t") == sympy.Symbol("x^+") * sympy.Symbol("x^-") + (
                sympy.exp(-t)
            )


class TestIntegrals:
    @pytest.mark.parametrize(
        ("text", "value"),
        [
            # the differential in a numerator ends the integral, so - \pi is outside it
            (r"\int_1^\infty \frac{dr}{r^2} - \pi", 1 - sympy.pi),
            (r"\int_0^1 \frac{d\theta}{1 + \theta^2}", sympy.pi / 4),
            (r"\int_0^1 \frac{x_1 \, dx_1}{2}", sympy.Rational(1, 4)),
            # an evaluation bar before the differential is inside the integrand
            (r"\int_0^{\pi} \left. \sin(x y) \right|_{y=1} dx", 2),
            (r"\int_0^1 \int_0^1 \left. x y z \right|_{z=2} \, dx \, dy", sympy.Rational(1, 2)),
        ],
    )
    def test_where_an_integral_ends(self, text, value):
        with Workbook() as wb:
            assert sympy.simplify(wb.expr(text) - value) == 0


class TestNamesOfTheirOwn:
    def test_primed_coordinates_are_defined(self):
        # as a Lorentz transformation writes the coordinates of another frame
        with Workbook() as wb:
            wb.exe(
                "\n".join(
                    [
                        r"\gamma := \frac{1}{\sqrt{1 - v^2/c^2}}",
                        "x := u t",
                        r"x' := \gamma (x - v t)",
                        r"t' := \gamma \left(t - \frac{v x}{c^2}\right)",
                    ]
                )
            )
            u, v, c = sympy.symbols("u v c")
            assert sympy.simplify(wb.expr(r"\frac{x'}{t'}") - (u - v) / (1 - u * v / c**2)) == 0

    def test_an_accented_name_is_defined(self):
        with Workbook() as wb:
            assert wb.exe("\\hat{p} := 2\n\\hat{p} + 1") == [3]

    def test_a_numbered_function_is_called(self):
        with Workbook() as wb:
            wb.exe("R_{21}(r) := r e^{-r}\n\\psi_1(x) := \\sin x")
            assert wb.expr("R_{21}(2)") == 2 * sympy.exp(-2)
            assert wb.expr(r"\psi_1(\pi/2) + \psi_{1}{\left(0 \right)}") == 1


class TestBrackets:
    def test_a_poisson_bracket_with_its_canonical_pair(self):
        with Workbook() as wb:
            assert wb.expr(r"\{x^2, p\}_{x, p}") == 2 * sympy.Symbol("x")
            assert wb.expr(r"\{x, p\}_{x, p} + \{p, x\}_{x, p}") == 0


t, r, x, theta = sympy.symbols("t r x theta")
a, f, A = (sympy.Function(name) for name in ("a", "f", "A"))


class TestLatexReadsBack:
    """LaTeX that RelativisticPy or SymPy prints reads back as the same value, so that a
    result written into a paper and checked there holds."""

    @pytest.mark.parametrize(
        "value",
        [
            a(t),
            a(t) ** 2,
            sympy.diff(a(t), t, 2),
            sympy.diff(a(t), t) ** 2,
            sympy.diff(f(t, r), t, r),
            sympy.exp(-2 * A(r)),
            sympy.Function("alpha")(r) * sympy.diff(sympy.Function("alpha")(r), r),
            sympy.sin(theta) ** 2 / a(t),
            sympy.Subs(sympy.diff(f(x), x), x, t),
            sympy.Function("psi_1")(x),  # \psi_{1}{\left(x \right)}
            sympy.Function("R_21")(r) ** 2,  # R_{21}^{2}{\left(r \right)}
        ],
    )
    def test_sympy_s_latex(self, value):
        with Workbook() as wb:
            wb.exe("Coordinates := [t, r, \\theta, \\phi]")
            assert sympy.simplify(wb.expr(sympy.latex(value)) - value) == 0

    @pytest.mark.parametrize(
        "metric",
        [
            "ds^2 = -dt^2 + a(t)^2 (dx^2 + dy^2 + dz^2)",
            r"ds^2 = -e^{2A(r)} dt^2 + e^{2B(r)} dr^2 + r^2 d\Omega^2",
        ],
    )
    @pytest.mark.parametrize("quantity", ["R", "G_{tt}"])
    def test_curvature_printed_by_relativisticpy(self, metric, quantity):
        with Workbook() as wb:
            if "dx" in metric:
                wb.exe("Coordinates := [t, x, y, z]")
            wb.exe(metric)
            value = wb.expr(quantity)
            assert sympy.simplify(wb.expr(latex(value)) - value) == 0
