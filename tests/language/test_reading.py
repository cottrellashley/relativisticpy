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
        # R^a_b has two indices, a up and b down, as TeX sets it: not one, named a_b
        assert str(schwarzschild.expr(unbraced)) == str(schwarzschild.expr(braced))

    def test_data_bound_with_mixed_indices_reads_back(self):
        with Workbook() as wb:
            wb["A^i_j"] = np.array([[1.0, 2.0], [3.0, 4.0]])
            assert wb.expr("A^i_i") == 5.0
            assert wb.expr("A^i_j").to_numpy().tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_names_written_with_an_underscore_stay_names(self):
        with Workbook() as wb:
            assert str(wb.expr("r_s := 2 G M\nx_0 := 3\n1 - r_s/r + x_0^2")) == "-2*G*M/r + 10"


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
