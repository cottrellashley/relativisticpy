r"""Families (design/notation.md, decision 5.2): on the left of :=, a one-letter subscript
that the value is written with makes a family, so that c_1, a_{n+1} and \psi_3(x) evaluate
it; a family defined through its own members is a recurrence, computed from the members
given; and a finite sum over the subscript of names the program defines is taken term by
term. Each was refused, or a symbol of its own, before."""

from __future__ import annotations

import pytest
import sympy

from relativisticpy import Workbook
from relativisticpy.errors import CompileError, EvaluationError
from relativisticpy.language import Source

a, d, n, x, alpha, beta = sympy.symbols("a d n x alpha beta")


def readings(wb: Workbook, source: str) -> list[str]:
    """How the families of the last statement of ``source`` were read."""
    return [d.text() for d in wb.explain(source)[-1].decisions if d.kind == "family or name"]


def matrix(value: object) -> sympy.Matrix:
    return sympy.Matrix(value.tolist())  # type: ignore[attr-defined]


class TestFamilies:
    def test_a_member_of_a_family(self, wb):
        wb.exe("c_n := 1/n^2")
        assert wb.expr("c_1 + c_2") == sympy.Rational(5, 4)
        assert wb.expr("c_n") == 1 / n**2  # the family at its label

    def test_members_written_in_latex(self, wb):
        wb.exe(r"E_k := \alpha + 2\beta \cos\frac{2\pi k}{6}")
        wb.exe(r"E_\pi := 2E_0 + 2E_1 + 2E_{-1}")  # E_\pi is a name of its own
        assert wb.expr(r"E_\pi - 6(\alpha + \beta)") == 2 * beta
        wb.exe("a_n := n^2")
        assert wb.expr(r"\frac{a_{n+1}}{a_n}") == (n + 1) ** 2 / n**2
        assert wb.expr(r"\lim_{n \to \infty} \frac{a_{n+1}}{a_n}") == 1

    def test_a_member_in_a_family_of_names(self, wb):
        # the subscript of \sigma_a is the member's: T_1 is \sigma_1 / 2
        wb.exe(
            r"\sigma_1 := \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}"
            "\n"
            r"\sigma_2 := \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}"
            "\n"
            r"\sigma_3 := \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}"
            "\n"
            r"T_a := \frac{\sigma_a}{2}"
        )
        casimir = sympy.Rational(3, 4) * sympy.eye(2)
        assert matrix(wb.expr("T_1^2 + T_2^2 + T_3^2")) == casimir

    def test_a_family_of_functions(self, wb):
        wb.exe(r"\psi_n(x) := \sqrt{\frac{2}{a}} \sin\left(\frac{n\pi x}{a}\right)")
        assert wb.expr(r"\psi_3(x)") == sympy.sqrt(2 / a) * sympy.sin(3 * sympy.pi * x / a)
        assert wb.expr(r"\int_0^a \psi_1(x)^* \psi_2(x) \, dx") == 0
        m = sympy.Symbol("m")
        assert wb.expr(r"\psi_m(x)") == sympy.sqrt(2 / a) * sympy.sin(m * sympy.pi * x / a)

    def test_a_member_called_with_a_script(self, wb):
        # \psi_n(x)^* is the conjugate of the call, which the parser reads with the star on
        # the bracket; it was \psi_n times the conjugate of x
        wb.exe(
            r"A := \left(\int_0^a x^2 (a - x)^2 \, dx\right)^{-1/2}"
            "\n"
            r"\Psi(x) := A\, x (a - x)"
            "\n"
            r"\psi_n(x) := \sqrt{\frac{2}{a}} \sin\left(\frac{n\pi x}{a}\right)"
            "\n"
            r"c_n := \int_0^a \psi_n(x)^* \Psi(x) \, dx"
        )
        assert wb.expr("c_1") == 8 * sympy.sqrt(15) / sympy.pi**3
        assert wb.expr("c_2") == 0

    def test_a_family_with_an_order_raised_on_it(self, wb):
        # E^{(0)}_n and E_n^{(0)}, as TeX sets them alike: the unperturbed energies
        wb.exe(r"E^{(0)}_n := \frac{n^2 \pi^2}{2}")
        assert wb.expr(r"E^{(0)}_2 - E_1^{(0)}") == 3 * sympy.pi**2 / 2

    def test_a_family_of_functions_in_plain_text(self, wb):
        wb.exe("psi_n(x) := sin(n*x)")
        assert wb.expr("psi_2(x)") == sympy.sin(2 * x)

    def test_names_that_are_no_members_are_symbols(self, wb):
        x_0, v = sympy.symbols("x_0 v")
        assert wb.expr("x_n := x_0 + n d\nx_0 + 1") == x_0 + 1  # a name the value writes
        assert wb.expr("x_3") == x_0 + 3 * d
        assert wb.expr("c_n := 1/n^2\nc_k") == sympy.Symbol("c_k")  # k binds nothing
        assert wb.expr("v_x := v \\cos x\nv_y := 2\nv_x + v_y") == v * sympy.cos(x) + 2

    def test_a_family_written_with_its_label_is_at_the_label(self, wb):
        # t_i as it stands is t at the label i, never at i = \sqrt{-1}
        wb.exe("h := 0.2\nt_i := i h")
        assert wb.expr("t_i") == 0.2 * sympy.Symbol("i")
        assert abs(wb.expr("t_3") - 0.6) < 1e-15

    def test_a_family_is_said_in_a_note_and_a_reading(self, wb):
        (step,) = list(wb.each("c_n := 1/n^2"))
        assert step.notes == ["c_n is a family in n: c_1 is c_n at n = 1"]
        assert readings(wb, "c_1") == [
            "family or name: c_1 is c_n at n = 1, since c_n is a family in n, defined on line 1"
        ]

    def test_a_subscript_that_is_an_index_makes_no_family(self, wb):
        wb.exe(r"\mathcal{A} := \mathbb{R}^{2}" "\n" r"a \in \mathcal{A}" "\n" "w_{a} := [1, 2]")
        assert wb.expr("w_{a} w_{a}") == 5
        wb.exe("n := 3\nb_n := 2 n")  # n has a value: b_n is a name of its own
        assert wb.expr("b_n") == 6


class TestRecurrences:
    def test_a_member_at_a_whole_number(self, wb):
        wb.exe("F_0 := 0\nF_1 := 1\nF_{n+2} := F_{n+1} + F_n")
        assert wb.expr("F_{10}") == 55

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("T_0 = 0\nT_n = 2 T_{n-1} + 1, \\quad n > 0\nT_n", 2**n - 1),
            ("a_0 = 0\na_1 = 1\na_n = 5 a_{n-1} - 6 a_{n-2}, \\quad n \\geq 2\na_n", 3**n - 2**n),
        ],
    )
    def test_a_closed_form_where_rsolve_finds_one(self, wb, source, expected):
        found = wb.exe(source)[-1]
        (label,) = found.free_symbols  # n, positive as the condition beside it says
        assert sympy.simplify(found - expected.subs(n, label)) == 0

    def test_a_member_needs_the_members_before_it(self, wb):
        wb.exe("C_0 = 1\nC_{n+1} = \\sum_{k=0}^{n} C_k C_{n-k}, \\quad n \\geq 0")
        assert wb.expr("C_{10}") == 16796

    def test_members_in_a_finite_sum(self, wb):
        # rsolve finds no closed form for the Airy series, so each term is computed
        wb.exe(
            "a_0 := 1\na_1 := 0\na_2 := 0\n"
            r"a_{n+2} := \frac{a_{n-1}}{(n + 2)(n + 1)}"
        )
        series = 1 + x**3 / 6 + x**6 / 180 + x**9 / 12960
        assert sympy.expand(wb.expr(r"\sum_{n=0}^{9} a_n x^n")) == series

    def test_an_iteration_of_numbers(self, wb):
        wb.exe(
            "f(t, y) := y - t^2 + 1\nh := 0.2\nt_i := i h\nw_0 := 0.5\n"
            "w_{i+1} := w_i + h f(t_i, w_i)"
        )
        assert abs(float(wb.expr("w_{10}")) - 4.8657845) < 5e-8  # Burden and Faires

    def test_names_that_hold_a_member_as_a_symbol_are_refused(self, wb):
        # Runge and Kutta's stages, defined before the recurrence, hold w_i as a symbol of
        # their own: computed, the recurrence would keep it
        wb.exe(
            "f(t, y) := y - t^2 + 1\nh := 0.2\nt_i := i h\nw_0 := 0.5\n"
            "k_1 := h f(t_i, w_i)\nk_2 := h f(t_i + h, w_i + k_1)\n"
            r"w_{i+1} := w_i + \frac{1}{2}(k_1 + k_2)"
        )
        with pytest.raises(CompileError) as raised:
            wb.expr("w_{10}")
        assert raised.value.message == (
            "w_i is computed from k_1 and k_2, whose values hold w_i as a symbol of their "
            "own, not as the member the recurrence computes"
        )

    def test_a_recurrence_with_no_member_given_is_refused(self, wb):
        wb.exe("x_{n+1} := 2 x_n")
        with pytest.raises(CompileError, match="none of its members is given"):
            wb.expr("x_3")

    def test_a_member_before_those_given_is_refused(self, wb):
        wb.exe("a_1 := 1\na_{n+1} := 2 a_n")
        with pytest.raises(EvaluationError, match="comes before the first member given"):
            wb.expr("a_{0}")

    def test_a_recurrence_is_said_in_a_note(self, wb):
        steps = list(wb.each("a_0 := 1\na_{n+1} := 2 a_n"))
        assert steps[-1].notes == [
            "a_{n+1} is defined by a recurrence: its members are computed from those given "
            "before them"
        ]


class TestSumsOfNames:
    def test_a_sum_over_the_subscript_of_names(self, wb):
        wb.exe(
            r"\lambda_1 := \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}"
            "\n"
            r"\lambda_2 := \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}"
        )
        total = wb.expr(r"\sum_{a=1}^{2} \left(\frac{\lambda_a}{2}\right)^2")
        assert matrix(total) == sympy.eye(2) / 4
        assert readings(wb, r"\sum_{a=1}^{2} \lambda_a") == [
            r"family or name: \sum_{a=1}^{2} \lambda_a is the sum of its 2 terms, since a is "
            r"the subscript of \lambda_1, …, \lambda_2"
        ]

    def test_a_sum_over_names_not_all_defined_is_kept(self, wb):
        wb.exe(r"\lambda_1 := 2")
        assert wb.expr(r"\sum_{a=1}^{2} \lambda_a") == 2 + sympy.Symbol("lambda_2")


class TestALabelInTheNameOfAFunction:
    """Finding 11 of the review of the reading stack: ``y_n(x) := P_n(x)`` defined no
    family, as its label is written only in the name of the function it calls, and
    ``y_2(x)`` was then the spherical Bessel function of the second kind. The label of a
    family may be the subscript of a function its value calls; and a name the program
    defines with a label owns its letter with any subscript."""

    def test_the_subscript_of_a_function_called_makes_a_family(self, wb):
        steps = list(wb.each(r"y_n(x) := P_n(x)"))
        assert steps[0].notes == ["y_n(x) is a family in n: y_1(x) is y_n(x) at n = 1"]
        assert wb.expr("y_2(x)") == sympy.Rational(3, 2) * x**2 - sympy.Rational(1, 2)
        assert wb.expr("y_0(x)") == 1
        assert wb.expr("y_m(x)") == sympy.legendre(sympy.Symbol("m"), x)

    def test_a_family_of_wavefunctions_written_with_hermites_polynomials(self, wb):
        wb.exe(r"\psi_n(x) := H_n(x) e^{-x^2/2}")
        expected = (4 * x**2 - 2) * sympy.exp(-(x**2) / 2)
        assert sympy.simplify(wb.expr(r"\psi_2(x)") - expected) == 0

    def test_a_letter_before_the_function_is_a_factor_and_may_be_the_label(self, wb):
        wb.exe(Source(r"c_n := nf(x)", latex=True))  # in LaTeX, n times f(x)
        assert wb.expr("c_3") == 3 * sympy.Function("f")(x)

    def test_the_letter_of_the_function_itself_is_no_label(self, wb):
        wb.exe(Source(r"c_f := 2f(x)", latex=True))  # f is the function, not a label
        assert wb.expr("c_f") == 2 * sympy.Function("f")(x)
        assert wb.expr("c_1") == sympy.Symbol("c_1")

    def test_a_name_defined_with_a_label_owns_its_letter(self, wb):
        wb.exe("y_n(x) := x^2")  # no family: n is not in its value
        value = wb.expr("y_2(x)")  # an unknown function, not the spherical Bessel function
        assert value == sympy.Function("y_2")(x)
        assert wb.expr("y_n(3)") == 9
