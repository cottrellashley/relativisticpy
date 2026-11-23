r"""Written forms: a call written as books write a capability is read as it, under the
notation decisions of design/notation.md; a book name no integration provides is refused,
saying what it is; and a call prints in its form, which reads back."""

from __future__ import annotations

import pytest
import sympy

from relativisticpy import Workbook, display, integrations
from relativisticpy.errors import CompileError, EvaluationError
from relativisticpy.integrations import Example, _forms, function
from relativisticpy.language import parse

x, z = sympy.symbols("x z")


def shape(source: str) -> _forms.Shape | None:
    return _forms.form_shape(parse(source).statements[0].expr)


def decisions(wb: Workbook, source: str, kind: str = "book name") -> list[tuple[str, str]]:
    (reading,) = wb.explain(source)
    return [(d.written, d.reading) for d in reading.decisions if d.kind == kind]


class TestShapes:
    @pytest.mark.parametrize(
        ("source", "head", "style", "scripts", "arguments"),
        [
            (r"\Gamma(x)", "Gamma", "letter", 0, 1),
            ("Γ(x)", "Gamma", "letter", 0, 1),
            ("Gamma(x)", "Gamma", "plain", 0, 1),
            (r"\mathrm{B}(a, b)", "B", "upright", 0, 2),
            (r"\operatorname{erfc}(x)", "erfc", "upright", 0, 1),
            ("J_0(1)", "J", "plain", 1, 1),
            (r"J_{\nu}(x)", "J", "plain", 1, 1),
            (r"J_{1/2}(x)", "J", "plain", 1, 1),
            (r"Y_l^m(\theta, \phi)", "Y", "plain", 2, 2),
            (r"L_n^{(\alpha)}(x)", "L", "plain", 2, 1),
            (r"\left(\tfrac{1}{2}\right)_3", "", "plain", 1, 1),
        ],
    )
    def test_a_call_is_taken_apart_as_written(self, source, head, style, scripts, arguments):
        found = shape(source)
        assert found is not None
        assert (found.head, found.style) == (head, style)
        assert (len(found.scripts), len(found.arguments)) == (scripts, arguments)

    def test_what_is_not_a_call_has_no_shape(self):
        assert shape("x^2") is None
        assert shape(r"J_{\nu} (x)") is not None  # a name with an operand: an operator
        assert shape(r"J_{\nu} (x)").kind == "operator"
        assert shape(r"T_{ab}(x)").kind == "operator"  # two indices: no call of T

    def test_braces_are_not_brackets(self):
        assert shape(r"\mathcal{L}\{f\}").bracket == "{"
        assert shape(r"\mathcal{L}(f)").bracket == "("


class TestReading:
    def test_gamma_with_one_argument_is_eulers(self, wb):
        assert wb.expr(r"\Gamma\left(\tfrac{1}{2}\right)") == sympy.sqrt(sympy.pi)
        assert wb.expr(r"\Gamma(z)\Gamma(1 - z)") == sympy.gamma(z) * sympy.gamma(1 - z)
        assert wb.expr("Γ(5)") == 24

    def test_the_reading_is_recorded_with_a_note(self, wb):
        assert decisions(wb, r"\Gamma(5)") == [(r"\Gamma(5)", "gamma, Euler's gamma function")]
        (reading,) = wb.explain(r"\Gamma(5)")
        assert reading.notes == [
            r"\Gamma is read as Euler's gamma function, gamma; for a function of your own, "
            r"define \Gamma first"
        ]

    def test_gamma_with_indices_stays_the_christoffel_symbol(self, wb):
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        r, M = sympy.symbols("r M")
        assert sympy.simplify(wb.expr(r"\Gamma^{t}_{tr}") - M / (r * (r - 2 * M))) == 0
        assert wb.expr(r"\Gamma(5)") == 24  # without indices, Euler's, with a metric in force

    def test_a_users_definition_always_wins(self, wb):
        wb.exe(r"\Gamma(x) := x^2")
        assert wb.expr(r"\Gamma(3)") == 9
        other = Workbook()
        other.exe(r"\Gamma := 2")
        assert other.expr(r"\Gamma(3)") == 6  # in LaTeX, a name with a value multiplies

    def test_the_letter_gamma_stays_an_unknown_function(self, wb):
        assert wb.expr(r"\gamma(r)") == sympy.Function("gamma")(sympy.Symbol("r"))

    def test_a_one_letter_name_called_with_the_coordinates_is_a_function_of_them(self, wb):
        wb.exe(r"ds^2 = -N(t)^2 {\rm d}t^2 + a(t)^2 {\rm d}x^2")
        t = sympy.Symbol("t")
        assert wb.expr("g_{tt}") == -(sympy.Function("N")(t) ** 2)
        (reading,) = wb.explain(r"\Gamma(t)")
        assert reading.decisions[0].reading == "a function of the coordinates"
        assert "elsewhere" in reading.notes[0]

    @pytest.mark.parametrize(
        "written", [r"U_{\text{eff}}(r)", r"T_{\rm eff}(x)", r"U_{eff}(r)", r"L_{\mathrm{in}}(x)"]
    )
    def test_a_label_in_a_script_is_not_a_parameter(self, wb, written):
        # the effective potential is not the Chebyshev polynomial of degree "eff"
        assert decisions(wb, written) == []

    def test_a_parameter_in_a_script_still_is(self, wb):
        assert [d[1].split(",")[0] for d in decisions(wb, r"P_\ell(x)")] == ["legendre"]
        assert [d[1].split(",")[0] for d in decisions(wb, r"H_{n+1}(x)")] == ["hermite"]

    def test_n_in_plain_text_stays_numerical_evaluation(self, wb):
        assert round(float(wb.expr("N(pi, 3)")), 2) == 3.14


class TestSpecialFunctions:
    @pytest.mark.parametrize(
        ("source", "value"),
        [
            ("J_0(1)", sympy.besselj(0, 1)),
            (r"J_{\nu}(x)", sympy.besselj(sympy.Symbol("nu"), x)),
            (r"J_{1/2}(x)", sympy.sqrt(2) * sympy.sin(x) / (sympy.sqrt(sympy.pi) * sympy.sqrt(x))),
            (r"K_{\nu}(x)", sympy.besselk(sympy.Symbol("nu"), x)),
            ("j_1(x)", sympy.sin(x) / x**2 - sympy.cos(x) / x),
            (r"\operatorname{Ai}(0)", sympy.airyai(0)),
            ("P_3(x)", (5 * x**3 - 3 * x) / 2),
            ("P_2^1(x)", sympy.assoc_legendre(2, 1, x)),
            ("H_3(x)", 8 * x**3 - 12 * x),
            (r"L_1^{(\alpha)}(x)", -x + sympy.Symbol("alpha") + 1),
            ("T_2(x)", 2 * x**2 - 1),
            (r"Y_1^0(\theta, \phi)",
             sympy.sqrt(3) * sympy.cos(sympy.Symbol("theta")) / (2 * sympy.sqrt(sympy.pi))),
            (r"\mathrm{B}(2, 3)", sympy.Rational(1, 12)),
            (r"\zeta(2)", sympy.pi**2 / 6),
            ("ζ(4)", sympy.pi**4 / 90),
            (r"\operatorname{erfc} x", sympy.erfc(x)),
            (r"\operatorname{Log}(i)", sympy.pi * sympy.I / 2),
            ("{}_2F_1(1, 1; 2; z)", -sympy.log(1 - z) / z),
            (r"\left(\tfrac{1}{2}\right)_3", sympy.Rational(15, 8)),
            ("W_0(e)", 1),
            (r"\Phi(0)", sympy.Rational(1, 2)),
            (r"u_{\pi/4}(\pi)", 1),
            (r"\operatorname{K}(0)", sympy.pi / 2),
        ],
    )  # fmt: skip
    def test_a_book_name_is_read_as_its_capability(self, wb, source, value):
        assert sympy.simplify(wb.expr(source) - value) == 0

    def test_the_counts_of_the_hypergeometric_function_are_checked(self, wb):
        with pytest.raises(EvaluationError, match="takes 3 parameters before its first"):
            wb.expr("{}_3F_1(1, 1; 2; z)")

    def test_a_small_letter_with_scripts_is_noted(self, wb):
        (reading,) = wb.explain("y_0(x)")
        assert reading.notes == [
            "y_0 is read as the spherical Bessel function of the second kind, of order n, yn; "
            "for a function of your own, define y_0 first"
        ]
        (reading,) = wb.explain("J_0(x)")
        assert reading.notes == []

    def test_a_small_letter_called_with_the_coordinates_is_a_function_of_them(self, wb):
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        assert wb.expr("u_c(t)") == sympy.Function("u_c")(sympy.Symbol("t"))

    def test_a_definition_wins(self, wb):
        wb.exe("y_1(x) := x^2")
        assert wb.expr("y_1(3)") == 9
        wb.exe(r"\zeta(s) := s + 1")
        assert wb.expr(r"\zeta(2)") == 3

    def test_a_greek_name_in_plain_text_is_called_only_with_brackets(self, wb):
        gamma, m, c = sympy.symbols("gamma m c")
        assert wb.expr("gamma m c^2") == gamma * m * c**2
        assert wb.expr("beta x") == sympy.Symbol("beta") * x
        assert wb.expr("zeta(2)") == sympy.pi**2 / 6

    def test_a_function_lifts_over_the_argument_that_is_a_tensor(self, wb):
        import numpy as np
        from scipy.special import jv

        wb["x_t"] = np.array([0.5, 1.5, 2.5])
        assert np.allclose(wb.expr("J_0(x_t)").to_numpy(), jv(0, [0.5, 1.5, 2.5]))
        assert "map besselj at 1" in wb.ir("J_0(x_t)")


class TestReservations:
    @pytest.mark.parametrize(
        ("source", "said"),
        [
            (r"\operatorname{Res}_{z=0} f(z)", "the residue of a function at a point"),
            (r"\epsilon_{ijk}", r"\epsilon_{ijk}, the Levi-Civita symbol, is not supported yet"),
        ],
    )
    def test_a_book_name_no_integration_provides_is_refused_saying_what_it_is(
        self, wb, source, said
    ):
        with pytest.raises(CompileError) as refused:
            wb.expr(source)
        assert said in refused.value.message

    def test_every_reserved_form_is_refused_in_latex(self):
        # in LaTeX, where a form books write is read: in plain text a name is called
        for reservation in integrations.reservations():
            for written in reservation.written:
                with pytest.raises(CompileError, match="is not supported yet"):
                    Workbook().expr(rf"\left({written}\right)")

    def test_n_of_a_number_in_latex_is_no_distribution_function(self, wb):
        # N(\pi) was the normal distribution function of pi (the review of the engines, 12):
        # in LaTeX the letter is the program's own, as the lapse N(t) is, and \Phi the function
        assert wb.expr(r"N(\pi)") == sympy.Function("N")(sympy.pi)
        assert wb.expr(r"\left(N(0)\right)") == sympy.Function("N")(0)
        *_, step = wb.each(r"N(\pi, 5)")
        assert step.values == [sympy.Function("N")(sympy.pi, 5)]
        assert step.notes[0].startswith(r"N is not defined, so N(\pi, 5) is an unknown function")
        assert str(wb.expr("N(pi, 5)")) == "3.1416"  # numerical evaluation, in plain text
        assert wb.expr(r"\Phi(0)") == sympy.Rational(1, 2)
        wb.exe(r"X \sim N(1, 4)")  # after \sim, the normal distribution
        assert wb.expr(r"\operatorname{Var}(X)") == 4

    @pytest.mark.parametrize(
        "source", [r"B(1, 2)", r"\beta(r)", r"\mathcal{L}(\phi)", r"L(x)", r"J_{\nu} (x)"]
    )
    def test_what_only_looks_like_a_book_name_is_read_as_before(self, wb, source):
        wb.expr(source)  # an unknown function, or a product

    def test_a_definition_of_the_name_wins_over_its_book_name(self, wb):
        wb.exe(r"\operatorname{rank}(A) := 2")
        assert wb.expr(r"\operatorname{rank}(B)") == 2

    def test_providing_a_capability_retires_its_reservation(self):
        @function(
            "Res",
            latex=r"\operatorname{Res}(f, a)",
            summary="the residue of a function at a point",
            examples=[Example(r"\operatorname{Res}(1/z, 0)", 1, "standard")] * 2,
            lifts=False,
        )
        class Residue:
            def exact(self, f, a):
                return sympy.residue(f, z, a)

        with integrations.installed(Residue):
            assert "Res" not in {r.name for r in integrations.reservations()}
            assert Workbook().expr(r"\operatorname{Res}(1/z, 0)") == 1
        assert "Res" in {r.name for r in integrations.reservations()}


class TestPrinting:
    @pytest.mark.parametrize(
        ("form", "printed"),
        [("text", "gamma(x)"), ("unicode", "Γ(x)"), ("latex", r"\Gamma\left(x\right)")],
    )
    def test_a_call_prints_in_its_form_and_reads_back(self, form, printed):
        assert getattr(display, form)(sympy.gamma(x)) == printed
        assert Workbook().expr(printed) == sympy.gamma(x)

    def test_a_power_of_a_call_reads_back(self):
        for form in ("text", "unicode", "latex"):
            shown = getattr(display, form)(sympy.gamma(x) ** 2)
            assert Workbook().expr(shown) == sympy.gamma(x) ** 2
