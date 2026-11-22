r"""Written forms: a call written as books write a capability is read as it, under the
notation decisions of design/notation.md; a book name no integration provides is refused,
saying what it is; and a call prints in its form, which reads back."""

from __future__ import annotations

import pytest
import sympy

from relativisticpy import Workbook, display, integrations
from relativisticpy.errors import CompileError
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

    def test_n_in_plain_text_stays_numerical_evaluation(self, wb):
        assert round(float(wb.expr("N(pi, 3)")), 2) == 3.14


class TestReservations:
    @pytest.mark.parametrize(
        ("source", "said"),
        [
            (r"\zeta(2)", r"\zeta, Riemann's zeta function, is not supported yet"),
            ("zeta(2)", "zeta, Riemann's zeta function, is not supported yet"),
            (r"\mathrm{B}(1, 2)", r"\mathrm{B}, the Beta function, is not supported yet"),
            ("J_0(1)", "J_0, the Bessel function of the first kind, is not supported yet"),
            (r"P_3(x)", "P_3, the Legendre polynomial, is not supported yet"),
            (r"Y_1^0(\theta, \phi)", "Y_1^0, the spherical harmonic, is not supported yet"),
            (r"\operatorname{erfc} x", "the complementary error function, is not supported yet"),
            (r"\Phi(1)", "the standard normal distribution function, is not supported yet"),
            (r"\operatorname{Res}_{z=0} f(z)", "the residue of a function at a point"),
            (r"\operatorname{tr} A", "the trace of a matrix written as an operator"),
            (r"\operatorname{rank}(A)", "the rank of a matrix, is not supported yet"),
            (r"\operatorname{Var}(X)", "the variance of a random variable"),
            (r"\mathcal{L}\{t^2\}", r"\mathcal{L}, the Laplace transform, is not supported yet"),
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
        # in LaTeX: in plain text N(x) stays numerical evaluation
        for reservation in integrations.reservations():
            for written in reservation.written:
                with pytest.raises(CompileError, match="is not supported yet"):
                    Workbook().expr(rf"\left({written}\right)")

    @pytest.mark.parametrize(
        "source", [r"B(1, 2)", r"\beta(r)", r"\mathcal{L}(\phi)", r"L(x)", r"J_{\nu} (x)"]
    )
    def test_what_only_looks_like_a_book_name_is_read_as_before(self, wb, source):
        wb.expr(source)  # an unknown function, or a product

    def test_a_definition_of_the_name_wins_over_its_reservation(self, wb):
        wb.exe(r"\zeta(s) := s + 1")
        assert wb.expr(r"\zeta(2)") == 3

    def test_providing_a_capability_retires_its_reservation(self):
        @function(
            "zeta",
            latex=r"\zeta(s)",
            summary="Riemann's zeta function",
            examples=[Example(r"\zeta(2)", sympy.pi**2 / 6, "DLMF 25.6.1")] * 2,
        )
        class Zeta:
            def exact(self, s):
                return sympy.zeta(s)

        with integrations.installed(Zeta):
            assert "zeta" not in {r.name for r in integrations.reservations()}
            assert Workbook().expr(r"\zeta(2)") == sympy.pi**2 / 6
        assert "zeta" in {r.name for r in integrations.reservations()}


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
