r"""The names books write with scripts, read by what each script labels
(design/notation.md, 5 and 6): a subscript that labels is part of the name, whatever the
base, as in \lambda_1, \delta_0, M_\odot and r_+; on the left of := a number or an order
raised on a name is part of it, as in \sigma^2; an index of a tensor in force, a declared
label, and a built-in tensor with its indices stay indices; and a definition of the
program's own always wins."""

from __future__ import annotations

import pytest
import sympy

from relativisticpy import Workbook
from relativisticpy.errors import CompileError, IndexNotationError

K, M, Q, a, k, t, x = sympy.symbols("K M Q a k t x")
PAULI = (
    r"\sigma_x := \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}"
    "\n"
    r"\sigma_y := \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}"
    "\n"
    r"\sigma_z := \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}"
)


def readings(wb: Workbook, source: str) -> list[str]:
    """How the scripts of the last statement of ``source`` were read."""
    return [d.text() for d in wb.explain(source)[-1].decisions if d.kind == "label or index"]


def matrix(value: object) -> sympy.Matrix:
    return sympy.Matrix(value.tolist())  # type: ignore[attr-defined]


class TestSubscriptsThatLabel:
    @pytest.mark.parametrize(
        ("definition", "use", "expected"),
        [
            (r"\lambda_1 := 2 \cosh K", r"\lambda_1^2", 4 * sympy.cosh(K) ** 2),
            (r"\delta_0 := -k a", r"\sin^2\delta_0", sympy.sin(k * a) ** 2),
            (r"V_{10} := 4", r"V_{10}^2", 16),
            (r"\psi_{200} := 3", r"2\psi_{200}", 6),
            (r"r_+ := M + \sqrt{M^2 - Q^2}", "r_+", M + sympy.sqrt(M**2 - Q**2)),
            (r"h_\times := 3", r"h_\times + 1", 4),
            (r"T_{\rm eff} := 5800", r"T_{\rm eff}", 5800),
        ],
    )
    def test_a_name_with_a_label_is_defined_and_read(self, wb, definition, use, expected):
        wb.exe(definition)
        assert wb.expr(use) == expected

    def test_a_label_names_a_matrix_where_no_index_can_be_meant(self, wb):
        # no coordinates, no space declared for x, and a matrix, which no vector of one
        # index has as its components: x labels \sigma
        wb.exe(PAULI)
        assert matrix(wb.expr(r"\sigma_x \sigma_y - \sigma_y \sigma_x")) == 2 * sympy.I * matrix(
            wb.expr(r"\sigma_z")
        )

    def test_a_definition_of_ones_own_wins_over_a_built_in_tensor(self, wb):
        wb.exe(r"ds^2 = -dt^2 + dx^2")  # a metric in force: R with indices is curvature
        wb.exe("M_\\odot := 2\nR_\\odot := 7")
        assert wb.expr(r"\frac{M_\odot}{R_\odot}") == sympy.Rational(2, 7)

    def test_a_built_in_tensor_claims_only_index_labels(self, wb):
        # \delta_0 is no Kronecker delta, which has two indices, and R_\oplus no curvature
        assert wb.expr(r"\delta_0 + R_\oplus") == sympy.Symbol("delta_0") + sympy.Symbol("R_oplus")
        wb.exe(r"ds^2 = -dt^2 + dx^2")
        assert wb.expr(r"R_\oplus") == sympy.Symbol("R_oplus")  # a mark never indexes
        with pytest.raises(IndexNotationError, match="R has 2 or 4 indices, but 1 is given"):
            wb.expr(r"R_{\mu}")

    def test_index_notation_is_read_as_before(self, wb):
        wb.exe("Coordinates := [t, x]\nT_{ab} := [[1, 2], [3, 4]]")
        assert wb.expr("T_{tx}") == 2
        wb.exe(r"g_{ab} := diag(-1, 1)")
        assert wb.expr("g_{tt}") == -1
        assert wb.expr(r"T_{ab} T^{ab}") == 1 - 2 * 2 - 3 * 3 + 4 * 4
        with pytest.raises(CompileError, match="needs coordinates"):
            Workbook().expr("T_{a} := [1, 2]")  # a vector: its components, not a name's

    def test_a_declared_label_stays_an_index(self, wb):
        wb.exe(r"\mathcal{A} := \mathbb{R}^{2}" "\n" r"a \in \mathcal{A}" "\n" "w_{a} := [1, 2]")
        assert wb.expr("w_{a} w_{a}") == 5

    def test_with_coordinates_a_letter_is_an_index_as_before(self, wb):
        wb.exe("Coordinates := [t, x]")
        with pytest.raises(CompileError, match="expected 2 components"):
            wb.expr(r"\sigma_x := \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}")

    def test_a_label_is_recorded_as_a_reading(self, wb):
        assert readings(wb, r"\lambda_1 := 2") == [
            r"label or index: \lambda_1 is the name \lambda_1, since its scripts label it, "
            "as no index does"
        ]
        wb.exe(r"\lambda_1 := 2")
        assert readings(wb, r"\lambda_1^2") == [
            r"label or index: \lambda_1 is the name \lambda_1, since \lambda_1 is defined on "
            "line 1"
        ]

    def test_a_greek_letter_with_a_label_is_no_statistic(self, wb):
        wb.exe(PAULI)
        (step,) = list(wb.each(r"\sigma_x \sigma_z"))
        assert not any("operatorname" in note for note in step.notes)


class TestRunsOfLetters:
    def test_a_subscript_binds_to_the_last_letter_of_a_run(self, wb):
        # in TeX, iE_1 is i times E_1 (decision 5.7)
        wb.exe("E_1 := 3")
        assert wb.expr(r"\alpha + iE_1 t") == sympy.Symbol("alpha") + 3 * sympy.I * t
        assert wb.expr(r"\alpha + ab_2") == sympy.Symbol("alpha") + sympy.Symbol(
            "a"
        ) * sympy.Symbol("b_2")

    def test_in_plain_text_a_run_with_a_subscript_is_one_name(self, wb):
        assert wb.expr("iE_1 + 1") == sympy.Symbol("iE_1") + 1


class TestWhatIsRaisedOnANameBeingDefined:
    def test_a_number_raised_is_part_of_the_name(self, wb):
        wb.exe(r"\sigma^2 := \frac{1}{4}")
        assert wb.expr(r"\sigma^2 + 1") == sympy.Rational(5, 4)
        assert wb.expr(r"\sigma") == sympy.Symbol("sigma")  # defines no \sigma

    def test_it_is_said_in_a_note(self, wb):
        (step,) = list(wb.each(r"\sigma^2 := \frac{1}{4}"))
        assert step.notes == [r"\sigma^2 is defined as a name of its own, not as a power of \sigma"]

    def test_a_matrix_named_with_a_number_raised(self, wb):
        wb.exe(r"\gamma^0 := \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}")
        assert matrix(wb.expr(r"(\gamma^0)^2")) == sympy.eye(2)

    def test_indices_raised_still_make_a_tensor(self, wb):
        wb.exe("Coordinates := [t, x]\nV^{a} := [1, 2]")
        assert wb.expr("V^{a}").components == sympy.Array([1, 2])


class TestAccentsAndStars:
    def test_an_accented_name_with_a_label(self, wb):
        wb.exe(r"\hat\beta_1 := 2")
        assert wb.expr(r"3 \hat\beta_1") == 6

    def test_an_accented_name_with_indices_is_refused_as_before(self, wb):
        with pytest.raises(CompileError, match="only a name can have a subscript"):
            wb.expr(r"\Box \bar h_{\mu\nu}")

    def test_a_star_on_a_defined_number_or_a_call_is_its_conjugate(self, wb):
        wb.exe("z := 1 + i\nf(x) := i x")
        assert wb.expr("z^*") == 1 - sympy.I
        assert wb.expr("f(2)^*") == -2 * sympy.I
        assert wb.expr("r^*") == sympy.Symbol("r^*")  # where r means nothing, a name

    def test_a_star_on_a_defined_label(self, wb):
        wb.exe(r"\psi_{200} := 1 + i")
        assert sympy.expand(wb.expr(r"\psi_{200}^* \psi_{200}")) == 2
