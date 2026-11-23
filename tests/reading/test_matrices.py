r"""The notation of matrices, read by its operands (design/notation.md, 6, 11, 12, 13 and
23): a superscript T is a transpose on a matrix and a power on a number, I is the identity
beside a matrix and a symbol elsewhere, ρ(A) is a spectral radius of a matrix and a
density's product elsewhere; and a definition of the program's own always wins."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
import sympy

from relativisticpy import Workbook
from relativisticpy.errors import CompileError, EvaluationError

A = r"A := \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}"
x, t, T, k = sympy.symbols("x t T k")


def matrix(value: object) -> sympy.Matrix:
    return sympy.Matrix(value.tolist())  # type: ignore[attr-defined]


def readings(wb: Workbook, source: str) -> list[str]:
    return [
        decision.text()
        for reading in wb.explain(source)
        for decision in reading.decisions
        if decision.kind == "matrix or number"
    ]


class TestMarks:
    @pytest.mark.parametrize("written", ["A^T", r"A^\top", r"A^\intercal", "Aᵀ", r"A^{\rm T}"])
    def test_a_superscript_t_on_a_matrix_is_its_transpose(self, wb, written):
        wb.exe(A)
        assert matrix(wb.expr(written)) == sympy.Matrix([[1, 3], [2, 4]])

    def test_where_t_has_a_value_it_is_a_power(self, wb):
        wb.exe(A + "\nT := 2")
        assert matrix(wb.expr("A^T")) == sympy.Matrix([[7, 10], [15, 22]])

    def test_on_a_symbol_t_is_a_power(self, wb):
        assert wb.expr("x^T") == x**T

    @pytest.mark.parametrize("written", [r"B^\dagger", "B^*", "B†"])
    def test_a_dagger_or_a_star_on_a_matrix_is_its_conjugate_transpose(self, wb, written):
        wb.exe(r"B := \begin{pmatrix} 1 & i \\ 0 & 2 \end{pmatrix}")
        assert matrix(wb.expr(written)) == sympy.Matrix([[1, 0], [-sympy.I, 2]])

    def test_a_dagger_is_recorded_as_a_reading(self, wb):
        wb.exe(A)
        assert readings(wb, r"A^\dagger") == [
            r"matrix or number: A^\dagger is the conjugate transpose of A, since A is a matrix"
        ]

    @pytest.mark.parametrize("written", ["A^+", "A^{+}"])
    def test_a_plus_on_a_matrix_is_its_pseudoinverse(self, wb, written):
        wb.exe(r"A := \begin{pmatrix} 1 & 2 \\ 2 & 4 \\ 3 & 6 \end{pmatrix}")
        expected = np.linalg.pinv([[1, 2], [2, 4], [3, 6]])
        assert np.allclose(np.array(matrix(wb.expr(written)), dtype=float), expected)

    def test_a_star_on_a_name_nothing_defines_is_part_of_it(self, wb):
        assert wb.expr("r^* + 1") == sympy.Symbol("r^*") + 1

    @pytest.mark.parametrize("written", [r"x^\top", r"a^\dagger", r"\mathbf{x}^T"])
    def test_a_transpose_or_a_dagger_of_what_is_no_matrix_is_refused(self, wb, written):
        with pytest.raises(CompileError, match="which is not a matrix"):
            wb.expr(written)


class TestFunctionsOfAMatrix:
    def test_e_to_a_matrix_is_its_exponential(self, wb):
        wb.exe(r"A := \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}")
        rotation = sympy.Matrix([[sympy.cos(t), sympy.sin(t)], [-sympy.sin(t), sympy.cos(t)]])
        assert matrix(wb.expr("e^{A t}")) == rotation
        assert matrix(wb.expr(r"\exp(A t)")) == rotation

    def test_the_exponential_agrees_with_scipy(self, wb):
        found = wb.expr(r"\exp\left(\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\right)")
        expected = scipy.linalg.expm(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert np.allclose(np.array(sympy.N(matrix(found)), dtype=float), expected)

    def test_the_logarithm_with_a_base(self, wb):
        wb.exe(r"\rho := \begin{pmatrix} \frac{1}{2} & \frac{1}{4} \\ \frac{1}{4} & \frac{1}{2} "
               r"\end{pmatrix}")  # fmt: skip
        entropy = wb.expr(r"-\operatorname{tr}(\rho \log_2 \rho)")
        values = np.linalg.eigvalsh([[0.5, 0.25], [0.25, 0.5]])
        assert float(entropy) == pytest.approx(-sum(v * np.log2(v) for v in values))

    @pytest.mark.parametrize("written", [r"\sqrt{V}", "V^{1/2}"])
    def test_a_square_root(self, wb, written):
        wb.exe(r"V := \operatorname{diag}(4, 9)")
        assert matrix(wb.expr(written)) == sympy.diag(2, 3)

    def test_a_power_by_a_symbol(self, wb):
        wb.exe(r"\Lambda := \begin{pmatrix} 1 & 0 \\ 0 & 6 \end{pmatrix}")
        assert matrix(wb.expr(r"\Lambda^k")) == sympy.diag(1, 6**k)
        assert matrix(wb.expr(r"\Lambda^{-1/2}")) == sympy.diag(1, 1 / sympy.sqrt(6))

    def test_a_whole_power_is_a_product_of_matrices(self, wb):
        wb.exe(A)
        assert matrix(wb.expr("A^2")) == sympy.Matrix([[7, 10], [15, 22]])
        assert readings(wb, "A^2") == []


class TestTheIdentity:
    @pytest.mark.parametrize("identity", ["I", r"\mathbb{1}", "𝟙", r"\mathbf{1}"])
    def test_beside_a_matrix_in_a_sum_it_is_the_identity(self, wb, identity):
        wb.exe(A)
        lam = sympy.Symbol("lambda")
        assert wb.expr(rf"\det(A - \lambda {identity})") == sympy.expand(
            (sympy.Matrix([[1, 2], [3, 4]]) - lam * sympy.eye(2)).det()
        )

    def test_it_is_said_in_a_note(self, wb):
        wb.exe(A)
        (step,) = list(wb.each("A - I"))
        assert any("I is read as the identity of the size of A" in note for note in step.notes)

    def test_the_reading_is_of_the_identity_not_of_its_term(self, wb):
        wb.exe(A)
        assert readings(wb, r"A - \lambda I") == [
            "matrix or number: I is the identity of the size of A, since nothing defines I, "
            "and it is added to the matrix A"
        ]

    def test_a_definition_of_i_wins(self, wb):
        wb.exe(A + "\n" + r"I := \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}")
        assert matrix(wb.expr("A - I")) == sympy.Matrix([[1, 1], [2, 4]])

    def test_beside_no_matrix_i_is_a_symbol(self, wb):
        assert wb.expr("I + x") == sympy.Symbol("I") + x

    def test_a_number_added_to_a_matrix_is_that_multiple_of_the_identity(self, wb):
        wb.exe(A)
        assert matrix(wb.expr("A^2 + 4 A + 8")) == sympy.Matrix([[19, 18], [27, 46]])

    def test_beside_a_matrix_that_is_not_square_it_is_refused(self, wb):
        wb.exe(r"B := \begin{pmatrix} 1 & 2 & 3 \end{pmatrix}")
        with pytest.raises(EvaluationError, match="not square"):
            wb.expr("B - I")

    def test_a_number_added_to_a_matrix_that_is_not_square_is_refused(self, wb):
        wb.exe(r"B := \begin{pmatrix} 1 & 2 & 3 \end{pmatrix}")
        with pytest.raises(EvaluationError, match="cannot add a scalar and an array"):
            wb.expr("B + 2")

    def test_equated_to_a_matrix_it_is_the_identity(self, wb):
        wb.exe(r"U := \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}")
        assert wb.expr(r"U^\dagger U = I").holds()
        wb.exe(A)
        assert wb.expr("A^2 - 5 A = 2").holds()  # Cayley and Hamilton: A^2 - 5A = 2I
        assert not wb.expr("A = 5").holds()
        assert wb.expr("A - A = 0").holds()  # the matrix of zeros, as before

    def test_solving_for_a_matrix_nothing_defines_is_refused(self, wb):
        wb.exe(r"A := \begin{pmatrix} 0 & 1 \\ -2 & -3 \end{pmatrix}")
        with pytest.raises(CompileError, match="P multiplies a matrix") as refused:
            wb.expr("solve(A^T P + P A = -I, P)")
        assert refused.value.notes[0] == "an unknown matrix is not supported yet"


class TestTheVectorOfOnes:
    def test_it_is_sized_by_the_matrix_it_meets(self, wb):
        wb.exe(A)
        assert wb.expr(r"\mathbf{1}^T A \mathbf{1}") == 10
        assert matrix(wb.expr(r"A \mathbf{1}")) == sympy.Matrix([[3], [7]])

    def test_it_is_said_in_a_note(self, wb):
        wb.exe(A)
        (step,) = list(wb.each(r"A \mathbf{1}"))
        assert any("vector of ones" in note for note in step.notes)

    def test_beside_a_vector_in_a_sum_it_is_the_vector_of_ones(self, wb):
        wb.exe(r"x := \begin{pmatrix} 1 \\ 2 \end{pmatrix}")
        assert matrix(wb.expr(r"x + \mathbf{1}")) == sympy.Matrix([[2], [3]])

    def test_beside_no_matrix_it_is_the_number_one(self, wb):
        assert wb.expr(r"\mathbf{1} + x") == 1 + x

    def test_beside_a_dot_or_a_cross_it_is_sized_by_the_vector_it_meets(self, wb):
        # the review's programs: \mathbf{1} \cdot \mathbf{w} was w, as if 1 times w
        wb.exe(r"\mathbf{w} := (1, 2, 3)")
        assert wb.expr(r"\mathbf{1}\cdot\mathbf{w}") == 6
        assert wb.expr(r"\mathbf{w} \cdot \mathbf{1}") == 6
        assert list(wb.expr(r"\mathbf{1} \times \mathbf{w}")) == [1, -2, 1]
        (step,) = list(wb.each(r"\mathbf{1}\cdot\mathbf{w}"))
        assert step.notes[0].split("\n")[0] == (
            r"\mathbf{1} is read as the vector of ones as long as \mathbf{w}; to name it, "
            "define it"
        )
        assert wb.expr(r"\mathbf{1} \cdot 3") == 3  # beside a number, the number 1

    def test_a_dot_with_it_agrees_with_its_transpose(self, wb):
        wb.exe(r"w := \begin{pmatrix} 0.2 \\ 0.3 \\ 0.5 \end{pmatrix}")
        assert wb.expr(r"\mathbf{1}^T w") == pytest.approx(1.0)
        assert wb.expr(r"\mathbf{1} \cdot w") == pytest.approx(1.0)

    def test_a_definition_of_it_wins(self, wb):
        # the portfolio's \mathbf{1}^T \Sigma^{-1} \mathbf{1}, with a bold one of its own
        wb.exe(A + "\n" + r"\mathbf{1} := \begin{pmatrix} 1 \\ 2 \end{pmatrix}")
        assert wb.expr(r"\mathbf{1}^T A \mathbf{1}") == 27
        assert wb.expr(r"\mathbf{1}^T \mathbf{1}") == 5
        assert readings(wb, r"\mathbf{1}^T A") == [
            r"matrix or number: \mathbf{1}^T is the transpose of \mathbf{1}, since \mathbf{1} "
            "is a matrix, and T has no value"
        ]

    def test_a_transpose_of_a_decorated_name_is_of_its_value(self, wb):
        wb.exe(r"\bar{\mathbf{e}} := \begin{pmatrix} 1 \\ 2 \end{pmatrix}")
        assert matrix(wb.expr(r"\bar{\mathbf{e}}^T")) == sympy.Matrix([[1, 2]])


class TestOneByOne:
    def test_a_row_times_a_matrix_times_a_column_is_a_number(self, wb):
        wb.exe(
            r"A := \begin{pmatrix} 0 & 1 \\ -2 & -3 \end{pmatrix}" "\n"
            r"B := \begin{pmatrix} 0 \\ 1 \end{pmatrix}" "\n"
            r"C := \begin{pmatrix} 1 & 0 \end{pmatrix}" "\n"
            "D := 0"
        )  # fmt: skip
        s = sympy.Symbol("s")
        assert sympy.simplify(wb.expr("C (s I - A)^{-1} B + D") - 1 / (s**2 + 3 * s + 2)) == 0

    def test_a_quadratic_form_is_a_number(self, wb):
        wb.exe(r"x := \begin{pmatrix} a \\ b \end{pmatrix}" "\n" + A)
        a, b = sympy.symbols("a b")
        assert wb.expr("x^T A x") == sympy.expand(a**2 + 5 * a * b + 4 * b**2)
        (step,) = list(wb.each("x^T A x"))
        assert any("1 × 1 matrix" in note for note in step.notes)


class TestProducts:
    SIGMA_X = r"\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}"

    def test_a_kronecker_product_of_matrices(self, wb):
        found = matrix(wb.expr(rf"{self.SIGMA_X} \otimes {self.SIGMA_X}"))
        assert found == sympy.Matrix(np.kron([[0, 1], [1, 0]], [[0, 1], [1, 0]]))

    def test_a_kronecker_product_is_a_factor_of_its_own(self, wb):
        wb.exe(f"X := {self.SIGMA_X}")
        chi = r"\begin{pmatrix} 0 \\ 1 \\ -1 \\ 0 \end{pmatrix}"
        assert matrix(wb.expr(rf"(X \otimes X) {chi}")) == sympy.Matrix([[0], [-1], [1], [0]])

    def test_a_direct_sum_of_matrices(self, wb):
        found = matrix(wb.expr(rf"{self.SIGMA_X} \oplus \begin{{pmatrix}} 5 \end{{pmatrix}}"))
        assert found == sympy.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 5]])

    def test_a_direct_sum_of_anything_else_is_refused(self, wb):
        with pytest.raises(CompileError, match="a direct sum, which is of matrices"):
            wb.expr(r"\mathbf{8} \oplus \mathbf{1}")

    def test_a_block_matrix(self, wb):
        wb.exe(r"P := \operatorname{diag}(2, 4)" "\n" r"q := \begin{pmatrix} 1 & 1 \end{pmatrix}")
        found = matrix(wb.expr(r"\begin{pmatrix} P & q^T \\ q & 0 \end{pmatrix}"))
        assert found == sympy.Matrix([[2, 0, 1], [0, 4, 1], [1, 1, 0]])


class TestNormsAndEntries:
    @pytest.mark.parametrize(("order", "numpy_order"), [("2", 2), ("F", "fro"), (r"\infty", np.inf),
                                                        ("1", 1)])  # fmt: skip
    def test_a_norm_of_a_matrix(self, wb, order, numpy_order):
        wb.exe(A)
        found = float(wb.expr(rf"\|A\|_{{{order}}}"))
        assert found == pytest.approx(np.linalg.norm([[1, 2], [3, 4]], numpy_order))

    def test_the_norm_of_a_vector(self, wb):
        wb.exe(r"x := \begin{pmatrix} 1 \\ -2 \\ 2 \end{pmatrix}")
        assert wb.expr(r"\|x\|") == 3
        assert wb.expr(r"\|x\|_1") == 5
        assert wb.expr(r"\|x\|_2^2") == 9

    def test_the_norm_of_a_matrix_without_an_order_says_which(self, wb):
        wb.exe(A)
        (step,) = list(wb.each(r"\|A\|"))
        assert any("the 2-norm of A" in note for note in step.notes)

    def test_an_entry_is_counted_from_one(self, wb):
        wb.exe(A)
        assert wb.expr("(A^2)_{12}") == 10
        assert wb.expr("A_{21}") == 3


class TestBookNamesOfAMatrix:
    def test_a_letter_is_a_function_of_a_matrix_only_with_a_matrix(self, wb):
        wb.exe(r"T := \begin{pmatrix} 0 & \frac{1}{4} \\ \frac{1}{4} & 0 \end{pmatrix}")
        assert wb.expr(r"\rho(T)") == sympy.Rational(1, 4)
        w = sympy.Symbol("w")
        assert wb.expr(r"\rho(1 + 3w)") == sympy.Function("rho")(1 + 3 * w)  # as before

    def test_a_definition_of_the_letter_wins(self, wb):
        wb.exe(r"\rho(y) := 2 y" "\n" + A)
        assert matrix(wb.expr(r"\rho(A)")) == 2 * sympy.Matrix([[1, 2], [3, 4]])

    def test_the_largest_and_a_counted_eigenvalue(self, wb):
        wb.exe(r"S := \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}")
        assert wb.expr(r"\lambda_{\max}(S)") == 3
        assert wb.expr(r"\lambda_{\min} S") == 1
        assert wb.expr(r"\lambda_2(S)") == 3

    def test_the_argument_of_an_operator_runs_over_the_factors_after_it(self, wb):
        wb.exe(A + "\n" + r"B := \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}")
        assert wb.expr(r"\operatorname{tr} A B") == 5  # tr(AB), as books read it
        assert wb.expr(r"\operatorname{tr} A \, B").tolist() == [[0, 5], [5, 0]]

    def test_a_function_of_a_matrix_nothing_defines_is_an_unknown_value(self, wb):
        # M may be a matrix of any size: its 1 × 1 case would prove what is claimed of it
        M = sympy.Symbol("M")
        tr, det = sympy.Function("tr"), sympy.Function("det")
        (step,) = list(wb.each(r"\operatorname{tr}(\ln M) = \ln(\det M)"))
        assert step.values == [sympy.Eq(tr(sympy.log(M)), sympy.log(det(M)))]
        assert step.notes == [
            r"\operatorname{tr}(\ln M) is left as an unknown value: nothing defines M, which "
            r"may be a matrix of any size; to compute it, define M first"
            "\n"
            r"\det M is left as an unknown value: nothing defines M, which may be a matrix of "
            "any size; to compute it, define M first"
        ]
        assert wb.expr(r"\operatorname{rank} M") == sympy.Function("rank")(M)
        assert readings(wb, r"\det M") == [
            r"matrix or number: \det M is an unknown value, since nothing defines M, which may "
            "be a matrix of any size"
        ]

    def test_a_claim_about_a_matrix_nothing_defines_relates(self, wb):
        report = wb.check(r"\[ \operatorname{tr}(\ln M) = \ln(\det M) \]")
        assert [finding.verdict for finding in report] == ["relates"]

    def test_of_a_scalar_the_program_defines_it_is_its_1_by_1_case(self, wb):
        wb.exe("a := 3\nCoordinates := [t, x]\nb > 0")
        assert [wb.expr(r"\det a"), wb.expr(r"\det(2)"), wb.expr(r"\det b")] == [
            3,
            2,
            sympy.Symbol("b", positive=True),
        ]
        (step,) = list(wb.each(r"\operatorname{tr} x"))
        assert step.values == [x]
        assert any("the trace of what is no matrix" in note for note in step.notes)

    def test_the_trace_of_a_tensor_of_spacetime_contracts_it_with_the_metric(self, wb):
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        for written in (r"\operatorname{Tr}(R_{\mu\nu})", r"\operatorname{tr} R_{\mu\nu}"):
            assert wb.expr(f"{written} = R") is sympy.true

    def test_the_trace_of_a_map_keeps_to_its_space(self, wb):
        wb.exe(r"\mathcal{V} := \mathbb{R}^{2}" "\n" r"a, b \in \mathcal{V}" "\n"
               r"A^{a}{}_{b} := [[1, 2], [3, 4]]")  # fmt: skip
        assert wb.expr(r"\operatorname{rank} A^{a}{}_{b}") == 2
        exponential = wb.expr(r"expm(A^{a}{}_{b})")
        assert [space.name for space in exponential.spaces] == ["V", "V"]
