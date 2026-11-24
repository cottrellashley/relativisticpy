r"""Typefaces that make names of their own (design/notation.md, decision 9): a bold name
and a plain one are two names where the program defines both, and where it defines one of
them, the other stands for it, with a note."""

import pytest

from relativisticpy import Workbook
from relativisticpy.errors import CompileError

STANDS_FOR = (
    "{written}, which the program does not define, is read as {defined}, which it does: a "
    "typeface makes a name of its own, so that defining {written} makes it one"
)


def notes(wb: Workbook, source: str) -> list[str]:
    """The notes of the one step of a one-statement program."""
    (step,) = list(wb.each(source))
    assert step.error is None, step.error
    return step.notes


class TestTwoNames:
    def test_a_plain_name_defined_after_a_bold_one_is_a_name_of_its_own(self, wb):
        # \mathbf{w} := (1, 2, 3), then w := 7: \mathbf{w} was 7, without a word
        wb.exe("\\mathbf{w} := (1, 2, 3)\nw := 7")
        assert list(wb.expr(r"\mathbf{w}")) == [1, 2, 3]
        assert wb.expr("w") == 7
        assert notes(wb, r"\mathbf{w}") == notes(wb, "w") == []
        assert set(wb.names()) >= {"w", r"\mathbf{w}"}

    def test_a_bold_name_defined_after_a_plain_one_is_a_name_of_its_own(self, wb):
        wb.exe("w := 7\n\\mathbf{w} := (1, 2, 3)\n\\mathbf{w} := (0, 0, 1)")
        assert wb.expr("w") == 7
        assert list(wb.expr(r"\mathbf{w}")) == [0, 0, 1]

    def test_a_vector_over_the_cube_of_its_length(self, wb):
        # the review's program: \frac{\mathbf{r}}{r^3} was 1/(x^2 + y^2 + z^2), r having
        # replaced \mathbf{r}, without a word
        wb.exe("\\mathbf{r} := (x, y, z)\nr := \\sqrt{x^2 + y^2 + z^2}")
        found = [str(component) for component in wb.expr(r"\frac{\mathbf{r}}{r^3}")]
        assert found == [f"{c}/(x**2 + y**2 + z**2)**(3/2)" for c in "xyz"]

    @pytest.mark.parametrize("bold", [r"\boldsymbol{\sigma}", r"\bm{\sigma}"])
    def test_bold_however_it_is_written(self, wb, bold):
        wb.exe(f"{bold} := (1, 2)\n\\sigma := 3")
        assert list(wb.expr(r"\boldsymbol{\sigma}")) == [1, 2]
        assert wb.expr(r"\sigma") == 3


class TestOneNameSaysSo:
    def test_a_bold_name_is_the_plain_one_the_program_defines_and_says_so(self, wb):
        wb.exe("w := 7")
        assert wb.expr(r"\mathbf{w}") == 7
        assert notes(wb, r"\mathbf{w}") == [STANDS_FOR.format(written=r"\mathbf{w}", defined="w")]

    def test_a_plain_name_is_the_bold_one_the_program_defines_and_says_so(self, wb):
        wb.exe(r"\mathbf{w} := (1, 2, 3)")
        assert list(wb.expr("w")) == [1, 2, 3]
        assert notes(wb, "w") == [STANDS_FOR.format(written="w", defined=r"\mathbf{w}")]
        assert notes(wb, r"\mathbf{w}") == []

    def test_a_calligraphic_letter_too(self, wb):
        # L := 5, then \mathcal{L} + 1 was 6, without a word
        wb.exe("L := 5")
        assert wb.expr(r"\mathcal{L} + 1") == 6
        assert notes(wb, r"\mathcal{L} + 1") == [
            STANDS_FOR.format(written=r"\mathcal{L}", defined="L")
        ]

    def test_a_name_written_with_more_than_its_typeface_says_nothing(self, wb):
        # each was said to stand for the program's name in another typeface
        wb.exe("\\mathbf{b}^{(1)} := (1, 2)\n\\mathcal{L} := [[1, 0], [0, 2]]")
        assert notes(wb, r"\mathbf{b}^{(1)}") == []  # the label makes the name b^(1)
        assert wb.expr(r"\lambda_{\max}(\mathcal{L})") == 2
        assert not any("typeface" in note for note in notes(wb, r"\lambda_{\max}(\mathcal{L})"))

    def test_the_variable_of_an_optimum_is_its_own(self, wb):
        wb.exe(
            r"A := \begin{pmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \end{pmatrix}" "\n"
            r"\mathbf{b} := \begin{pmatrix} 1 \\ 2 \end{pmatrix}"
        )  # fmt: skip
        program = (
            r"\arg\min_{\mathbf{x}} \|\mathbf{x}\|_2 \quad \text{subject to} \quad "
            r"A\mathbf{x} = \mathbf{b}"
        )
        assert not any("typeface" in note for note in notes(wb, program))

    def test_a_definition_that_fails_leaves_the_names_as_they_were(self, wb):
        wb.exe(r"\mathbf{w} := (1, 2, 3)")
        with pytest.raises(CompileError):
            wb.exe("w := eig(x)")
        assert list(wb.expr("w")) == [1, 2, 3]
        assert notes(wb, "w") == [STANDS_FOR.format(written="w", defined=r"\mathbf{w}")]
