r"""The one resolver of names with a subscript, and of names with brackets after them
(:mod:`relativisticpy.reading.subscripts`; design/notation.md, decision 5): what ``c_1``,
``x_i``, ``f_x``, ``H_n``, ``X_k`` and ``x[n]`` mean is decided in one place, in one order,
with a note where the program defines the base as two things. Before, each layer of the
compiler had its own test, and the order of the layers decided."""

from __future__ import annotations

import pytest
import sympy

from relativisticpy import Workbook
from relativisticpy.errors import CompileError
from relativisticpy.language import parse
from relativisticpy.reading.subscripts import PRECEDENCE, Kind, picks_a_member, subscripted

n, x, y = sympy.symbols("n x y")


def notes(wb: Workbook, source: str) -> list[str]:
    """The notes the last statement of ``source`` is read with."""
    return list(wb.explain(source)[-1].notes)


def written(source: str):
    return subscripted(parse(source).statements[0].expr)


class TestWhatIsWritten:
    """A name with one subscript is taken apart as the parser read it, whether the lexer
    joined it into one name or it is written with braces."""

    def test_a_joined_name_and_a_braced_one_are_one_form(self):
        joined, braced = written("c_1"), written("c_{1}")
        assert (joined.base, joined.shown, joined.joined) == ("c", "1", True)
        assert (braced.base, braced.shown, braced.joined) == ("c", "1", False)
        assert joined.subscript == braced.subscript

    def test_the_subscript_is_an_expression(self):
        assert written("a_{n+1}").shown == "n+1"
        assert written(r"\lambda_1").base == "lambda"
        assert written("f_{xy}").labels == ("x", "y")
        assert written("f_xy").labels == ("x", "y")
        assert written("T_1^2") is None  # the lexer joins T_1, the name the 2 is raised on
        assert len(written("T_{1}^2").raised) == 1

    def test_what_is_no_name_with_one_subscript(self):
        assert written("x^2") is None
        assert written("x") is None
        assert written(r"T^{\mu}{}_{\nu\rho}") is not None  # one subscript, of two labels
        assert written(r"T_{\mu}{}_{\nu}") is None

    def test_a_subscript_picks_a_member_by_what_it_is(self):
        nothing = lambda name: False  # noqa: E731
        assert picks_a_member(written("c_1").subscript, nothing)
        assert picks_a_member(written("c_{-1}").subscript, nothing)
        assert picks_a_member(written("c_{n+1}").subscript, nothing)
        assert not picks_a_member(written("c_k").subscript, nothing)
        assert picks_a_member(written("c_k").subscript, lambda name: name == "k")


class TestPrecedence:
    """The readings are tried in one order, a user's definition first."""

    def test_the_order_is_written_down(self):
        assert PRECEDENCE[0] is Kind.DEFINED
        assert PRECEDENCE[-1] is Kind.LABEL
        assert PRECEDENCE.index(Kind.MEMBER) < PRECEDENCE.index(Kind.ELEMENT)
        assert PRECEDENCE.index(Kind.PARTIAL) < PRECEDENCE.index(Kind.BOOK)

    def test_a_name_the_program_defines_wins(self, wb):
        wb.exe("c_n := 1/n^2\nc_1 := 5")
        assert wb.expr("c_1") == 5
        assert wb.expr("c_2") == sympy.Rational(1, 4)

    def test_a_member_of_a_family(self, wb):
        wb.exe("c_n := 1/n^2")
        assert wb.expr("c_3") == sympy.Rational(1, 9)
        assert wb.expr("c_{3}") == sympy.Rational(1, 9)
        assert wb.expr("c_k") == sympy.Symbol("c_k")  # a letter nothing binds

    def test_a_partial_derivative(self, wb):
        wb.exe("f(x, y) := x^2 y")
        assert wb.expr("f_x") == 2 * x * y
        assert wb.expr("f_{xy}") == 2 * x
        assert wb.expr("f_x := 7\nf_x") == 7  # a definition of the name wins

    def test_a_book_name_of_an_integer(self, wb):
        wb.exe(r"n \in \mathbb{N}")
        assert wb.expr("H_n") == sympy.harmonic(sympy.Symbol("n", integer=True, nonnegative=True))
        assert wb.expr("H_0") == sympy.Symbol("H_0")  # a number keeps it a name

    def test_a_label_is_a_name_of_its_own(self, wb):
        assert wb.expr(r"\lambda_1 + r_s") == sympy.Symbol("lambda_1") + sympy.Symbol("r_s")

    def test_an_element_of_a_list_outside_a_sum_is_refused(self, wb):
        wb.exe("x := [2, 4, 4]")
        with pytest.raises(CompileError, match="x is a list"):
            wb.expr("x_1")
        with pytest.raises(CompileError, match="x is a list"):
            wb.expr("x_{1}")


class TestAFamilyAndAListOfOneLetter:
    """Finding 9 of the review of the reading stack: with ``c_n := 1/n^2`` and
    ``c := [10, 20, 30]``, ``\\sum_{i=1}^{3} c_i`` was 60, the list's, while
    ``c_1 + c_2 + c_3`` and ``\\sum_{n=1}^{3} c_n`` were 49/36, the family's: the letter of
    the index decided. A family comes before a list (decision 5), whatever the letter, and
    the reading says so."""

    PROGRAM = "c_n := \\frac{1}{n^2}\nc := [10, 20, 30]"

    def test_the_sum_agrees_with_its_terms_whatever_its_index(self, wb):
        wb.exe(self.PROGRAM)
        terms = wb.expr("c_1 + c_2 + c_3")
        assert terms == sympy.Rational(49, 36)
        assert wb.expr(r"\sum_{i=1}^{3} c_i") == terms
        assert wb.expr(r"\sum_{n=1}^{3} c_n") == terms

    def test_the_reading_says_which_was_taken(self, wb):
        wb.exe(self.PROGRAM)
        said = "c_1 is read as a member of the family c_n, not as an element of the list c"
        assert said in notes(wb, "c_1")
        said = "c_i is read as a member of the family c_n, not as an element of the list c"
        assert said in notes(wb, r"\sum_{i=1}^{3} c_i")

    def test_a_list_alone_is_summed_as_before(self, wb):
        wb.exe("c := [10, 20, 30]")
        assert wb.expr(r"\sum_{i=1}^{3} c_i") == 60
        assert notes(wb, r"\sum_{i=1}^{3} c_i") == []
