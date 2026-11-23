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
from relativisticpy.language import Source, parse
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


class TestASumOverAFamilyAgreesWithItsTerms:
    """Finding 9: a finite sum over a family is taken term by term, through the lookup its
    members written out use, so that it has the members the program gives. Before,
    ``a_0 := 1`` beside ``a_n := 1/n`` made ``a_0 + a_1 + a_2 + a_3`` 17/6 and
    ``\\sum_{n=0}^{3} a_n`` zoo, the family's value at 0."""

    def test_a_member_the_program_gives_is_summed(self, wb):
        wb.exe("a_0 := 1\na_n := \\frac{1}{n}")
        terms = wb.expr("a_0 + a_1 + a_2 + a_3")
        assert terms == sympy.Rational(17, 6)
        assert wb.expr(r"\sum_{n=0}^{3} a_n") == terms
        assert wb.expr(r"\prod_{n=0}^{2} (1 + a_n)") == 6

    def test_a_member_given_after_the_family_is_summed(self, wb):
        wb.exe("c_n := \\frac{1}{n^2}\nc_1 := 5")
        terms = wb.expr("c_1 + c_2 + c_3")
        assert terms == sympy.Rational(193, 36)
        assert wb.expr(r"\sum_{n=1}^{3} c_n") == terms

    def test_a_sum_past_a_given_member_whose_terms_cannot_be_taken_is_refused(self, wb):
        wb.exe("a_0 := 1\na_n := \\frac{1}{n}")
        with pytest.raises(CompileError) as raised:
            wb.expr(r"\sum_{n=0}^{N} a_n")
        assert raised.value.message == (
            "the sum runs over a_0, which the program gives apart from a_n := ..., and the sum "
            "has limits that are no whole numbers: it cannot be taken term by term"
        )
        assert raised.value.hint == (
            r"write the members given apart from the sum, as a_0 + \sum_{n=1}^{...} a_n"
        )
        big = sympy.Symbol("N")
        assert wb.expr(r"a_0 + \sum_{n=1}^{N} a_n") == 1 + sympy.harmonic(big)  # the hint works

    def test_a_sum_that_starts_after_the_given_members_has_the_familys_value(self, wb):
        wb.exe("a_0 := 1\na_n := \\frac{1}{n^2}")
        assert wb.expr(r"\sum_{n=1}^{\infty} a_n") == sympy.pi**2 / 6

    def test_a_recurrence_is_summed_from_the_members_given(self, wb):
        wb.exe("a_0 := 1\na_1 := 1\na_{n+2} := a_{n+1} + a_n")
        assert wb.expr(r"\sum_{n=0}^{5} a_n") == 20

    def test_a_list_and_a_family_in_one_sum(self, wb):
        wb.exe("x := [1, 2, 3]\ny_n := n^2")
        assert wb.expr(r"\sum_{i=1}^{3} x_i y_i") == 36  # was 6 y_i
        assert wb.expr(r"\sum_{i=0}^{2} x_i y_i") == 14  # counted from the lower limit
        with pytest.raises(CompileError, match="a sum over a part of a list"):
            wb.expr(r"\sum_{i=1}^{2} x_i y_i")


class TestASubscriptTheIndexOfASumRunsThrough:
    """Finding 10: a subscripted name that the index of a sum runs through is expanded, or
    the sum is refused; it is never one constant. Before, ``\\sum_{n=1}^{3} a_n`` was
    ``3 a_n``, and ``\\sum_{k=1}^{3} c_{2k}`` of a family c was ``3 c_2k``."""

    def test_names_of_their_own_are_written_out(self, wb):
        a = sympy.symbols("a_1 a_2 a_3")
        assert wb.expr(r"\sum_{n=1}^{3} a_n") == sum(a)
        assert wb.expr(r"\prod_{n=1}^{3} a_n") == a[0] * a[1] * a[2]
        assert wb.expr(r"\sum_{n=3}^{2} a_n") == 0  # no terms

    def test_the_names_the_program_defines_are_used(self, wb):
        wb.exe("\\lambda_1 := 2\n\\lambda_2 := 3")
        assert wb.expr(r"\sum_{a=1}^{2} \lambda_a^2") == 13
        assert wb.expr(r"\sum_{a=1}^{3} \lambda_a^2") == 13 + sympy.Symbol("lambda_3") ** 2

    def test_a_book_letter_under_a_sum_is_a_name_for_each_term_and_says_so(self, wb):
        wb.exe(r"n \in \mathbb{N}")
        integer = sympy.Symbol("n", integer=True, nonnegative=True)
        assert wb.expr("H_n") == sympy.harmonic(integer)
        names = sympy.symbols("H_1 H_2 H_3")
        assert wb.expr(r"\sum_{n=1}^{3} H_n") == sum(names)  # was 3 H_n
        assert notes(wb, r"\sum_{n=1}^{3} H_n") == [
            "H_n is a name for each n of the sum, H_1, …, H_3, as the coefficients of a "
            "series are; the harmonic number is harmonic(n)"
        ]
        assert wb.expr(r"\sum_{n=1}^{3} harmonic(n)") == sympy.Rational(13, 3)  # as it says

    def test_an_expression_in_the_index_picks_a_member(self, wb):
        k = sympy.Symbol("k")
        wb.exe("c_n := \\frac{1}{n^2}")
        assert wb.expr("c_{2k}") == 1 / (4 * k**2)
        assert wb.expr(r"\sum_{k=1}^{3} c_{2k}") == sympy.Rational(49, 144)
        assert wb.expr("b_n := c_{2n}\nb_1") == sympy.Rational(1, 4)  # was the symbol c_2n

    def test_a_member_given_at_the_value_of_an_expression_is_used(self, wb):
        wb.exe("c_n := \\frac{1}{n^2}\nc_2 := 7")
        terms = wb.expr("c_{2} + c_{4} + c_{6}")
        assert wb.expr(r"\sum_{k=1}^{3} c_{2k}") == terms == sympy.Rational(1021, 144)

    @pytest.mark.parametrize(
        ("written", "why"),
        [
            (r"\sum_{n=1}^{N} a_n", "has limits that are no whole numbers"),
            (r"\sum_{n=1}^{\infty} a_n \sin(n x)", "has limits that are no whole numbers"),
            (r"\sum_{n=1}^{200} a_n", "has over 100 terms"),
            (r"\sum_{n=3}^{1} a_n", "has limits that run backwards"),
        ],
    )
    def test_a_sum_whose_terms_cannot_be_taken_is_refused(self, wb, written, why):
        with pytest.raises(CompileError) as raised:
            wb.expr(written)
        assert raised.value.message == (
            "a_n names nothing, and the sum runs over its subscript: read as written, it "
            f"would be one symbol in every term, and the sum {why}"
        )
        wb.exe("a_n := n")  # the hint works: a family is summed by its value
        big = sympy.Symbol("N")
        assert sympy.expand(wb.expr(r"\sum_{n=1}^{N} a_n") - big * (big + 1) / 2) == 0

    def test_a_label_then_a_number_is_no_expression(self, wb):
        # C_{\mathrm{A}0}, the initial concentration of A, is a name, not C at A times 0
        wb.exe("C_\\mathrm{A}(t) := \\frac{C_{\\mathrm{A}0}}{1 + k C_{\\mathrm{A}0} t}")
        initial, rate = sympy.symbols("C_A0 k")
        assert wb.expr(r"C_\mathrm{A}(0)") == initial
        assert wb.expr(r"C_\mathrm{A}(1)") == initial / (1 + rate * initial)

    def test_a_list_and_a_name_in_one_sum(self, wb):
        wb.exe("x := [1, 2, 3]")
        y = sympy.symbols("y_1 y_2 y_3")
        assert wb.expr(r"\sum_{i=1}^{3} x_i y_i") == y[0] + 2 * y[1] + 3 * y[2]  # was 6 y_i


def latex(wb: Workbook, source: str) -> list[object]:
    """What a program written as LaTeX shows, every statement LaTeX."""
    return wb.exe(Source(source, latex=True))


class TestBracketsAfterAList:
    """Finding 14: brackets after a list, in a sum, multiplied it: with
    ``x := [1, 2, 3, 4]``, ``\\sum_{n=0}^{3} x[n]`` was ``[6, 12, 18, 24]``. They read the
    element, counted from the lower limit of the sum that ranges over the whole list
    (decision 17), and are refused anywhere else."""

    def test_in_a_sum_over_the_whole_list_they_are_its_elements(self, wb):
        wb.exe("x := [1, 2, 3, 4]")
        assert latex(wb, r"\sum_{n=0}^{3} x[n]") == [10]
        assert latex(wb, r"\sum_{n=1}^{4} x[n]") == [10]  # counted from the lower limit
        assert latex(wb, r"\sum_{n=1}^{4} x[n]^2") == [30]
        assert latex(wb, r"\prod_{n=0}^{3} x[n]") == [24]

    def test_a_discrete_fourier_transform_written_with_brackets(self, wb):
        wb.exe("x := [1, 2, 3, 4]")
        (value,) = latex(wb, r"\sum_{n=0}^{3} x[n] e^{-2\pi i n/4}")
        assert sympy.simplify(value - (-2 + 2 * sympy.I)) == 0

    def test_a_sum_over_a_part_of_the_list_is_refused(self, wb):
        wb.exe("x := [1, 2, 3, 4]")
        with pytest.raises(CompileError, match="a sum over a part of a list"):
            latex(wb, r"\sum_{n=0}^{2} x[n]")

    @pytest.mark.parametrize("written", ["x[2]", "x[n]", r"x[n] + \alpha"])
    def test_outside_a_sum_they_are_refused(self, wb, written):
        wb.exe("x := [1, 2, 3, 4]")
        with pytest.raises(CompileError) as raised:
            latex(wb, written)
        assert raised.value.message == "a list is not indexed with [ ]"
        assert raised.value.notes == [
            "the elements of a list are read in a sum over all of it, counted from its lower "
            r"limit, as in \sum_{n=0}^{3} x[n]"
        ]

    def test_a_bracket_that_holds_no_point_multiplies_as_before(self, wb):
        wb.exe("x := [1, 2]")
        epsilon = sympy.Symbol("epsilon")
        (value,) = latex(wb, r"x[1 + \epsilon]")
        assert list(value) == [1 + epsilon, 2 + 2 * epsilon]


class TestASequenceAndAFamilyOfOneLetter:
    """Finding 16: a sequence and a family of the same letter did not see each other:
    after ``x[n] := n^2``, ``\\sum_{n=0}^{3} x_n`` was 12, x_n read as the derivative of
    x(n); and after ``x_n := n^2``, ``\\sum_{n=0}^{3} x[n]`` was 6x. One lookup reads
    both, however the member is written."""

    def test_a_sequence_written_with_a_subscript(self, wb):
        latex(wb, "x[n] := n^2")
        assert latex(wb, r"\sum_{n=0}^{3} x_n") == [14]
        assert latex(wb, "x_3") == [9]
        assert latex(wb, "x_{n-1}") == [(n - 1) ** 2]
        assert notes(wb, "x_3") == ["x_3 is read as x[3], the sequence x at 3"]

    def test_a_sequence_defined_with_the_subscript_of_another(self, wb):
        latex(wb, "x[n] := n^2\ny[n] := x_n + 1")
        assert latex(wb, "y[3]") == [10]  # was 2n + 1, of the derivative of x(n)

    def test_a_family_written_with_brackets(self, wb):
        wb.exe("x_n := n^2")
        assert latex(wb, r"\sum_{n=0}^{3} x[n]") == [14]
        assert latex(wb, "x[3]") == [9]
        assert wb.expr("x[3]") == 9  # in plain text too
        assert latex(wb, "x[n-1]") == [(n - 1) ** 2]

    def test_a_member_the_program_gives_is_the_same_however_it_is_written(self, wb):
        wb.exe("x_n := n^2\nx_3 := 100")
        assert latex(wb, "x[3]") == [100]
        assert latex(wb, r"\sum_{n=0}^{3} x[n]") == latex(wb, "x_0 + x_1 + x_2 + x_3") == [105]


class TestBracketsAfterAnExpectation:
    """Finding 25: ``E[X]`` of a random variable was claimed as a sequence before
    probability read it: in plain text it was refused as Python's indexing, and in LaTeX
    it carried two notes that contradicted each other."""

    def test_e_of_a_random_variable_in_square_brackets_is_its_expectation(self, wb):
        wb.exe(r"X \sim \mathcal{N}(2, 9)")
        assert wb.expr("E(X)") == 2
        assert wb.expr("E[X]") == 2  # was refused: "indexes as Python does"
        alpha = sympy.Symbol("alpha")
        assert wb.expr(r"E[X] + \alpha") == alpha + 2
        assert notes(wb, r"E[X] + \alpha") == [
            "E[X] is read as an expectation, as E acts on random variables"
        ]
        assert wb.expr(r"P[X > 2]") == sympy.Rational(1, 2)

    def test_e_of_anything_else_multiplies_as_before(self, wb):
        wb.exe(r"X \sim \mathcal{N}(2, 9)")
        energy, k, alpha = sympy.symbols("E k alpha")
        assert latex(wb, r"E[k] + \alpha") == [energy * k + alpha]
