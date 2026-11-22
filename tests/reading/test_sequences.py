r"""Sequences, read as signal processing writes them (design/notation.md, 31): ``x[n]`` with
square brackets is a sequence where the name is one, or where nothing defines δ or u, which
are the unit impulse and the unit step; a list, a tensor or data is read as before. Sums
over a sequence's index, difference equations at initial rest, and the transforms of
sequences."""

from __future__ import annotations

import pytest
import sympy

from relativisticpy import Workbook, display
from relativisticpy.errors import CompileError, ParseError, RelativisticPyError
from relativisticpy.integrations.sequences import impulse, step
from relativisticpy.language import parse, to_sexpr
from relativisticpy.language.source import Source

n, z, a, b, w = sympy.symbols("n z a b omega")


def notes(wb: Workbook, source: str) -> list[str]:
    return [note for each in wb.each(Source(source, latex=True)) for note in each.notes]


def latex(wb: Workbook, source: str) -> list[object]:
    """What a program written as LaTeX shows, every statement LaTeX."""
    return wb.exe(Source(source, latex=True))


def same(found: object, expected: object) -> bool:
    """Whether two values are equal, their symbols compared by name, whatever is assumed of
    them."""
    found, expected = (plain(sympy.sympify(value)) for value in (found, expected))
    return sympy.simplify(found - expected) == 0


def plain(value: sympy.Basic) -> sympy.Basic:
    return value.xreplace({s: sympy.Symbol(s.name) for s in value.free_symbols})


class TestParsing:
    def test_a_name_with_a_point_in_brackets_is_defined_as_the_compiler_reads_it(self):
        statement = parse(r"x[n] := a^n u[n]").statements[0]
        assert to_sexpr(statement) == "(:= (* x (list n)) (* (* (a ^[n]) u) (list n)))"

    def test_after_a_name_the_reader_decides_and_the_parser_refuses_the_rest(self):
        assert to_sexpr(parse("y[2]")) == "(* y (list 2))"
        with pytest.raises(ParseError, match=r"\[1, 2\]\[0\] indexes as Python does"):
            parse("[1, 2][0]")


class TestDefinitions:
    def test_a_sequence_is_a_function_of_its_label(self, wb):
        values = latex(wb, "x[n] := a^n u[n]\nx[3]\nx[-2]\nx[n - 1]")
        assert values[:2] == [a**3, 0]
        assert values[2] == a ** (n - 1) * step(n - 1)

    def test_one_built_of_impulses_is_its_values(self, wb):
        latex(wb, r"h[n] := \delta[n] - \delta[n-1]")
        assert [wb.expr(f"h[{k}]") for k in (-1, 0, 1, 2)] == [0, 1, -1, 0]

    def test_written_with_a_cosine_it_is_the_samples_of_the_cosine(self, wb):
        latex(wb, r"x[n] := \cos(\omega_0 n)")
        assert latex(wb, "x[0]\nx[2]") == [1, sympy.cos(2 * sympy.Symbol("omega_0"))]

    def test_plain_text_reads_it_too(self, wb):
        assert wb.exe("x[n] := 2^n u[n]\nx[3] + x[-1]") == [8]

    def test_the_sequence_named_alone_is_at_its_label(self, wb):
        assert latex(wb, "x[n] := n u[n]\nx") == [n * step(n)]

    def test_a_definition_at_a_point_is_refused(self, wb):
        with pytest.raises(CompileError, match=r"a sequence is defined at its label"):
            latex(wb, "x[0] := 1")

    def test_defined_with_equals_where_it_has_no_meaning_yet(self, wb):
        assert latex(wb, "x[n] = 3^n u[n]\nx[2]") == [9]


class TestImpulseAndStep:
    def test_undefined_delta_and_u_are_the_impulse_and_the_step(self, wb):
        assert latex(wb, r"\delta[0] + \delta[3] + u[0] - u[-1] + u[5]") == [3]

    def test_a_note_says_so(self, wb):
        assert notes(wb, r"u[n - 2]") == [
            "u[n - 2] is read as the unit step, step; for a sequence of your own, define u[n] "
            ":= ... first"
        ]

    def test_where_a_sequence_is_used_its_definition_said_it(self, wb):
        assert len(notes(wb, r"x[n] := \delta[n]")) == 1
        assert notes(wb, "x[3]") == []

    def test_a_definition_of_ones_own_wins(self, wb):
        latex(wb, r"u := 4")
        assert latex(wb, "u[2]") == [8]  # u times 2, as before

    def test_a_point_that_is_no_integer_keeps_delta_a_variation(self, wb):
        found = latex(wb, r"\delta[(-g^{-1})^{-1/2}]")[0]
        assert found.has(sympy.Symbol("delta"))  # a product, as Carroll writes a variation
        assert not found.has(impulse)

    def test_they_print_as_books_write_them_and_read_back(self, wb):
        value = step(n - 1) * impulse(n)
        assert display.latex(value) == r"\delta\left[n\right] u\left[n - 1\right]"
        assert display.unicode(step(n - 1)) == "u[n − 1]"
        assert display.text(value) == "impulse(n)*step(n - 1)"
        for shown in (display.latex(value), display.unicode(value), display.text(value)):
            assert wb.expr(shown) == value


class TestWhatIsNoSequence:
    def test_a_list_is_indexed_as_before(self, wb):
        wb.exe("x := [1, 2, 3]")
        with pytest.raises(ParseError, match=r"x\[1\] indexes as Python does"):
            wb.exe("x[1]")
        with pytest.raises(CompileError, match=r"a list is not indexed with \[ \]"):
            latex(wb, "x[1]")

    def test_brackets_after_a_name_nothing_defines_multiply_it_with_a_note(self, wb):
        assert latex(wb, "c[n + 1]") == [sympy.Symbol("c") * (n + 1)]
        assert notes(wb, "c[n]") == [
            "c[n] is read as c times n, as nothing defines c; for a sequence, define c[n] := "
            "... first"
        ]

    def test_brackets_after_a_function_of_one_variable_are_its_value(self, wb):
        wb.exe("f(x) := x^2")
        assert latex(wb, "f[3]") == [9]
        assert notes(wb, "f[k]") == ["f[k] is read as f(k), f being a function"]

    def test_an_asterisk_between_sequences_is_their_convolution(self, wb):
        # as tests/reading/test_convolutions.py says in full (design/notation.md, 27)
        latex(wb, "x[n] := \\delta[n]\nh[n] := u[n]")
        assert latex(wb, "x[n] * h[n]") == [step(n)]
        assert latex(wb, r"x[n] \cdot h[n]") == [impulse(n) * step(n)]

    def test_a_functional_is_no_sequence(self, wb):
        # one name at one point, a value written with no sequence: S[q] = ... as before
        assert notes(wb, r"S[q] = \int_0^1 q^2 \, dt") == [
            "S[q] is read as S times q, as nothing defines S; for a sequence, define S[n] := "
            "... first"
        ]

    def test_brackets_holding_more_than_an_integer_are_read_as_before(self, wb):
        assert latex(wb, "x[t^2]") == [sympy.Symbol("x") * sympy.Symbol("t") ** 2]


class TestSums:
    def test_an_impulse_picks_one_term(self, wb):
        assert latex(wb, r"\sum_{k=-\infty}^{\infty} 2^k \delta[k - 3]") == [8]

    def test_a_step_cuts_the_sum_to_where_it_is_1(self, wb):
        program = "h[n] := (-0.8)^n u[n]\n\\sum_{n=-\\infty}^{\\infty} |h[n]|"
        assert latex(wb, program) == [pytest.approx(5)]

    def test_the_convolution_sum_of_impulses(self, wb):
        program = (
            r"x[n] := \delta[n] + 2\delta[n-1] + 3\delta[n-2]" "\n"
            r"h[n] := \delta[n] + \delta[n-1]" "\n"
            r"y[n] := \sum_{k=-\infty}^{\infty} x[k] h[n-k]" "\n"
            "[y[0], y[1], y[2], y[3], y[4]]"
        )  # fmt: skip
        assert latex(wb, program) == [[1, 3, 5, 3, 0]]

    def test_the_convolution_of_two_exponentials_under_what_is_assumed(self, wb):
        program = (
            "n \\in \\mathbb{Z}\nn \\geq 0\na \\neq b\nx[n] := a^n u[n]\nh[n] := b^n u[n]\n"
            r"\sum_{k=-\infty}^{\infty} x[k] h[n-k]"
        )
        *_, found = latex(wb, program)
        assert same(found, (a ** (n + 1) - b ** (n + 1)) / (a - b))

    def test_without_limits_a_sum_over_a_sequences_index_is_over_every_integer(self, wb):
        latex(wb, r"x[n] := \left(\frac{1}{2}\right)^n u[n - 1]")
        assert latex(wb, r"\sum_n x[n]") == [1]
        assert notes(wb, r"\sum_n x[n]") == [
            r"\sum_{n} runs over every integer, as signal processing sums over the index of a "
            "sequence"
        ]

    def test_a_pulse_of_steps_sums_to_its_length(self, wb):
        # u[n] - u[n - 4] is 1 from 0 to 3: not a sum from 0 on less one from 4 on, two
        # sums without end whose difference has no value
        latex(wb, r"x[n] := u[n] - u[n-4]")
        assert latex(wb, r"\sum_n x[n]") == [4]
        assert latex(wb, r"\sum_{n=-\infty}^{\infty} x[n]") == [4]
        latex(wb, r"X(e^{j\omega}) := \sum_n x[n] e^{-j\omega n}")
        assert latex(wb, r"X(e^{j 0})") == [4]

    def test_a_sum_without_a_value_is_kept_as_written(self, wb):
        latex(wb, r"x[n] := 2 u[n] - u[n-4]")
        (found,) = latex(wb, r"\sum_n x[n]")
        assert isinstance(found, sympy.Sum)

    def test_a_finite_sum_of_exponentials_is_its_value(self, wb):
        program = (
            "j := \\sqrt{-1}\nN := 8\nx[n] := \\cos\\left(\\frac{2\\pi n}{N}\\right)\n"
            r"X[k] := \sum_{n=0}^{N-1} x[n] e^{-j 2\pi k n / N}" "\n[X[0], X[1], X[2]]"
        )  # fmt: skip
        assert latex(wb, program) == [[0, 4, 0]]


class TestDifferenceEquations:
    def test_a_sequence_written_with_its_own_members_is_at_initial_rest(self, wb):
        latex(wb, r"y[n] := \frac{1}{2} y[n-1] + \delta[n]")
        assert latex(wb, "y[-1]\ny[0]\ny[3]") == [0, 1, sympy.Rational(1, 8)]
        assert same(latex(wb, "y[n]")[0], sympy.Rational(1, 2) ** n * step(n))

    def test_a_note_says_it_is_at_rest(self, wb):
        assert notes(wb, r"y[n] := 2 y[n-1] + \delta[n - 1]")[0] == (
            "y[n] is defined by a difference equation, at initial rest, as books take it: y is "
            "0 before its input starts"
        )

    def test_written_with_equals_it_defines_the_sequence_too(self, wb):
        latex(wb, "x[n] := u[n]\ny[n] = x[n] + \\frac{1}{2} y[n-1]")
        assert same(latex(wb, "y[n]")[0], (2 - sympy.Rational(1, 2) ** n) * step(n))
        assert latex(wb, "y[1]") == [sympy.Rational(3, 2)]

    def test_of_second_order(self, wb):
        latex(wb, r"y[n] := \frac{3}{4} y[n-1] - \frac{1}{8} y[n-2] + \delta[n]")
        expected = 2 * sympy.Rational(1, 2) ** n - sympy.Rational(1, 4) ** n
        assert same(latex(wb, "y[n]")[0], expected * step(n))

    def test_solve_gives_the_sequence_at_rest(self, wb):
        (found,) = latex(wb, r"solve(y[n] - \frac{1}{2} y[n-1] = \delta[n], y[n])")
        assert len(found) == 1
        assert same(found[0], sympy.Rational(1, 2) ** n * step(n))

    def test_an_input_that_never_starts_is_refused(self, wb):
        latex(wb, "y[n] := y[n-1] + 1")
        with pytest.raises(RelativisticPyError, match="its input never starts"):
            latex(wb, "y[3]")


class TestSystems:
    def test_names_nothing_defines_at_points_make_a_difference_equation(self, wb):
        program = r"y[n] - \frac{3}{4} y[n-1] + \frac{1}{8} y[n-2] = x[n]"
        assert notes(wb, program) == [
            f"{program} is read as a difference equation in the sequences x and y, which "
            "nothing defines"
        ]

    def test_their_capital_letters_are_their_transforms(self, wb):
        program = (
            r"y[n] - \frac{3}{4} y[n-1] + \frac{1}{8} y[n-2] = x[n]" "\n"
            r"H(z) := \frac{Y(z)}{X(z)}" "\nH(z)\npoles(H(z))\nzeros(H(z))"
        )  # fmt: skip
        found = latex(wb, program)
        assert same(found[1], 1 / (1 - sympy.Rational(3, 4) / z + sympy.Rational(1, 8) / z**2))
        assert found[2] == [sympy.Rational(1, 4), sympy.Rational(1, 2)]
        assert found[3] == [0, 0]

    def test_a_note_says_how_a_capital_letter_is_read(self, wb):
        latex(wb, "y[n] = x[n] + x[n - 1]")
        assert notes(wb, "Y(z)")[0] == (
            "Y(z) is read as the z-transform of y, of the difference equation on line 1, its "
            "sequences at rest before they start"
        )


class TestTransforms:
    def test_the_z_transform_of_a_sequence_is_over_every_n(self, wb):
        found = latex(wb, r"\mathcal{Z}\{\delta[n - 1] + u[n]\}(z)" "\n"
                      r"\mathcal{Z}\{-a^n u[-n - 1]\}(z)")  # fmt: skip
        assert same(found[0], 1 / z + z / (z - 1))
        assert same(found[1], z / (z - a))

    def test_the_inverse_gives_a_sequence_back(self, wb):
        program = (
            "|z| > \\frac{1}{2}\n"
            r"x[n] := \mathcal{Z}^{-1}\left\{\frac{1}{\left(1 - \frac{1}{4}z^{-1}\right)"
            r"\left(1 - \frac{1}{2}z^{-1}\right)}\right\}" "\nx[3]\nx[-1]\nx[n]"
        )  # fmt: skip
        found = latex(wb, program)
        assert found[:2] == [sympy.Rational(15, 64), 0]
        expected = (2 * sympy.Rational(1, 2) ** n - sympy.Rational(1, 4) ** n) * step(n)
        assert same(found[2], expected)

    def test_a_sum_of_a_sequence_under_its_region_of_convergence(self, wb):
        program = (
            "|z| > \\frac{1}{2}\nx[n] := \\left(\\frac{1}{2}\\right)^n u[n]\n"
            r"X(z) := \sum_{n=-\infty}^{\infty} x[n] z^{-n}" "\nX(z)"
        )  # fmt: skip
        assert same(latex(wb, program)[0], z / (z - sympy.Rational(1, 2)))


class TestFrequencyResponses:
    def test_defined_at_e_to_the_j_omega_it_is_a_function_of_omega(self, wb):
        program = (
            "j := \\sqrt{-1}\n|a| < 1\n\\omega \\in \\mathbb{R}\nh[n] := a^n u[n]\n"
            r"H(e^{j\omega}) := \sum_n h[n] e^{-j\omega n}" "\nH(e^{j\\omega})\nH(e^{j\\pi})"
        )  # fmt: skip
        found = latex(wb, program)
        assert same(found[0], 1 / (1 - a * sympy.exp(-sympy.I * w)))
        assert same(found[1], 1 / (1 + a))

    def test_a_note_says_what_it_is_a_function_of(self, wb):
        said = notes(wb, r"H(e^{j\omega}) := 1 + e^{-j\omega}")
        assert said == [
            r"H(e^{j\omega}) defines H as a function of \omega: H at e^{j\theta} is its value "
            r"at \theta"
        ]

    def test_called_elsewhere_than_on_the_unit_circle_it_is_refused(self, wb):
        latex(wb, r"H(e^{j\omega}) := 1 + e^{-j\omega}")
        with pytest.raises(CompileError, match="so it is called at a point written so"):
            latex(wb, "H(2)")

    def test_the_magnitude_and_the_phase(self, wb):
        program = (
            "j := \\sqrt{-1}\n-\\pi < \\omega < \\pi\n"
            r"H(e^{j\omega}) := 1 + 2e^{-j\omega} + e^{-2j\omega}" "\n"
            r"-\frac{d}{d\omega} \arg H(e^{j\omega})" "\n"
            r"\left|H(e^{j\pi/2})\right|"
        )  # fmt: skip
        found = latex(wb, program)
        assert sympy.simplify(found[0]) == 1
        assert same(found[1], 2)


class TestAssumptions:
    def test_a_sum_that_holds_a_case_of_equal_parameters_apart_is_given_in_general(self, wb):
        program = "n \\in \\mathbb{N}\nx[n] := a^n u[n]\nh[n] := b^n u[n]\n\\sum_k x[k] h[n - k]"
        said = "assuming a ≠ b, as the sum takes the case a = b apart: there it is b^n (n + 1)"
        assert said in notes(wb, program)

    def test_a_bound_on_a_modulus_is_an_assumption(self, wb):
        assert notes(wb, r"|z| > \frac{1}{2}") == [
            r"|z| > \frac{1}{2} is an assumption: from here on, |z| > 1/2"
        ]
