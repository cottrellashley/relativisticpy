r"""Convolutions, read as signal processing writes them (design/notation.md, 27): ``*`` or
``\ast`` between two functions applied at one variable, or two sequences at points of one
index, is their convolution, with a note; between anything else it multiplies, and ``z^*``
is a conjugate. ``\circledast`` is the circular convolution."""

from __future__ import annotations

import pytest
import sympy

from relativisticpy import Workbook
from relativisticpy.errors import CompileError
from relativisticpy.integrations.sequences import impulse, step
from relativisticpy.language.source import Source

t, n = sympy.symbols("t n")
f, g = sympy.Function("f"), sympy.Function("g")


@pytest.fixture
def wb():
    with Workbook() as workbook:
        yield workbook


def latex(wb: Workbook, source: str) -> list[object]:
    """What a program written as LaTeX shows, every statement LaTeX."""
    return wb.exe(Source(source, latex=True))


def notes(wb: Workbook, source: str) -> list[str]:
    return [note for step in wb.each(Source(source, latex=True)) for note in step.notes]


def same(found: object, expected: object) -> bool:
    """Whether two values are equal, their symbols compared by name, whatever is assumed of
    them."""
    found, expected = (sympy.sympify(value) for value in (found, expected))
    plain = {s: sympy.Symbol(s.name) for s in found.free_symbols if isinstance(s, sympy.Symbol)}
    return sympy.simplify(found.xreplace(plain) - expected) == 0


class TestOfFunctions:
    def test_an_asterisk_between_functions_of_one_variable_convolves_them(self, wb):
        latex(wb, "t > 0")
        (value,) = latex(wb, r"\left(e^{-t} u(t)\right) * \left(e^{-2t} u(t)\right)")
        assert same(value, sympy.exp(-t) - sympy.exp(-2 * t))

    def test_the_impulse_is_its_identity(self, wb):
        assert latex(wb, r"f(t) * \delta(t)") == [f(t)]
        wb.exe(r"h(t) := e^{-t}")
        assert latex(wb, r"h(t) \ast \delta(t - 1)") == [sympy.exp(1 - t)]

    def test_it_is_said_in_a_note(self, wb):
        said = notes(wb, r"f(t) * \delta(t)")[0].split("\n")
        assert said == [
            r"f(t) * \delta(t) is read as the convolution of f(t) and \delta(t), the integral "
            "over every τ of the first at τ times the second at t - τ, as signal processing "
            "writes it; for their product, write them side by side, without *",
            r"\delta(t) is read as the Dirac delta, DiracDelta, as signal processing writes it; "
            r"for a function of your own, define \delta(t) := ... first",
        ]
        reading = str(wb.explain(r"f(t) * g(t)"))
        assert (
            "convolution or product: f(t) * g(t) is their convolution, in t, since * stands "
            "between two functions applied at t"
        ) in reading

    def test_in_plain_text_too(self, wb):
        (value,) = wb.exe("f(t)*g(t)")
        assert isinstance(value, sympy.Integral)
        assert value.limits[0][1:] == (-sympy.oo, sympy.oo)

    @pytest.mark.parametrize(
        "source",
        [
            "2*x",
            "x * y",
            "sin(t)*cos(t)",
            "f(t) * g(s)",
            "f(t) g(t)",
            r"f(t) \cdot g(t)",
            "z^* * z",
        ],
    )
    def test_between_anything_else_it_multiplies(self, wb, source):
        (value,) = latex(wb, source)
        assert not value.has(sympy.Integral)
        assert not any("convolution" in note for note in notes(wb, source))


class TestOfSequences:
    @pytest.fixture
    def signals(self, wb):
        latex(
            wb,
            "x[n] := \\delta[n] + 2\\delta[n-1] + 3\\delta[n-2]\nh[n] := \\delta[n] + \\delta[n-1]",
        )
        return wb

    def test_an_asterisk_between_sequences_of_one_index_convolves_them(self, signals):
        (value,) = latex(signals, "x[n] * h[n]")
        expected = impulse(n) + 3 * impulse(n - 1) + 5 * impulse(n - 2) + 3 * impulse(n - 3)
        assert sympy.simplify(value - expected) == 0
        latex(signals, r"y[n] := x[n] \ast h[n]")
        assert latex(signals, "[y[0], y[1], y[2], y[3], y[4]]") == [[1, 3, 5, 3, 0]]

    def test_of_steps(self, wb):
        (value,) = latex(wb, "n \\geq 0\nu[n] * u[n]")[-1:]
        assert same(value, n + 1)

    def test_at_points_of_two_indices_it_multiplies(self, signals):
        k = sympy.Symbol("k")
        (value,) = latex(signals, "x[n] * h[k]")
        assert (
            sympy.simplify(
                value
                - (impulse(n) + 2 * impulse(n - 1) + 3 * impulse(n - 2))
                * (impulse(k) + impulse(k - 1))
            )
            == 0
        )
        assert latex(signals, r"\delta[n] \cdot u[n]") == [impulse(n) * step(n)]


class TestCircular:
    @pytest.fixture
    def signals(self, wb):
        latex(
            wb,
            "x[n] := \\delta[n] + 2\\delta[n-1] + 3\\delta[n-2] + 4\\delta[n-3]\n"
            "h[n] := \\delta[n] + \\delta[n-1]",
        )
        return wb

    def test_at_a_point_of_the_length_of_the_longer_sequence(self, signals):
        assert latex(signals, r"(x \circledast h)[0]") == [5]
        assert latex(signals, "(x ⊛ h)[3]") == [7]
        assert notes(signals, r"(x \circledast h)[0]") == [
            r"(x \circledast h) is the circular convolution of the length of the longer "
            "sequence, from 0 to its last member that is not 0; for another, write it as the "
            r"subscript, as in x \circledast_N h"
        ]

    def test_of_the_length_its_subscript_writes(self, signals):
        assert latex(signals, r"(x \circledast_8 h)[0]") == [1]
        assert latex(signals, "N := 4\n" r"(x \circledast_N h)[0]")[-1:] == [5]

    def test_as_a_sequence_of_its_index(self, signals):
        latex(signals, r"y[n] := x[n] \circledast h[n]")
        assert latex(signals, "[y[0], y[1], y[2], y[3]]") == [[5, 3, 5, 7]]

    def test_a_sequence_that_does_not_end_needs_its_length(self, wb):
        latex(wb, "s[n] := 2^{-n} u[n]")
        with pytest.raises(Exception, match=r"does not end, so its circular convolution needs"):
            latex(wb, r"(s \circledast s)[0]")
        assert latex(wb, r"(s \circledast_2 s)[0]") == [sympy.Rational(5, 4)]

    def test_named_alone_it_is_computed_at_a_point(self, signals):
        with pytest.raises(CompileError, match="computed at a point") as caught:
            latex(signals, r"x \circledast h")
        assert caught.value.hint == r"write it at a point, as (x \circledast h)[n]"
