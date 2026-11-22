r"""What books assume where a program says nothing (design/notation.md, decision 1).

Symbols stay general. Where an integral or a limit would hold conditions on its parameters,
a modulus of one, or be left undone for one, it is given under the assumption books make, a
scale positive and the variable of an integral over real limits real, with the fewest
parameters assumed; a note says so, and so does the step's data, ``assuming``. What the
program says of a name decides, and nothing is assumed of it.
"""

import json

import sympy

from relativisticpy.cli import main

a, b, s, x = sympy.symbols("a b s x")


def last(wb, source):
    """The last step of a program, which must not fail."""
    *_, step = wb.each(source)
    assert step.error is None, step.error
    return step


class TestIntegrals:
    def test_a_condition_of_convergence(self, wb):
        step = last(wb, r"\int_0^\infty e^{-s t} \, dt")
        assert step.values == [1 / s]
        assert step.notes == [
            "assuming s > 0, as books take a scale to be; the program does not say what s is"
        ]
        assert step.assuming == ["s > 0"]

    def test_the_fewest_parameters_are_assumed(self, wb):
        step = last(wb, r"\int_{-\infty}^{\infty} e^{-a x^2 + b x} \, dx")
        expected = sympy.sqrt(sympy.pi) * sympy.exp(b**2 / (4 * a)) / sympy.sqrt(a)
        assert sympy.simplify(step.values[0] - expected) == 0
        assert step.assuming == ["a > 0"]  # b may be anything

    def test_several_scales(self, wb):
        step = last(wb, r"\int_{-\infty}^{\infty} e^{-p^2/(2 m k T)} \, dp")
        assert step.assuming == ["T > 0", "k > 0", "m > 0"]
        assert step.notes == [
            "assuming T > 0, k > 0 and m > 0, as books take scales to be; the program does not "
            "say what T, k and m are"
        ]

    def test_a_modulus_in_the_integrand(self, wb):
        source = r"\int_0^a \left| \sqrt{\frac{2}{a}} \sin\left(\frac{\pi x}{a}\right) \right|^2 dx"
        step = last(wb, source)
        assert step.values == [1]
        assert step.assuming == ["a > 0"]

    def test_the_symbols_of_the_value_are_the_programs(self, wb):
        assert wb.expr(r"\int_0^\infty e^{-s t} \, dt - \frac{1}{s}") == 0

    def test_a_value_defined_carries_its_note_where_it_is_shown(self, wb):
        wb.exe(r"F := \int_0^\infty e^{-s t} \, dt")
        assert last(wb, "F^2").assuming == ["s > 0"]

    def test_what_no_assumption_settles_keeps_its_conditions(self, wb):
        step = last(wb, r"\int_{-\infty}^{\infty} e^{-i \omega t} \, dt")
        assert isinstance(step.values[0], sympy.Piecewise)
        assert step.assuming == []
        assert step.notes[-1].startswith("this holds under the conditions SymPy gives")

    def test_what_is_done_assumes_nothing(self, wb):
        step = last(wb, r"\int_0^1 x^2 \, dx")
        assert (step.values, step.notes, step.assuming) == ([sympy.Rational(1, 3)], [], [])


class TestWhatTheProgramSays:
    def test_a_declaration_decides(self, wb):
        wb.exe(r"s \in \mathbb{R}")  # real, but not said to be positive
        step = last(wb, r"\int_0^\infty e^{-s t} \, dt")
        assert step.assuming == []
        assert isinstance(step.values[0], sympy.Piecewise)

    def test_complex_analysis_keeps_z_complex(self, wb):
        wb.exe(r"z \in \mathbb{C}")
        assert last(wb, r"\int_0^\infty e^{-z t} \, dt").assuming == []

    def test_a_stated_assumption_needs_no_note(self, wb):
        step = last(wb, "s > 0\n" r"\int_0^\infty e^{-s t} \, dt")
        assert (step.values, step.notes, step.assuming) == (
            [1 / sympy.Symbol("s", positive=True)],
            [],
            [],
        )

    def test_bounds_decide_the_conditions(self, wb):
        step = last(wb, "x > 1\n" r"\int_0^\infty e^{-(x - 1) t} \, dt")
        positive = sympy.Symbol("x", positive=True)
        assert step.values == [1 / (positive - 1)]
        assert step.assuming == []


class TestLimits:
    def test_a_limit_that_depends_on_the_sign_of_a_parameter(self, wb):
        source = (
            r"\lim_{n \to \infty} \binom{n}{k} \left(\frac{\lambda}{n}\right)^{k}"
            r" \left(1 - \frac{\lambda}{n}\right)^{n - k}"
        )
        step = last(wb, source)
        k, lam = sympy.symbols("k lambda", positive=True)
        found = step.values[0].xreplace(
            {s: sympy.Symbol(s.name, positive=True) for s in step.values[0].free_symbols}
        )
        assert sympy.simplify(found - lam**k * sympy.exp(-lam) / sympy.factorial(k)) == 0
        assert step.assuming == ["k > 0"]


class TestTheData:
    def test_json_says_what_a_value_is_given_under(self, capsys):
        assert main(["-c", r"\int_0^\infty e^{-s t} \, dt", "--format", "json"]) == 0
        record = json.loads(capsys.readouterr().out)
        assert record["assuming"] == ["s > 0"]
        assert record["notes"] == [
            "assuming s > 0, as books take a scale to be; the program does not say what s is"
        ]
