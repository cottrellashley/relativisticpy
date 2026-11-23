r"""What books assume where a program says nothing (design/notation.md, decision 1).

Symbols stay general. Where an integral or a limit would hold conditions on its parameters,
a modulus of one, or be left undone for one, it is given under the assumption books make, a
scale positive and the variable of an integral over real limits real, with the fewest
parameters assumed; a note says so, and so does the step's data, ``assuming``. What the
program says of a name decides, and nothing is assumed of it.
"""

import json
import signal
import time

import pytest
import sympy

from relativisticpy.cli import main
from relativisticpy.errors import EvaluationError

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
        assert last(wb, r"\int_0^\infty e^{-z t} \, dt").assuming == []
        wb.exe(r"w \in \mathbb{C}")
        assert last(wb, r"\int_0^\infty e^{-w t} \, dt").assuming == []

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

    def test_a_variable_away_from_0_before_a_positive_one(self, wb):
        step = last(
            wb,
            r"K(x, t) := \frac{1}{\sqrt{4\pi t}} e^{-\frac{x^2}{4t}}"
            "\n"
            r"\lim_{t \to 0^+} K(x, t)",
        )
        assert step.values == [0]
        assert step.notes == ["assuming x ≠ 0, real; the program does not say what x is"]
        assert step.assuming == ["x ≠ 0, real"]

    @pytest.mark.parametrize(
        ("source", "said"),
        [
            ("limit(abs(x)/x, x, 0)", "it is -1 from below and 1 from above"),
            ("limit(1/x, x, 0)", "it is -oo from below and oo from above"),  # SymPy: zoo
            ("limit(floor(x), x, 1)", "it is 0 from below and 1 from above"),
            ("limit(sin(1/x), x, 0)", "the values keep between -1 and 1"),
        ],
    )
    def test_the_function_is_from_both_sides_as_the_notation_is(self, wb, source, said):
        with pytest.raises(EvaluationError, match="the limit does not exist") as raised:
            wb.expr(source)
        assert said in raised.value.message

    def test_the_function_takes_a_side(self, wb):
        assert wb.exe("limit(1/x, x, 0, '-')\nlimit(1/x, x, 0, '+')") == [-sympy.oo, sympy.oo]
        assert wb.expr("limit(floor(x), x, 1, '-')") == 0
        with pytest.raises(EvaluationError, match="from above, '\\+', from below, '-', or"):
            wb.expr("limit(1/x, x, 0, 'up')")

    def test_the_side_is_the_only_text_the_function_takes(self, wb):
        for source in ("limit('1/x', x, 0)", "limit(1/x, 'x', 0)", "limit(1/x, x, '0')"):
            with pytest.raises(EvaluationError, match="text in quotes is not a value"):
                wb.expr(source)


class TestTheData:
    def test_json_says_what_a_value_is_given_under(self, capsys):
        assert main(["-c", r"\int_0^\infty e^{-s t} \, dt", "--format", "json"]) == 0
        record = json.loads(capsys.readouterr().out)
        assert record["assuming"] == ["s > 0"]
        assert record["notes"] == [
            "assuming s > 0, as books take a scale to be; the program does not say what s is"
        ]


class TestSums:
    """A sum with symbolic or infinite limits is in closed form where SymPy finds one
    within a time budget, with the index an integer (design/notation.md, decision 4)."""

    def test_a_closed_form_is_the_value(self, wb):
        n, N = sympy.symbols("n N")
        assert wb.expr(r"\sum_{n=1}^{\infty} \frac{1}{n^2}") == sympy.pi**2 / 6
        assert wb.expr(r"\sum_{l=0}^{N-1} (2l + 1)") == N**2
        assert wb.expr(r"\prod_{k=1}^{n} k") == sympy.factorial(n)

    def test_the_index_is_an_integer(self, wb):
        assert wb.expr(r"\sum_{n=1}^{N} \sin(n \pi)") == 0
        assert wb.expr(r"\sum_{n=1}^{N} (-1)^{2n}") == sympy.Symbol("N")

    def test_a_geometric_series_of_exponentials_under_what_books_assume(self, wb):
        step = last(wb, r"\sum_{n=0}^{\infty} e^{-a n}")
        assert sympy.simplify(step.values[0] - 1 / (1 - sympy.exp(-a))) == 0
        assert step.assuming == ["a > 0"]

    def test_a_stated_assumption_needs_no_note(self, wb):
        step = last(wb, "r > 0\n" r"\sum_{t=1}^{\infty} \frac{C}{(1 + r)^t}")
        positive = sympy.Symbol("r", positive=True)
        assert sympy.simplify(step.values[0] - sympy.Symbol("C") / positive) == 0
        assert (step.notes, step.assuming) == ([], [])

    @pytest.mark.parametrize(
        ("source", "value"),
        [
            (r"\sum_{n=1}^{\infty} \frac{1}{n (2n + 1)}", 2 - 2 * sympy.log(2)),
            (r"\sum_{n=1}^{\infty} \frac{1}{n (n + 1/2)}", 4 - 4 * sympy.log(2)),
            (r"\sum_{n=1}^{\infty} \frac{1}{n (n + 3/2)}",
             sympy.Rational(16, 9) - 4 * sympy.log(2) / 3),
            (r"\sum_{n=0}^{\infty} \frac{1}{(n + 1/2)(n + 1)}", 4 * sympy.log(2)),
            (r"\sum_{n=1}^{\infty} \frac{1}{n (n + 1/2)(n + 1)}", 6 - 8 * sympy.log(2)),
            (r"\sum_{n=1}^{\infty} \frac{1}{n (n + a)}", sympy.harmonic(a) / a),
        ],
    )  # fmt: skip
    def test_a_closed_form_that_is_undefined_is_none(self, wb, source, value):
        # SymPy gives nan, or zoo - lerchphi(1, 1, a + 1), of sums that converge
        found = wb.expr(source)
        assert not found.has(sympy.nan, sympy.zoo)
        assert sympy.simplify(found - value) == 0

    def test_a_series_that_diverges_says_so_and_not_that_its_time_ran_out(self, wb):
        step = last(wb, r"\sum_{n=2}^{\infty} \frac{1}{n \ln n}")
        assert isinstance(step.values[0], sympy.Sum)
        assert step.notes == [
            "the sum diverges, as SymPy's tests of convergence show",
            "a sum is left as written: SymPy found no closed form for it",
        ]

    def test_a_series_whose_terms_do_not_tend_to_0_is_infinite(self, wb):
        step = last(wb, "a < 0\n" r"\sum_{n=0}^{\infty} e^{-a n}")
        assert step.values == [sympy.oo]
        assert step.notes == ["the sum diverges: its terms tend to ∞, not to 0"]

    @pytest.mark.parametrize("terms", ["(-1)^n", r"\cos n"])
    def test_a_series_whose_terms_tend_to_no_limit_diverges(self, wb, terms):
        # Grandi's series: said to diverge only where SymPy's own test did in its second
        step = last(wb, rf"\sum_{{n=0}}^{{\infty}} {terms}")
        assert isinstance(step.values[0], sympy.Sum)
        assert step.notes == [
            "the sum diverges: its terms do not tend to 0",
            "a sum is left as written: SymPy found no closed form for it",
        ]

    def test_a_series_that_converges_under_a_condition_is_given_under_it(self, wb):
        # as docs/language.md says, and as books give it: 1/(1 - r) for |r| < 1
        r = sympy.Symbol("r")
        step = last(wb, r"\sum_{n=0}^{\infty} r^n")
        assert step.values == [1 / (1 - r)]
        assert step.notes == [
            "assuming |r| < 1, where the series converges; the program does not say"
        ]
        assert step.assuming == ["|r| < 1"]
        step = last(wb, "0 < r < 1\n" r"\sum_{n=0}^{\infty} r^n")  # assumed: no note
        assert [str(value) for value in step.values] == ["1/(1 - r)"]
        assert step.notes == []
        step = last(wb, r"\sum_{n=0}^{\infty} 2^n")  # no parameter: nothing to assume
        assert step.values == [sympy.oo]

    @pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="no SIGALRM here")
    def test_one_past_its_budget_says_that_its_time_ran_out(self, wb, monkeypatch):
        from relativisticpy.runtime import core

        def searching(*args, **kwargs):
            time.sleep(2)
            raise AssertionError("the budget should have stopped this")

        monkeypatch.setattr(core, "_CLOSED_FORM_BUDGET", 0.05)
        monkeypatch.setattr(core.assumptions, "summed", searching)
        step = last(wb, r"\sum_{n=1}^{\infty} \frac{1}{n^n}")
        assert step.notes == [
            "SymPy found no closed form of the sum in the 0.05 s it is given",
            "a sum is left as written: SymPy found no closed form for it",
        ]

    def test_one_with_no_closed_form_is_kept_as_written(self, wb):
        step = last(wb, r"\sum_{n=1}^{\infty} \frac{1}{n^n}")
        assert isinstance(step.values[0], sympy.Sum)
        assert step.notes == ["a sum is left as written: SymPy found no closed form for it"]


class TestProducts:
    """An infinite product SymPy leaves undone is the limit of its partial products."""

    @pytest.mark.parametrize(
        ("source", "value"),
        [
            (r"\prod_{n=2}^{\infty} \left(1 - \frac{1}{n^2}\right)", sympy.Rational(1, 2)),
            # Wallis's product
            (r"\prod_{n=1}^{\infty} \frac{4n^2}{4n^2 - 1}", sympy.pi / 2),
            (
                r"\prod_{n=1}^{\infty} \left(1 + \frac{1}{n^2}\right)",
                sympy.sinh(sympy.pi) / sympy.pi,
            ),
        ],
    )
    def test_its_value_is_the_limit_of_its_partial_products(self, wb, source, value):
        step = last(wb, source)
        assert sympy.simplify(step.values[0] - value) == 0
        assert step.notes == []

    def test_one_whose_partial_products_tend_to_0_or_infinity_diverges(self, wb):
        step = last(wb, r"\prod_{n=1}^{\infty} \frac{n+1}{n}")
        assert step.values == [sympy.oo]
        assert step.notes == ["the product diverges: its partial products tend to ∞"]
        step = last(wb, r"\prod_{n=2}^{\infty} \left(1 - \frac{1}{n}\right)")
        assert step.values == [0]
        assert step.notes == ["the product diverges: its partial products tend to 0"]

    def test_one_kept_as_written_is_called_a_product(self, wb):
        step = last(wb, r"\prod_{n=1}^{\infty} \left(1 + \frac{1}{n^n}\right)")
        assert isinstance(step.values[0], sympy.Product)
        assert step.notes == ["a product is left as written: SymPy found no closed form for it"]
