"""Results checked by another route: SymPy's closed form of a definite integral or a series
of numbers against mpmath's value, an antiderivative differentiated back. A confident
disagreement is refused, saying what disagreed; numbers that do not settle it say nothing,
so that a right answer is never refused for a failure of the numbers."""

import pytest
import sympy

from relativisticpy import Workbook
from relativisticpy.errors import EvaluationError
from relativisticpy.language.source import Source
from relativisticpy.runtime import crosscheck

x, n = sympy.symbols("x n")


@pytest.mark.parametrize(
    ("program", "said"),
    [
        # SymPy's antiderivatives jump where the integrands do: it gives 0 and -π²/2
        (
            "integrate(sign(sin(x)), (x, 0, 3 pi))",
            "SymPy's closed form of the integral, 0, disagrees with numerical integration, "
            "which gives 3.141592654",
        ),
        (
            r"\int_0^{\pi} \arctan(\tan x) \, dx",
            "SymPy's closed form of the integral, −π²/2, which is −4.934802201, disagrees "
            "with numerical integration, which gives 0",
        ),
    ],
)
def test_a_closed_form_that_numbers_confidently_refute_is_refused(program, said):
    source = Source(program, latex=program.startswith("\\"))
    with pytest.raises(EvaluationError) as raised:
        Workbook().exe(source)
    assert raised.value.message == said
    assert "split the integral there" in (raised.value.hint or "")


@pytest.mark.parametrize(
    ("program", "value"),
    [
        (r"\int_0^\infty \frac{\sin x}{x} dx", sympy.pi / 2),  # oscillates: numbers unsure
        (r"\int_{-1}^{2} |x| \, dx", sympy.Rational(5, 2)),  # bends at 0
        (r"\int_0^{2\pi} \frac{1}{2 + \cos x} dx", 2 * sympy.pi / sympy.sqrt(3)),
        (r"\int_0^1 \frac{1}{\sqrt{x}} dx", 2),  # a singularity at an end
        (r"\sum_{n=1}^{\infty} \frac{(-1)^{n+1}}{n}", sympy.log(2)),
    ],
)
def test_a_right_closed_form_is_given(program, value):
    assert sympy.simplify(Workbook().expr(Source(program, latex=True)) - value) == 0


def test_a_closed_form_of_a_series_that_numbers_refute_is_refused():
    with pytest.raises(EvaluationError, match="disagrees with the series summed numerically"):
        crosscheck.series(sympy.pi**2 / 5, 1 / n**2, n, 1, sympy.oo)
    crosscheck.series(sympy.pi**2 / 6, 1 / n**2, n, 1, sympy.oo)


def test_an_antiderivative_that_differentiates_to_another_integrand_is_refused():
    with pytest.raises(EvaluationError, match="differentiated back, is not"):
        crosscheck.antiderivative(x**3 / 2, x**2, x)
    a = sympy.Symbol("a")
    crosscheck.antiderivative(sympy.log(x) / a, 1 / (a * x), x)  # with a parameter
    crosscheck.antiderivative(sympy.log(sympy.Abs(x)), 1 / x, x)  # as books write it


def test_an_antiderivative_is_checked_only_where_the_program_says_its_variable_lies():
    # x - x²/2 is the antiderivative of |x - 1| where x < 1, as the program assumes
    crosscheck.integral(x - x**2 / 2, sympy.Abs(x - 1), (x,), (x < 1,))
    # each symbol is checked at numbers it may be: of its sign, and whole where it is
    for facts, sign in (({"negative": True}, -1), ({"positive": True}, 1)):
        point = crosscheck._assumed(sympy.Symbol("y", **facts), sympy.Rational(1, 2))
        assert sympy.sign(point) == sign
    assert crosscheck._assumed(sympy.Symbol("k", integer=True), sympy.Rational(1, 2)).is_integer
