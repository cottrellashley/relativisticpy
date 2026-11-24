r"""Integrals SymPy cannot do as they are written, computed all the same: under the bounds the
program assumes, where SymPy fails without them; with the integrand simplified where the
variable lies, as \sqrt{1 + \sinh^2 x} is \cosh x for a real x; and over a half-line, as a
geometric series of exponentials, term by term, as the integrals of Bose and Einstein, and
of Fermi and Dirac, are."""

import pytest
import sympy

from relativisticpy import Workbook
from relativisticpy.errors import EvaluationError
from relativisticpy.language.source import Source

x = sympy.Symbol("x")


def shown(program: str) -> tuple[list[object], list[str]]:
    """The values of a program's last statement, read as LaTeX, and its notes."""
    *_, last = Workbook().each(Source(program, latex=True))
    assert last.error is None, last.error
    return last.values or [], last.notes


def value(program: str) -> object:
    values, _ = shown(program)
    return values[-1]


def test_the_bounds_assumed_decide_what_sympy_cannot():
    # geom.gauss-bonnet.torus: SymPy failed on a - r >= 0 in the integral of |a + r cos θ|
    program = r"""a > r > 0
ds^2 = r^2 d\theta^2 + (a + r\cos\theta)^2 d\phi^2
K := \frac{R}{2}
\int_0^{2\pi} \int_0^{2\pi} K \sqrt{g} \, d\theta \, d\phi"""
    assert value(program) == 0


def test_an_integrand_is_simplified_where_its_variable_lies():
    # geom.curves.catenary-length: the arc length of the catenary, kept as written before
    assert value(r"\int_0^1 \sqrt{1 + \sinh^2 x} \, dx") == sympy.sinh(1)


@pytest.mark.parametrize(
    ("program", "expected"),
    [
        # statmech.quantum.planck-integral: Γ(4)ζ(4) (Bose and Einstein)
        (r"\int_0^\infty \frac{x^3}{e^x - 1} dx", sympy.pi**4 / 15),
        # (1 - 2^{-3}) Γ(4)ζ(4) (Fermi and Dirac)
        (r"\int_0^\infty \frac{x^3}{e^x + 1} dx", 7 * sympy.pi**4 / 120),
        (
            r"\int_0^\infty \frac{\sqrt{x}}{e^x - 1} dx",
            sympy.sqrt(sympy.pi) * sympy.zeta(sympy.Rational(3, 2)) / 2,
        ),
        (r"\int_0^\infty \frac{1}{e^x + 1} dx", sympy.log(2)),
        (r"\int_0^\infty \frac{x e^{-x}}{1 - e^{-x}} dx", sympy.pi**2 / 6),
        (r"\int_0^\infty \frac{x^2}{e^{2x} - 1} dx", sympy.zeta(3) / 4),
        # the sum of 1/(n^2 + 1) over n from 1
        (r"\int_0^\infty \frac{\sin x}{e^x - 1} dx", (sympy.pi / sympy.tanh(sympy.pi) - 1) / 2),
    ],
)
def test_an_integral_over_a_half_line_is_a_series_of_exponentials(program, expected):
    found = value(program)
    assert found == expected or abs(sympy.N(found - expected, 30)) < 1e-25
    assert not found.has(sympy.Integral)


@pytest.mark.parametrize(
    "program",
    [
        r"\frac{\hbar}{\pi^2 c^3} \int_0^\infty \frac{\omega^3}{e^{\hbar\omega/\tau} - 1} d\omega",
        # statmech.quantum.stefan-boltzmann, where c in the integrand kept it as written
        r"""\langle s \rangle := \frac{1}{e^{\hbar \omega / \tau} - 1}
D(\omega) := \frac{\omega^2}{\pi^2 c^3}
\int_0^\infty \hbar \omega \, \langle s \rangle \, D(\omega) \, d\omega""",
    ],
)
def test_a_scale_in_the_exponential_is_taken_positive_as_books_take_it(program):
    hbar, tau, c = sympy.symbols("hbar tau c")
    values, notes = shown(program)
    found = values[-1].xreplace({s: sympy.Symbol(s.name) for s in values[-1].free_symbols})
    assert sympy.simplify(found - sympy.pi**2 * tau**4 / (15 * c**3 * hbar**3)) == 0
    assert notes == [
        "assuming ℏ > 0 and τ > 0, as books take scales to be; the program does not say what "
        "ℏ and τ are"
    ]


def test_an_integral_that_diverges_is_not_given_a_value_by_its_series():
    assert value(r"\int_0^\infty \frac{1}{e^x - 1} dx") == sympy.oo


def test_limits_written_as_the_variable_equal_to_a_value_are_those_values():
    # the integral was given the equations x = 0 and x = 1, and SymPy failed on them
    assert value(r"\int_{x=0}^{x=1} x^2 \, dx") == sympy.Rational(1, 3)
    assert value(r"\int_{t=1}^{t=2} \frac{1}{t} \, dt") == sympy.log(2)
    *_, last = Workbook().each(Source(r"\int_{y=0}^{y=1} x \, dx", latex=True))
    assert last.error is not None
    assert last.error.message == (
        "a limit of the integral over x is written y = 0, which does not say what x is"
    )


def test_an_integral_sympy_fails_on_for_a_limit_that_is_not_real_is_refused():
    # arXiv 2611.26576, line 6712: SymPy compares the limit -i∞, which is not real, and the
    # failure was an internal error
    *_, last = Workbook().each(
        Source(r"\int_{z}^{-i \, \infty} p^{m - 1} e^{-i k p} \, dp", latex=True)
    )
    assert type(last.error) is EvaluationError
    assert (
        last.error.message == "SymPy could not compute this integral, whose limit −∞i is not real"
    )
    # where it compares no limit, SymPy computes one: -1/p is 0 at i∞
    assert value(r"\int_{1}^{i \infty} \frac{1}{p^2} \, dp") == 1
