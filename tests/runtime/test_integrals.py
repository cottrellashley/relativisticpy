r"""Integrals SymPy cannot do as they are written, computed all the same: under the bounds the
program assumes, where SymPy fails without them."""

from relativisticpy import Workbook
from relativisticpy.language.source import Source


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
