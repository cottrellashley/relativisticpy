r"""LaTeX in, mathematics out (ported from the original test suite, plus more)."""

import pytest
from sympy import (
    Array,
    Function,
    Integral,
    Product,
    Sum,
    Symbol,
    cos,
    diff,
    integrate,
    oo,
    pi,
    root,
    sin,
    sqrt,
)

from helpers import same

x, y, z, t, tau, theta = (Symbol(n) for n in ("x", "y", "z", "t", "tau", "theta"))
f, g, h = Function("f"), Function("g"), Function("h")


def test_sums_and_products_stay_unevaluated(wb):
    assert wb.expr(r"\sum_{x=0}^{10} x") == Sum(x, (x, 0, 10))
    assert wb.expr(r"\sum_{x=1}^{oo} 1/x**2") == Sum(1 / x**2, (x, 1, oo))
    assert wb.expr(r"\prod_{x=0}^{10} x") == Product(x, (x, 0, 10))
    assert wb.expr(r"doit(\sum_{n=1}^{\infty} \frac{1}{n^2})") == pi**2 / 6


def test_limits(wb):
    assert wb.expr(r"\lim_{x -> 0} sin(x)/x") == 1
    assert wb.expr(r"\lim_{ x -> oo } 1/x**2") == 0
    assert wb.expr(r"\lim_{ x \to 10 } 1/x**2") == pi**0 / 100
    assert wb.expr(r"\lim_{x \to 0^+} \frac{1}{x}") == oo
    assert wb.expr(r"\lim_{x \to 0^-} \frac{1}{x}") == -oo


def test_evaluation_bars(wb):
    assert wb.expr(r" \frac{x}{10} * 1/x |_{x = oo}") == pi**0 / 10
    assert wb.expr(r" \frac{x}{10} * 1/x |_{x = 10}") == pi**0 / 10
    assert (
        wb.expr(r" \frac{ diff(f(x), x) }{10} * 1/diff(f(x), x) |_{ diff(f(x), x) = 10}")
        == pi**0 / 10
    )
    assert wb.expr(r"\left. x^2 + y \right|_{x=3, y=1}") == 10


def test_an_evaluation_bar_substitutes_at_once(wb):
    assert wb.expr(r"x - y |_{x=y, y=x}") == y - x
    boosted = wb.expr(r"t^2 - x^2 |_{t = \gamma (t - v x), x = \gamma (x - v t)}")
    v, gamma = Symbol("v"), Symbol("gamma")
    assert same(boosted, (gamma * (t - v * x)) ** 2 - (gamma * (x - v * t)) ** 2)


def test_fractions_and_roots(wb):
    assert wb.expr(r"\frac{x}{10} * 1/x") == pi**0 / 10
    assert wb.expr(r"\frac{x}{oo} * x**2") == 0
    assert wb.expr(r"\frac12") == pi**0 / 2
    assert wb.expr(r"\sqrt{ x }").equals(root(x, 2))
    assert wb.expr(r"\sqrt{ 10.23 * y }").equals(root(10.23 * y, 2))
    assert wb.expr(r"\sqrt{ sin(x)**2 }").equals(root(sin(x) ** 2, 2))
    assert wb.expr(r"\sqrt[3]{x}") == root(x, 3)


def test_statements_separated_by_newline_commands(wb):
    assert wb.expr(r" A := \prod_{x=0}^{10} x \newline A") == Product(x, (x, 0, 10))


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("diff(f(x), x)", diff(f(x), x)),
        ("diff(f(x), x, 4)", diff(f(x), x, 4)),
        ("diff(f(x) ** x, x, x)", diff(f(x) ** x, x, x)),
        ("f'(x)", diff(f(x), x)),
        ("f''(x)", diff(f(x), x, 2)),
        ("f'''''''''''''''''(x)", diff(f(x), x, 17)),
        (r"\frac{\partial^2{x^2}}{\partial{\tau}^2}", diff(x**2, tau, 2)),
        (
            r"\frac{\partial^2{\sqrt{ - x**3 + y**2 - z }}}{\partial{x}^2}",
            diff(sqrt(-(x**3) + y**2 - z), x, 2),
        ),
        (r"d{f(x)}/d{x}", diff(f(x), x)),
        (r"(d{f(x)})/(d{x})", diff(f(x), x)),
        (r"d^3{f(x) + x**6 - y*7}/d{x}^3", diff(f(x) + x**6, x, 3)),
        (r"\frac{d^2{x^2}}{d{\tau}^2}", diff(x**2, tau, 2)),
        (
            r"\frac{ d^2{-(1 - t / (x**2*h(tau)))} }{d{tau}^2}",
            diff(-(1 - t / (x**2 * h(tau))), tau, 2),
        ),
        (r"\pdv{f(x)}{x}", diff(f(x), x)),
        (r"\pdv[3]{f(x) + x**6 - y*7}{x}", diff(f(x) + x**6, x, 3)),
        (
            r"\pdv{x}( \frac{ \partial^2{ \sqrt{ - x^3 + y^2 - z } } }{ \partial{y}^2 } )",
            diff(diff(sqrt(-(x**3) + y**2 - z), y, 2), x),
        ),
        (r"\frac{d}{dx} x^3", 3 * x**2),
        (r"\frac{d^2 y}{dx^2}", 0),
        (r"\frac{\partial^2}{\partial x \partial y} x^2 y^3", 6 * x * y**2),
        ("sin'(x)", diff(sin(x), x)),
    ],
)
def test_derivatives(wb, source, expected):
    assert same(wb.expr(source), expected)


def test_lagrange_notation_for_defined_functions(wb):
    assert wb.expr("f(x) := x^3\nf'(2)") == 12
    assert wb.expr("f''(x)") == 6 * x


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("integrate(f(x), x)", integrate(f(x), x)),
        ("integrate(f(x), (x, 0, 1))", integrate(f(x), (x, 0, 1))),
        ("∫ f(x) dx", integrate(f(x), x)),
        ("∫_0^1 f(x) dx", integrate(f(x), (x, 0, 1))),
        (r"\int f(x) \, dx", integrate(f(x), x)),
        (r"\int_0^1 f(x) \, dx", integrate(f(x), (x, 0, 1))),
        (r"\iint f(x, y) \, dx \, dy", integrate(f(x, y), x, y)),
        (r"\iint_0^1 f(x, y) \, dx \, dy", integrate(f(x, y), (x, 0, 1), (y, 0, 1))),
        (r"\iiint f(x, y, z) \, dx \, dy \, dz", integrate(f(x, y, z), x, y, z)),
        (
            r"\iiint_0^1 f(x, y, z) \, dx \, dy \, dz",
            integrate(f(x, y, z), (x, 0, 1), (y, 0, 1), (z, 0, 1)),
        ),
        (r"\int_0^\infty e^{-x^2} \, dx", sqrt(pi) / 2),
        (r"\int_0^1 \int_0^x x y \, dy \, dx", Integral(x * y, (y, 0, x), (x, 0, 1)).doit()),
        (r"\int \frac{dx}{x^2}", -1 / x),
        (r"\int \frac{x^2 dx}{2}", x**3 / 6),
    ],
)
def test_integrals(wb, source, expected):
    assert wb.expr(source) == expected


def test_matrices(wb):
    assert wb.expr(r"\begin{matrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \\ \end{matrix}") == Array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    )
    assert wb.expr(r"\begin{matrix} 1.123 & 2.12 \\ 4.123 & 5.32 \end{matrix}") == Array(
        [[1.123, 2.12], [4.123, 5.32]]
    )
    assert wb.expr(r"\begin{matrix} f(x) & 2 \\ 4 & h(y, x) \end{matrix}") == Array(
        [[f(x), 2], [4, h(y, x)]]
    )
    assert wb.expr(r"\begin{matrix} f(x) & 2 & 3 \end{matrix}") == Array([[f(x), 2, 3]])
    assert wb.expr(r"\begin{vmatrix} a & b \\ c & d \end{vmatrix}") == Symbol("a") * Symbol(
        "d"
    ) - Symbol("b") * Symbol("c")


def test_a_definition_with_equiv(wb):
    source = r"""
        C \equiv \begin{matrix}
            1 & 2 & 3 \\
            0 & 0 & x \\
            y & f(x) & 0 \\
        \end{matrix}
        \newline
        C
    """
    assert wb.expr(source) == Array([[1, 2, 3], [0, 0, x], [y, f(x), 0]])


def test_cases(wb):
    source = r"""
        H(x) := \begin{cases} 1 & \text{if } x \geq 0 \\ 0 & \text{otherwise} \end{cases}
        H(3) + H(-3)
    """
    assert wb.expr(source) == 1


def test_letters_in_latex_are_products(wb):
    G, M, c, r = (Symbol(n) for n in "GMcr")
    assert wb.expr(r"\frac{2GM}{c^2 r}") == 2 * G * M / (c**2 * r)
    assert wb.expr("GM") == Symbol("GM")  # plain text: one name
    assert wb.expr(r"\mathrm{GM} + \theta") == Symbol("GM") + theta  # \mathrm keeps a name whole


def test_a_script_on_letters_binds_to_the_last_even_a_tensor(wb):
    wb.exe(r"g_{\mu\nu} := DeSitter(L)")  # Rg_{\mu\nu} is R g_{\mu\nu}, with R nonzero here
    assert wb.expr(r"G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2} Rg_{\mu\nu}")


def test_functions_without_parentheses(wb):
    omega = Symbol("omega")
    assert wb.expr(r"\sin\omega t") == sin(omega * t)
    assert wb.expr(r"\sin^2\theta + \cos^2\theta") == sin(theta) ** 2 + cos(theta) ** 2
    assert wb.expr("sin x") == sin(x)


def test_accents(wb):
    assert wb.expr(r"\bar{x} + \hat{n}") == Symbol("xbar") + Symbol("nhat")
    assert wb.expr(r"\dot{a}") == Symbol("adot")
    assert wb.expr(r"\dot{x(t)}") == diff(Function("x")(t), t)
    assert wb.expr(r"\ddot{x(t)}") == diff(Function("x")(t), t, 2)


def test_unicode(wb):
    assert wb.expr("θ² + π") == theta**2 + pi
    assert wb.expr("√(x + 1) · 2") == 2 * sqrt(x + 1)
