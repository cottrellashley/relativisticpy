r"""LaTeX in, mathematics out (ported from the original test suite, plus more)."""

import pytest
from sympy import (
    Array,
    Function,
    Integral,
    Rational,
    Sum,
    Symbol,
    cos,
    diff,
    exp,
    factorial,
    integrate,
    oo,
    pi,
    root,
    sin,
    sqrt,
)

from helpers import same
from relativisticpy.errors import CompileError
from relativisticpy.language import Source

x, y, z, t, tau, theta = (Symbol(n) for n in ("x", "y", "z", "t", "tau", "theta"))
f, g, h = Function("f"), Function("g"), Function("h")


def test_sums_and_products_are_carried_out_where_a_closed_form_is_found(wb):
    n = Symbol("n")
    assert wb.expr(r"\sum_{x=0}^{10} x") == 55
    assert wb.expr(r"\prod_{x=1}^{4} x") == 24
    assert wb.expr(r"\sum_{x=1}^{n} x") == n**2 / 2 + n / 2  # to a symbol (decision 4)
    assert wb.expr(r"\sum_{x=1}^{oo} 1/x**2") == pi**2 / 6
    assert wb.expr(r"\sum_{x=1}^{oo} 1/x**x") == Sum(1 / x**x, (x, 1, oo))  # none: as written
    assert wb.expr(r"doit(\sum_{n=1}^{\infty} \frac{1}{n^2})") == pi**2 / 6
    wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
    assert wb.expr(r"\sum_{n=0}^{2} R_{\mu\nu}").nonzero() == {}  # no Sum(0, ...) left


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


def test_a_bar_where_the_form_is_zero_over_zero_evaluates_it_cancelled(wb):
    assert wb.expr(r"\left. \frac{x^2 - 1}{x - 1} \right|_{x = 1}") == 2
    assert wb.expr(r"\left. \frac{1}{x - 1} - \frac{2}{x^2 - 1} \right|_{x = 1}") == pi**0 / 2
    assert wb.expr(r"\left. \tan(x) \cos(x) \right|_{x = \pi/2}") == 1
    # in a form no one wrote: the surface gravity of the Schwarzschild horizon, 1/(4M),
    # which the curriculum had counted as right while it was nan
    wb.exe(r"g_{\mu\nu} := Schwarzschild(M)" "\n" r"\xi^\mu := [1, 0, 0, 0]")
    gravity = wb.expr(
        r"\left. \sqrt{-\frac{1}{2} (\nabla_\mu \xi_\nu)(\nabla^\mu \xi^\nu)} \right|_{r = 2M}"
    )
    assert same(gravity**2, 1 / (16 * Symbol("M") ** 2))


def test_fractions_and_roots(wb):
    assert wb.expr(r"\frac{x}{10} * 1/x") == pi**0 / 10
    assert wb.expr(r"\frac{x}{oo} * x**2") == 0
    assert wb.expr(r"\frac12") == pi**0 / 2
    assert wb.expr(r"\sqrt{ x }").equals(root(x, 2))
    assert wb.expr(r"\sqrt{ 10.23 * y }").equals(root(10.23 * y, 2))
    assert wb.expr(r"\sqrt{ sin(x)**2 }").equals(root(sin(x) ** 2, 2))
    assert wb.expr(r"\sqrt[3]{x}") == root(x, 3)


def test_statements_separated_by_newline_commands(wb):
    assert wb.expr(r" A := \prod_{x=1}^{n} x \newline A") == factorial(Symbol("n"))


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
        # d(...) against its brackets was a call of d: sin^2(theta)/theta, and theta
        (r"\frac{d(\sin^2\theta)}{d\theta}", diff(sin(theta) ** 2, theta)),
        (r"\frac{d(\theta^2)}{d\theta}", 2 * theta),
        (r"\frac{\mathrm{d}(\cos\theta)}{\mathrm{d}\theta}", -sin(theta)),
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


def test_eulers_operator(wb):
    # D_x f(x) was the symbol D_x times f(x), and Df(x) D times f(x): Stewart writes both
    # for the derivative (design/notation.md, 18)
    wb.exe("f(x) := x^3")
    (step,) = list(wb.each(Source("D_x f(x)", latex=True)))
    assert step.values == [3 * x**2]
    assert step.notes == [
        "there are no coordinates, so D_x is the derivative by x, d/dx, as Euler writes it"
    ]
    (step,) = list(wb.each(Source("Df(x)", latex=True)))
    assert step.values == [3 * x**2]
    assert step.notes == [
        r"D f(x) is read as Euler's D, the derivative f'(x); for D times f(x), write "
        r"D \cdot f(x)"
    ]
    D = Symbol("D")
    assert wb.expr(r"D \cdot f(x)") == D * x**3  # as the note says
    assert wb.expr(Source("D_x + 1", latex=True)) == Symbol("D_x") + 1  # alone, a name
    wb.exe("D := 2")
    assert wb.expr(Source("D f(x)", latex=True)) == 2 * x**3  # D with a value multiplies


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # (x^3)' was x'^3 and (e^{-x^2})' e'^(-x^2): the prime was put on the base inside
        ("(x^3)'", diff(x**3, x)),
        ("(e^{-x^2})'", diff(exp(-(x**2)), x)),
        (r"(\sin\theta)'", diff(sin(theta), theta)),
        (r"\left(x^2 + 1\right)''", 2),
        ("(x^3)'(2) + 1", 13),
        ("(3)'", 0),
    ],
)
def test_primes_after_brackets_differentiate_by_the_one_variable(wb, source, expected):
    *_, step = wb.each(source)
    assert step.values == [expected]
    if expected not in (0, 13):
        assert "the one variable it is written with" in step.notes[0]


def test_primes_after_brackets_of_several_variables_are_refused(wb):
    with pytest.raises(CompileError, match="its prime does not say which variable") as error:
        wb.expr("(x^2 y)'")
    assert error.value.hint == r"write the variable, as in \frac{d}{dx}(x^2 y)"


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


def test_unicode_fractions_divide(wb):
    # ½ and the fraction slash of 3⁄4 were refused
    assert wb.expr("½ + ⅓") == Rational(5, 6)
    assert wb.expr("(x + 1)⁄(x − 1)") == (x + 1) / (x - 1)


def test_unicode_spells_latexs_functions_in_letters(wb):
    # sin²x + cos²x was c o s² x + i n² s x, the i of sin taken for the imaginary unit
    assert wb.expr("sin²x + cos²x") == sin(x) ** 2 + cos(x) ** 2
    assert wb.expr("cos²θ") == cos(theta) ** 2
    assert wb.expr("∫_0^π sin x dx") == 2
    assert wb.expr("lim_{x→0} sin(x)/x") == 1
    wb.exe("y(x) := x² sin x")
    assert wb.expr("y'(x)") == diff(x**2 * sin(x), x)
    # applied to nothing, it is a name, with a note; in LaTeX too, never s i n
    assert wb.expr("θ + sin") == theta + Symbol("sin")
    assert wb.expr(r"\theta sin x") == theta * sin(x)
