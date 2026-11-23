r"""LaTeX forms an agent writes in algebra and calculus, each held to its value computed by
SymPy on its own, or to a refusal: brackets with powers and primes, derivatives written
as fractions, products written side by side, powers of functions, nested scripts,
absolute values, roots, ``\cdot`` and ``\times`` between numbers and names, and decimals
in exponents. A form read otherwise than as it is written gives a silent wrong answer,
the worst bug there is, so each is read in LaTeX, as a paper or an agent writes it."""

from __future__ import annotations

from typing import Any

import pytest
import sympy
from sympy import Abs, Function, Rational, acos, asin, atan, cos, diff, exp, log, pi, sin, sqrt

from relativisticpy import Workbook
from relativisticpy.errors import RelativisticPyError
from relativisticpy.language import Source

a, b, c, n, x, y, z, r, t = sympy.symbols("a b c n x y z r t")
theta, phi, alpha, beta = sympy.symbols("theta phi alpha beta")

#: What a form that must be refused has in the place of a value.
REFUSED = object()

BRACKETS = [
    # a power after brackets is of all they hold (x y^2 and a b/c before)
    (r"(xy)^2", x**2 * y**2),
    (r"\left(xy\right)^2", x**2 * y**2),
    (r"a (bc)^{-1}", a / (b * c)),
    (r"(2ab)^3", 8 * a**3 * b**3),
    (r"(ab)^n", (a * b) ** n),
    (r"(xyz)^2", x**2 * y**2 * z**2),
    (r"(xy)^{1/2}", sqrt(x * y)),
    (r"\frac{1}{(xy)^2}", 1 / (x**2 * y**2)),
    (r"e^{(xy)^2}", exp(x**2 * y**2)),
    (r"(\alpha\beta)^2", alpha**2 * beta**2),
    (r"\Bigl(xy\Bigr)^2", x**2 * y**2),
    (r"\left[xy\right]^2", x**2 * y**2),
    (r"\left[1 + x^2\right]^{-1}", 1 / (1 + x**2)),
    (r"(xy)^2(ab)^3", x**2 * y**2 * a**3 * b**3),
    (r"\frac{d}{dx}(xy)^2", diff((x * y) ** 2, x)),
    (r"\int_0^1 (xy)^2 \, dx", sympy.integrate((x * y) ** 2, (x, 0, 1))),
    (r"(x+1)^2(x-1)", (x + 1) ** 2 * (x - 1)),
    (r"(x^2)^3", x**6),
    (r"(-x)^2", x**2),
    (r"-2^2", -4),
    (r"[xy]^2", REFUSED),  # square brackets of one item are a list (decision 31)
    # a name that multiplies the brackets after it raises them alone (\pi^2 (x+1)^2 before)
    (r"\pi(x + 1)^2", pi * (x + 1) ** 2),
    (r"x(x + 1)^2 + \alpha", x * (x + 1) ** 2 + alpha),
    (r"(xy)(x + 1)", x * y * (x + 1)),
    # a name in brackets of its own multiplies the brackets after it (a(b) and an error
    # that \beta is a function before)
    (r"(a)(b)", a * b),
    (r"(\alpha)(\beta)", alpha * beta),
    (r"(a)(b)(c)", a * b * c),
    (r"2(\beta)", 2 * beta),
    (r"\alpha (\beta)", alpha * beta),
    (r"(\zeta)(2)", 2 * sympy.Symbol("zeta")),
    # a function the program defines is called by the brackets after it, whatever spaces
    # TeX does not show are between them (3 x^2 before)
    ("f(x) := x^2\nf (3)", 9),
    ("f(x) := x^2\n(f)(3)", 9),
    ("f(x) := x^2\nf \\left(3\\right)", 9),
    ("y(x) := x^3\ny (x + 1)", (x + 1) ** 3),
    # a letter after a subscript of a number starts a factor, as TeX sets it (x_1x_2 was a
    # name of its own)
    (r"x_1x_2", sympy.Symbol("x_1") * sympy.Symbol("x_2")),
    (r"x_1y", sympy.Symbol("x_1") * y),
    (r"x_1x_2x_3", sympy.Symbol("x_1") * sympy.Symbol("x_2") * sympy.Symbol("x_3")),
    ("x_1 := 3\nx_2 := 5\nx_1x_2", 15),
    ("x_n := n^2\nx_1x_3 + x_2", 13),
    (r"x_{12}", sympy.Symbol("x_12")),
]

PRIMES = [
    # a prime after brackets is the derivative of the bracket (x'^3 and e'^(-x^2) before)
    (r"(x^3)'", diff(x**3, x)),
    (r"(e^{-x^2})'", diff(exp(-(x**2)), x)),
    (r"(\sin\theta)'", cos(theta)),
    (r"(x^2 + 1)''", 2),
    (r"\left(x^3\right)'", 3 * x**2),
    (r"(\theta^2)'", 2 * theta),
    (r"(x^2 + 1)'(3)", 6),
    (r"(xy)'", REFUSED),  # by x or by y?
    ("f(x) := x^3\nf'(x)", 3 * x**2),
    ("f(x) := x^3\n(f(x))'", 3 * x**2),
]

F_XY = "f(x, y) := x^2 y^3\n"

DERIVATIVES = [
    # Stewart numbers the variables of a function of several: f_1, f_{12} and D_1 f were
    # names of their own, and D_1 times f
    (F_XY + "f_1", 2 * x * y**3),
    (F_XY + "f_2", 3 * x**2 * y**2),
    (F_XY + "f_{12}", 6 * x * y**2),
    (F_XY + "f_{11}", 2 * y**3),
    (F_XY + "f_1(1, 2)", 16),
    (F_XY + "D_1 f", 2 * x * y**3),
    (F_XY + "D_2 f(x, y)", 3 * x**2 * y**2),
    (F_XY + "2 D_1 f", 4 * x * y**3),
    (F_XY + r"D_1 \cdot f", sympy.Symbol("D_1") * x**2 * y**3),  # set apart, a product
    ("f(x, y, z) := x y z^2\nf_{13}", 2 * y * z),
    ("h(r, \\theta) := r^2 \\sin\\theta\nh_2", r**2 * cos(theta)),
    ("g(x) := x^2\ng_1", sympy.Symbol("g_1")),  # of one variable, a name of its own
    (F_XY + "f_3", sympy.Symbol("f_3")),  # f has no third variable
    (F_XY + "f_1 := 5\nf_1", 5),  # a name the program defines wins
    # d with brackets against it, atop a fraction (sin^2(theta)/theta before)
    (r"\frac{d(\sin^2\theta)}{d\theta}", diff(sin(theta) ** 2, theta)),
    (r"\frac{d(x^2)}{dx}", 2 * x),
    (r"\frac{d(\cos\phi)}{d\phi}", -sin(phi)),
    (r"\frac{d^2(x^3)}{dx^2}", 6 * x),
    (r"\frac{\partial (x^2 y)}{\partial x}", 2 * x * y),
    (r"\frac{d(\theta^2)}{d\theta}", 2 * theta),
    (r"\frac{\mathrm{d}(\sin\theta)}{\mathrm{d}\theta}", cos(theta)),
    (r"\frac{d}{dx}\left(x^2\right)", 2 * x),
    (r"\frac{d}{d\theta}\sin^2\theta", diff(sin(theta) ** 2, theta)),
    ("f(x) := x^3\nD_x f(x)", 3 * x**2),  # Euler's D (decision 18)
    ("f(x) := x^3\nDf(x)", 3 * x**2),
]

FRACTIONS_AND_PRODUCTS = [
    (r"\tfrac{1}{2}x", x / 2),
    (r"\frac12 x", x / 2),
    (r"\frac{1}{2}xy", x * y / 2),
    (r"\dfrac{x}{2}y", x * y / 2),
    (r"\frac{a}{b}c", a * c / b),
    (r"2\pi r", 2 * pi * r),
    (r"\pi r^2", pi * r**2),
    (r"(x+1)(x-1)", (x + 1) * (x - 1)),
    (r"ab^2c", a * b**2 * c),
    (r"\frac{x+1}{2}(x-1)", (x + 1) * (x - 1) / 2),
    (r"3!", 6),
]

FUNCTION_POWERS = [
    (r"\sin^{-1} x", asin(x)),
    (r"\sin^{-1}(1/2)", pi / 6),
    (r"\ln^2 x", log(x) ** 2),
    (r"\sin^2(2x)", sin(2 * x) ** 2),
    (r"\cos^2 x + \sin^2 x", 1),
    (r"\tan^{-1}(1)", pi / 4),
    (r"\log^2(x)", log(x) ** 2),
    (r"\sin x^2", sin(x**2)),
    (r"\sin(x)^2", sin(x) ** 2),
    (r"\cos^{-1}(0)", pi / 2),
    (r"\sin 2x", sin(2 * x)),
    (r"\sin^3 2x", sin(2 * x) ** 3),
    (r"\ln(x)^2", log(x) ** 2),
    (r"\sqrt{x}^3", x ** Rational(3, 2)),
    (r"\log_{10} 1000", 3),
    (r"\sin^{-1}x + \cos^{-1}x", asin(x) + acos(x)),
    (r"\arctan 1", atan(1)),
]

SCRIPTS = [
    (r"x^{y^2}", x ** (y**2)),
    (r"x^{2^3}", x**8),
    (r"2^{2^2}", 16),
    (r"e^{-x^2/2}", exp(-(x**2) / 2)),
    (r"x^{-1/2}", 1 / sqrt(x)),
    (r"e^{i\pi}", -1),
    (r"x_1^2", sympy.Symbol("x_1") ** 2),
    (r"x^2_1", sympy.Symbol("x_1") ** 2),
    (r"x^{10}", x**10),
    (r"x^{2n}", x ** (2 * n)),
    ("J^0", sympy.Symbol("J^0")),  # a component, never J to the power 0 (decision 6)
    (r"{\partial}_{i}F^{0i} = 4\pi J^0", REFUSED),  # F is no tensor
]

ABSOLUTE_VALUES_AND_ROOTS = [
    (r"|x - 1|", Abs(x - 1)),
    (r"\left|x\right|", Abs(x)),
    (r"|-3|", 3),
    (r"2|x|", 2 * Abs(x)),
    (r"\left|\frac{x}{2}\right|", Abs(x) / 2),
    (r"\sqrt{xy}", sqrt(x * y)),
    (r"\sqrt[3]{8}", 2),
    (r"\sqrt{x}y", sqrt(x) * y),
    (r"\sqrt[n]{x}", x ** (1 / n)),
    (r"\sqrt{4}^3", 8),
    (r"\sqrt{2}\sqrt{8}", 4),
    (r"\sqrt[3]{x^2}", sympy.root(x**2, 3)),
]

NUMBERS = [
    (r"2 \cdot 3", 6),
    (r"2 \times 3 + 4", 10),
    (r"2 \cdot x", 2 * x),
    (r"x \times y", x * y),
    (r"3 \times 10^8", 300000000),
    (r"2.5 \times 10^{-3}", 0.0025),
    (r"a \cdot b^2", a * b**2),
    (r"2 \cdot 3^2", 18),
    (r"3 \times (2 + 1)", 9),
    (r"\frac{3 \times 10^8}{2}", 150000000),
    (r"x^{0.5}", x**0.5),
    (r"2^{0.5}", 2**0.5),
    (r"e^{0.1}", float(sympy.exp(sympy.Float(0.1)))),
    (r"10^{-0.5}", 10**-0.5),
    (r"e^{-0.5 x}", exp(-0.5 * x)),
    (r"1.5^2", 2.25),
    (r"(0.5)^2", 0.25),
    (r"0.5^{-1}", 2.0),
]

RELATIONS = [
    # a comparison among equalities is links of their chain (2 equal to True before)
    ("1 + 1 = 2 < 3", [sympy.true, sympy.true]),
    ("1 + 1 = 2 > 3", [sympy.true, sympy.false]),
    ("2 = 2 = 2", [sympy.true, sympy.true]),
]

CONJUGATES = [
    # a bar over a quantity the program defines, or declares complex, is its conjugate, and
    # so is a star on brackets (the name zbar, and (3 + 4i) to the power of a name *, before)
    ("z := 3 + 4i\n\\bar{z}", 3 - 4 * sympy.I),
    ("z := 3 + 4i\n\\bar z", 3 - 4 * sympy.I),
    ("z := 3 + 4i\n\\overline{z}", 3 - 4 * sympy.I),
    ("z := \\frac{1}{1 + i}\n\\bar{z}", (1 + sympy.I) / 2),
    ("z := e^{i \\pi / 3}\n\\bar{z}", exp(-sympy.I * pi / 3)),
    (r"\overline{3 + 4i}", 3 - 4 * sympy.I),
    (r"(3 + 4i)^*", 3 - 4 * sympy.I),
    (r"(2i)^*", -2 * sympy.I),
    (r"(5)^*", 5),
    (r"\left(\frac{1}{1 + i}\right)^*", (1 + sympy.I) / 2),
    (r"(e^{i \pi / 3})^*", exp(-sympy.I * pi / 3)),
    ("w \\in \\mathbb{C}\n\\bar{w}", sympy.conjugate(sympy.Symbol("w", complex=True))),
    ("w \\in \\mathbb{C}\nw^*", sympy.conjugate(sympy.Symbol("w", complex=True))),
    # a bar on a name that means nothing is a name of its own, and one the program gives wins
    (r"\bar{q}", sympy.Symbol("qbar")),
    ("\\bar{x} := 7\n\\bar{x}", 7),
    ("x := [1, 2, 3, 6]\n\\bar{x}", sympy.Symbol("xbar")),
]

OPERATOR_NAMES = [
    # a name written as an operator that nothing defines applies to what follows it, an
    # unknown function, as it does to brackets (8 lb and arsinh times x before)
    (r"\operatorname{lb} 8", Function("lb")(8)),
    (r"\operatorname{foo} x", Function("foo")(x)),
    (r"\operatorname{foo} x y", Function("foo")(x * y)),
    (r"2 \operatorname{sinc} x + 1", 2 * Function("sinc")(x) + 1),
    (r"\operatorname{foo} \cdot x", sympy.Symbol("foo") * x),  # set apart, a product
    (r"\operatorname{erf} x", sympy.erf(x)),
    # the ISO 80000-2 names of the inverse hyperbolic functions (an error before)
    (r"\operatorname{arsinh} x", sympy.asinh(x)),
    (r"\operatorname{arcosh}(x)", sympy.acosh(x)),
    (r"\operatorname{artanh} \frac{3}{5}", sympy.log(2)),
]

CHEMISTRY = [
    # the upright p of chemists (decision 39)
    (
        "K_\\mathrm{a} := 1.8 \\times 10^{-5}\n\\mathrm{p}K_\\mathrm{a}",
        -log(Rational(18, 10**6), 10),
    ),
]


def case(group: list[tuple[str, Any]]) -> list[Any]:
    return [pytest.param(source, expected, id=source) for source, expected in group]


def value(source: str) -> Any:
    """What a program gives, read in LaTeX in every statement: its values, or the error."""
    with Workbook() as wb:
        try:
            return wb.exe(Source(source, latex=True))
        except RelativisticPyError as error:
            return error


def same(found: Any, expected: Any) -> bool:
    """Whether a value is the one expected: to rounding error, where either is a decimal."""
    try:
        difference = sympy.simplify(sympy.sympify(found) - sympy.sympify(expected))
    except (sympy.SympifyError, TypeError):
        return False
    if difference == 0:
        return True
    return bool(difference.is_number) and abs(complex(sympy.N(difference))) < 1e-9


@pytest.mark.parametrize(
    ("source", "expected"),
    case(BRACKETS)
    + case(PRIMES)
    + case(DERIVATIVES)
    + case(FRACTIONS_AND_PRODUCTS)
    + case(FUNCTION_POWERS)
    + case(SCRIPTS)
    + case(ABSOLUTE_VALUES_AND_ROOTS)
    + case(NUMBERS)
    + case(CONJUGATES)
    + case(OPERATOR_NAMES)
    + case(CHEMISTRY),
)
def test_a_form_gives_what_sympy_computes_or_is_refused(source, expected):
    found = value(source)
    if expected is REFUSED:
        assert isinstance(found, RelativisticPyError), f"{source} gives {found}"
        return
    assert not isinstance(found, RelativisticPyError), f"{source}: {found}"
    assert same(found[-1], expected), f"{source} gives {found[-1]}, not {expected}"


@pytest.mark.parametrize(("source", "expected"), case(RELATIONS))
def test_a_chain_of_relations_gives_a_truth_value_per_link(source, expected):
    assert value(source) == expected


SQUARE = "A := \\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}\n"
NINE = "A := \\begin{bmatrix} 1 & 2 & 3 \\\\ 4 & 5 & 6 \\\\ 7 & 8 & 9 \\end{bmatrix}\n"
SHEAR = "A := \\begin{pmatrix} 1 & 1 \\\\ 0 & 1 \\end{pmatrix}\n"
natural_t = sympy.Symbol("t", integer=True, nonnegative=True)

TRANSPOSES = [
    # a t raised on a matrix, after it or before it, is its transpose, as Artin and Lang
    # write it, where t means nothing: it was the t-th power, silently, and for a 3 x 3
    # matrix a crash; and a tilde over one, as Goldstein and Griffiths write it, the name
    # Atilde
    (SQUARE + "A^t", sympy.Matrix([[1, 3], [2, 4]])),
    (SQUARE + "A^{t}", sympy.Matrix([[1, 3], [2, 4]])),
    (NINE + "A^t", sympy.Matrix([[1, 4, 7], [2, 5, 8], [3, 6, 9]])),
    (SQUARE + "{}^{t}A", sympy.Matrix([[1, 3], [2, 4]])),
    (SQUARE + "{}^tA", sympy.Matrix([[1, 3], [2, 4]])),
    (SQUARE + "\\tilde{A}", sympy.Matrix([[1, 3], [2, 4]])),
    (SQUARE + "\\tilde A", sympy.Matrix([[1, 3], [2, 4]])),
    # where t means something, A^t is a power, and a definition of \tilde{A} wins
    (SHEAR + "t \\in \\mathbb{N}\nA^t", sympy.Matrix([[1, natural_t], [0, 1]])),
    (SHEAR + "t := 3\nA^t", sympy.Matrix([[1, 3], [0, 1]])),
    (SQUARE + "\\tilde{A} := 5\n\\tilde{A}", 5),
]


@pytest.mark.parametrize(("source", "expected"), case(TRANSPOSES))
def test_a_transpose_is_read_as_books_write_it(source, expected):
    found = value(source)
    assert not isinstance(found, RelativisticPyError), f"{source}: {found}"
    if isinstance(expected, sympy.Matrix):
        assert sympy.Matrix(found[-1]) == expected, f"{source} gives {found[-1]}"
    else:
        assert same(found[-1], expected), f"{source} gives {found[-1]}, not {expected}"


def test_a_name_with_brackets_after_it_stays_a_call_where_it_is_a_function():
    # f(x)^2 is the square of the call, and 2Gm(r) is 2 G times m(r)
    f, m = Function("f"), Function("m")
    assert same(value(r"f(x)^2")[-1], f(x) ** 2)
    assert same(value(r"2Gm(r)")[-1], 2 * sympy.Symbol("G") * m(r))
