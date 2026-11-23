"""Scalar expressions and calculus (ported from the original test suite)."""

import re

import pytest
from sympy import (
    Array,
    E,
    Function,
    Rational,
    Symbol,
    cos,
    diff,
    dsolve,
    expand,
    integrate,
    limit,
    oo,
    pi,
    simplify,
    sin,
    solve,
)

from relativisticpy import EvaluationError
from relativisticpy.errors import CompileError

x, y = Symbol("x"), Symbol("y")
f, P, Q = Function("f"), Function("P"), Function("Q")


def test_numbers(wb):
    assert wb.expr("2 * 3") == 6
    assert wb.expr("-2 * 3") == -6
    assert wb.expr("-2 * -3") == 6
    assert wb.expr("2 - 1") == 1
    assert wb.expr("-2 - -1") == -1
    assert wb.expr("2.5 - 1") == 1.5
    assert wb.expr("-2.5 - -1.5") == -1.0


def test_division_is_exact(wb):
    assert str(wb.expr("1/2")) == "1/2"
    assert str(wb.expr("(1/2) * x")) == "x/2"


def test_constants(wb):
    assert wb.expr("2 - 1 - pi") == 1 - pi
    assert wb.expr("-2 - 1 + e") == -3 + E
    assert wb.expr("-2 - 1 + oo") == oo


def test_functions_and_operators_written_in_letters_are_latexs(wb):
    # sin 2x was sin(2) x, log_2 8 the name log_2 times 8, and int_0^1 x^2 dx and
    # min_x x^2 - 4x + 7 names times x^2
    assert wb.expr("sin 2x") == sin(2 * x)
    assert wb.expr("sin^2 x + cos^2 x") == sin(x) ** 2 + cos(x) ** 2
    assert wb.expr("sin x / cos x") == sin(x) / cos(x)
    assert wb.expr("log_2 8") == 3
    assert wb.expr("e^ln(2)") == 2  # was e^l n(2)
    assert wb.expr("2^log_2(8)") == 8
    assert wb.expr("12 mod 5") == wb.expr("mod(12, 5)") == 2
    assert wb.expr("min_x x^2 - 4x + 7") == 3
    assert wb.expr("int_0^1 x^2 dx") == Rational(1, 3)
    assert wb.expr("int_0^oo exp(-x) dx") == 1
    assert wb.expr("lim_(x->0) sin(x)/x") == 1
    assert wb.expr("sum_{n=1}^oo 1/n^2") == pi**2 / 6


def test_a_solidus_between_differentials_is_leibnizs_fraction(wb):
    # d/dx sin(x) was d sin(x)/dx, d(x^3)/dx a function d at x^3, and df/dx a name over dx
    assert wb.expr("d/dx sin(x)") == cos(x)
    assert wb.expr("d/dx (x^3)") == 3 * x**2
    assert wb.expr("d(x^3)/dx") == 3 * x**2
    assert wb.expr("d^2/dx^2 (x^4)") == 12 * x**2
    assert wb.expr("-d/dx (x^2)") == -2 * x
    wb.exe("f(x) := x^3")
    assert wb.expr("df/dx") == 3 * x**2
    assert wb.expr("x df/dx") == 3 * x**3
    wb.exe("g(x, y) := x^2 y")
    assert wb.expr("∂g/∂x") == 2 * x * y
    assert wb.expr("(del g)/(del x)") == 2 * x * y  # AsciiMath writes ∂ as del
    assert wb.expr("∂²g/∂x∂y") == 2 * x


def test_sum_and_product_called_as_sympy_calls_them_are_latexs(wb):
    # sum(1/n^2, (n, 1, oo)) was an unknown function, refused for the tuple in it
    assert wb.expr("sum(1/n^2, (n, 1, oo))") == pi**2 / 6
    assert wb.expr("summation(k, (k, 1, 10))") == 55
    assert wb.expr("product(k, (k, 1, 5))") == 120
    assert wb.expr("sum(1/(n^2 m^2), (n, 1, oo), (m, 1, oo))") == pi**4 / 36
    wb.exe("a_n := 1/n")
    assert wb.expr("sum(a_n, (n, 1, 3))") == wb.expr(r"\sum_{n=1}^{3} a_n") == Rational(11, 6)
    with pytest.raises(CompileError, match="its index and the index's limits"):
        wb.expr("sum([1, 2, 3])")
    wb.exe("product(a, b) := a b")  # the program's own wins
    assert wb.expr("product(2, 3)") == 6


def test_oo_is_infinity_in_every_notation(wb):
    # \int_0^oo was the integral to o of o e^{-x}, and 2^oo was 2^o o
    assert wb.expr(r"\int_0^oo e^{-x} dx") == 1
    assert wb.expr("∫_0^oo e^(−x) dx") == 1
    assert wb.expr("2^oo") == oo


def test_symbols(wb):
    assert wb.expr("2 * x") == 2 * x
    assert wb.expr("-2.5 * x * y") == -2.5 * x * y
    assert wb.expr("y - x") == y - x
    assert wb.expr("-1.5 * x - y") == -1.5 * x - y


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("integrate(x**2, x)", integrate(x**2, x)),
        ("diff(x**3 + 2*x**2 - 1, x)", diff(x**3 + 2 * x**2 - 1, x)),
        ("diff(x**2*y**3, y)", diff(x**2 * y**3, y)),
        ("simplify((x + 1)**2 - x**2)", simplify((x + 1) ** 2 - x**2)),
        ("expand((x + y)*(x - y))", expand((x + y) * (x - y))),
        ("lim((x**2 - 1)/(x - 1), x, 1)", limit((x**2 - 1) / (x - 1), x, 1)),
        ("lim(1/x^2, x, 0)", limit(1 / x**2, x, 0, "+-")),  # from both sides, as \lim is
        ("sin(x)", sin(x)),
    ],
)
def test_calculus_functions(wb, source, expected):
    assert wb.expr(source) == expected


def test_solve(wb):
    assert wb.expr("solve(x + 2*y - 3, x)") == solve(x + 2 * y - 3, x)
    assert wb.expr("solve([x + y - 2, x - y + 1], [x, y])") == solve([x + y - 2, x - y + 1], [x, y])


def test_diag(wb):
    assert wb.expr("diag(x)") == Array([[x]])
    assert wb.expr("diag(x**2, y**x, 1, 10*x)") == Array(
        [[x**2, 0, 0, 0], [0, y**x, 0, 0], [0, 0, 1, 0], [0, 0, 0, 10 * x]]
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("2 [1, 2]", Array([2, 4])),
        ("-[1, 2]", Array([-1, -2])),
        ("[2, 4]/2", Array([1, 2])),
        ("x \\cdot (1, 2)", Array([x, 2 * x])),
        ("[1, 2] + [3, 4]", Array([4, 6])),
        ("[[1, 2], [3, 4]] [[0, 1], [1, 0]]", Array([[2, 1], [4, 3]])),
        (
            r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}",
            Array([[x + 2 * y], [3 * x + 4 * y]]),
        ),
        ("diag(1, 2)^2", Array([[1, 0], [0, 4]])),
        ("[[1, 2], [3, 4]]^{-1}", Array([[-2, 1], [Rational(3, 2), Rational(-1, 2)]])),
    ],
)
def test_lists_are_vectors_and_matrices(wb, source, expected):
    assert wb.expr(source) == expected


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("[1, 2] + 1", "cannot add a scalar and an array"),
        ("[1, 2] + [1, 2, 3]", "cannot add arrays of shapes (2,) and (3,)"),
        ("[1, 2] [3, 4]", "cannot multiply arrays of shapes (2,) and (2,)"),
        ("[1, 2]^2", "only a square matrix can be raised to a power"),
        ("[[1], [2, 3]] + 1", "only rectangular lists"),
        (
            "Coordinates := [x, y]\nT_{ab} := diag(1, 2)\ndiag(1, 2) T_{ab}",
            "a tensor can be multiplied by scalars, not by an array",
        ),
    ],
)
def test_arithmetic_errors(wb, source, message):
    with pytest.raises(EvaluationError, match=re.escape(message)):
        wb.expr(source)


def test_long_expressions(wb):
    # Thousands of terms or factors, as in a long expansion, are no deeper to compile.
    n = 3000
    total = wb.expr(" + ".join(f"x^{{{i}}}" for i in range(n)))
    assert len(total.args) == n
    assert wb.expr(" ".join(["x"] * n)) == x**n
    assert wb.expr(r"\int_0^1 " + " ".join(["x"] * 100) + r" \, dx") == Rational(1, 101)


def test_definitions_are_remembered(wb):
    a = diff(f(x), x) + P(x) * f(x) - Q(x)
    # P(x) f(x), side by side: P(x)*f(x) is their convolution (design/notation.md, 27)
    assert wb.expr("a := diff(f(x), x) + P(x) f(x) - Q(x)\ndsolve(a, f(x))") == dsolve(a, f(x))
    assert wb.expr("b := f(x)\ndiag(a, b)") == Array([[a, 0], [0, f(x)]])
