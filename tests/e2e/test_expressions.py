"""Scalar expressions and calculus (ported from the original test suite)."""

import re

import pytest
from sympy import (
    Array,
    E,
    Function,
    Rational,
    Symbol,
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
        ("lim(1/x, x, 0)", limit(1 / x, x, 0, "+-")),  # from both sides, as \lim is
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
