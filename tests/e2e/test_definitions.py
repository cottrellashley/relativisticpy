"""Declarations, assumptions and complex numbers (ported)."""

import sympy
from sympy import I, Symbol, expand, factor, simplify, solve, symbols


def test_assumptions(wb):
    wb.expr("Constants := [C, G]")
    assert wb.expr("C + G") == symbols("C", constant=True) + symbols("G", constant=True)
    wb.expr("Integers := [n, m]")
    assert wb.expr("n + m") == symbols("n", integer=True) + symbols("m", integer=True)
    wb.expr("Reals := [x, y]")
    assert wb.expr("x + y") == symbols("x", real=True) + symbols("y", real=True)
    wb.expr("Complex := [z, w]")
    assert wb.expr("z + w") == symbols("z", complex=True) + symbols("w", complex=True)


def test_assumptions_reach_sympy(wb):
    wb.expr("Positive := [a]")
    assert wb.expr("sqrt(a^2)") == Symbol("a", positive=True)
    assert wb.expr("sqrt(b^2)") == sympy.sqrt(Symbol("b") ** 2)


def test_declared_symbols_in_calculations(wb):
    wb.exe("Coordinates := [t, r, theta, phi]\nConstants := [C, G]\nIntegers := [n, m]")
    C, G = symbols("C G", constant=True)
    n, m = symbols("n m", integer=True)
    assert wb.expr("C*G + n*m") == C * G + n * m
    assert wb.expr("factor(n**2 - 2*n*m + m**2)") == factor(n**2 - 2 * n * m + m**2)
    assert wb.expr("expand((t + r)**2)") == expand((Symbol("t") + Symbol("r")) ** 2)


def test_imaginary_unit(wb):
    assert wb.expr("i**2") == -1
    assert wb.expr("Re(1 + i)") == 1
    assert wb.expr("Im(1 + i)") == 1


def test_complex_algebra(wb):
    x, y = Symbol("x"), Symbol("y")
    assert wb.expr("expand((x + i*y)**2)") == expand((x + I * y) ** 2)
    assert wb.expr("simplify((x + i*y)**2 - x**2 - 2*i*x*y + y**2)") == 0
    assert wb.expr("factor(x**2 - 2*i*x*y + y**2)") == factor(x**2 - 2 * I * x * y + y**2)
    assert wb.expr("solve(x**2 + 2*i*x + 1, x)") == solve(x**2 + 2 * I * x + 1, x)


def test_destructuring(wb):
    x, y = Symbol("x"), Symbol("y")
    assert wb.expr("a, b := symbols('x y')\na - i*b") == x - I * y
    expected = solve([x - I * y, sympy.im(y), sympy.im(x)])
    assert wb.expr("x, y := symbols('x y')\nsolve([x - i*y, Im(y), Im(x)])") == expected
    assert simplify(wb.expr("x*y") - x * y) == 0
