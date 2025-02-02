import pytest
import sympy as sp
from relativisticpy.workbook.workbook import Workbook

# Do not import from sympy - only import from the SAME interface which Workbook points to implements everything.
from relativisticpy.symengine import (
    Symbol,
    Function,
    sin,
    integrate,
    solve,
    diff,
    expand,
    limit,
    simplify,
    dsolve,
    SymbolArray,
    pi,
    oo,
    E
)

# Workbook.expr() => is used for one line solver function - like a "unit test" for workbook module.


@pytest.fixture
def var_setup():
    x = Symbol("x")
    y = Symbol("y")
    return x, y


@pytest.fixture
def func_setup():
    f = Function("f")
    P = Function("P")
    Q = Function("Q")
    return f, P, Q


@pytest.fixture
def workbook_setup():
    return Workbook()


def test_mul_numbers(workbook_setup):
    wb = workbook_setup
    assert int(wb.expr("2 * 3")) == 6
    assert int(wb.expr("-2 * 3")) == -6
    assert int(wb.expr("-2 * -3")) == 6


def test_mul_symbols(workbook_setup, var_setup):
    wb = workbook_setup
    x, _ = var_setup
    assert wb.expr("2 * x") == 2 * x


def test_sub_symbols(workbook_setup, var_setup):
    wb = workbook_setup
    x, _ = var_setup
    assert simplify(wb.expr("x - 1") - (x - 1)) == 0


def test_integrate(workbook_setup, var_setup):
    wb = workbook_setup
    x, _ = var_setup
    assert simplify(wb.expr("integrate(x**2, x)") - integrate(x**2, x)) == 0


def test_series(workbook_setup, var_setup):
    wb = workbook_setup
    x, _ = var_setup
    assert wb.expr("sin(x)") - sin(x) == 0


def test_mul_symbols(workbook_setup, var_setup):
    wb = workbook_setup
    x, y = var_setup

    # Symbol and Number
    assert wb.expr("2 * x") - 2 * x == 0
    assert wb.expr("-2 * x") + 2 * x == 0
    assert wb.expr("-2.5 * x") + 2.5 * x == 0

    # Symbol and Symbol
    assert wb.expr("y * x") - y * x == 0
    assert wb.expr("-y * x") + y * x == 0
    assert wb.expr("-2.5 * x * y") + 2.5 * x * y == 0


def test_sub_numbers(workbook_setup):
    wb = workbook_setup
    # Integer Subtraction
    assert wb.expr("2 - 1") == 1
    assert wb.expr("-2 - 1") == -3
    assert wb.expr("-2 - -1") == -1

    # Float Subtraction
    assert wb.expr("2.5 - 1") == 1.5
    assert wb.expr("-2.5 - 1") == -3.5
    assert wb.expr("-2.5 - -1.5") == -1.0

def test_constant_numbers(workbook_setup):
    wb = workbook_setup
    # Integer Subtraction
    assert wb.expr("2 - 1 - pi") == 1 - pi
    assert wb.expr("-2 - 1 + e") == -3 + E
    assert wb.expr("-2 - 1 + oo") == oo

def test_sub_symbols(workbook_setup, var_setup):
    wb = workbook_setup
    x, y = var_setup

    # Symbol and Number
    assert wb.expr("x - 1") - (x - 1) == 0
    assert wb.expr("1 - x") - (1 - x) == 0
    assert wb.expr("-1.5 - x") - (-1.5 - x) == 0

    # Symbol and Symbol
    assert wb.expr("y - x") - (y - x) == 0
    assert wb.expr("-y - x") - (-y - x) == 0
    assert wb.expr("-1.5 * x - y") - (-1.5 * x - y) == 0


def test_solve(workbook_setup, var_setup):
    wb = workbook_setup
    x, y = var_setup

    # Simple Linear Equations
    # FAILING => need a standard way/function to call to get back same outputs
    assert wb.expr("solve(x + 2*y - 3, x)") == solve(x + 2 * y - 3, x)
    assert wb.expr("solve([x + y - 2, x - y + 1], [x, y])") == solve(
        [x + y - 2, x - y + 1], [x, y]
    )


def test_diff(workbook_setup, var_setup):
    wb = workbook_setup
    x, y = var_setup

    # Simple Differentiation
    assert wb.expr("diff(x**2, x)") - diff(x**2, x) == 0
    assert wb.expr("diff(x**3 + 2*x**2 - 1, x)") - diff(x**3 + 2 * x**2 - 1, x) == 0

    # Multiple Variables
    assert wb.expr("diff(x*y, x)") - diff(x * y, x) == 0
    assert wb.expr("diff(x**2*y**3, y)") - diff(x**2 * y**3, y) == 0


def test_simplify(workbook_setup, var_setup):
    wb = workbook_setup
    x, y = var_setup

    # Simple Simplification
    assert (
        simplify(wb.expr("simplify(x**2 + 2*x + 1)") - simplify(x**2 + 2 * x + 1))
        == 0
    )
    assert (
        simplify(
            wb.expr("simplify((x + 1)**2 - x**2)") - simplify((x + 1) ** 2 - x**2)
        )
        == 0
    )

    # Multiple Variables
    assert (
        simplify(
            wb.expr("simplify(x*y + x**2*y**2)") - simplify(x * y + x**2 * y**2)
        )
        == 0
    )
    assert (
        simplify(
            wb.expr("simplify(x**2*y**3 / (x*y))") - simplify(x**2 * y**3 / (x * y))
        )
        == 0
    )


def test_expand(workbook_setup, var_setup):
    wb = workbook_setup
    x, y = var_setup

    # Simple expansion
    assert simplify(wb.expr("expand((x + y)**2)") - expand((x + y) ** 2)) == 0
    assert simplify(wb.expr("expand((x + y)**3)") - expand((x + y) ** 3)) == 0

    # Multiple Variables
    assert simplify(wb.expr("expand(x*(y + 1))") - expand(x * (y + 1))) == 0
    assert simplify(wb.expr("expand((x + y)*(x - y))") - expand((x + y) * (x - y))) == 0


def test_limit(workbook_setup, var_setup):
    wb = workbook_setup
    x, _ = var_setup

    # Simple Limits
    assert (
        wb.expr("lim((x**2 - 1)/(x - 1), x, 1)") - limit((x**2 - 1) / (x - 1), x, 1)
        == 0
    )
    assert wb.expr("lim(1/x, x, 0)") == limit(1 / x, x, 0)

def test_diag(workbook_setup, var_setup):
    wb = workbook_setup
    x, y = var_setup

    assert SymbolArray([[x]]) == wb.expr("diag(x)")  # 1D
    assert SymbolArray([[x**2, 0], [0, y**x]]) == wb.expr("diag(x**2, y**x)")  # 2D
    assert SymbolArray([[x**2, 0, 0], [0, y**x, 0], [0, 0, 1]]) == wb.expr(
        "diag(x**2, y**x, 1)"
    )  # 3D
    assert SymbolArray(
        [[x**2, 0, 0, 0], [0, y**x, 0, 0], [0, 0, 1, 0], [0, 0, 0, 10 * x]]
    ) == wb.expr(
        "diag(x**2, y**x, 1, 10*x)"
    )  # 4D

def test_cache(workbook_setup, var_setup, func_setup):
    wb = workbook_setup
    x, y = var_setup
    f, P, Q = func_setup

    a = diff(f(x), x) + P(x) * f(x) - Q(x)

    assert (
        wb.expr(
            """
                        a := diff(f(x), x) + P(x)*f(x) - Q(x)
                        dsolve(a, f(x)) 
            """
        )
        == dsolve(a, f(x))
    )

    assert (
        wb.expr(
            """
                        a := diff(f(x), x) + P(x)*f(x) - Q(x)
                        b := f(x)
                        diag(a, b)
            """
        )
        == SymbolArray([[a, 0], [0, f(x)]])
    )
