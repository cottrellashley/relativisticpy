import pytest
import sympy as sp
from relativisticpy.workbook.workbook import Workbook

# Do not import from sympy - only import from the SAME interface which Workbook points to implements everything.


def test_function_call():
    wb = Workbook()

    res = wb.expr(
        """
        f() := x**2
        f()
    """
    )
    assert str(res) == 'x**2'


def test_function_arg_calls():
    wb = Workbook()

    res = wb.expr(
        """
        h(x, y) := x**2 - y*8 + y*x
        h(v**2, 7)
    """
    )
    assert str(res) == 'v**4 + 7*v**2 - 56'


def test_polynomial_function():
    wb = Workbook()

    res = wb.expr(
        """
        f(x) := 3*x**3 - 2*x**2 + x - 5
        f(2)
    """
    )
    assert str(res) == '13'


def test_logarithm_and_exponential_function():
    wb = Workbook()

    res = wb.expr(
        """
            g(x) := e**x - ln(x)
            g(1) + exp(ln(x) + 10)
    """
    )
    assert str(res) == 'x*exp(10) + E'


def test_logarithm_and_exponential_function_state_bug():
    wb = Workbook()

    res = wb.expr(
        """
            g(x) := x
            g(1) + exp(ln(x) + 10)
    """
    )
    assert str(res) == 'x*exp(10) + 1'


def test_trigonometric_function():
    wb = Workbook()

    res = wb.expr(
        """
        t(x) := sin(x)**2 + cos(x)**2
        t(pi/4)
    """
    )
    assert str(res) == '1'


def test_absolute_function():
    wb = Workbook()

    res = wb.expr(
        """
            abs_func(x) := |x - 3|
            abs_func_two(x) := |-x**2|
            |-x + abs_func(-1) + abs_func_two(10)|
    """
    )
    assert str(res) == 'Abs(x - 104)'


def test_piecewise_function():
    wb = Workbook()

    res = wb.expr(
        """
            piecewise(x) := { x**2, x > 0; -x, x <= 0 }
            piecewise(2) * piecewise(-2)
    """
    )
    print(res)
    assert str(res) == 'v**4 + 7*v**2 - 56'


def test_composition_of_functions():
    wb = Workbook()

    res = wb.expr(
        """
            f(x) := x**2 + 2*x - 1
            g(x) := 2*sin(x)
            h(x) := f(g(x))
            h(pi/2)
    """
    )
    assert str(res) == '7'


def test_differential_equation_function():
    wb = Workbook()

    res = wb.expr(
        """
            f : (x) -> { 
                diff(y(x), x, 2) + 2*diff(y(x), x) + y(x) = 10 
            }
            dsolve(f(x), y(x))
    """
    )
    assert str(res) == 'Eq(y(x), (C1 + C2*x)*exp(-x) + 10)'


def test_equation_subs_function():
    wb = Workbook()

    res = wb.expr(
        """
            y(x) = 10 * x |_{x = 10}
    """
    )
    assert str(res) == 'Eq(y(10), 100)'


def test_equation_function():
    wb = Workbook()

    res = wb.expr(
        """
            y(x) = 10
    """
    )
    assert str(res) == 'Eq(y(x), 10)'


def test_rational_functions():
    wb = Workbook()

    res = wb.expr(
        """
            r(x) := (x**2 - 4) / (x - 2)
            r(3)
    """
    )
    assert str(res) == '5.0'


def test_factorial_functions():

    wb = Workbook()
    res = wb.expr(
        """
            factorial_func(n) := n!
            comb(n, k) := n! / (k! * (n - k)!)

            comb(factorial_func(3), 4)
    """
    )
    assert str(res) == '15'
