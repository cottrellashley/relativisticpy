import pytest
import sympy as sp
from relativisticpy.workbook.workbook import Workbook

# Do not import from sympy - only import from the SAME interface which Workbook points to implements everything.
from relativisticpy.symengine import (
    Symbol,
    Function,
    sin,
    I,
    oo,
    integrate,
    Sum,
    Product,
    solve,
    factor,
    diff,
    expand,
    limit,
    simplify,
    apart,
    cancel,
    collect,
    sqrt,
    root,
    dsolve,
    SymbolArray,
)

@pytest.fixture
def vars_x_y_z_t_r_theta_phi_tau():
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    t = Symbol("t")
    r = Symbol("r")
    theta = Symbol("theta")
    phi = Symbol("phi")
    tau = Symbol("tau")
    return x, y, z, t, r, theta, phi, tau


@pytest.fixture
def func_f_g_h():
    f = Function("f")
    g = Function("g")
    h = Function("h")
    return f, g, h

def test_imaginary_numbers(vars_x_y_z_t_r_theta_phi_tau):
    wb = Workbook()
    x, y, z, t, r, theta, phi, tau = vars_x_y_z_t_r_theta_phi_tau

    # Test that 'i' is recognized as the imaginary unit
    assert wb.expr("i**2") == -1

    # Test that 'Re' and 'Im' functions work correctly
    assert wb.expr("Re(1 + i)") == 1
    assert wb.expr("Im(1 + i)") == 1

    # Test that equations involving imaginary numbers can be solved
    res = wb.expr(
    """
        x, y := symbols('x y')
        solve([x - i*y, Im(y), Im(x)])
    """
    )
    assert res == [(0, 0)]

def test_imaginary_numbers_with_sympy_functions(vars_x_y_z_t_r_theta_phi_tau):
    wb = Workbook()
    x, y, z, t, r, theta, phi, tau = vars_x_y_z_t_r_theta_phi_tau

    # Test the 'expand' function with imaginary numbers
    assert wb.expr("expand((x + i*y)**2)") == expand((x + I*y)**2)

    # Test the 'simplify' function with imaginary numbers
    assert wb.expr("simplify((x + i*y)**2 - x**2 - 2*i*x*y - y**2)") == simplify((x + I*y)**2 - x**2 - 2*I*x*y - y**2)

    # Test the 'factor' function with imaginary numbers
    assert wb.expr("factor(x**2 - 2*i*x*y + y**2)") == factor(x**2 - 2*I*x*y + y**2)

    # # Test the 'collect' function with imaginary numbers
    # assert wb.expr("collect(x**2 + i*x*y + i*x*y + y**2, x)") == collect(x**2 + I*x*y + I*x*y + y**2, x)

    # # Test the 'apart' function with imaginary numbers
    # assert wb.expr("apart((x**2 + 2*i*x + 1)/(x + i), x)") == apart((x**2 + 2*I*x + 1)/(x + I), x)

    # # Test the 'cancel' function with imaginary numbers
    # assert wb.expr("cancel((x**2 + 2*i*x + 1)/(x + i))") == cancel((x**2 + 2*I*x + 1)/(x + I))

    # Test the 'solve' function with imaginary numbers
    assert wb.expr("solve(x**2 + 2*i*x + 1, x)") == solve(x**2 + 2*I*x + 1, x)
