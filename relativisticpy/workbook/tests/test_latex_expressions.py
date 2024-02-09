import pytest
import sympy as sp
from relativisticpy.workbook.workbook import Workbook

# Do not import from sympy - only import from the SAME interface which Workbook points to implements everything.
from relativisticpy.symengine import (
    Symbol,
    Function,
    sin,
    oo,
    integrate,
    Sum,
    Product,
    solve,
    diff,
    expand,
    limit,
    simplify,
    dsolve,
    SymbolArray,
)

# Workbook.expr() => is used for one line solver function - like a "unit test" for workbook module.


@pytest.fixture
def vars_x_y_z_t_r_theta_phi():
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    t = Symbol("t")
    r = Symbol("r")
    theta = Symbol("theta")
    phi = Symbol("phi")
    return x, y, z, t, r, theta, phi

@pytest.fixture
def func_f_g_h():
    f = Function("f")
    g = Function("g")
    h = Function("h")
    return f, g, h

def test_summation_function(vars_x_y_z_t_r_theta_phi):
    wb = Workbook()
    x, y, z, t, r, theta, phi = vars_x_y_z_t_r_theta_phi

    assert wb.expr("\\sum_{x=0}^{10} x")[0].value == Sum(x, (x, 0, 10))
    assert wb.expr("\\sum_{x=1}^{oo} 1/x**2")[0].value == Sum(1/x**2, (x, 1, oo))

def test_product_function(vars_x_y_z_t_r_theta_phi):
    wb = Workbook()
    x, y, z, t, r, theta, phi = vars_x_y_z_t_r_theta_phi

    assert wb.expr("\\prod_{x=0}^{10} x")[0].value == Product(x, (x, 0, 10))
    assert wb.expr("\\prod_{x=1}^{oo} 1/x**2")[0].value == Product(1/x**2, (x, 1, oo))

def test_matrix_begin_function(vars_x_y_z_t_r_theta_phi, func_f_g_h):
    wb = Workbook()
    x, y, z, t, r, theta, phi = vars_x_y_z_t_r_theta_phi
    f, g, h = func_f_g_h
    assert wb.expr("""
                        C = \\begin{matrix} 
                        1 &  2  & 3 \\\\
                        0 &  0  & x \\\\
                        y & f(x) & 0 \\\\
                        \\end{matrix} 
                        \\newline
                        C  
                        """)[0].value == SymbolArray([[1, 2, 3], [0, 0, x], [y, f(x), 0]])
