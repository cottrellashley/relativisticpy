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
    root,
    dsolve,
    SymbolArray,
)

# Workbook.expr() => is used for one line solver function - like a "unit test" for workbook module.

## Derivatives on tensors
# Euler-Lagrange equation:
# \frac{\partial L}{\partial x^{\alpha}} - \frac{d}{d\tau}\frac{\partial L}{\partial \frac{\partial x^{\alpha}}{\partial \tau}}
#
#
# \frac{d}{d\tau} <================================================= WE'RE looking at a new object here, not already defined within tensor objects. I.e. an operator which has multiplication defined as taking the derivative.
# Geodesic Equation
# x^{\mu} := [t(\tau), r(\tau), \theta(\tau), \phi(\tau)]
# Geo^{\mu} := \frac{d^2{x^{\mu}}}{d{\tau^2}} + \Gamma^{\mu}_{\alpha \beta} \frac{d{x^{\alpha}}}{d\tau}\frac{d{x^{\beta}}}{d{\tau}}
#
# Euler-Lagrage of Geodesic
# L := \sqrt{-g_{\mu \nu}\frac{d{x^{\mu}}}{d{\tau}}\frac{d {x^{\nu}}}{d{\tau}}}
#
# L := (1/2)*x^2 -

# Functions
# f : (x) -> {
#   d := x**2
#   }

# Substitution
# d{f(x)}/d{x} |_{x = 2}

# subs(f(x), x, 1)

# \pdv{f}{x}
# \pdv[n]{f}{x}		
# \pdv{x}(\frac{x}{x^{2} + 1})

# f(x) |_{x = 2}
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


def test_summation_command(vars_x_y_z_t_r_theta_phi_tau):
    wb = Workbook()
    x, y, z, t, r, theta, phi, tau = vars_x_y_z_t_r_theta_phi_tau

    assert wb.expr("\\sum_{x=0}^{10} x") == Sum(x, (x, 0, 10))
    assert wb.expr("\\sum_{x=1}^{oo} 1/x**2") == Sum(1 / x**2, (x, 1, oo))


def test_product_command(vars_x_y_z_t_r_theta_phi_tau):
    wb = Workbook()
    x, y, z, t, r, theta, phi, tau = vars_x_y_z_t_r_theta_phi_tau

    assert wb.expr("\\prod_{x=0}^{10} x") == Product(x, (x, 0, 10))
    assert wb.expr("\\prod_{x=1}^{oo} 1/x**2") == Product(1 / x**2, (x, 1, oo))


def test_limit_command(vars_x_y_z_t_r_theta_phi_tau):
    wb = Workbook()
    x, y, z, t, r, theta, phi, tau = vars_x_y_z_t_r_theta_phi_tau

    assert int(wb.expr("\\lim_{x -> 0} sin(x)/x")) == 1
    assert float(wb.expr("\\lim_{ x -> oo } 1/x**2")) == 0
    assert float(wb.expr("\\lim_{ x \\to 10 } 1/x**2")) == 1/100

def test_substitution_command(vars_x_y_z_t_r_theta_phi_tau):
    wb = Workbook()
    x, y, z, t, r, theta, phi, tau = vars_x_y_z_t_r_theta_phi_tau

    assert float(wb.expr(" \\frac{x}{10} * 1/x |_{x = oo}")) == 1/10
    assert float(wb.expr(" \\frac{x}{10} * 1/x |_{x = 10}")) == 1/10
    assert float(wb.expr(" \\frac{ diff(f(x), x) }{10} * 1/diff(f(x), x) |_{ diff(f(x), x) = 10}")) == 1/10

def test_frac_command(vars_x_y_z_t_r_theta_phi_tau):
    wb = Workbook()
    x, y, z, t, r, theta, phi, tau = vars_x_y_z_t_r_theta_phi_tau

    assert float(wb.expr("\\frac{x}{10} * 1/x")) == 1/10
    assert int(wb.expr("\\frac{x}{oo} * x**2")) == 0
    assert float(wb.expr("\\frac{1}{2} * 2/1")) == 1

def test_sqrt_command(vars_x_y_z_t_r_theta_phi_tau):
    wb = Workbook()
    x, y, z, t, r, theta, phi, tau = vars_x_y_z_t_r_theta_phi_tau

    assert wb.expr("\sqrt{ x }").equals(root( x , 2))
    assert wb.expr("\sqrt{ x**2 }").equals(root( x**2 , 2))
    assert wb.expr("\sqrt{ 10.23 * y  }").equals(root( 10.23 * y , 2))
    assert wb.expr("\sqrt{ sin(10) }").equals(root( sin(10) , 2))
    assert wb.expr("\sqrt{ sin(x**2) }").equals(root( sin(x**2) , 2))
    assert wb.expr("\sqrt{ sin(x)**2 }").equals(root( sin(x)**2 , 2))

def test_newline_command(vars_x_y_z_t_r_theta_phi_tau):
    wb = Workbook()
    x, y, z, t, r, theta, phi, tau = vars_x_y_z_t_r_theta_phi_tau

    assert wb.expr(" A := \\prod_{x=0}^{10} x \\newline A") == Product(x, (x, 0, 10))
    assert wb.expr(" A := \\prod_{x=0}^{10} x \\newline A") == wb.expr(""" A := \\prod_{x=0}^{10} x 
                                                                          A""")

def test_all_derivative_commands(vars_x_y_z_t_r_theta_phi_tau, func_f_g_h):
    wb = Workbook()
    x, y, z, t, r, theta, phi, tau = vars_x_y_z_t_r_theta_phi_tau
    f, g, h = func_f_g_h

    # using diff built in function
    assert wb.expr("diff(f(x), x)") == diff(f(x), x)
    assert wb.expr("diff(f(x), x, 4)") == diff(f(x), x, 4)
    assert wb.expr("diff(f(x) ** x, x, x)") == diff(f(x)**x, x, x)

    # New - to be implemented
    assert wb.expr(" f'(x) ") == diff(f(x), x)
    assert wb.expr(" f''(x) ") == diff(f(x), x, 2)
    assert wb.expr(" f'''(x) ") == diff(f(x), x, 3)

    # New - to be implemented
    assert wb.expr(" f'(x) ") == diff(f(x), x)
    assert wb.expr(" f''(x) ") == diff(f(x), x, 2)
    assert wb.expr(" f'''(x) ") == diff(f(x), x, 3)
    assert wb.expr(" f'''''''''''''''''(x) ")  == diff(f(x), x, 17)
    # using partial 
    assert wb.expr(" \partial{f(x)} / \partial{x} ") == diff(f(x), x)
    assert wb.expr(" \\frac{\partial^2{x^2}{\partial{\\tau}^2} ") == diff( x**2 , tau, 2)
    assert wb.expr(" \\frac{\partial^2{{\sqrt{ - x**3 + y**2 - z }}{\partial{x}^2} ") == diff( sqrt( - x**3 + y**2 - z ), x, 2)

    # using d operator and a division operator /
    assert wb.expr(" d{f(x)}/d{x} ") == diff(f(x), x)
    assert wb.expr(" (d{f(x)})/(d{x}) ") == diff(f(x), x)
    assert wb.expr(" d^3{f(x) + x**6 - y*7}/d{x}^3 ") == diff(f(x) + x**6 - 7*x, x, 3)

    # using d operator and a \frac operator
    assert wb.expr("\\frac{d^2{x^2}{d{\\tau}^2} ") == diff(x**2, tau, 2) 
    assert wb.expr("\\frac{ d^2{-(1 - t / (x**2*r(tau)))} }{d{tau}^2} ") == diff( -(1 - t / (x**2*r(tau))) , tau, 2)
    assert wb.expr("\\frac{ d^2{\sqrt{ - x**3 + y**2 - z }} }{d{x}^2} ") == diff( sqrt( - x**3 + y**2 - z ), x, 2)

    # using \pdv operator which is a physics package in latex
    assert wb.expr(" \pdv{f(x)}{x} ") == diff(f(x), x)
    assert wb.expr(" \pdv[3]{f(x) + x**6 - y*7}{x}	") == diff(f(x) + x**6 - 7*x, x, 3)
    assert wb.expr(" \pdv{x}( \\frac{ \partial^2{ \sqrt{ - x^3 + y^2 - z } } }{ \partial{y}^2 }  ) ") == diff( diff( sqrt( - x**3 + y**2 - z ), y, 2) , x)

def test_undecided_derivative_syntax(vars_x_y_z_t_r_theta_phi_tau, func_f_g_h):
    wb = Workbook()
    x, y, z, t, r, theta, phi, tau = vars_x_y_z_t_r_theta_phi_tau
    f, g, h = func_f_g_h

    # using diff built in function
    assert wb.expr(" d f(x)/d x ") == diff(f(x), x)


def test_matrix_building(vars_x_y_z_t_r_theta_phi_tau, func_f_g_h):
    wb = Workbook()
    x, y, z, t, r, theta, phi, tau = vars_x_y_z_t_r_theta_phi_tau
    f, g, h = func_f_g_h

    int_matrix = (
        """
                        \\begin{matrix} 
                        1 &  2  & 3 \\\\
                        4 &  5  & 6 \\\\
                        7 &  8  & 9 \\\\
                        \\end{matrix} 
        """,
        SymbolArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    )

    float_matrix = (
        """
                        \\begin{matrix} 
                        1.123 &  2.12  & 3.87 \\\\
                        4.123 &  5.32  & 6.12 \\\\
                        7.123 &  8.12  & 9.12 \\\\
                        \\end{matrix} 
        """,
        SymbolArray([[1.123, 2.12, 3.87], [4.123, 5.32, 6.12], [7.123, 8.12, 9.12]]),
    )

    symbol_matrix = (
        """
                        \\begin{matrix} 
                        x &  2  & 3 \\\\
                        4 &  y  & 6 \\\\
                        7 &  8  & z \\\\
                        \\end{matrix} 
        """,
        SymbolArray([[x, 2, 3], [4, y, 6], [7, 8, z]]),
    )

    function_matrix = (
        """
                        \\begin{matrix} 
                        f(x) &  2  & 3 \\\\
                        4 &  h(y, x)  & 6 \\\\
                        7 &  8  & g(t, theta, phi) \\\\
                        \\end{matrix} 
        """,
        SymbolArray([[f(x), 2, 3], [4, h(y, x), 6], [7, 8, g(t, theta, phi)]]),
    )

    line_matrix = (
        """
                        \\begin{matrix} 
                        f(x)  &  2  & 3
                        \\end{matrix} 
        """,
        SymbolArray([[f(x), 2, 3]]),
    )

    line2_matrix = (
        """
                        \\begin{matrix} 
                        f(x)  &  2  & 3 \\\\
                        \\end{matrix} 
         """,
        SymbolArray([[f(x), 2, 3]]),
    )

    two_line_matrix = (
        """
                        \\begin{matrix} 
                        f(x)  &  2  & 3 \\\\
                        f(x)  &  2  & 3 
                        \\end{matrix} 
        """,
        SymbolArray([[f(x), 2, 3], [f(x), 0, x]]),
    )

    assert wb.expr(int_matrix[0] == int_matrix[1])
    assert wb.expr(float_matrix[0] == float_matrix[1])
    assert wb.expr(symbol_matrix[0] == symbol_matrix[1])
    assert wb.expr(function_matrix[0] == function_matrix[1])
    assert wb.expr(line_matrix[0] == line_matrix[1])
    assert wb.expr(line2_matrix[0] == line2_matrix[1])
    assert wb.expr(two_line_matrix[0] == two_line_matrix[1])
