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

# f(x) |_{x = 2}
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

    assert wb.expr("\\sum_{x=0}^{10} x") == Sum(x, (x, 0, 10))
    assert wb.expr("\\sum_{x=1}^{oo} 1/x**2") == Sum(1/x**2, (x, 1, oo))

def test_product_function(vars_x_y_z_t_r_theta_phi):
    wb = Workbook()
    x, y, z, t, r, theta, phi = vars_x_y_z_t_r_theta_phi

    assert wb.expr("\\prod_{x=0}^{10} x") == Product(x, (x, 0, 10))
    assert wb.expr("\\prod_{x=1}^{oo} 1/x**2") == Product(1/x**2, (x, 1, oo))

def test_tensor_generation_from_latex_derivative(vars_x_y_z_t_r_theta_phi, func_f_g_h):
    wb = Workbook()
    x, y, z, t, r, theta, phi = vars_x_y_z_t_r_theta_phi
    f, g, h = func_f_g_h
    assert wb.expr("""
                        C \equiv \\begin{matrix} 
                        1 &  2  & 3 \\\\
                        0 &  0  & x \\\\
                        y & f(x) & 0 \\\\
                        \\end{matrix} 
                        \\newline
                        C 
                        """) == SymbolArray([[1, 2, 3], [0, 0, x], [y, f(x), 0]])


def future():
    Lagrangian = """
                    Coordinates := [t(tau), r(tau), theta(tau), phi(tau)]
                    g_{mu}_{nu} := [[-(1 - (2 * G * M) / (c**2*r(tau))), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (c**2*r(tau))), 0, 0],[0, 0, r(tau)**2, 0],[0, 0, 0, r(tau)**2 * sin(theta(tau)) ** 2]]
                    x^{\\alpha} := [t(tau), r(tau), theta(tau), phi(tau)]

                    Gamma^{a}_{c}_{f} := (1/2)*g^{a}^{b}*(d_{c}*g_{b}_{f} + d_{f}*g_{b}_{c} - d_{b}*g_{c}_{f})

                    T^{mu} :=  \\frac{d^2{x^{mu}}}{d{tau}^2} + Gamma^{mu}_{alpha beta} * \\frac{d{x^{alpha}}}{d{tau}} * \\frac{d{x^{beta}}}{d{tau}}
                    L = ( - g_{mu nu}*\\frac{ d{ x^{\mu} } }{ d{tau} }*\\frac{ d{ x^{nu} } }{ d{tau} } )**(1/2)
                    L
            """
    K_Scalar_From_scratch = """

                    Coordinates := [t, r, theta, phi]

                    g_{mu}_{nu} := [ 
                                    [-A(r),0,0,0], 
                                    [0,B(r),0,0], 
                                    [0,0,r**2,0], 
                                    [0,0,0,r**2*sin(theta)**2]
                                ]

                    # Now we have defined the metric above, we can call any individual component of the Ricci tensor itself (as it is metric dependent)
                    eq0 = Ric_{mu:0}_{nu:0}
                    eq1 = Ric_{mu:1}_{nu:1}
                    eq2 = Ric_{mu:2}_{nu:2}

                    eq5 = (eq0*B(r) + eq1*A(r))*(r*B(r))

                    B = RHS( dsolve(eq5, B(r)) )

                    eq6 = simplify( subs(eq2, B(r), B) )

                    A = RHS( dsolve(eq6, A(r)) )

                    g_{mu}_{nu} := [
                                    [A,0,0,0], 
                                    [0,1/A,0,0], 
                                    [0,0,r**2,0], 
                                    [0,0,0,r**2*sin(theta)**2]
                                ]

                    # Step 5: We prove that C_1 and C_2 equations are in terms of c, G, M by comparing with Newton at large radius

                    a = C^{t}_{r r}
                    solve( a*c**2 + G*M/r**2 ) # This shows us what 

                                
                    g_{mu}_{nu} := [[-(1 - (2 * G * M) / (c**2*r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (c**2*r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]

                    Gamma^{a}_{c}_{f} = (1/2)*g^{a}^{b}*(d_{c}*g_{b}_{f} + d_{f}*g_{b}_{c} - d_{b}*g_{c}_{f})

                    Riemann^{a}_{m}_{b}_{n} = d_{b}*Gamma^{a}_{n}_{m} + Gamma^{a}_{b}_{l}*Gamma^{l}_{n}_{m} - d_{n}*Gamma^{a}_{b}_{m} - Gamma^{a}_{n}_{l}*Gamma^{l}_{b}_{m}

                    Ricci_{m}_{n} = Riemann^{a}_{m}_{a}_{n}

                    TempOne^{a}^{f}^{h}^{i} = g^{i}^{d}*(g^{h}^{c}*(g^{f}^{b}*Riemann^{a}_{b}_{c}_{d}))

                    TempTwo_{a}_{f}_{h}_{i} = g_{a}_{n}*Riemann^{n}_{f}_{h}_{i}

                    S = TempOne^{a}^{f}^{h}^{i}*TempTwo_{a}_{f}_{h}_{i}

                    S

        """
    pass