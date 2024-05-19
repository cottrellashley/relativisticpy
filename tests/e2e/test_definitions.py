import pytest
import sympy as sp
from relativisticpy.workbook.workbook import Workbook

# Do not import from sympy - only import from the SAME interface which Workbook points to implements everything.
from relativisticpy.symengine import (
    symbols,
    integrate,
    factor,
    diff,
    expand,
    simplify,
)

def test_constant_assumptions():
    wb = Workbook()

    # Test that constants can be defined with assumptions and used in calculations
    wb.expr("Constants := [C, G]")
    assert wb.expr("C + G") == symbols('C', constant=True) + symbols('G', constant=True)

    # Test that integer variables can be defined with assumptions and used in calculations
    wb.expr("Integers := [n, m]")
    assert wb.expr("n + m") == symbols('n', integer=True) + symbols('m', integer=True)

    # Test that real variables can be defined with assumptions and used in calculations
    wb.expr("Reals := [x, y]")
    assert wb.expr("x + y") == symbols('x', real=True) + symbols('y', real=True)

    # Test that complex variables can be defined with assumptions and used in calculations
    wb.expr("Complexes := [z, w]")
    assert wb.expr("z + w") == symbols('z', complex=True) + symbols('w', complex=True)

def test_involved_calculations():
    wb = Workbook()

    # Define variables
    wb.expr("Coordinates := [t, r, theta, phi]")
    wb.expr("Constants := [C, G]")
    wb.expr("Integers := [n, m]")
    wb.expr("Reals := [x, y]")
    wb.expr("Complexes := [z, w]")

    # Test calculations involving various operations and functions
    assert wb.expr("t*r + theta*phi") == symbols('t')*symbols('r') + symbols('theta')*symbols('phi')
    assert wb.expr("C*G + n*m") == symbols('C', constant=True)*symbols('G', constant=True) + symbols('n', integer=True)*symbols('m', integer=True)
    assert wb.expr("x**y + z*w") == symbols('x', real=True)**symbols('y', real=True) + symbols('z', complex=True)*symbols('w', complex=True)

    # Test calculations involving sympy functions
    assert wb.expr("expand((t + r)**2)") == expand((symbols('t') + symbols('r'))**2)
    assert wb.expr("simplify((C + G)**2 - C**2 - 2*C*G - G**2)") == simplify((symbols('C', constant=True) + symbols('G', constant=True))**2 - symbols('C', constant=True)**2 - 2*symbols('C', constant=True)*symbols('G', constant=True) - symbols('G', constant=True)**2)
    assert wb.expr("factor(n**2 - 2*n*m + m**2)") == factor(symbols('n', integer=True)**2 - 2*symbols('n', integer=True)*symbols('m', integer=True) + symbols('m', integer=True)**2)