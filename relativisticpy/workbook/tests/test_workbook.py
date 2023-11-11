import pytest
import sympy as sp
from relativisticpy.workbook.workbook import Workbook

def test_mul_numbers():
    mul_workbook = Workbook()
    assert int(mul_workbook.expr('2 * 3')) == 6
    assert int(mul_workbook.expr('-2 * 3')) == -6
    assert int(mul_workbook.expr('-2 * -3')) == 6

def test_mul_symbols():
    mul_symbol_workbook = Workbook()
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    assert mul_symbol_workbook.expr('2 * x') == 2 * x

def test_sub_symbols():
    sub_symbol_workbook = Workbook()
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    assert sp.simplify(sub_symbol_workbook.expr('x - 1') - (x - 1)) == 0

def test_integrate():
    integrate_workbook = Workbook()
    x = sp.Symbol('x')
    assert sp.simplify(integrate_workbook.expr('integrate(x**2, x)') - sp.integrate(x**2, x)) == 0

def test_series():
    w = Workbook()
    x = sp.Symbol('x')
    assert sp.simplify(w.expr('sin(x)') - sp.sin(x)) == 0

def test_mul_symbols():
    mul_symbol_workbook = Workbook()
    x = sp.Symbol('x')
    y = sp.Symbol('y')

    # Symbol and Number
    assert sp.simplify(mul_symbol_workbook.expr('2 * x') - 2 * x) == 0
    assert sp.simplify(mul_symbol_workbook.expr('-2 * x') + 2 * x) == 0
    assert sp.simplify(mul_symbol_workbook.expr('-2.5 * x') + 2.5 * x) == 0

    # Symbol and Symbol
    assert sp.simplify(mul_symbol_workbook.expr('y * x') - y * x) == 0
    assert sp.simplify(mul_symbol_workbook.expr('-y * x') + y * x) == 0
    assert sp.simplify(mul_symbol_workbook.expr('-2.5 * x * y') + 2.5 * x * y) == 0

def test_sub_numbers():
    sub_workbook = Workbook()
    # Integer Subtraction
    assert sub_workbook.expr('2 - 1') == 1
    assert sub_workbook.expr('-2 - 1') == -3
    assert sub_workbook.expr('-2 - -1') == -1

    # Float Subtraction
    assert sub_workbook.expr('2.5 - 1') == 1.5
    assert sub_workbook.expr('-2.5 - 1') == -3.5
    assert sub_workbook.expr('-2.5 - -1.5') == -1.0

def test_sub_symbols():
    sub_symbol_workbook = Workbook()
    x = sp.Symbol('x')
    y = sp.Symbol('y')

    # Symbol and Number
    assert sp.simplify(sub_symbol_workbook.expr('x - 1') - (x - 1)) == 0
    assert sp.simplify(sub_symbol_workbook.expr('1 - x') - (1 - x)) == 0
    assert sp.simplify(sub_symbol_workbook.expr('-1.5 - x') - (-1.5 - x)) == 0

    # Symbol and Symbol
    assert sp.simplify(sub_symbol_workbook.expr('y - x') - (y - x)) == 0
    assert sp.simplify(sub_symbol_workbook.expr('-y - x') - (-y - x)) == 0
    assert sp.simplify(sub_symbol_workbook.expr('-1.5 * x - y') - (-1.5 * x - y)) == 0

def test_solve():
    solve_workbook = Workbook()
    x = sp.Symbol('x')
    y = sp.Symbol('y')

    # Simple Linear Equations
    # FAILING => need a standard way/function to call to get back same outputs
    assert solve_workbook.expr('solve(x + 2*y - 3, x)') == sp.solve(x + 2*y - 3, x) 
    assert solve_workbook.expr('solve([x + y - 2, x - y + 1], [x, y])') == sp.solve([x + y - 2, x - y + 1], [x, y])

def test_diff():
    diff_workbook = Workbook()
    x = sp.Symbol('x')
    y = sp.Symbol('y')

    # Simple Differentiation
    assert sp.simplify(diff_workbook.expr('diff(x**2, x)') - sp.diff(x**2, x)) == 0
    assert sp.simplify(diff_workbook.expr('diff(x**3 + 2*x**2 - 1, x)') - sp.diff(x**3 + 2*x**2 - 1, x)) == 0

    # Multiple Variables
    assert sp.simplify(diff_workbook.expr('diff(x*y, x)') - sp.diff(x*y, x)) == 0
    assert sp.simplify(diff_workbook.expr('diff(x**2*y**3, y)') - sp.diff(x**2*y**3, y)) == 0

def test_simplify():
    simplify_workbook = Workbook()
    x = sp.Symbol('x')
    y = sp.Symbol('y')

    # Simple Simplification
    assert sp.simplify(simplify_workbook.expr('simplify(x**2 + 2*x + 1)') - sp.simplify(x**2 + 2*x + 1)) == 0
    assert sp.simplify(simplify_workbook.expr('simplify((x + 1)**2 - x**2)') - sp.simplify((x + 1)**2 - x**2)) == 0

    # Multiple Variables
    assert sp.simplify(simplify_workbook.expr('simplify(x*y + x**2*y**2)') - sp.simplify(x*y + x**2*y**2)) == 0
    assert sp.simplify(simplify_workbook.expr('simplify(x**2*y**3 / (x*y))') - sp.simplify(x**2*y**3 / (x*y))) == 0

def test_expand():
    expand_workbook = Workbook()
    x = sp.Symbol('x')
    y = sp.Symbol('y')

    # Simple expansion
    assert sp.simplify(expand_workbook.expr('expand((x + y)**2)') - sp.expand((x + y)**2)) == 0
    assert sp.simplify(expand_workbook.expr('expand((x + y)**3)') - sp.expand((x + y)**3)) == 0

    # Multiple Variables
    assert sp.simplify(expand_workbook.expr('expand(x*(y + 1))') - sp.expand(x*(y + 1))) == 0
    assert sp.simplify(expand_workbook.expr('expand((x + y)*(x - y))') - sp.expand((x + y)*(x - y))) == 0

def test_limit():
    limit_workbook = Workbook()
    x = sp.Symbol('x')
    y = sp.Symbol('y')

    # Simple Limits
    assert sp.simplify(limit_workbook.expr('limit((x**2 - 1)/(x - 1), x, 1)') - sp.limit((x**2 - 1)/(x - 1), x, 1)) == 0
    assert limit_workbook.expr('limit(1/x, x, 0)') == sp.limit(1/x, x, 0)
