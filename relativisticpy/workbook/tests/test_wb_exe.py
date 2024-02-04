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
)

# Workbook.expr() => is used for one line solver function - like a "unit test" for workbook module.


@pytest.fixture
def var_setup():
    x = Symbol("x")
    y = Symbol("y")
    return x, y


@pytest.fixture
def calculus_basic_result():
    x = Symbol("x")
    y = Symbol("y")
    r = Symbol("r")
    theta = Symbol("theta")
    f = Function("f")
    g = Function("g")
    result = r**2*f(x)*sin(y)**2 + r*(r**2*f(x)*sin(theta)**2 + diff(f(x), x))*g(x)**2 + diff(f(x), x)
    return result


@pytest.fixture
def workbook_setup():
    return Workbook()

@pytest.mark.skip(reason="TDD =====> Implement TODO: Multi-line-array <======== ")
def test_calculus_basic(workbook_setup, calculus_basic_result):
    wb = workbook_setup
    result = calculus_basic_result
    assert wb.exe("relativisticpy/workbook/tests/gr_test_scripts/calculus_basic.txt")[0] == result