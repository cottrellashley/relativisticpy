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
    sqrt,
    root,
    dsolve,
    SymbolArray,
)


def test_single_argument_function():
    wb = Workbook()
    res = wb.expr(
    """
        f : (x) -> { 
            x^2
        }

        f(3)
    """
    )
    assert str(res) == '9'

def test_multiple_argument_function():
    wb = Workbook()
    res = wb.expr(
    """
        f : (x, y) -> { 
            x*y
        }

        f(3, 4)
    """
    )
    assert str(res) == '12'

def test_nested_function_calls():
    wb = Workbook()
    res = wb.expr(
    """
        f : (x) -> { 
            g : (y) -> {
                y^2
            }

            x + g(x)
        }

        f(3)
    """
    )
    assert str(res) == '12'

def test_function_with_no_arguments():
    wb = Workbook()
    res = wb.expr(
    """
        f : () -> { 
            42
        }

        f()
    """
    )
    assert str(res) == '42'

def test_nested_function_with_variable_definition():
    wb = Workbook()
    res = wb.expr(
    """
        f : (x) -> { 
            m := x^2 
            s := m^2

            g : (y) -> {
                n := y^2
                n^2
            }

            s + g(2)
        }

        f(2)
    """
    )
    assert str(res) == '32'

def test_variable_scope_within_function():
    wb = Workbook()
    res = wb.expr(
    """
        f : (x) -> { 
            y := x + 1
            y
        }

        f(2) + y
    """
    )
    assert str(res) == 'y + 3', "Variable y defined inside function should not be accessible outside"

def test_variable_scope_within_nested_function():
    wb = Workbook()
    res = wb.expr(
    """
        f : (x) -> { 
            y := x + 1

            g : (z) -> {
                z + y
            }

            g(2)
        }

        f(2) + y
    """
    )
    assert str(res) == 'y + 5', "Variable y defined inside outer function should not be accessible outside"

def test_function_scope():
    wb = Workbook()
    res = wb.expr(
    """
        f : (x) -> { 
            g : (z) -> {
                z + x
            }

            g(2)
        }

        f(2) + g(2)
    """
    )
    assert str(res) == 'g(2) + 4', "Function g defined inside function f should not be accessible outside"