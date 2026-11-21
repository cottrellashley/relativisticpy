"""Equations between tensors: of any shape, and numerical ones to rounding error."""

import numpy as np
import sympy
from sympy import Array, symbols

from relativisticpy.tensors import Index, Indices, Space, numerical
from relativisticpy.tensors.equation import TensorEquation

x, y = symbols("x y")


def test_an_equation_of_any_shape():
    data = Array([[x, 0], [0, y], [1, 0], [0, 1]])  # four days of two assets
    other = Array([[x, 0], [0, y], [1, 0], [0, 2]])
    equation = TensorEquation(data, other)
    assert not equation
    assert equation.equations() == [sympy.Eq(1, 2, evaluate=False)]
    # A zero side has the other's shape, whatever shape it was given with.
    assert len(TensorEquation(data, Array.zeros(4, 4)).equations()) == 4
    assert len(TensorEquation(data, 0).equations()) == 4


def test_components_are_named_by_their_spaces():
    days = Space("days", 2, basis=("mon", "tue"))
    assets = Space("assets", 2, basis=("AAPL", "MSFT"))
    indices = Indices([Index("t"), Index("a")])
    equation = TensorEquation(np.zeros((2, 2)), np.eye(2), indices, spaces=(days, assets))
    assert str(equation).splitlines()[0].startswith("mon,AAPL:")
    assert "_{mon AAPL}" in equation.latex()


def test_numbers_agree_to_rounding_error_in_what_they_were_computed_from():
    masses = np.array([9.109e-31, 1.673e-27])
    assert not TensorEquation(masses, 2 * masses)
    assert not TensorEquation(masses, 0)
    residual = np.array([8.6e-12, -3e-12])  # E[PP] - m m - C, of numbers near 10^4
    assert not TensorEquation(residual, 0)
    with numerical.operands(1e4):
        decided = TensorEquation(residual, 0)
    assert decided
    assert not decided.components()
    assert TensorEquation(residual, 0, scale=1e4)
