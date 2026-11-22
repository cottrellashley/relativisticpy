"""Equations between tensors: of any shape, and numerical ones to rounding error; and
whether a claim of any kind holds."""

import time

import numpy as np
import pytest
import sympy
from sympy import Array, Function, symbols

import relativisticpy
from relativisticpy import IndexEquation, Workbook, display, holds
from relativisticpy.errors import RelativisticPyError
from relativisticpy.tensors import Index, Indices, Space, numerical
from relativisticpy.tensors.equation import TensorEquation, UndecidedError

x, y, M, r = symbols("x y M r")


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
    assert str(equation).splitlines()[1] == "  _{t:mon a:AAPL}: 0 ≠ 1"
    assert r"_{\text{mon}\,\text{AAPL}}" in equation.latex()
    assert equation.failures() == {("mon", "AAPL"): (0.0, 1.0), ("tue", "MSFT"): (0.0, 1.0)}


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


class TestHolds:  # API-1, DOC-8
    def test_holds_is_true_false_or_undecided(self):
        assert TensorEquation(Array([x, 0]), Array([x, 0])).holds() is True
        assert TensorEquation(Array([x, 0]), Array([x, 1])).holds() is False
        f = Function("f")  # f(x) and f(x, y): numbers cannot stand for f, nor SymPy decide
        assert TensorEquation(Array([f(x)]), Array([f(x, y)])).holds() is None

    def test_bool_agrees_and_refuses_what_is_not_decided(self):
        assert bool(TensorEquation(Array([1, 2]), Array([1, 2]))) is True
        assert bool(TensorEquation(Array([1, 2]), Array([1, 3]))) is False
        f = Function("f")
        undecided = TensorEquation(Array([f(x)]), Array([f(x, y)]), source="F_{a} = H_{a}")
        with pytest.raises(UndecidedError, match="not decided") as raised:
            bool(undecided)
        assert "holds()" in raised.value.hint
        assert issubclass(UndecidedError, RelativisticPyError)
        assert issubclass(UndecidedError, TypeError)
        assert "not shown equal" in repr(undecided)

    def test_an_unknown_function_is_no_identity(self):
        f = Function("f")
        equation = TensorEquation(Array([sympy.Derivative(f(r), r, 2) + f(r)]), Array([0]))
        assert equation.holds() is False

    def test_the_analysis_counts_components_once_by_symmetry(self):
        sides = Array([[1, x], [x, 2]]), Array([[1, y], [y, 2]])
        analysis = TensorEquation(*sides).analysis()
        assert (analysis.total, analysis.failing) == (3, 1)
        assert analysis.failures[0].members == [(1, (0, 1)), (1, (1, 0))]

    def test_holds_answers_for_every_kind_of_claim(self):
        assert holds(True) is True
        assert holds(sympy.false) is False
        assert holds(sympy.Eq(25.0, 24, evaluate=False)) is False
        assert holds(sympy.Eq(x**2 - 1, (x - 1) * (x + 1))) is True
        assert holds(sympy.Eq(2 * M - r, -r)) is False
        assert holds(sympy.Ne(x, x + 1)) is True
        assert holds(x < 1) is None
        assert holds(sympy.Integer(2) < 3) is True
        assert holds(np.bool_(True)) is True
        with pytest.raises(TypeError, match="not a claim"):
            holds(x + 1)
        assert relativisticpy.holds is holds

    def test_holds_answers_for_the_language_s_equations(self):
        wb = Workbook()
        wb.expr(r"""
            Coordinates := [t, r, \theta, \phi]
            g_{\mu\nu} := Schwarzschild(M)
        """)
        assert holds(wb.expr(r"R_{\mu\nu} = 0")) is True
        assert holds(wb.expr(r"\Gamma^{a}_{bc} = 0")) is False
        notation = Workbook().expr(r"R_{\mu\nu} = 0")
        assert isinstance(notation, IndexEquation)
        assert holds(notation) is False
        assert bool(notation) is False

    def test_failures_are_keyed_by_coordinate(self):  # API-23
        t = sympy.Symbol("t")
        equation = TensorEquation(Array([[0, 0], [0, 0]]), Array([[2 * M / r - 1, 0], [0, 1]]),
                                  coordinates=(t, r))  # fmt: skip
        assert equation.failures() == {("t", "t"): (0, 2 * M / r - 1), ("r", "r"): (0, 1)}
        assert equation.labels(0) == ["t", "r"]


class TestDisplay:  # API-2, API-3
    def test_an_equation_says_whether_it_holds_and_where_not(self):
        t = sympy.Symbol("t")
        equation = TensorEquation(
            Array([[0, 0], [0, 0]]),
            Array([[2 * M / r - 1, 0], [0, r / (r - 2 * M)]]),
            Indices(r"_{\mu\nu}"),
            (t, r),
            source=r"R_{\mu\nu} = g_{\mu\nu}",
        )
        assert repr(equation) == (
            "R_{μν} = g_{μν}: differs in 2 of 3 components\n"
            "  R_{tt} = 0 but g_{tt} = 2M/r − 1\n"
            "  R_{rr} = 0 but g_{rr} = r/(r − 2M)"
        )
        assert display.text(equation).splitlines()[1] == "  R_{tt} = 0 but g_{tt} = 2*M/r - 1"
        assert r"R_{tt} &= 0 \neq g_{tt} = \frac{2 M}{r} - 1" in equation.latex()
        holding = TensorEquation(
            Array([1, 2]), Array([1, 2]), Indices("^{a}"), source="v^{a} = w^{a}"
        )
        assert repr(holding) == "v^a = w^a: holds in all 2 components"
        assert holding._repr_latex_() == "$$ v^{a} = w^{a} \\tag*{holds in all 2 components} $$"

    def test_numbers_are_summarised_in_milliseconds(self):
        data = np.random.default_rng(1).normal(size=(500, 500))
        equation = TensorEquation(data, data.T, Indices("_{ij}"), source="P_{ij} = P_{ji}")
        start = time.perf_counter()
        shown, typeset = repr(equation), equation.latex()
        assert time.perf_counter() - start < 1.0
        lines = shown.splitlines()
        assert lines[0].startswith("P_{ij} = P_{ji}: differs in 249,500 of 250,000 components")
        assert "largest difference 6.62, at P_{150 263}" in lines[0]
        assert lines[1] == "  P_{0 1} = 0.82162 but P_{1 0} = −1.3703"
        assert lines[-1] == "  … and 249,490 more: .failures() lists them all"
        assert typeset.count("\\\\") <= 12
