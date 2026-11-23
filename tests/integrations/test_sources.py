"""Where the answers of the examples come from.

An example's answer is known from outside the integration: its source says where, a place
in a book, or the library called directly, as ``"scipy.special.jv(0, 1.0)"``. A source
that is such a call is run here, and gives the answer: the answer is the library's, not
what the language was seen to give. A source may write ``x`` and ``f(x)`` for a symbol
and a function of SymPy's.
"""

from __future__ import annotations

import ast
import builtins
import cmath
import fractions
import math
import statistics
from typing import Any

import mpmath
import numpy
import pytest
import scipy
import scipy.integrate
import scipy.linalg
import scipy.optimize
import scipy.special
import scipy.stats
import sympy

from relativisticpy import integrations
from relativisticpy.integrations._conformance import same

#: The libraries a source calls, by the names it calls them by.
LIBRARIES = {
    module.__name__: module
    for module in (math, cmath, fractions, statistics, numpy, scipy, sympy, mpmath)
}


def call(source: str) -> ast.Expression | None:
    """A source as the Python it is, where it is one expression that calls a library and
    names nothing besides but symbols of one letter; ``None`` for a place in a book, or a
    call with words around it."""
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError:
        return None
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    others = {name for name in names - LIBRARIES.keys() if not hasattr(builtins, name)}
    if not names & LIBRARIES.keys() or any(len(name) > 1 for name in others):
        return None
    return tree


class Letters(dict[str, Any]):
    """The letters of a source: a symbol of SymPy's each, or a function where it is called."""

    def __init__(self, tree: ast.Expression) -> None:
        super().__init__()
        self.called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

    def __missing__(self, name: str) -> Any:
        if name in LIBRARIES or hasattr(builtins, name):
            raise KeyError(name)
        return sympy.Function(name) if name in self.called else sympy.Symbol(name)


def value(tree: ast.Expression) -> Any:
    """What a source gives when it is run: arrays and tuples as lists, to compare."""
    return plain(eval(compile(tree, "<source>", "eval"), dict(LIBRARIES), Letters(tree)))


def plain(found: Any) -> Any:
    if isinstance(found, numpy.ndarray | numpy.generic):
        return found.tolist()
    if isinstance(found, list | tuple):
        return [plain(each) for each in found]
    return found


def an_answer(found: Any) -> bool:
    """Whether what a source gives is a value to compare: a number, text, a list or what
    SymPy and mpmath compute with; not a report, as SciPy's of an optimisation is."""
    kinds = (bool, int, float, complex, str, list, fractions.Fraction, sympy.Basic, mpmath.mpf)
    return isinstance(found, (*kinds, mpmath.mpc))


CALLS = [
    pytest.param(example, tree, id=f"{capability.name}: {example.source}")
    for capability in (*integrations.capabilities(), *integrations.conversions())
    for example in capability.examples
    if (tree := call(example.source)) is not None
]


@pytest.mark.parametrize(("example", "tree"), CALLS)
def test_a_library_called_directly_gives_the_answer(example, tree):
    found = value(tree)
    if not an_answer(found):
        pytest.skip(f"{example.source} gives a {type(found).__name__}, to read the answer from")
    assert same(found, example.answer, max(example.tolerance, 1e-12)), (
        f"{example.source} gives {found}, and the example's answer is {example.answer}"
    )


def test_every_capability_has_an_answer_from_outside():
    # of the examples of 72 capabilities, each had said "standard" (the review of the engines, 29)
    without = [
        capability.name
        for capability in integrations.capabilities()
        if all(
            example.source.strip().lower().startswith("standard") for example in capability.examples
        )
    ]
    assert without == []


def test_what_is_a_call_is_told_from_a_place_in_a_book():
    assert call("DLMF 10.8.1") is None
    assert call("standard") is None
    assert call("Strang (2016), section 6.1") is None
    assert call("scipy.special.jv(0, 1.0)") is not None
    assert call("max(abs(numpy.linalg.eigvals([[1, 2], [3, 4]])))") is not None
    assert call("numpy.fft.fft(x), x = numpy.arange(8)") is None  # words around the call
    assert call("minimize_scalar(lambda x: np.cosh(x)).fun") is None  # names of its own


def test_the_letters_of_a_source_are_symbols_and_functions():
    x, f = sympy.Symbol("x"), sympy.Function("f")
    assert value(call("sympy.factor(x**2 - 1)")) == (x - 1) * (x + 1)
    solved = value(call("sympy.dsolve(f(x).diff(x) - f(x), ics={f(0): 2})"))
    assert solved == sympy.Eq(f(x), 2 * sympy.exp(x))
    assert value(call("scipy.integrate.quad(lambda x: x, 0, 2)[0]")) == 2.0
