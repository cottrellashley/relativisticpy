"""The curriculum (``curriculum/``): every case is well formed, and every case recorded as
passing still passes. ``benchmarks/curriculum.py`` runs the rest and keeps the record."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import sympy

BENCHMARKS = Path(__file__).parent.parent / "benchmarks"


def _curriculum():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("curriculum", BENCHMARKS / "curriculum.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["curriculum"] = module
    spec.loader.exec_module(module)
    return module


curriculum = _curriculum()
CASES = curriculum.load()
PASSING = curriculum.passing()


def test_every_case_is_well_formed():
    assert CASES, "the curriculum has no cases"
    assert curriculum.problems(CASES) == []


def test_every_case_recorded_as_passing_exists():
    assert {case.id for case in CASES} >= PASSING


@pytest.mark.parametrize(
    "case", [case for case in CASES if case.id in PASSING], ids=lambda case: case.id
)
def test_a_case_that_passed_still_passes(case):
    result = curriculum.run(case)
    assert result.passed, result.describe()


# -- what counts as the answer ---------------------------------------------------------------

s, t, x = sympy.symbols("s t x")
y = sympy.Function("y")
SINC = sympy.sin(sympy.pi * s) / (sympy.pi * s)


def test_a_piecewise_that_fills_a_removable_point_is_the_answer():
    filled = sympy.Piecewise((SINC, sympy.Ne(s, 0)), (1, True))
    line = (s > -sympy.oo) & (s < sympy.oo) & sympy.Ne(s, 0)
    assert curriculum.equal(filled, SINC)
    assert curriculum.equal(sympy.Piecewise((SINC, line), (1, True)), SINC)


def test_a_piecewise_with_another_value_or_condition_is_not():
    assert not curriculum.equal(sympy.Piecewise((SINC, sympy.Ne(s, 0)), (0, True)), SINC)
    assert not curriculum.equal(sympy.Piecewise((SINC, s > 0), (1, True)), SINC)
    assert not curriculum.equal(sympy.Piecewise((SINC, s > -sympy.oo), (0, True)), SINC)


def test_equations_are_compared_by_the_difference_of_their_sides():
    solved = sympy.Eq(y(t), sympy.Symbol("C1") * sympy.exp(t))
    assert curriculum.equal(sympy.Eq(y(t), sympy.exp(t) * sympy.Symbol("C1")), solved)
    assert curriculum.equal(sympy.Eq(sympy.Symbol("C1") * sympy.exp(t), y(t)), solved)
    assert curriculum.equal(sympy.Eq(y(t) - sympy.Symbol("C1") * sympy.exp(t), 0), solved)
    assert not curriculum.equal(sympy.Eq(y(t), sympy.Symbol("C1") * sympy.exp(2 * t)), solved)


def test_a_result_left_undone_is_not_its_value():
    assert not curriculum.equal(sympy.Integral(x, (x, 0, 1)), sympy.Rational(1, 2))
    assert curriculum.equal(sympy.Integral(x, (x, 0, 1)), sympy.Integral(x, (x, 0, 1)))
