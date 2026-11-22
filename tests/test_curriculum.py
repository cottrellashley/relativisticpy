"""The curriculum (``curriculum/``): every case is well formed, and every case recorded as
passing still passes. ``benchmarks/curriculum.py`` runs the rest and keeps the record."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

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
