"""The notation matrix (``curriculum/notations/``): every specification is well formed,
every oracle agrees with its second route, and every cell recorded as passing still
passes. ``benchmarks/notations.py`` runs the rest and keeps the record.

The cells are run by the harness itself, in a process of its own, as the command line runs
them: each cell in a fresh process, several at a time. The whole record runs by default,
since it takes seconds; ``RELATIVISTICPY_NOTATIONS_SAMPLE=N`` runs one recorded cell in N,
the same cells every time, for when it no longer does.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import sympy

BENCHMARKS = Path(__file__).parent.parent / "benchmarks"
SAMPLE = int(os.environ.get("RELATIVISTICPY_NOTATIONS_SAMPLE", "1"))


def _notations():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("notations", BENCHMARKS / "notations.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["notations"] = module
    spec.loader.exec_module(module)
    return module


notations = _notations()
CONSTRUCTS = notations.load()
CELLS = [cell for construct in CONSTRUCTS for cell in construct.cells()]
PASSING = notations.passing()
RECORDED = list(notations.recorded_cells(SAMPLE if SAMPLE > 1 else None))


def _harness(*options: str, outcomes: Path) -> dict[str, tuple[str, str]]:
    """What the harness says of each cell, or of each oracle, run as the command line
    runs it."""
    command = [sys.executable, str(BENCHMARKS / "notations.py"), *options]
    ran = subprocess.run(
        [*command, "--outcomes", str(outcomes)],
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    assert outcomes.is_file(), ran.stdout + ran.stderr
    found = {}
    for line in outcomes.read_text(encoding="utf-8").splitlines():
        identifier, outcome, reason = line.split("\t", 2)
        found[identifier] = (outcome, reason)
    return found


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """The outcome of each cell recorded as passing, or of the sample of them."""
    sample = ["--sample", str(SAMPLE)] if SAMPLE > 1 else []
    written = tmp_path_factory.mktemp("notations") / "cells.tsv"
    return _harness("--recorded", *sample, outcomes=written)


@pytest.fixture(scope="module")
def oracles(tmp_path_factory):
    """Whether each operand's oracle gives an answer, and its second route the same."""
    return _harness("--oracles", outcomes=tmp_path_factory.mktemp("notations") / "oracles.tsv")


def test_every_specification_is_well_formed():
    assert CONSTRUCTS, "the notation matrix has no constructs"
    assert notations.problems(CONSTRUCTS) == []


def test_every_construct_checks_its_oracle_by_a_second_route():
    assert [c.id for c in CONSTRUCTS if not c.second_route.strip()] == []


def test_every_cell_has_an_identifier_of_its_own():
    assert len({cell.id for cell in CELLS}) == len(CELLS)


def test_every_cell_recorded_as_passing_exists():
    assert {cell.id for cell in CELLS} >= PASSING


@pytest.mark.parametrize(
    ("construct", "operand"),
    [(construct, operand) for construct in CONSTRUCTS for operand in construct.operands],
    ids=lambda value: getattr(value, "id", None) or value.name,
)
def test_an_oracle_agrees_with_its_second_route(construct, operand, oracles):
    outcome, reason = oracles[f"{construct.id}//{operand.name}"]
    assert outcome == "passed", reason


@pytest.mark.parametrize("cell", RECORDED, ids=lambda cell: cell.id)
def test_a_cell_that_passed_still_passes(cell, recorded):
    outcome, reason = recorded[cell.id]
    again = f"uv run python benchmarks/notations.py --failures --cell '{cell.id}'"
    assert outcome == "passed", f"{outcome}: {reason}\n{cell.program}\n{again}"


# -- reading a specification -----------------------------------------------------------------

SPECIFICATION = r"""# Halves

A half of a quantity.

## Names

```python
x = symbols("x")
```

## Operands

| operand | written in | `top` | `(top)` | separates |
|---|---|---|---|---|
| `x` | latex, text | `x` | `x` | |
| | python | `x` | | |
| `modulus` | latex | `\|x\|` | `\|x\|` | a bar in a cell |
| | python | `Abs(x)` | | |

## Oracle

```python
expected = S(top) / 2
```

## Second route

```python
assert expected * 2 == top
```

## Forms

### Built up

**Form** · `frac` · Lamport (1994), §3.3.1

```latex
\frac{«top»}{2}
```

### A solidus, in plain text

**Form** · `slash` · the solidus of plain text

```text
«(top)»/2
```

### The physics package's

**Form** · `physics` · the physics package

```latex
\flatfrac{«top»}{2}
```

**Refused** · `ParseError` · `unknown LaTeX command` · no command of the language's
**Only of** · `x` · one operand shows it
"""


@pytest.fixture
def halves(tmp_path):
    folder = tmp_path / "algebra"
    folder.mkdir()
    (folder / "half.md").write_text(SPECIFICATION, encoding="utf-8")
    return notations.read(folder / "half.md")


def test_a_construct_is_named_after_its_folder_and_its_file(halves):
    assert halves.id == "algebra.half"
    assert halves.slots == ("top", "(top)")
    assert notations.problems([halves]) == []


def test_a_form_is_filled_with_each_operand_written_in_its_notation(halves):
    cells = {cell.id: cell.program for cell in halves.cells()}
    assert cells == {
        "algebra.half/frac/x": r"\frac{x}{2}",
        "algebra.half/frac/modulus": r"\frac{|x|}{2}",  # a bar, written \| in the table
        "algebra.half/slash/x": "x/2",  # the modulus is not written in plain text
        "algebra.half/physics/x": r"\flatfrac{x}{2}",
    }


def test_an_oracle_is_given_the_operand_alone(halves):
    frac, _, _, physics = halves.cells()
    assert frac.oracle == 'x = symbols("x")\ntop = (x)\nexpected = S(top) / 2\n'
    assert physics.oracle == "error = 'ParseError'\nmessage = 'unknown LaTeX command'\n"


def test_a_program_in_latex_is_given_as_latex(halves):
    frac, _, slash, _ = halves.cells()
    assert frac.case().latex
    assert (slash.case().program, slash.case().latex) == ("x/2", False)


def test_a_cell_is_run_as_the_curriculum_runs_a_case():
    cells = {cell.id: cell for cell in CELLS}
    answered = notations.run(cells["algebra.fraction/frac/numbers"])
    refused = notations.run(cells["calculus.definite-integral/physics-dd/x^2"])
    assert (answered.outcome, refused.outcome) == ("passed", "passed"), answered.describe()
    assert cells["calculus.definite-integral/physics-dd/x^2"].form.refused == "ParseError"


def test_what_is_wrong_with_a_specification_is_said(tmp_path):
    folder = tmp_path / "algebra"
    folder.mkdir()
    wrong = SPECIFICATION.replace(
        "«top»}{2}\n```\n\n### A solidus", "«bottom»}{2}\n```\n\n### A solidus"
    )
    wrong = wrong.replace('x = symbols("x")', 'top = symbols("top")').replace(
        "`ParseError`", "`Oops`"
    )
    (folder / "half.md").write_text(wrong, encoding="utf-8")
    said = "\n".join(notations.problems([notations.read(folder / "half.md")]))
    assert "the form frac has «bottom», which no operand writes" in said
    assert "the slot top is also a name of ## Names" in said
    assert "must be refused with Oops, which is no error" in said


def test_forms_that_disagree_are_grouped_by_the_answer_they_gave(halves):
    frac, modulus, slash, physics = halves.cells()
    outcomes = [
        notations.Outcome(frac, "passed"),
        notations.Outcome(slash, "wrong", "not the answer", "expected x/2\ngot      2/x"),
        notations.Outcome(modulus, "passed"),
        notations.Outcome(physics, "wrong", "should have been refused", "expected ParseError"),
    ]
    (found,) = notations.disagreements(outcomes)
    assert found.operand.name == "x"
    assert found.answers == {None: ("frac",), "2/x": ("slash",)}
    assert found.expected == "x/2"


RECIPROCALS = r"""# Reciprocals

One over a number, and a set of two numbers.

## Operands

| operand | written in | `n` | separates |
|---|---|---|---|
| `two` | latex, text | `2` | |
| | python | `2` | |
| `physics` | latex | `\dd{x}` | a command the language refuses, so the cell must be refused |
| | python | `None` | |

## Oracle

```python
if n is None:
    error = "ParseError"
else:
    expected = [Rational(1, n), n]
    in_any_order = True
```

## Second route

```python
assert n is None or expected[0] * n == 1
```

## Forms

### Built up

**Form** · `frac` · Lamport (1994), §3.3.1

```latex
[\frac{1}{«n»}, «n»]
```

### A list in plain text, the other way round

**Form** · `slash` · the solidus of plain text

```text
[«n», 1/«n»]
```
"""


@pytest.fixture
def reciprocals(tmp_path, monkeypatch):
    """A construct in a curriculum of its own, which its cells are run in."""
    monkeypatch.setattr(notations.curriculum, "ROOT", tmp_path / "curriculum")
    folder = tmp_path / "curriculum" / "notations" / "algebra"
    folder.mkdir(parents=True)
    (folder / "reciprocal.md").write_text(RECIPROCALS, encoding="utf-8")
    return notations.read(folder / "reciprocal.md")


def test_an_operand_whose_oracle_sets_an_error_must_be_refused(reciprocals):
    assert notations.problems([reciprocals]) == []
    checked = [notations.check(reciprocals, operand) for operand in reciprocals.operands]
    assert [each.outcome for each in checked] == ["passed", "passed"]
    outcomes = {cell.id: notations.run(cell).outcome for cell in reciprocals.cells()}
    assert outcomes == {
        "algebra.reciprocal/frac/two": "passed",
        "algebra.reciprocal/frac/physics": "passed",  # refused, with a ParseError
        "algebra.reciprocal/slash/two": "passed",  # the same items, in the other order
    }


def test_an_oracle_may_compare_in_any_order_or_up_to_a_constant():
    x, y = sympy.symbols("x y")
    equal = notations.curriculum.equal
    in_any_order = notations._comparison(equal, in_any_order=True)
    assert in_any_order([2, 1], [1, 2])
    assert in_any_order({2: 1, 1: 2}, [1, 1, 2])  # a value repeated as often as it counts
    assert not in_any_order([1, 1], [1, 2])
    assert not in_any_order([1, 2, 3], [1, 2])
    assert not equal([2, 1], [1, 2])  # without it, the order is the answer's
    antiderivative = notations._comparison(equal, up_to_a_constant=x)
    assert antiderivative(x**3 / 3 + 5 * y, x**3 / 3)
    assert not antiderivative(x**3 / 3 + x, x**3 / 3)
    assert not antiderivative(sympy.Integral(x**2, x), x**3 / 3)  # left undone


def test_an_oracle_that_asks_for_a_comparison_it_cannot_have_is_broken(tmp_path, monkeypatch):
    monkeypatch.setattr(notations.curriculum, "ROOT", tmp_path / "curriculum")
    folder = tmp_path / "curriculum" / "notations" / "algebra"
    folder.mkdir(parents=True)
    for wrong, said in (
        ("in_any_order = True", "in_any_order is set, but expected is no list"),
        ("up_to_a_constant = 2", "up_to_a_constant is set, but to no symbol"),
        ('error = "Oops"', "the oracle's error Oops is no error"),
    ):
        written = SPECIFICATION.replace("expected = S(top) / 2", f"expected = S(top) / 2\n{wrong}")
        (folder / "half.md").write_text(written, encoding="utf-8")
        construct = notations.read(folder / "half.md")
        checked = notations.check(construct, construct.operands[0])
        assert (checked.outcome, checked.reason) == ("broken", said)


def test_the_sample_is_chosen_by_identifier_alone():
    sample = notations.sample(CELLS, 3)
    assert sample == notations.sample(CELLS, 3)
    assert 0 < len(sample) < len(CELLS)
    assert sample == notations.sample(CELLS[::-1], 3)[::-1]
