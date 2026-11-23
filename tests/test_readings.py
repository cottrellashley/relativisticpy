"""The benchmark of readings: every statement, read a line each, and two runs compared."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("readings", ROOT / "benchmarks/readings.py")
assert spec is not None
assert spec.loader is not None
readings = importlib.util.module_from_spec(spec)
sys.modules.setdefault("readings", readings)
spec.loader.exec_module(readings)


def test_a_statement_is_read_as_the_operations_it_compiles_to():
    rows = readings.read_all({"here": [("a program", ["f(x) := x^2\nf(3) + \\sin y"])]})
    assert [text for _, text, _ in rows] == ["f(x) := x^2", "f(3) + \\sin y"]
    assert rows[0][2] == "-"  # a definition shows nothing
    assert "call sin" in rows[1][2]


def test_a_statement_refused_is_read_as_its_error():
    rows = readings.read_all({"here": [("a program", ["\\frac{1}{"])]})
    assert rows[0][2].startswith("!")


def test_two_runs_compared_list_only_what_is_read_otherwise(tmp_path, capsys):
    before, after = tmp_path / "before.tsv", tmp_path / "after.tsv"
    before.write_text("a#0\tx + y\t111\tadd(x, y)\na#1\tz\t222\tz\n")
    after.write_text("a#0\tx + y\t111\tadd(x, y)\na#1\tz\t333\t! refused\n")
    readings.compare(before, after, limit=10)
    shown = capsys.readouterr().out
    assert "a#1" in shown
    assert "a#0" not in shown
    assert "1 statements read otherwise (1 read -> refused)" in shown
