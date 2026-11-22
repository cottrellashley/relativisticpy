r"""The editor's engine: a program run in copies of a workbook, from its first changed
statement, and a part of a statement computed as the workbook stands at it."""

import time

import pytest

from relativisticpy import Workbook
from relativisticpy.live.engine import Engine

SCHWARZSCHILD = r"""g_{\mu\nu} := Schwarzschild(M)
K := R_{abcd} R^{abcd}
K
\Gamma^{r}_{\theta\theta}"""


def texts(result):
    return [[value["text"] for value in entry["values"]] for entry in result["statements"]]


def test_a_run_gives_each_statement_as_the_command_line_does():
    result = Engine().run(SCHWARZSCHILD)
    assert result["ok"]
    assert result["source"] == SCHWARZSCHILD
    first, second, third, fourth = result["statements"]
    assert first["statement"] == r"g_{\mu\nu} := Schwarzschild(M)"
    assert [d["name"] for d in first["defines"]] == ["t", "r", "theta", "phi", "g"]
    assert (first["start"], first["end"], first["line"]) == (0, 30, 1)
    assert second["defines"] == [{"name": "K", "kind": "value", "written": "K"}]
    assert third["values"][0]["latex"] == r"\frac{48 M^{2}}{r^{6}}"
    assert fourth["values"][0]["text"] == "2*M - r"
    assert fourth["typeset"] == r"\Gamma^{r}_{\theta\theta}"


def test_an_error_says_where_it_is_and_the_rest_still_runs():
    source = "x := 2\n\\frca{1}{2}\nx + 1\nR_{ab}"
    _, unread, third, fourth = Engine().run(source)["statements"]
    assert not unread["read"]
    error = unread["error"]
    assert error["type"] == "ParseError"
    assert source[error["start"] : error["end"]] == "\\frca"
    assert error["rendered"].startswith("ParseError: unknown LaTeX command \\frca")
    assert error["hint"] == "did you mean \\frac?"
    assert texts({"statements": [third]}) == [["3"]]
    assert fourth["error"]["message"] == "the Ricci tensor needs coordinates"
    assert source[fourth["error"]["start"] : fourth["error"]["end"]] == "R_{ab}"


def test_a_run_never_changes_the_workbook_it_was_given():
    wb = Workbook()
    wb.expr("x := 2")
    engine = Engine(wb)
    assert texts(engine.run("x := 3; x^2")) == [[], ["9"]]
    assert engine.run("y := ")["statements"][0]["error"] is not None
    assert "y" not in wb
    assert wb.expr("x^2") == 4


def test_the_statements_before_the_first_change_are_not_run_again():
    engine = Engine()
    engine.run(SCHWARZSCHILD)
    kept = engine.statements[:3]
    started = time.perf_counter()
    result = engine.run(SCHWARZSCHILD + "\nK + 1")
    assert time.perf_counter() - started < 1.0
    assert engine.statements[:3] == kept
    assert texts(result)[-1] == ["(48*M^2 + r^6)/r^6"]
    changed = SCHWARZSCHILD.replace("K\n", "K + 2\n")
    result = engine.run(changed)
    assert engine.statements[:2] == kept[:2]
    assert engine.statements[2] is not kept[2]
    assert texts(result)[2] == ["(48*M^2 + 2*r^6)/r^6"]


def test_a_statement_that_grows_runs_again():
    engine = Engine()
    assert texts(engine.run("x := 1\nx")) == [[], ["1"]]
    assert texts(engine.run("x := 12\nx")) == [[], ["12"]]
    assert texts(engine.run("x := 12\nx\n")) == [[], ["12"]]


def test_what_a_rerun_reads_alone_must_read_as_it_did():
    engine = Engine()
    engine.run(r"\begin{aligned} u &= 1 \\ v &= u + 1 \end{aligned}")
    result = engine.run(r"\begin{aligned} u &= 1 \\ v &= u + 2 \end{aligned}")
    assert result["ok"]
    assert [entry["statement"] for entry in result["statements"]] == ["u &= 1", "v &= u + 2"]


def test_a_program_that_cannot_be_read_at_all_says_why():
    result = Engine().run("x := [1, 2")
    assert not result["ok"]
    assert result["statements"][0]["error"]["type"] == "ParseError"


def test_a_time_limit_stops_a_statement_and_says_what_did_not_run():
    slow = r"g_{\mu\nu} := Kerr(M, a)" + "\nR_{abcd} R^{abcd}\nx := 1\ny := 2"
    engine = Engine()
    result = engine.run(slow, limit=1.0)
    assert not result["ok"]
    over = result["statements"][-1]
    assert over["error"]["message"] == "over the time limit of 1 s"
    assert over["statement"] == "R_{abcd} R^{abcd}"
    assert result["notes"] == [
        "the time limit of 1 s was reached: 2 statements, from line 3 on, did not run"
    ]
    assert len(engine.statements) == 1  # the definition of the metric stands


@pytest.fixture(scope="module")
def ran():
    engine = Engine()
    engine.run(SCHWARZSCHILD + "\n" + r"\frac{d}{dr}\left(1 - \frac{2M}{r}\right) = \frac{2M}{r^2}")
    return engine


def test_a_part_is_computed_as_the_workbook_stands_at_its_statement(ran):
    source = ran.source
    start = source.index("R_{abcd}")
    answer = ran.compute(source, start, start + len("R_{abcd} R^{abcd}"))
    assert answer["part"] == "R_{abcd} R^{abcd}"
    assert [value["text"] for value in answer["values"]] == ["48*M^2/r^6"]
    assert answer["error"] is None


def test_a_selection_is_widened_to_the_expression_that_holds_it(ran):
    source = ran.source
    start = source.index("M}{r}\\right)")
    answer = ran.compute(source, start, start + len("M}{r"))
    assert answer["part"] == r"\frac{2M}{r}"
    assert source[answer["start"] : answer["end"]] == answer["part"]
    assert [value["text"] for value in answer["values"]] == ["2*M/r"]


def test_a_part_is_simplified_expanded_or_judged(ran):
    source = ran.source
    start = source.index(r"\frac{d}{dr}")
    end = len(source)
    claim = ran.compute(source, start, end, "holds")
    assert claim["holds"] is True
    derivative = ran.compute(source, start, source.index(" = \\frac{2M}{r^2}"), "simplify")
    assert [value["text"] for value in derivative["values"]] == ["2*M/r^2"]
    expanded = ran.compute(source, start, start + len(r"\frac{d}{dr}\left(1 - \frac{2M}{r}\right)"))
    assert expanded["values"][0]["latex"] == r"\frac{2 M}{r^{2}}"


def test_whether_a_part_holds_says_when_it_is_no_claim(ran):
    source = ran.source
    start = source.index("R_{abcd}")
    answer = ran.compute(source, start, start + len("R_{abcd} R^{abcd}"), "holds")
    assert answer["holds"] is None
    assert answer["notes"] == ["this is an expression, not a claim: it has a value"]


def test_a_part_of_another_program_runs_it_first():
    engine = Engine()
    engine.run("x := 2")
    answer = engine.compute("x := 3\nx^2", 7, 10)
    assert [value["text"] for value in answer["values"]] == ["9"]
    assert engine.source == "x := 3\nx^2"


def test_a_part_of_no_statement_is_refused(ran):
    start = ran.source.index("K\n")
    answer = ran.compute(ran.source, start - 3, start + 3)
    assert answer["error"]["message"] == "select a part of one statement, among those that ran"
