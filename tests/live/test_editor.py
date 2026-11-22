r"""The editor's Python side, without a browser: from source to results, stale runs, errors
with their spans, computing a part of a statement, time limits, what it carries, and the
Python objects its programs read."""

import json
import math
import re
import threading
import time

import pytest

pytest.importorskip("anywidget", reason="the editor needs the live extra, with anywidget")

from relativisticpy import Workbook
from relativisticpy.errors import UsageError
from relativisticpy.live import editor as editing
from relativisticpy.live import worker as working
from relativisticpy.live.editor import Editor

KERR = r"g_{\mu\nu} := Kerr(M, a)" + "\nR_{abcd} R^{abcd}"


@pytest.fixture(scope="module")
def vacuum():
    wb = Workbook()
    wb.expr(r"g_{\mu\nu} := Schwarzschild(M)")
    return wb


@pytest.fixture(scope="module")
def editor(vacuum):
    made = Editor(vacuum)
    yield made
    made.close()


class Sent:
    """What an editor sends its widget, as the widget would receive it."""

    def __init__(self, editor):
        self.messages = []
        self.arrived = threading.Condition()
        editor.send = self.send

    def send(self, content, buffers=None):
        with self.arrived:
            self.messages.append(json.loads(json.dumps(content)))
            self.arrived.notify_all()

    def wait(self, count=1, timeout=60.0):
        with self.arrived:
            self.arrived.wait_for(lambda: len(self.messages) >= count, timeout)
        return self.messages


def values(statements):
    return [[value["text"] for value in entry["values"]] for entry in statements]


# -- what it carries ----------------------------------------------------------------------


def test_it_carries_temml_and_loads_nothing_from_the_network():
    module, style = Editor._esm, Editor._css
    assert module.startswith("var temml=function(){")
    assert f'version:"{editing.TEMML}"' in module
    assert module.rstrip().endswith("export default { render };")
    addresses = set(re.findall(r"https?://[^\s'\")]+", module + style))
    assert addresses <= {"http://www.w3.org/1998/Math/MathML", "http://www.w3.org/2000/svg"}
    targets = re.findall(r"url\(\s*['\"]?([^'\")]*)", style)
    assert all(target.startswith(("data:", "%23", "#")) for target in targets)
    assert "url(data:font/woff2;base64," in style
    own = (editing.STATIC / "editor.js").read_text(encoding="utf-8")
    assert "fetch(" not in own
    assert "import " not in own.replace("export default", "")
    licence = (editing.STATIC / "temml" / "LICENSE").read_text(encoding="utf-8")
    assert licence.startswith("The MIT License (MIT)\n\nCopyright (c) 2020 Ron Kok")


def test_its_state_is_data(editor):
    assert editor.trait_metadata("source", "sync")
    assert editor.trait_metadata("statements", "sync")
    statements = editor.run(r"R_{abcd} R^{abcd}" + "\nx := ")
    assert editor.statements == statements
    assert json.loads(json.dumps(statements)) == statements
    state = editor.get_state(["source", "statements"])
    assert state["source"] == editor.source
    assert state["statements"] == statements


# -- runs ---------------------------------------------------------------------------------


def test_each_statement_has_its_values_as_display_gives_them(editor):
    first, second = editor.run(r"R_{abcd} R^{abcd}" + "\n" + r"\Gamma^{r}_{\theta\theta}")
    assert first["values"][0]["kind"] == "expression"
    assert first["values"][0]["latex"] == r"\frac{48 M^{2}}{r^{6}}"
    assert (first["start"], first["end"], first["line"]) == (0, 17, 1)
    assert second["typeset"] == r"\Gamma^{r}_{\theta\theta}"
    assert second["values"][0]["text"] == "2*M - r"


def test_an_error_is_where_it_is_with_its_notes_and_hint(editor):
    source = "x := 2\n\\frca{1}{2}\nx + 1"
    first, unread, third = editor.run(source)
    error = unread["error"]
    assert source[error["start"] : error["end"]] == "\\frca"
    assert (error["line"], error["column"]) == (2, 1)
    assert error["hint"] == "did you mean \\frac?"
    assert error["rendered"].startswith("ParseError: unknown LaTeX command \\frca")
    assert values([first, third]) == [[], ["3"]]


def test_a_half_typed_definition_never_reaches_the_workbook(vacuum, editor):
    editor.run("y := 3\nK := R_{abcd} R^{abcd}\nz := ")
    assert "y" not in vacuum
    assert "K" not in vacuum


def test_a_run_starts_from_the_workbook_as_it_is_then():
    wb = Workbook()
    with_workbook = Editor(wb)
    try:
        assert with_workbook.run("w")[0]["values"][0]["text"] == "w"
        wb.expr("w := 7")
        assert with_workbook.run("w^2")[0]["values"][0]["text"] == "49"
    finally:
        with_workbook.close()


def test_the_widget_is_answered_with_the_id_it_asked_with(editor):
    sent = Sent(editor)
    editor._heard(editor, {"type": "run", "id": "view:1", "source": "u := 4\nu^2"}, [])
    (answer,) = sent.wait()
    assert answer["type"] == "ran"
    assert answer["id"] == "view:1"
    assert answer["source"] == "u := 4\nu^2"
    assert values(answer["statements"]) == [[], ["16"]]


def test_a_run_made_stale_by_more_typing_is_dropped():
    slow = Editor(time_limit=60)
    try:
        sent = Sent(slow)
        slow.run("x := 1")  # the worker has started
        slow._heard(slow, {"type": "run", "id": "v:1", "source": KERR}, [])
        time.sleep(0.5)
        started = time.monotonic()
        slow._heard(slow, {"type": "run", "id": "v:2", "source": "x := 2\nx^2"}, [])
        (answer,) = sent.wait()
        assert time.monotonic() - started < 10
        assert answer["id"] == "v:2"
        assert values(answer["statements"]) == [[], ["4"]]
        time.sleep(0.5)
        assert len(sent.messages) == 1
    finally:
        slow.close()


def test_a_runaway_computation_is_stopped_at_its_time_limit():
    quick = Editor(time_limit=1.0)
    try:
        started = time.monotonic()
        statements = quick.run(KERR + "\nx := 1")
        assert time.monotonic() - started < 10
        over = statements[-1]
        assert over["error"]["message"] == "over the time limit of 1 s"
        assert over["statement"] == "R_{abcd} R^{abcd}"
        assert values(quick.run("x := 2\nx + 1")) == [[], ["3"]]  # and it goes on
    finally:
        quick.close()


# -- parts of a statement ------------------------------------------------------------------


def test_a_part_of_a_statement_is_computed_where_it_stands(editor):
    source = "f := (x + 1)^2\n" + r"\frac{d}{dx} (x + 1)^2 = 2x + 2"
    editor.run(source)
    start = source.index(r"\frac{d}{dx}")
    expanded = editor.compute(0, len("f := (x + 1)^2"), "expand")
    assert expanded["part"] == "(x + 1)^2"
    assert [value["text"] for value in expanded["values"]] == ["x^2 + 2*x + 1"]
    claim = editor.compute(start, len(source), "holds")
    assert claim["holds"] is True
    latex = editor.compute(start, source.index(" = 2x"), "latex")
    assert latex["values"][0]["latex"] == "2 x + 2"


def test_a_part_must_be_in_one_statement(editor):
    source = "x := 1\ny := 2"
    editor.run(source)
    answer = editor.compute(3, 10)
    assert answer["error"]["message"] == "select a part of one statement, among those that ran"


def test_the_widget_asks_for_parts_too(editor):
    sent = Sent(editor)
    source = "s := 2\ns^3"
    asked = {"type": "compute", "id": "v:9", "source": source, "start": 7, "end": 10}
    editor._heard(editor, {**asked, "action": "compute"}, [])
    (answer,) = sent.wait()
    assert (answer["type"], answer["id"], answer["part"]) == ("computed", "v:9", "s^3")
    assert [value["text"] for value in answer["values"]] == ["8"]


# -- what it is given, and where it computes -----------------------------------------------


def test_what_it_is_given_is_checked():
    with pytest.raises(UsageError, match="an editor's workbook is a Workbook, not dict"):
        Editor({})
    with pytest.raises(UsageError, match="an editor's source is a str"):
        Editor(source=3)
    with pytest.raises(UsageError, match="a time limit is a positive number of seconds"):
        Editor(time_limit=0)


def test_a_workbook_no_worker_can_take_is_computed_in_the_notebook(monkeypatch):
    def refuse(workbook, objects=None):
        raise TypeError("cannot pickle 'generator' object")

    monkeypatch.setattr(editing, "portable", refuse)
    here = Editor()
    try:
        statements = here.run("x := 2\nx^2")
        assert values(statements) == [[], ["4"]]
        answer = here._wait("run", {"text": "x"})
        assert answer["notes"][-1].startswith("this editor computes in the notebook's own process")
        assert isinstance(here._computations, working.InPlace)
    finally:
        here.close()


# -- Python objects ---------------------------------------------------------------------------


def quarter(n):
    """A function of the notebook's own, which no other process can import."""
    return n / 4


def test_its_programs_read_python_objects_which_go_to_the_worker_as_data():
    np = pytest.importorskip("numpy")
    signal = np.array([1.0, 2.0, 6.0])
    reading = Editor(scope={"signal": signal, "cos": math.cos})
    try:
        program = r"x_t := \texttt{signal}" "\n" r"\mathbb{E}_t[x_t]" "\n" r"\texttt{cos}(0)"
        assert values(reading.run(program)) == [[], ["3.0"], ["1.0"]]
        assert isinstance(reading._computations, working.Computations)  # cos goes by name
        signal[0] = 4.0  # changed in place: read again as the next run starts
        assert values(reading.run()) == [[], ["4.0"], ["1.0"]]
    finally:
        reading.close()


def test_a_function_no_worker_can_import_is_computed_in_the_notebook_saying_why():
    here = Editor(scope={"quarter": quarter})
    try:
        assert values(here.run(r"\texttt{quarter}(3)")) == [["0.75"]]
        assert isinstance(here._computations, working.InPlace)
        note = here._wait("run", {"text": "1"})["notes"][-1]
        assert "cannot be sent to a worker" in note
        assert r"the Python function \texttt{quarter} is defined in" in note
    finally:
        here.close()


def test_the_workbooks_scope_is_read_where_the_editor_has_none():
    with_scope = Editor(Workbook(scope={"G": 2}))
    without = Editor()
    try:
        assert values(with_scope.run(r"\texttt{G} + 1")) == [["3"]]
        (entry,) = without.run(r"\texttt{G} + 1")
        assert "no Python scope was given" in entry["error"]["message"]
        assert entry["error"]["hint"].startswith("give the workbook one")
    finally:
        with_scope.close()
        without.close()


def test_closing_it_stops_its_worker():
    closing = Editor()
    closing.run("x := 1")
    worker = closing._computations._worker
    process = worker._process
    closing.close()
    process.wait(timeout=30)
    assert process.returncode is not None
