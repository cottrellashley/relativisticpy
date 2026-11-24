"""The MCP server: its tools, resources and prompts, as a client sees them, its sessions,
and the bounds it keeps: no Python, files only where allowed, time limits and sizes.

Each test serves the server in memory, with the SDK's own client, except one that starts
it as a client does, over standard input and output."""

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="the MCP server needs the mcp extra")

import anyio
from mcp import Client, StdioServerParameters

from relativisticpy import cli
from relativisticpy import help as documentation
from relativisticpy import mcp as server
from relativisticpy.checking import Verdict

ROOT = Path(__file__).parent.parent
POSIX = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
#: A statement that runs for minutes, in Python, so that a time limit can stop it.
SLOW = "factor(expand((x+y+z)^60) + 1)"
SCHWARZSCHILD = r"g_{\mu\nu} := Schwarzschild(M)"


def serve(test, **settings):
    """What ``test(client, pool)`` returns, run against a server in memory."""

    async def main():
        built, pool = server._build(server.Settings(**settings))
        async with Client(built, raise_exceptions=True) as client:
            return await test(client, pool)

    return anyio.run(main)


async def call(client, tool, **arguments):
    """A tool's result: its structured content, or the text of its error."""
    result = await client.call_tool(tool, arguments)
    if result.is_error:
        return Failure(result.content[0].text)
    if result.structured_content is not None:
        return result.structured_content
    return result.content[0].text


class Failure(str):
    """The text of a call that failed as a whole."""


def values(result):
    """The text of each value of each statement of a computation."""
    return [[value["text"] for value in entry["values"]] for entry in result["statements"]]


# -- the tools ------------------------------------------------------------------------------


def test_the_tools_say_what_they_take_and_what_they_give():
    async def test(client, pool):
        return {tool.name: tool for tool in (await client.list_tools()).tools}

    tools = serve(test)
    assert set(tools) == {"compute", "verify", "check", "help", "list_sessions", "reset_session"}
    for tool in tools.values():
        assert tool.title, tool.name
        assert len(tool.description) > 100, tool.name
        for name, schema in tool.input_schema.get("properties", {}).items():
            assert schema.get("description"), f"{tool.name}({name})"
    assert tools["compute"].input_schema["required"] == ["source"]
    assert tools["verify"].input_schema["required"] == ["claim"]
    statement = tools["compute"].output_schema["$defs"]["Statement"]["properties"]
    assert list(statement) == [
        "file", "line", "column", "end_line", "end_column", "statement", "values", "defines",
        "notes", "assuming", "error",
    ]  # fmt: skip
    assert tools["help"].output_schema is None  # Markdown, to read
    assert tools["verify"].annotations.read_only_hint
    assert tools["reset_session"].annotations.destructive_hint
    # Each says the limit it is bound by.
    assert "60 s at most" in tools["compute"].description


def test_the_descriptions_say_what_an_agent_needs_to_act_on_a_result():
    async def test(client, pool):
        tools = {tool.name: tool.description for tool in (await client.list_tools()).tools}
        return client.instructions, tools

    instructions, tools = serve(test)
    compute, verify = tools["compute"], tools["verify"]
    told = {
        "the notation read": ["LaTeX", "plain text", "Unicode", ":=", r"\mathrm{mass}"],
        "where the rest of it is": ["topic reference"],
        "what is no answer": ["complete", "reasons", "integral", "time budget", "approximate"],
        "how to rephrase": ["evalf(", "hint", "fix the statement"],
        "how sessions work": ["session", "list_sessions", "reset_session"],
        "the time limits": ["60 s at most", "time_limit", "over the time limit"],
    }
    for what, words in told.items():
        assert all(word in compute for word in words), what
    for verdict in ("holds", "differs", "relates", "not decided", "error"):
        assert f"- {verdict}:" in verify
    assert "keeps no session" in verify
    assert "never guesses" in instructions
    assert "60 s at most" in instructions


def test_the_fail_on_verdicts_are_the_checks():
    names = {verdict.value for verdict in Verdict} | {"time limit", "empty"}
    assert set(server.FailOn.__args__) == names


def test_compute_gives_each_statement_as_the_command_line_does(capsys):
    program = f"{SCHWARZSCHILD}\n\\Gamma^{{r}}_{{tt}}\nR_{{abcd}} R^{{abcd}}\nx := 2; x^3 = 8"

    async def test(client, pool):
        return await call(client, "compute", source=program)

    result = serve(test)
    assert result["ok"]
    assert result["complete"]
    assert result["session"] is None
    assert not result["truncated"]
    assert values(result) == [[], ["M*(r - 2*M)/r^3"], ["48*M^2/r^6"], [], ["True"]]
    kretschmann = result["statements"][2]["values"][0]
    assert kretschmann["latex"] == r"\frac{48 M^{2}}{r^{6}}"
    assert kretschmann["unicode"] == "48M²/r⁶"
    defined = [definition["name"] for definition in result["statements"][0]["defines"]]
    assert defined == ["t", "r", "theta", "phi", "g"]
    # The entries are the lines of relativisticpy --format json, the same program run.
    assert cli.main(["-c", program, "--format", "json"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    command_line = [{**line, "file": "<input>"} for line in lines]
    assert result["statements"] == command_line


def test_compute_says_what_a_value_is_given_under():
    program = "a > 0\n" r"\int_0^\infty e^{-s t} \, dt"

    async def test(client, pool):
        return await call(client, "compute", source=program)

    result = serve(test)
    stated, integral = result["statements"]
    assert stated["notes"] == ["a > 0 is an assumption: from here on, a is positive"]
    assert (stated["assuming"], integral["assuming"]) == ([], ["s > 0"])
    assert values(result) == [[], ["1/s"]]


def test_compute_says_whether_each_value_is_an_answer():
    program = "x := 0.5; x^2\n" r"\int_0^1 \sin(\sin x) \, dx" "\nfoo(2)"

    async def test(client, pool):
        return await call(client, "compute", source=program)

    result = serve(test)
    assert result["ok"]
    assert not result["complete"]  # every statement ran, and not every value is an answer
    _, square, integral, unknown = result["statements"]
    flags = [
        (value["complete"], value["reasons"], value["approximate"])
        for entry in (square, integral, unknown)
        for value in entry["values"]
    ]
    assert flags == [
        (True, [], True),
        (False, ["integral"], False),
        (False, ["unknown function"], False),
    ]
    assert integral["notes"] == ["an integral is left as written: SymPy could not compute it"]


def test_an_error_comes_whole_and_the_statements_after_it_still_run():
    program = "x := 2\n\\frac{1}{2\nR_{ab}\nx + 1"

    async def test(client, pool):
        return await call(client, "compute", source=program)

    result = serve(test)
    assert not result["ok"]
    parse, compile = (entry["error"] for entry in result["statements"][1:3])
    assert parse == {
        "type": "ParseError",
        "kind": "ParseError",
        "message": "unclosed '{' in the argument of \\frac",
        "file": "<input>",
        "line": 2,
        "column": 9,
        "end_line": 2,
        "end_column": 10,
        "text": "{",
        "notes": [],
        "hint": "add '}' where the argument ends",
    }
    assert compile["type"] == "CompileError"
    assert compile["message"] == "the Ricci tensor needs coordinates"
    assert compile["hint"]
    assert values(result)[3] == ["3"]


def test_a_session_keeps_its_definitions_until_it_is_reset():
    async def test(client, pool):
        defined = await call(client, "compute", source=SCHWARZSCHILD, session="bh")
        kretschmann = await call(client, "compute", source="R_{abcd} R^{abcd}", session="bh")
        alone = await call(client, "compute", source="R_{abcd} R^{abcd}")
        listed = await call(client, "list_sessions")
        reset = await call(client, "reset_session", session="bh")
        again = await call(client, "reset_session", session="bh")
        after = await call(client, "compute", source="R_{abcd} R^{abcd}", session="bh")
        return defined, kretschmann, alone, listed, reset, again, after

    defined, kretschmann, alone, listed, reset, again, after = serve(test)
    assert defined["session"] == "bh"
    assert defined["statements"][0]["file"] == "<input 1>"
    assert values(kretschmann) == [["48*M^2/r^6"]]
    assert alone["statements"][0]["error"]["message"] == "the Riemann tensor needs coordinates"
    (session,) = listed["sessions"]
    assert session["session"] == "bh"
    assert session["statements"] == 2
    assert not session["busy"]
    assert {d["name"]: d["kind"] for d in session["defined"]}["g"] == "metric"
    assert listed["limits"]["time_limit"] == 60
    assert listed["limits"]["files"] == []
    assert reset == {"session": "bh", "reset": True, "notes": []}
    assert not again["reset"]
    assert again["notes"]
    assert after["statements"][0]["error"] is not None  # a new session knows no metric


def test_a_session_is_named_plainly_and_there_are_only_so_many():
    async def test(client, pool):
        badly = await call(client, "compute", source="x", session="no spaces")
        first = await call(client, "compute", source="x := 1", session="one")
        second = await call(client, "compute", source="x := 2", session="two")
        return badly, first, second

    badly, first, second = serve(test, max_sessions=1)
    assert isinstance(badly, Failure)
    assert "cannot name a session" in badly
    assert "hint:" in badly
    assert first["ok"]
    assert isinstance(second, Failure)
    assert "at most 1 sessions at once, and has 1: one" in second
    assert "reset_session" in second


def test_programs_get_no_python():
    marker = ROOT / "tests" / "mcp-python-ran"
    payload = f'__import__("pathlib").Path("{marker.as_posix()}").touch()'

    async def test(client, pool):
        named = await call(client, "compute", source=r"\texttt{os} + 1")
        reached = await call(client, "compute", source=r"\texttt{os.path.sep}", session="one")
        quoted = await call(client, "compute", source=f"simplify('{payload}')")
        return named, reached, quoted

    named, reached, quoted = serve(test)
    for refused in (named, reached):  # no Python scope: the program reaches nothing in Python
        error = refused["statements"][0]["error"]
        assert error["type"] == "CompileError"
        assert "names a Python object, but no Python scope was given" in error["message"]
        assert "MCP server" in " ".join(error["notes"])
        assert error["hint"].startswith("give the workbook one, as Workbook(scope=globals())")
    assert (
        quoted["statements"][0]["error"]["message"]
        == "text in quotes is not a value to compute with"
    )
    assert not marker.exists()


# -- verify ------------------------------------------------------------------------------------


def test_verify_says_whether_a_claim_holds():
    async def test(client, pool):
        verify = {
            "holds": {"claim": r"\Gamma^{r}_{\theta\theta} = 2M - r", "context": SCHWARZSCHILD},
            "differs": {
                "claim": r"R_{abcd} R^{abcd} = \frac{12 M^2}{r^6}",
                "context": SCHWARZSCHILD,
            },
            "chain": {"claim": "(x+1)^2 = x^2 + 2x + 1 = x^2 + 1", "context": "Reals := [x]"},
            "relates": {"claim": r"\frac{d}{dx} \sin^2 x = 2\cos x"},
            "declared": {"claim": r"\frac{d}{dx} \sin^2 x = 2\cos x", "context": "Reals := [x]"},
            "approximately": {"claim": r"\pi \approx 3.14"},
            "unread": {"claim": r"\frac{1}{2 = x"},
            "defines": {"claim": "E = m c^2"},
            "expression": {"claim": "x^2 + 1"},
            "broken context": {"claim": "a = 2", "context": "a := \\frca{1}{2}"},
        }
        return {
            name: await call(client, "verify", **arguments) for name, arguments in verify.items()
        }

    results = serve(test)
    verdicts = {name: result["verdict"] for name, result in results.items()}
    assert verdicts == {
        "holds": "holds",
        "differs": "differs",
        "chain": "differs",
        "relates": "relates",
        "declared": "differs",
        "approximately": "not decided",
        "unread": "error",
        "defines": "error",
        "expression": "error",
        "broken context": "error",
    }
    difference = results["differs"]["links"][0]["difference"]
    assert difference["latex"] == r"\frac{36 M^{2}}{r^{6}}"
    assert [link["verdict"] for link in results["chain"]["links"]] == ["holds", "differs"]
    assert results["chain"]["links"][1]["difference"]["text"] == "2*x"
    relates = results["relates"]["links"][0]
    assert relates["detail"].startswith("nothing defines x:")
    assert relates["difference"]["latex"]
    unread = results["unread"]["links"][0]
    assert unread["check"] == "not parsed"
    assert unread["hint"]
    assert unread["column"]
    assert "nothing to verify" in results["defines"]["links"][0]["detail"]
    assert "not a claim" in results["expression"]["links"][0]["detail"]
    broken = results["broken context"]
    assert broken["context"][0]["check"] == "not parsed"
    assert broken["notes"]


def test_compute_and_verify_agree_on_an_equation_of_index_notation():
    # R_{ab} = 0 with no metric relates the Ricci tensor of whatever metric there is: verify
    # says it relates, and compute gave it holds false, which an agent reads as untrue
    async def test(client, pool):
        computed = await call(client, "compute", source="R_{ab} = 0; R_{ab} = R_{ba}")
        verified = await call(client, "verify", claim="R_{ab} = 0")
        return computed, verified

    computed, verified = serve(test)
    law, identity = (entry["values"][0] for entry in computed["statements"])
    assert verified["verdict"] == "relates"
    assert (law["kind"], law["holds"]) == ("index equation", None)
    assert law["identity"] is False
    assert "it relates its tensors, as a law does" in computed["statements"][0]["notes"][0]
    assert (identity["holds"], identity["identity"]) == (True, True)
    assert computed["statements"][1]["notes"] == []


def test_compute_says_an_equation_of_index_notation_it_cannot_tell_is_not_decided():
    # Carroll's (1.69) failed inside RelativisticPy as compute described it: T_{[μνρ]σ} is
    # read as a symbol, beside a sum of tensors, and SymPy cannot subtract the two sides
    carroll = (
        r"T_{[\mu\nu\rho]\sigma} = {1\over 6}\left(T_{\mu\nu\rho\sigma} - T _{\mu\rho\nu\sigma}"
        r" + T_{\rho\mu\nu\sigma} - T_{\nu\mu\rho\sigma} + T_{\nu\rho\mu\sigma}"
        r" - T_{\rho\nu\mu\sigma} \right)"
    )

    async def test(client, pool):
        return await call(client, "compute", source=carroll)

    computed = serve(test)
    [entry] = computed["statements"]
    assert entry["error"] is None
    [value] = entry["values"]
    assert (value["kind"], value["holds"], value["identity"]) == ("index equation", None, None)
    assert "is an identity of index notation is not decided" in entry["notes"][0]


def test_compute_verify_and_check_give_an_error_alike():
    claims = [r"\frac{1}{2 = x", r"\lim_{x \to \infty} \sin(x) = 0", r"e^x \geq 1 + x"]
    notebook = "\n\n".join(f"$${claim}$$" for claim in claims)

    async def test(client, pool):
        computed = await call(client, "compute", source="\n".join(claims))
        verified = [await call(client, "verify", claim=claim) for claim in claims]
        checked = await call(client, "check", document=notebook, format="markdown", all=True)
        return computed, verified, checked

    computed, verified, checked = serve(test)

    def essence(error):
        return {key: error[key] for key in ("type", "message", "text", "notes", "hint")}

    errors = [essence(entry["error"]) for entry in computed["statements"]]
    assert [error["type"] for error in errors] == [
        "ParseError",
        "EvaluationError",
        "UndecidedError",
    ]
    assert [essence(result["links"][0]["error"]) for result in verified] == errors
    assert [essence(finding["error"]) for finding in checked["findings"]] == errors
    assert [result["verdict"] for result in verified] == ["error", "error", "not decided"]


def test_verify_refuses_what_is_no_claim_with_an_error_as_well():
    async def test(client, pool):
        expression = await call(client, "verify", claim="x^2 + 1")
        definition = await call(client, "verify", claim="E = m c^2")
        return expression, definition

    for result in serve(test):
        [link] = result["links"]
        assert result["verdict"] == "error"
        assert link["error"]["type"] == "UsageError"
        assert (link["error"]["message"], link["error"]["hint"]) == (link["detail"], link["hint"])
        assert link["error"]["column"] == 1


# -- check ---------------------------------------------------------------------------------------

PAPER = r"""\documentclass{article}
\begin{document}
The metric is
\begin{equation} ds^2 = -dt^2 + a(t)^2 dx^2 \end{equation}
so that
\begin{equation} \Gamma^x_{tx} = \frac{\dot a}{a} \end{equation}
and, mistakenly,
\begin{equation} \Gamma^t_{xx} = a \end{equation}
\end{document}
"""


def test_check_reports_on_a_paper_given_as_text():
    async def test(client, pool):
        failing = await call(client, "check", document=PAPER, name="paper.tex")
        everything = await call(client, "check", document=PAPER, name="paper.tex", all=True)
        strict = await call(client, "check", document=PAPER, strict=True)
        relaxed = await call(client, "check", document=PAPER, fail_on=["not parsed"])
        part = await call(client, "check", document=PAPER, lines="8", all=True)
        notebook = await call(
            client, "check", document="# Notes\n\n$$ 2 + 2 = 5 $$\n", format="markdown"
        )
        neither = await call(client, "check")
        bad_lines = await call(client, "check", document=PAPER, lines="nine")
        return failing, everything, strict, relaxed, part, notebook, neither, bad_lines

    failing, everything, strict, relaxed, part, notebook, neither, bad_lines = serve(test)
    assert failing["counts"] == {"holds": 1, "differs": 1, "defines": 1}
    (finding,) = failing["findings"]
    assert (finding["file"], finding["line"], finding["column"]) == ("paper.tex", 8, 18)
    assert finding["verdict"] == "differs"
    assert finding["statement"] == r"\Gamma^t_{xx} = a"
    assert finding["difference"]["text"] == "a(t)*a'(t) - a(t)"
    assert failing["listed"] == "failing"
    assert "all=true lists every statement" in failing["notes"][0]
    assert [f["verdict"] for f in everything["findings"]] == ["defines", "holds", "differs"]
    assert strict["failed"]
    assert strict["policy"] == [
        "differs",
        "empty",
        "ill-formed",
        "not parsed",
        "time limit",
    ]
    assert not relaxed["failed"]
    assert relaxed["policy"] == ["not parsed"]
    assert [f["verdict"] for f in part["findings"]] == ["differs"]
    assert part["context"] == 2
    assert notebook["counts"] == {"differs": 1}
    assert notebook["findings"][0]["line"] == 3
    assert isinstance(neither, Failure)
    assert "one of them" in neither
    assert isinstance(bad_lines, Failure)
    assert "--lines takes line numbers" in bad_lines


def test_check_reads_files_only_in_the_folders_allowed(tmp_path):
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "paper.tex").write_text(PAPER, encoding="utf-8")
    (papers / "chapter.tex").write_text("\\begin{equation} 1 + 1 = 2 \\end{equation}\n")
    (papers / "main.tex").write_text("\\input{chapter}\n\\input{../secret}\n")
    (tmp_path / "secret.tex").write_text("\\begin{equation} s = 1 \\end{equation}\n")
    (papers / "link.tex").symlink_to(tmp_path / "secret.tex")
    (papers / "notes.md").write_text("$$ 2 + 2 = 4 $$\n")

    async def test(client, pool):
        paths = {
            "absolute": str(papers / "paper.tex"),
            "relative": "paper.tex",
            "markdown": "notes.md",
            "outside": str(tmp_path / "secret.tex"),
            "climbing": str(papers / ".." / "secret.tex"),
            "linked": str(papers / "link.tex"),
            "including": str(papers / "main.tex"),
            "missing": str(papers / "none.tex"),
            "folder": str(papers),
        }
        return {name: await call(client, "check", path=path) for name, path in paths.items()}

    def refused():
        async def test(client, pool):
            return await call(client, "check", path=str(papers / "paper.tex"))

        return serve(test)

    results = serve(test, allowed=(papers.resolve(),))
    assert results["absolute"]["counts"] == {"holds": 1, "differs": 1, "defines": 1}
    assert results["absolute"]["file"] == str((papers / "paper.tex").resolve())
    assert results["relative"]["counts"] == results["absolute"]["counts"]
    assert results["markdown"]["counts"] == {"holds": 1}
    for name in ("outside", "climbing", "linked"):
        assert isinstance(results[name], Failure), name
        assert "is not in a folder this server may read" in results[name], name
    assert "secret.tex is outside the folders this server may read" in results["including"]
    assert results["missing"].startswith("Error executing tool check: no file")
    assert "is a folder, not a file" in results["folder"]
    without = refused()
    assert "this server reads no files" in without
    assert "--allow-files" in without


# -- the bounds ----------------------------------------------------------------------------


@POSIX
def test_a_statement_over_the_time_limit_stops_and_the_session_keeps_its_definitions():
    async def test(client, pool):
        started = time.monotonic()
        slow = await call(
            client, "compute", source=f"y := 3\n{SLOW}\ny + 1", session="s", time_limit=1
        )
        seconds = time.monotonic() - started
        later = await call(client, "compute", source="y^2", session="s")
        return slow, seconds, later

    slow, seconds, later = serve(test)
    assert seconds < 5
    assert not slow["ok"]
    defined, over = slow["statements"]
    assert defined["error"] is None
    assert over["error"]["type"] == "NotComputedError"
    assert over["error"]["message"] == "over the time limit of 1 s"
    assert over["statement"] == SLOW
    assert over["line"] == 2
    assert over["error"]["notes"] == ["the definitions before it are kept"]
    assert slow["notes"] == [
        "the time limit of 1 s was reached: 1 statement, from <input 1>:3:1 on, did not run"
    ]
    assert values(later) == [["9"]]


@POSIX
def test_a_check_or_a_claim_over_the_time_limit_says_what_it_did():
    paper = (
        "\\begin{equation} 1 + 1 = 2 \\end{equation}\n"
        f"\\begin{{equation}} {SLOW} \\end{{equation}}\n"
        "\\begin{equation} 2 + 2 = 4 \\end{equation}\n"
    )

    async def test(client, pool):
        started = time.monotonic()
        report = await call(client, "check", document=paper, all=True, time_limit=2)
        claim = await call(client, "verify", claim=f"{SLOW} = 1", time_limit=1)
        return time.monotonic() - started, report, claim

    seconds, report, claim = serve(test)
    assert seconds < 8  # not the 30 s each statement of a check may take
    assert [finding["verdict"] for finding in report["findings"]] == ["holds", "not computed"]
    assert report["findings"][1]["reason"] == "time limit"
    assert report["notes"] == [
        "the time limit of 2 s was reached: the equations from <document>:3 on were not "
        "checked; give it longer, or check fewer lines at a time, with lines"
    ]
    assert claim["verdict"] == "not decided"
    assert claim["links"][0]["reason"] == "time limit"


@POSIX
def test_a_worker_that_does_not_stop_is_stopped(monkeypatch):
    monkeypatch.setattr(server, "GRACE", 0.5)

    async def test(client, pool):
        await call(client, "compute", source="y := 3", session="frozen")
        os.kill(pool.sessions["frozen"].worker.pid, signal.SIGSTOP)  # deaf to every alarm
        started = time.monotonic()
        stopped = await call(client, "compute", source="y + 1", session="frozen", time_limit=1)
        seconds = time.monotonic() - started
        listed = await call(client, "list_sessions")
        anew = await call(client, "compute", source="y + 1", session="frozen")
        return stopped, seconds, listed, anew

    stopped, seconds, listed, anew = serve(test)
    assert isinstance(stopped, Failure)
    assert seconds < 5
    message, note, hint = stopped.removeprefix("Error executing tool compute: ").split("\n")
    assert message == (
        "the computation took longer than its time limit of 1 s, and did not stop when "
        "asked to, so it was stopped"
    )
    assert note == "note: the session frozen was stopped with it, and what it defined is lost"
    assert hint.startswith("hint: run its definitions again; give it longer with time_limit")
    assert listed["sessions"] == []
    assert values(anew) == [["y + 1"]]  # a new session, which knows no y


@POSIX
def test_a_call_that_is_cancelled_interrupts_its_computation():
    async def test(client, pool):
        await call(client, "compute", source="y := 3", session="s")
        started = time.monotonic()
        with anyio.move_on_after(1):
            await call(client, "compute", source=SLOW, session="s")
        after = await call(client, "compute", source="y^2", session="s")
        return time.monotonic() - started, after

    seconds, after = serve(test)
    assert seconds < 5
    assert values(after) == [["9"]]


def test_a_result_too_long_is_cut_with_a_note():
    async def test(client, pool):
        return await call(client, "compute", source="expand((x + 1)^300); x := 2; x^2")

    result = serve(test, max_output=2000)
    assert result["truncated"]
    assert result["ok"]
    assert len(json.dumps(result, ensure_ascii=False, indent=2)) <= 2000
    assert result["notes"][-1].startswith("the result was cut to fit 2,000 characters")
    assert "characters]" in result["statements"][0]["values"][0]["text"]


@pytest.mark.parametrize("limit", [1000, 1100, 1500, 2000, 2500, 4000])
def test_a_result_cut_to_size_fits_it_with_its_note(limit):
    entry = {"statement": "x" * 900, "values": [{"text": "y" * 3000}] * 3, "notes": ["n"] * 40}
    result = {"statements": [entry] * 30, "ok": True, "notes": [], "truncated": False}
    cut = server._capped(result, limit)
    assert cut["truncated"]
    assert server._size(cut) <= limit
    assert cut["notes"][-1].startswith(f"the result was cut to fit {limit:,} characters")


def test_input_too_long_is_refused(monkeypatch):
    monkeypatch.setattr(server, "MAX_INPUT", 10)

    async def test(client, pool):
        return await call(client, "compute", source="x + y + z + w")

    assert "more than the 10 this server takes" in serve(test)


def test_the_settings_are_checked():
    with pytest.raises(server.UsageError, match="positive number of seconds"):
        server.Settings(time_limit=0)
    with pytest.raises(server.UsageError, match="1000 or more"):
        server.Settings(max_output=10)
    assert server.Settings(time_limit=10).limit(None) == 10
    assert server.Settings(time_limit=10).limit(100) == 10
    assert server.Settings(time_limit=10).limit(2) == 2


# -- help, resources and prompts ----------------------------------------------------------------


def test_help_is_the_documentation_offline():
    async def test(client, pool):
        index = await call(client, "help")
        language = await call(client, "help", topic="reference")
        alias = await call(client, "help", topic="check")
        unknown = await call(client, "help", topic="tensers")
        function = await call(client, "help", topic="besselj")
        misspelled = await call(client, "help", topic="besselk2")
        return index, language, alias, unknown, function, misspelled

    index, language, alias, unknown, function, misspelled = serve(test)
    for topic, title in documentation.topics():
        assert f"- {topic}: {title}" in index
    assert "g_{\\mu\\nu} := Schwarzschild(M)" in index
    assert "help with the name of a function, as besselj" in index
    assert language == documentation.page("reference")
    assert language.startswith("# The language reference")
    assert alias == documentation.page("checking")
    assert isinstance(unknown, Failure)
    assert "there is no help on 'tensers'" in unknown
    assert "did you mean tensors?" in unknown
    assert function == documentation.page("besselj")
    assert function.startswith("# besselj\n")
    assert isinstance(misspelled, Failure)
    assert "did you mean besselk?" in misspelled


def test_every_page_is_a_resource():
    async def test(client, pool):
        listed = (await client.list_resources()).resources
        templates = (await client.list_resource_templates()).resource_templates
        read = await client.read_resource("relativisticpy://docs/reference")
        alias = await client.read_resource("relativisticpy://docs/check")
        function = await client.read_resource("relativisticpy://docs/besselj")
        with pytest.raises(Exception, match="there is no help on 'nothing'"):
            await client.read_resource("relativisticpy://docs/nothing")
        return listed, templates, read, alias, function

    listed, templates, read, alias, function = serve(test)
    assert function.contents[0].text == documentation.page("besselj")
    assert {str(resource.uri) for resource in listed} == {
        f"relativisticpy://docs/{topic}" for topic, _ in documentation.topics()
    }
    assert all(resource.mime_type == "text/markdown" for resource in listed)
    assert [template.uri_template for template in templates] == ["relativisticpy://docs/{topic}"]
    assert read.contents[0].text == documentation.page("reference")
    assert alias.contents[0].text == documentation.page("checking")


def test_the_prompts_start_the_work_they_are_named_after():
    async def test(client, pool):
        listed = (await client.list_prompts()).prompts
        derivation = await client.get_prompt(
            "verify_derivation", {"derivation": "a = b\nb = c", "context": "Reals := [a]"}
        )
        paper = await client.get_prompt("check_paper", {"document": "paper.tex"})
        return listed, derivation, paper

    listed, derivation, paper = serve(test)
    assert {prompt.name for prompt in listed} == {"verify_derivation", "check_paper"}
    text = derivation.messages[0].content.text
    assert "verify tool" in text
    assert "a = b\nb = c" in text
    assert "Reals := [a]" in text
    assert "check tool" in paper.messages[0].content.text


# -- the command ----------------------------------------------------------------------------------


def test_a_client_starts_the_server_as_a_command():
    parameters = StdioServerParameters(
        command=sys.executable, args=["-m", "relativisticpy", "mcp"], cwd=str(ROOT)
    )

    async def main():
        async with Client(parameters) as client:
            tools = await client.list_tools()
            result = await client.call_tool(
                "compute", {"source": f"{SCHWARZSCHILD}; R_{{abcd}} R^{{abcd}}"}
            )
            return [tool.name for tool in tools.tools], result

    names, result = anyio.run(main)
    assert "compute" in names
    assert values(result.structured_content) == [[], ["48*M^2/r^6"]]
    assert json.loads(result.content[0].text) == result.structured_content


def test_the_command_has_help_of_its_own(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["mcp", "--help"])
    assert stopped.value.code == 0
    out = capsys.readouterr().out
    for option in (
        "--allow-files",
        "--time-limit",
        "--max-output",
        "--max-sessions",
        "--memory-limit",
    ):
        assert option in out
    assert "claude mcp add relativisticpy -- relativisticpy mcp" in out
    assert "relativisticpy mcp" in cli.help_text(prompt=False)
    ran = subprocess.run(
        [sys.executable, "-m", "relativisticpy.mcp", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ran.returncode == 0
    assert "usage: relativisticpy mcp [options]" in ran.stdout


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--time-limit", "0"], "takes a positive number of seconds"),
        (["--max-output", "ten"], "takes a positive number of characters"),
        (["--allow-files", "/no/such/folder"], "--allow-files takes a folder"),
        (["--max-output", "10"], "max_output is 1000 or more"),
    ],
)
def test_a_command_line_that_is_wrong_is_refused(capsys, arguments, message):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["mcp", *arguments])
    assert stopped.value.code == 2
    assert message in capsys.readouterr().err


def test_the_folders_allowed_are_given_on_the_command_line(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "papers").mkdir()
    command = server.parser()
    here = server.settings_from(command.parse_args(["--allow-files"]), command)
    both = server.settings_from(
        command.parse_args(["--allow-files", "papers", "--allow-files", str(tmp_path)]), command
    )
    assert here.allowed == (tmp_path.resolve(),)
    assert both.allowed == ((tmp_path / "papers").resolve(), tmp_path.resolve())
    assert server.settings_from(command.parse_args([]), command).allowed == ()
    unlimited = command.parse_args(["--memory-limit", "0"])
    assert server.settings_from(unlimited, command).memory_limit is None


def test_without_the_sdk_the_command_says_what_to_install(monkeypatch, capsys):
    import importlib.util

    found = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *rest: None if name == "mcp" else found(name)
    )
    assert cli.main(["mcp"]) == 1
    assert 'pip install "relativisticpy[mcp]"' in capsys.readouterr().err


# -- the documentation ---------------------------------------------------------------------------


def launches(arguments):
    """Whether a client's command starts the server: ``relativisticpy mcp``, by itself or
    by uvx, with options the server takes."""
    if arguments[0] == "uvx":
        assert arguments[1] == "--from"
        assert arguments[2].startswith("relativisticpy[mcp] @ git+https://github.com/")
        arguments = arguments[3:]
    assert arguments[:2] == ["relativisticpy", "mcp"]
    server.parser().parse_args(arguments[2:])
    return True


def test_every_configuration_in_the_docs_starts_the_server():
    page = (ROOT / "docs" / "mcp.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```json\n(.*?)```", page, re.DOTALL)
    assert len(blocks) == 3
    for block in blocks:
        configuration = json.loads(block)
        servers = configuration.get("mcpServers") or configuration.get("servers")
        for entry in servers.values():
            assert launches([entry["command"], *entry["args"]])
    added = re.findall(r"^ *claude mcp add relativisticpy -- (.*)$", page + readme, re.MULTILINE)
    assert len(added) == 4
    for line in added:
        assert launches(shlex.split(line))
