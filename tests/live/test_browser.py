r"""The editor in a browser: a marimo notebook with the editor, served by ``marimo run``
and driven by Playwright in Chromium, from editor_in_browser.cjs. A program is typed, its
results are typeset, an error is marked and explained under the pointer, a part of a
statement is computed from the menu a right click opens, and another cell reacts.

It needs marimo (``uv run --group notebooks``), Node and Node's Playwright with a
Chromium, as CI installs them, and is skipped without them, unless
``RELATIVISTICPY_BROWSER_TESTS=required`` asks for it to fail instead.
"""

import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

HERE = Path(__file__).parent
REQUIRED = os.environ.get("RELATIVISTICPY_BROWSER_TESTS") == "required"

NOTEBOOK = r"""
import marimo

app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from relativisticpy import Workbook
    from relativisticpy.live import Editor, show

    return Editor, Workbook, mo, show


@app.cell
def _(Editor, Workbook, mo):
    wb = Workbook()
    wb.expr(r"g_{\mu\nu} := Schwarzschild(M)")
    editor = mo.ui.anywidget(Editor(wb, source=r"\Gamma^{r}_{\theta\theta}"))
    editor
    return (editor,)


@app.cell
def _(editor, mo):
    mo.md(f"statements: {len(editor.value.get('statements', []))}")
    return


@app.cell
def _(show):
    show("x := 2\nx^2 + 1\n\\frca{1}{2}")
    return
"""


def missing(why):
    """Skip for want of a tool, unless the browser tests are required."""
    if REQUIRED:
        pytest.fail(why)
    pytest.skip(why)


def playwright_modules():
    """Where Node's Playwright is installed: as NODE_PATH, the global modules or the
    ``playwright`` command name it."""
    places = [Path(p) for p in os.environ.get("NODE_PATH", "").split(os.pathsep) if p]
    npm = shutil.which("npm")
    if npm:
        ran = subprocess.run([npm, "root", "-g"], capture_output=True, text=True, check=False)
        if ran.returncode == 0 and ran.stdout.strip():
            places.append(Path(ran.stdout.strip()))
    command = shutil.which("playwright")
    if command:  # node_modules/.bin/playwright, or a link to node_modules/playwright/cli.js
        places += [Path(command).parent.parent, Path(command).resolve().parent.parent]
    for place in places:
        if (place / "playwright" / "package.json").is_file():
            return place
    return None


def chromiums():
    """Executables of Chromium to try, after Playwright's own."""
    found = [os.environ[name] for name in ("CHROMIUM",) if os.environ.get(name)]
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        path = shutil.which(name)
        if path:
            found.append(path)
    cache = Path.home() / ".cache" / "ms-playwright"
    found += [str(p) for p in sorted(cache.glob("chromium-*/chrome-linux*/chrome"), reverse=True)]
    return found


def free_port():
    with contextlib.closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def tools():
    marimo = pytest.importorskip("marimo", reason="marimo comes with uv run --group notebooks")
    pytest.importorskip("anywidget", reason="the editor needs the live extra, with anywidget")
    node = shutil.which("node")
    if node is None:
        missing("the browser test needs Node")
    modules = playwright_modules()
    if modules is None:
        missing("the browser test needs Node's Playwright: npm install -g playwright")
    return marimo, node, modules


@pytest.fixture(scope="module")
def notebook(tools, tmp_path_factory):
    """The notebook, served by marimo run, as a reader opens it: its address."""
    folder = tmp_path_factory.mktemp("notebook")
    (folder / "live.py").write_text(NOTEBOOK.lstrip(), encoding="utf-8")
    port = free_port()
    command = [sys.executable, "-m", "marimo", "run", "live.py", "--headless", "--no-token"]
    command += ["--host", "127.0.0.1", "--port", str(port)]
    served = subprocess.Popen(
        command, cwd=folder, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    address = f"http://127.0.0.1:{port}/"
    deadline = time.monotonic() + 90
    try:
        while True:
            try:
                with urllib.request.urlopen(address, timeout=5) as answer:
                    if answer.status == 200:
                        break
            except OSError:
                if served.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(served.stderr.read().decode()[-2000:]) from None
                time.sleep(0.5)
        yield address
    finally:
        served.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            served.wait(timeout=20)
        if served.poll() is None:
            served.kill()
        served.stderr.close()


@pytest.fixture(scope="module")
def seen(tools, notebook):
    """What the browser saw of the editor and the notebook."""
    _, node, modules = tools
    environment = {**os.environ, "NODE_PATH": str(modules)}
    script = HERE / "editor_in_browser.cjs"
    ran = subprocess.run(
        [node, str(script), notebook, *chromiums()],
        capture_output=True,
        text=True,
        env=environment,
        timeout=600,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    observed = json.loads(ran.stdout.strip().splitlines()[-1])
    if "skip" in observed:
        missing(f"the browser test needs a Chromium that Playwright can start: {observed['skip']}")
    assert "failure" not in observed, f"{observed.pop('failure')}, having seen {observed}"
    return observed


def test_the_editor_typesets_what_is_typed(seen):
    assert seen["errors"] == []
    assert seen["keysSeen"] == 0  # the notebook's own shortcuts never take the editor's keys
    assert seen["first"] == ["2 M - r"]
    assert seen["statements"] == 4
    assert seen["written"][:3] == [
        r"\Gamma^{r}_{\theta\theta}",
        "K := R_{abcd} R^{abcd}",
        "K r^6",
    ]
    assert seen["kretschmann"] == ["48 M^{2}"]
    assert seen["definition"] == "defines K"
    assert seen["math"] >= 5


def test_an_error_is_marked_and_explained_under_the_pointer(seen):
    assert seen["error"] == "ParseError: unknown LaTeX command \\frca"
    assert seen["marked"] == ["\\frca"]
    assert "unknown LaTeX command \\frca" in seen["tip"]
    assert "hint: did you mean \\frac?" in seen["tip"]


def test_a_right_click_computes_a_part_of_a_statement(seen):
    assert seen["menu"] == ["Compute", "Simplify", "Expand", "Does it hold?", "Copy LaTeX"]
    assert seen["answer"] == {
        "asked": "Simplified",
        "part": ["R_{abcd} R^{abcd}"],
        "values": [r"\frac{48 M^{2}}{r^{6}}"],
        "under": 1,
    }
    assert seen["resultMenu"][-1] == "Copy text"


def test_another_cell_reacts_to_what_the_editor_computes(seen):
    assert seen["reacted"] == 1


def test_show_is_typeset_by_marimo(seen):
    assert seen["shown"] == 3
    assert seen["shownError"].startswith("ParseError: unknown LaTeX command \\frca")
