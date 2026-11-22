r"""The editor: mathematics computed as it is typed, in a widget for marimo, Jupyter and
VS Code.

The source is on the left, in LaTeX, plain text or Unicode, and on the right each
statement is typeset with its results, what it defines, its notes and its error. About
300 ms after typing stops, the program runs, in a worker process of its own, from the
first statement that changed, in a copy of the workbook the editor was given, with a
time limit (see :mod:`relativisticpy.live.worker`). An error is underlined where it is,
and its notes and hint show under the pointer. A part of a statement, selected, can be
computed, simplified, expanded or judged, and its LaTeX copied, from the menu a right
click opens; a result has the same menu.

It typesets with Temml 0.14.0, which it carries, under its MIT licence, in
``relativisticpy/live/static/temml``: nothing is loaded from the network.
"""

from __future__ import annotations

import base64
import sys
import threading
from pathlib import Path
from typing import Any

import anywidget
import traitlets

from relativisticpy.errors import UsageError
from relativisticpy.live.worker import (
    GRACE,
    MEMORY_LIMIT,
    STARTUP,
    TIME_LIMIT,
    Computations,
    Deliver,
    InPlace,
    Job,
    portable,
)
from relativisticpy.workbook import Workbook

__all__ = ["Editor"]

#: Where the widget's script, style and typesetter are.
STATIC = Path(__file__).parent / "static"

#: The version of Temml the editor carries.
TEMML = "0.14.0"


def _module() -> str:
    """The widget's module: Temml, then the editor, which typesets with it."""
    temml = (STATIC / "temml" / "temml.min.js").read_text(encoding="utf-8")
    editor = (STATIC / "editor.js").read_text(encoding="utf-8")
    return f"{temml}\n{editor}"


def _style() -> str:
    """The widget's style: Temml's, with its font of script letters inside it, then the
    editor's."""
    font = base64.b64encode((STATIC / "temml" / "Temml.woff2").read_bytes()).decode("ascii")
    temml = (STATIC / "temml" / "Temml-Local.css").read_text(encoding="utf-8")
    temml = temml.replace("url('Temml.woff2')", f"url(data:font/woff2;base64,{font})")
    editor = (STATIC / "editor.css").read_text(encoding="utf-8")
    return f"{temml}\n{editor}"


class Editor(anywidget.AnyWidget):
    r"""An editor in which mathematics is computed as it is typed, for marimo, Jupyter and
    VS Code.

    ``workbook`` is the session each run starts from, a copy of it as it is when the run
    starts, so that a half-typed definition never reaches it; without one, an empty
    workbook. ``source`` is the program to start with. Each run, and each part computed,
    may take ``time_limit`` seconds, and its worker ``memory_limit`` MiB, where the system
    enforces it (``None`` for no limit).

    In marimo, wrap it, so that other cells react to what it computes::

        editor = mo.ui.anywidget(Editor(wb))
        editor.value["statements"]

    In Jupyter, it shows itself. Its state is data: ``source``, and ``statements``, one
    for each statement of the program as last run, as ``relativisticpy --format json``
    gives it, each value as :func:`relativisticpy.display.data` describes it, with where
    the statement and its error are in the source. From Python, :meth:`run` computes the
    source at once, and :meth:`compute` a part of it.

    Python objects, read by name as in ``\texttt{signal}``, come in a later version, with
    the scope a workbook is given (see ``design/integrations.md``).
    """

    _esm = _module()
    _css = _style()

    source = traitlets.Unicode("").tag(sync=True)
    statements = traitlets.List(traitlets.Dict()).tag(sync=True)

    def __init__(
        self,
        workbook: Workbook | None = None,
        *,
        source: str = "",
        time_limit: float = TIME_LIMIT,
        memory_limit: int | None = MEMORY_LIMIT,
        **options: Any,
    ) -> None:
        if workbook is not None and not isinstance(workbook, Workbook):
            raise UsageError(f"an editor's workbook is a Workbook, not {type(workbook).__name__}")
        if not isinstance(source, str):
            raise UsageError(f"an editor's source is a str, not {type(source).__name__}")
        if isinstance(time_limit, bool) or not isinstance(time_limit, int | float):
            raise UsageError(f"a time limit is a number of seconds, not {time_limit!r}")
        if not time_limit > 0:
            raise UsageError(f"a time limit is a positive number of seconds, not {time_limit}")
        super().__init__(source=source, **options)
        self.workbook = workbook if workbook is not None else Workbook()
        self.time_limit = float(time_limit)
        self._given: int | None = None  # which state of the workbook the computations have
        self._computations: Computations | InPlace
        if sys.platform == "emscripten":  # a browser's Python starts no process
            self._computations = InPlace(self.workbook, time_limit=self.time_limit)
        else:
            self._computations = Computations(time_limit=self.time_limit, memory_limit=memory_limit)
        self._computations.warm()
        self.on_msg(self._heard)

    # -- from Python ----------------------------------------------------------------------

    def run(self, source: str | None = None) -> list[dict[str, Any]]:
        """Compute the source, or ``source``, which becomes the editor's, at once, as the
        widget does once typing stops; wait for it, and give the ``statements``, which it
        also shows."""
        if source is not None:
            self.source = source
        answer = self._wait("run", {"text": self.source})
        self.statements = answer["statements"]
        return list(self.statements)

    def compute(self, start: int, end: int, action: str = "compute") -> dict[str, Any]:
        """Compute the part ``source[start:end]`` of a statement, or the smallest
        expression that holds it, in the workbook as it stands at that statement, as the
        menu of a selection does: ``compute``, ``simplify``, ``expand``, ``holds`` (does
        it hold?) or ``latex``. Its answer, as data."""
        arguments = {"text": self.source, "start": start, "end": end, "action": action}
        return self._wait("compute", arguments)

    def close(self) -> None:
        """Stop the editor's worker, and close the widget."""
        computations = getattr(self, "_computations", None)  # none, if it was never made
        if computations is not None:
            computations.close()
        super().close()

    # -- from the widget --------------------------------------------------------------------

    def _heard(self, widget: Any, content: dict[str, Any], buffers: Any) -> None:
        """A message of the widget's: a run, as typing stops, or a part to compute."""
        kind = content.get("type")
        ident = str(content.get("id", ""))
        text = str(content.get("source", ""))
        if kind == "run":
            self._submit("run", {"text": text}, self._answerer(ident))
        elif kind == "compute":
            arguments = {
                "text": text,
                "start": int(content.get("start", 0)),
                "end": int(content.get("end", 0)),
                "action": str(content.get("action", "compute")),
            }
            self._submit("compute", arguments, self._answerer(ident))

    def _answerer(self, ident: str) -> Deliver:
        """Where the answer of a widget's request goes: back to the widget, with its id."""

        def deliver(kind: str, data: dict[str, Any]) -> None:
            self.send({"type": kind, "id": ident, **data})

        return deliver

    def _submit(self, kind: str, arguments: dict[str, Any], deliver: Deliver) -> None:
        """Ask for a job, with the workbook as it is, if it has changed since it was last
        given. It is taken here, in the thread a notebook runs its cells in, where nothing
        changes it while it is taken."""
        revision = self.workbook._revision
        state = None
        if revision != self._given and isinstance(self._computations, Computations):
            try:
                state = portable(self.workbook)
            except Exception as error:  # what no other process can take is computed here
                self._computations.close()
                self._computations = InPlace(
                    self.workbook, time_limit=self.time_limit, note=_kept_here(error)
                )
        self._given = revision
        self._computations.submit(Job(kind, arguments, deliver, revision, state))

    def _wait(self, kind: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Do a job, and wait for its answer."""
        done = threading.Event()
        answer: dict[str, Any] = {}

        def deliver(_: str, data: dict[str, Any]) -> None:
            answer.update(data)
            done.set()

        self._submit(kind, arguments, deliver)
        if not done.wait(STARTUP + self.time_limit + 2 * GRACE):
            raise UsageError("the editor's worker did not answer", hint="close it and make another")
        return answer


def _kept_here(error: Exception) -> str:
    """The note of an editor that computes in the notebook's own process, and why."""
    return (
        "this editor computes in the notebook's own process, where a time limit may not "
        f"stop it: its workbook cannot be sent to a worker ({type(error).__name__}: {error})"
    )
