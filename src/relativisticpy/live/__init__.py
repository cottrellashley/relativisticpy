r"""Live mathematics: programs shown in a notebook as they are computed.

:func:`show` runs a program and gives back what a notebook shows: each statement typeset
with its results, its notes and its errors, in place, as ``relativisticpy --render``
typesets a Markdown notebook, but as HTML, in marimo and in Jupyter::

    from relativisticpy.live import show

    show(r'''
        g_{\mu\nu} := Schwarzschild(M)
        R_{abcd} R^{abcd}
    ''')

:class:`~relativisticpy.live.editor.Editor` is a widget, for marimo, Jupyter and VS Code:
the source on the left, each statement typeset with its results on the right, computed
as it is typed, in a process of its own, with a time limit. It needs the ``live`` extra,
which adds anywidget: ``pip install "relativisticpy[live]"``. ``relativisticpy help
live`` says more.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from relativisticpy import display
from relativisticpy.errors import RelativisticPyError, UsageError
from relativisticpy.language import Source, Span
from relativisticpy.language import syntax as ast
from relativisticpy.workbook import Step, Workbook

if TYPE_CHECKING:
    from relativisticpy.live.editor import Editor

__all__ = ["Editor", "Shown", "show"]


def __getattr__(name: str) -> Any:
    """The :class:`~relativisticpy.live.editor.Editor`, imported when it is first asked
    for, since it needs anywidget, which ``show`` does not."""
    if name != "Editor":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from relativisticpy.live.editor import Editor
    except ImportError as error:
        if error.name not in ("anywidget", "ipywidgets", "traitlets"):
            raise
        raise ImportError(
            'the editor needs the live extra, with anywidget: pip install "relativisticpy[live]"'
        ) from error
    return Editor


def show(
    source: str | Source,
    *,
    workbook: Workbook | None = None,
    name: str = "<input>",
    **objects: Any,
) -> Shown:
    r"""Run a program and show each statement typeset with its results, its notes and its
    errors, as ``relativisticpy --render`` does, but as HTML, in marimo and in Jupyter.

    The statements run one at a time, as :meth:`Workbook.each` runs them: one that
    fails, or cannot be read, shows its error where it is, as
    :meth:`~relativisticpy.errors.RelativisticPyError.render` gives it, and those after
    it still run. With ``workbook``, the program runs in that workbook, so that what one
    call defines, the next uses, as from one cell to the next; without one, it runs in a
    new workbook, which :attr:`Shown.workbook` keeps. ``name`` is what the program is
    called where an error says where it is.

        >>> shown = show("x := 2; x^2 + 1")
        >>> shown.values
        [5]

    ``objects``, Python objects passed by keyword, as in
    ``show(r"\hat{x} := \mathcal{F}[\texttt{signal}]", signal=signal)``, are the place
    of a later version, in which ``\texttt{signal}`` reads the object called ``signal``
    (see ``design/integrations.md``). This one does not read them yet, and refuses them,
    rather than run a program that names them without them. Data from Python can be
    bound to a workbook today, as in ``wb["x_t"] = signal``, and the workbook passed.
    """
    if objects:
        named = ", ".join(sorted(objects))
        raise UsageError(
            f"show does not read Python objects yet, so it cannot take {named}",
            notes=[r"reading an object by name, as \texttt{signal}, comes in a later version"],
            hint="bind the data to a workbook, as in wb['x_t'] = signal, and pass it: "
            "show(source, workbook=wb)",
        )
    if not isinstance(source, str | Source):
        raise UsageError(f"show takes the program as a str, not {type(source).__name__}")
    if workbook is not None and not isinstance(workbook, Workbook):
        raise UsageError(f"workbook is a Workbook, not {type(workbook).__name__}")
    workbook = workbook if workbook is not None else Workbook()
    text = source if isinstance(source, Source) else Source(source, name)
    from relativisticpy.cli import _defined, _defines_text, _snapshot

    steps: list[Step] = []
    defines: list[str | None] = []
    try:
        before = _snapshot(workbook)
        for step in workbook.each(text, recover=True):
            after = _snapshot(workbook)
            defined = _defined(before, after) if step.error is None else []
            steps.append(step)
            defines.append(_defines_text(defined, "unicode") if defined else None)
            before = after
    except RelativisticPyError as error:  # a program that cannot be run at all
        span = Span(text, 0, len(text.text))
        steps.append(Step(span, ast.Statement(span=span), error=error))
        defines.append(None)
    return Shown(text, tuple(steps), workbook, tuple(defines))


@dataclass(frozen=True, eq=False)
class Shown:
    """A program run by :func:`show`, and what a notebook shows of it.

    Its ``steps`` are the :class:`~relativisticpy.workbook.Step` of each statement, as
    :meth:`Workbook.each` gives them; :attr:`values` and :attr:`errors` are theirs, in
    order, and the ``workbook`` it ran in keeps what it defined. In marimo and Jupyter it
    shows itself: each statement typeset with its results, what it defines, its notes and
    its error, as :meth:`html` gives them; ``repr`` gives the same as text.
    """

    source: Source
    steps: tuple[Step, ...]
    workbook: Workbook
    defines: tuple[str | None, ...]  # what each step defines, in words, if anything

    @property
    def values(self) -> list[Any]:
        """The value of every expression, in order, as :meth:`Workbook.exe` gives them,
        but carrying on past the statements that failed."""
        return [value for step in self.steps for value in step.values or []]

    @property
    def errors(self) -> list[RelativisticPyError]:
        """The error of every statement that failed, in order."""
        return [step.error for step in self.steps if step.error is not None]

    @property
    def ok(self) -> bool:
        """Whether every statement ran."""
        return not self.errors

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def html(self, math: Callable[[str], str] | None = None) -> str:
        """The program as HTML: each statement typeset with its results, what it defines,
        its notes and its error. ``math`` gives the HTML of a displayed formula from its
        LaTeX; by default ``\\[ ... \\]``, which MathJax typesets in Jupyter."""
        formula = math or _mathjax
        parts = [f"<div class='rp-shown'><style>{_STYLE}</style>"]
        for step, defines in zip(self.steps, self.defines, strict=True):
            parts.append(_step_html(step, defines, formula))
        parts.append("</div>")
        return "".join(parts)

    def _repr_html_(self) -> str:
        return self.html()

    def _display_(self) -> Any:
        """What marimo shows: :meth:`html`, its formulas typeset by marimo's own KaTeX."""
        import marimo as mo

        return mo.Html(self.html(lambda latex: str(mo.md("\\[" + latex + "\\]").text)))

    def __repr__(self) -> str:
        """Each statement and what came of it, as text, as the command line prints it."""
        lines: list[str] = []
        for step, defines in zip(self.steps, self.defines, strict=True):
            if step.error is not None:
                lines.extend(step.error.render().split("\n"))
                continue
            lines.append(" ".join(step.span.text.split()))
            for value in step.values or []:
                lines.extend("  " + line for line in display.unicode(value).split("\n"))
            if defines:
                lines.append(f"  {defines}")
            lines.extend(f"  note: {note}" for note in step.notes)
        return "\n".join(lines)


#: How a program shown looks, in light and dark themes alike.
_STYLE = (
    ".rp-shown .rp-step{border-left:3px solid rgba(127,127,127,.35);margin:.5em 0;"
    "padding:0 0 0 .8em}"
    ".rp-shown .rp-step.rp-failed{border-left-color:#d32f2f}"
    ".rp-shown .rp-value{margin-left:1.2em}"
    ".rp-shown .rp-said{margin:.2em 0 .2em 1.2em;opacity:.72;font-size:.9em}"
    ".rp-shown .rp-code,.rp-shown .rp-error{font-family:monospace;white-space:pre;"
    "overflow-x:auto;margin:.3em 0}"
    ".rp-shown .rp-error{color:#d32f2f}"
)


def _mathjax(latex: str) -> str:
    """A displayed formula for MathJax, as Jupyter typesets it."""
    return "\\[" + html.escape(latex, quote=False) + "\\]"


def _written(step: Step) -> str:
    """The LaTeX a statement is typeset from: as it is written, as ``--render`` leaves
    it, in an alignment where it has one, as a row of ``\\begin{aligned}`` has."""
    text = step.span.text
    if "&" in text.replace("\\&", ""):
        return "\\begin{aligned}" + text + "\\end{aligned}"
    return text


def _step_html(step: Step, defines: str | None, formula: Callable[[str], str]) -> str:
    """One statement as HTML, with what came of it."""
    failed = step.error is not None
    parts = [f"<div class='rp-step{' rp-failed' if failed else ''}'>"]
    if type(step.statement) is ast.Statement:  # a statement that could not be read
        parts.append(f"<div class='rp-code'>{html.escape(step.span.text)}</div>")
    else:
        parts.append(f"<div class='rp-written'>{formula(_written(step))}</div>")
    for value in step.values or []:
        parts.append(f"<div class='rp-value'>{_value_html(value, formula)}</div>")
    if defines:
        parts.append(f"<div class='rp-said'>{html.escape(defines)}</div>")
    for note in step.notes:
        parts.append(f"<div class='rp-said'>note: {html.escape(note)}</div>")
    if step.error is not None:
        parts.append(f"<div class='rp-error'>{html.escape(step.error.render())}</div>")
    parts.append("</div>")
    return "".join(parts)


def _value_html(value: Any, formula: Callable[[str], str]) -> str:
    """A result as HTML: a table of numbers, or else typeset."""
    table = display.jupyter_html(value)
    return table if table is not None else formula(display.latex(value))
