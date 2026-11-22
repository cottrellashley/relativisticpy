"""Write docs/examples.md: every example of examples/, run, with its results.

A Markdown example is a notebook, shown as ``relativisticpy FILE --render`` prints it,
with its results typeset under each display; a Python example is shown with what it
prints. The examples come in the order of examples/README.md, which says what each is::

    uv run python tools/examples.py      # write docs/examples.md again

The tests check that the page is what this writes.
"""

from __future__ import annotations

import ast
import contextlib
import io
import re
import runpy
import sys
from collections.abc import Mapping
from pathlib import Path

from relativisticpy import Workbook

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
PAGE = ROOT / "docs" / "examples.md"

#: A row of the table in examples/README.md: the file, and what it is.
_ROW = re.compile(r"^\| \[(?P<file>[^\]]+)\]\([^)]+\) \| (?P<what>.+?) \|$", re.MULTILINE)
_HEADING = re.compile(r"^(#+) ", re.MULTILINE)

_HEADER = """\
# Examples

> **For:** anyone who learns from worked examples. **You will learn:** what
> RelativisticPy makes of whole derivations, in General Relativity, quantum mechanics and
> finance, with every result shown.

Each example is a file in the `examples` folder of the repository. A Markdown example is
a notebook: its displays are the program, and its results are typeset under each
display, as `relativisticpy examples/NAME.md --render` prints them. A Python example is
shown with what it prints. Every example runs in the tests, and this page is written
from them, by `tools/examples.py`.
"""

_FOOTER = """\
**Next:** [Jupyter and notebooks](jupyter.md), to write notebooks of your own, or the
[guides](language.md) behind each step.
"""


def examples() -> list[tuple[str, str]]:
    """Every example, as its file name and what it is, in the order of the README."""
    return [(row["file"], row["what"]) for row in _ROW.finditer(_read(EXAMPLES / "README.md"))]


def printed(name: str) -> str:
    """What a Python example prints."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return buffer.getvalue()


def slug(title: str) -> str:
    """The anchor of a heading, as the docs site and GitHub make it."""
    return re.sub(r"[\s]+", "-", re.sub(r"[^\w\s-]", "", title).strip().lower())


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _demoted(markdown: str) -> str:
    """Markdown with each heading one level down, outside its code blocks."""
    lines, fenced = [], False
    for line in markdown.split("\n"):
        if line.startswith("```"):
            fenced = not fenced
        lines.append(line if fenced else _HEADING.sub(r"#\1 ", line))
    return "\n".join(lines)


def _notebook(name: str) -> tuple[str, str]:
    """A Markdown example, with its results, its headings one level down; and its title."""
    with Workbook() as workbook:
        rendered = workbook.markdown(_read(EXAMPLES / name), name=f"examples/{name}")
    title = rendered.split("\n", 1)[0].removeprefix("# ")
    return _demoted(rendered).strip(), title


def _script(name: str, output: str) -> tuple[str, str]:
    """A Python example, with what it prints; and its title, its docstring's first line."""
    source = _read(EXAMPLES / name)
    docstring = ast.get_docstring(ast.parse(source)) or name
    title = docstring.split("\n", 1)[0].rstrip(".")
    code = f"```python\n{source.rstrip()}\n```"
    printout = f"```text\n{output.rstrip()}\n```"
    return f"## {title}\n\n{code}\n\nIt prints:\n\n{printout}", title


def page(outputs: Mapping[str, str] | None = None) -> str:
    """The page. ``outputs`` gives what the Python examples print, where it is known."""
    outputs = outputs or {}
    sections, rows = [], []
    for name, what in examples():
        if name.endswith(".md"):
            text, title = _notebook(name)
        else:
            text, title = _script(name, outputs[name] if name in outputs else printed(name))
        sections.append(text)
        rows.append(f"| [{title}](#{slug(title)}) | `{name}` | {what} |")
    table = "\n".join(["| Example | File | What it shows |", "|---|---|---|", *rows])
    return "\n\n".join([_HEADER.rstrip(), table, *sections, _FOOTER.rstrip()]) + "\n"


def main() -> int:
    PAGE.write_text(page(), encoding="utf-8")
    print(f"wrote {PAGE.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
