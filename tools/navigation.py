"""An mkdocs hook: the navigation of the docs, and the files it gives language models.

- A page the navigation names that is not written yet, of those in :data:`PENDING`, is
  left out of it until it is, so that the site builds meanwhile.
- After the build, the site's root gets ``llms.txt``, an index of the pages in the order
  of the navigation, as https://llmstxt.org describes; ``llms-full.txt``, every page in
  one file; and a Markdown copy of each page beside its HTML, ``index.html.md``, which
  the links of ``llms.txt`` point to.

The page's description in ``llms.txt`` is what its header says the reader will learn, or
else its first sentence.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

#: Pages that the navigation names before they are written, by someone else.
PENDING = frozenset({"cli.md"})

#: Pages listed under "Optional" in llms.txt, which a reader short of room can skip.
OPTIONAL = frozenset({"architecture.md", "roadmap.md"})

#: What an agent needs before any page, under the summary of llms.txt.
FACTS = r"""
It is a Python package and a language. Key facts:

- Install: `pip install "relativisticpy[numeric] @ git+https://github.com/cottrellashley/relativisticpy"`
  while PyPI still serves the earlier 1.x releases. The extras are `numeric` (NumPy and
  SciPy), `data` (with pandas and scikit-learn), `jax`, `mcp` (the MCP server) and
  `live` (the editor in notebooks).
- Run a program or a Markdown notebook with `relativisticpy FILE`, code with
  `relativisticpy -c "..."`, and check the equations of a LaTeX paper with
  `relativisticpy check paper.tex`. From Python, `Workbook().expr(source)` gives the
  value of a program's last expression, and `Workbook().exe(source)` every value.
- Learn the language offline: `relativisticpy help` lists the topics,
  `relativisticpy help reference` prints the whole language on one page, and
  `relativisticpy help NAME` one function in full.
- Plain text (`g_{mu nu}`, `x**2`), LaTeX (`g_{\mu\nu}`, `\frac{1}{2}`) and Unicode
  (`θ`, `∂`) mix freely in one program.
- `:=` always defines. `=` defines only what has no meaning yet; otherwise it is a
  claim, which is checked, or an equation.
- In Python, write LaTeX in a raw string, `r"\frac{1}{2}"`: in an ordinary string,
  Python turns `\f`, `\t`, `\n`, `\r`, `\b`, `\a` and `\v` into control characters.
- For programs: `--format json` on the command line, and `display.data(value)`,
  `report.to_dict()` and `error.to_dict()` in Python. The exit status is 0 when
  everything succeeded, 1 when a statement or a check failed, and 2 for a command line
  that is wrong.
- For agents: `relativisticpy mcp` serves the tools `compute`, `verify` and `check` over
  MCP. Every answer is right, or says it is not one: a refusal is an `error` whose `hint`
  says what to write instead, and a value kept as written, as an integral with no closed
  form, has `"complete": false` and its `reasons`.
""".strip()

_LEARN = re.compile(r"\*\*You will learn:\*\*\s*(?P<text>.+?)(?:\s*\*\*|\n\n|$)", re.DOTALL)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_FENCED = re.compile(r"^```.*?^```[ \t]*$", re.DOTALL | re.MULTILINE)
_QUOTED = re.compile(r"\n>[ \t]?")


def pending(nav: Sequence[Any], docs: Path) -> list[Any]:
    """The navigation without the pages of :data:`PENDING` that are not written yet."""
    kept: list[Any] = []
    for item in nav:
        if isinstance(item, Mapping):
            ((title, value),) = item.items()
            if isinstance(value, list):
                kept.append({title: pending(value, docs)})
            elif not (value in PENDING and not (docs / value).is_file()):
                kept.append(item)
        elif not (item in PENDING and not (docs / item).is_file()):
            kept.append(item)
    return kept


def pages(nav: Sequence[Any], section: str | None = None) -> list[tuple[str | None, str, str]]:
    """Every page of the navigation, in order: its section, its title and its file."""
    found: list[tuple[str | None, str, str]] = []
    for item in nav:
        ((title, value),) = item.items() if isinstance(item, Mapping) else ((item, item),)
        if isinstance(value, list):
            found += pages(value, title)
        else:
            found.append((section, title, value))
    return found


def url(page: str) -> str:
    """Where a page is on the site, relative to its root: ``language.md`` is at
    ``language/``, and ``index.md`` at the root."""
    stem = page.removesuffix(".md")
    return "" if stem == "index" else f"{stem.removesuffix('/index')}/"


def description(markdown: str) -> str:
    """What a page's header says its reader will learn, or else the first sentence of its
    prose."""
    found = _LEARN.search(_QUOTED.sub(" ", markdown))  # a header quoted over several lines
    if found is not None:
        text = found["text"]
    else:
        prose = [
            paragraph.strip()
            for paragraph in _FENCED.sub("", markdown).split("\n\n")
            if paragraph.strip()[:1].isalpha() or paragraph.strip()[:2] == "**"
        ]
        text = re.split(r"(?<=\.)\s", prose[0].replace("*", ""))[0] if prose else ""
    text = _LINK.sub(r"\1", " ".join(text.split()))
    return text.rstrip(".:") + "."


def llms_txt(
    entries: Iterable[tuple[str | None, str, str]], docs: Path, site: str, name: str, summary: str
) -> str:
    """The index of the site, for language models: the site's name, its summary, the key
    facts, then each page, grouped as the navigation groups them."""
    groups: dict[str, list[str]] = {}
    for section, title, page in entries:
        group = "Optional" if page in OPTIONAL else section or "Docs"
        text = (docs / page).read_text(encoding="utf-8")
        link = f"{site}{url(page)}index.html.md"
        groups.setdefault(group, []).append(f"- [{title}]({link}): {description(text)}")
    ordered = sorted(groups, key=lambda group: (group == "Optional", group != "Docs"))
    lines = [f"# {name}", "", f"> {summary}", "", FACTS]
    for group in ordered:
        lines += ["", f"## {group}", "", *groups[group]]
    return "\n".join(lines) + "\n"


def llms_full_txt(entries: Iterable[tuple[str | None, str, str]], docs: Path, site: str) -> str:
    """Every page, in the order of the navigation, each after a line saying where it is."""
    parts = []
    for _, _, page in entries:
        text = (docs / page).read_text(encoding="utf-8").strip()
        parts.append(f"<!-- docs/{page} · {site}{url(page)} -->\n\n{text}\n")
    return "\n".join(parts)


def on_config(config: Any, **kwargs: Any) -> Any:
    config["nav"] = pending(config["nav"], Path(config["docs_dir"]))
    return config


def on_post_build(config: Any, **kwargs: Any) -> None:
    docs, site = Path(config["docs_dir"]), Path(config["site_dir"])
    root = config["site_url"] or "/"
    entries = pages(config["nav"])
    index = llms_txt(entries, docs, root, config["site_name"], config["site_description"])
    (site / "llms.txt").write_text(index, encoding="utf-8")
    (site / "llms-full.txt").write_text(llms_full_txt(entries, docs, root), encoding="utf-8")
    for _, _, page in entries:
        copy = site / url(page) / "index.html.md"
        copy.write_text((docs / page).read_text(encoding="utf-8"), encoding="utf-8")
