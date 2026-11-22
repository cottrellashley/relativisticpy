"""The documentation, offline: what ``relativisticpy help`` prints.

The pages of the documentation are Markdown. :func:`topics` lists them and :func:`page`
gives one, from the installed package, where they are shipped in ``relativisticpy/_docs``,
or, in a source tree, from its ``docs`` folder. The language reference is generated from
the language itself, so it is there even where its page is not. A topic may be asked for
by another name a person might use, as ``check`` for ``checking`` (:data:`ALIASES`), and
a topic that does not exist is told what it may have meant.

:func:`page` also gives one function of the language in full, by its name, as
``besselj``: its forms, its arguments, its examples with their answers and what computes
it, generated from what its integration declares, as the reference is
(:func:`relativisticpy.reference.capability`). A page comes first; ``latex()``, with
brackets, is always the function.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping
from pathlib import Path

from relativisticpy.errors import UsageError

__all__ = ["ALIASES", "page", "resolve", "topics"]

_TITLE = re.compile(r"^# (.+)$", re.MULTILINE)
_NAME = re.compile(r"[A-Za-z0-9_-]+")
#: A function's name, alone or with brackets after it, as ``latex()``.
_FUNCTION = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?P<called>\s*\(\s*\))?")

#: The topic of the language reference, which :mod:`relativisticpy.reference` writes.
_REFERENCE = "reference"

#: Other names a person may ask for a topic by, and the topic each names. A name that is
#: a page of its own is that page.
ALIASES: Mapping[str, str] = {
    "check": "checking",
    "checks": "checking",
    "verdicts": "checking",
    "papers": "checking",
    "start": "getting-started",
    "install": "getting-started",
    "notebooks": "getting-started",
    "command-line": "cli",
    "commands": "cli",
    "options": "cli",
    "prompt": "cli",
    "repl": "cli",
    "exit-codes": "cli",
    "json": "cli",
    "render": "cli",
    "api": "python",
    "tensor": "tensors",
    "spacetimes": "tensors",
    "geometry": "tensors",
    "data": "spaces",
    "editor": "live",
    "widget": "live",
    "widgets": "live",
    "marimo": "live",
    "anywidget": "live",
    "show": "live",
    "integration": "integrations",
    "plugins": "integrations",
    "libraries": "integrations",
    "extending": "integrations",
    "capabilities": "integrations",
    "agents": "mcp",
    "ai": "mcp",
    "claude": "mcp",
    "server": "mcp",
    "tools": "mcp",
}


def _folder() -> Path | None:
    """Where the pages are: shipped in the package, or the source tree's ``docs``."""
    here = Path(__file__).resolve().parent
    for folder in (here / "_docs", here.parent.parent / "docs"):
        if folder.is_dir():
            return folder
    return None


def topics() -> list[tuple[str, str]]:
    """Every page, as its topic and its title: ``("tensors", "Tensors and geometry")``."""
    folder = _folder()
    found = []
    if folder is not None:
        for path in sorted(folder.glob("*.md")):
            title = _TITLE.search(path.read_text(encoding="utf-8"))
            found.append((path.stem, title.group(1).strip() if title else path.stem))
    if all(topic != _REFERENCE for topic, _ in found):
        from relativisticpy.reference import TITLE

        found = sorted([*found, (_REFERENCE, TITLE)])
    return found


def resolve(topic: str) -> str:
    """The topic a name asks for: itself if it is a page, written in any case, or the page
    it is another name for (``check`` is ``checking``); unchanged otherwise."""
    folder = _folder()
    name = topic.strip().lower()
    if folder is None or _NAME.fullmatch(name) is None:  # a topic, never a path to elsewhere
        return name
    if (folder / f"{name}.md").is_file():
        return name
    alias = ALIASES.get(name)
    if alias is not None and (folder / f"{alias}.md").is_file():
        return alias
    return name


def page(topic: str) -> str:
    """The Markdown of one page, by its topic or another name for it, or of one function
    of the language, by its name: ``besselj``, or ``latex()`` where a page has the name."""
    asked = _FUNCTION.fullmatch(topic.strip())
    if asked is None or asked["called"] is None:
        found = _page(resolve(topic))
        if found is not None:
            return found
    described = _function(asked["name"]) if asked is not None else None
    if described is not None:
        return described
    if _folder() is None:
        raise UsageError(
            f"there is no help on {topic!r}: the documentation is not installed",
            hint="read it online, at https://relativisticpy.org; relativisticpy help reference "
            "prints the language, which is always there",
        )
    known = [name for name, _ in topics()]
    raise UsageError(
        f"there is no help on {topic!r}",
        hint=f"{_meant(topic, known)}the topics are {', '.join(known)}; the name of a "
        "function, as besselj, gives its help",
    )


def _page(name: str) -> str | None:
    """The Markdown of the page of a topic, if there is one."""
    folder = _folder()
    named = _NAME.fullmatch(name) is not None
    path = folder / f"{name}.md" if folder is not None and named else None
    if path is not None and path.is_file():
        return path.read_text(encoding="utf-8")
    if name == _REFERENCE:  # generated from the language, wherever its page is not shipped
        from relativisticpy.reference import markdown

        return markdown()
    return None


def _function(name: str) -> str | None:
    """The help of the function called ``name``, by any of its names, or in another case
    where only one function is called so, as ``BesselJ`` is ``besselj``; ``None`` where no
    function is."""
    from relativisticpy import integrations, reference

    try:
        return reference.capability(name)
    except KeyError:
        pass
    alike = {
        each.name
        for each in integrations.capabilities()
        if any(other.lower() == name.lower() for other in each.names)
    }
    return reference.capability(alike.pop()) if len(alike) == 1 else None


def _meant(topic: str, known: list[str]) -> str:
    """What a topic that does not exist may have meant: the closest topic, other name or
    function, as ``did you mean tensors? ``; nothing where none is close."""
    from relativisticpy import integrations

    names = [*known, *(alias for alias, target in ALIASES.items() if target in known)]
    names += [name for each in integrations.capabilities() for name in each.names]
    spelled: dict[str, list[str]] = {}
    for name in names:
        spelled.setdefault(name.lower(), []).append(name)
    close = difflib.get_close_matches(topic.strip().lower(), list(spelled), n=1, cutoff=0.6)
    if not close:
        return ""
    meant = [
        resolve(name) if name.lower() in known or name in ALIASES else name
        for name in spelled[close[0]]
    ]
    return f"did you mean {' or '.join(dict.fromkeys(meant))}? "
