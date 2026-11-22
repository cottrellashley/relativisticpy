"""The documentation, offline: what ``relativisticpy help`` prints.

The pages of the documentation are Markdown. :func:`topics` lists them and :func:`page`
gives one, from the installed package, where they are shipped in ``relativisticpy/_docs``,
or, in a source tree, from its ``docs`` folder. The language reference is generated from
the language itself, so it is there even where its page is not.
"""

from __future__ import annotations

import re
from pathlib import Path

from relativisticpy.errors import UsageError

__all__ = ["page", "topics"]

_TITLE = re.compile(r"^# (.+)$", re.MULTILINE)
_NAME = re.compile(r"[A-Za-z0-9_-]+")

#: The topic of the language reference, which :mod:`relativisticpy.reference` writes.
_REFERENCE = "reference"


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


def page(topic: str) -> str:
    """The Markdown of one page, by its topic."""
    folder = _folder()
    named = _NAME.fullmatch(topic) is not None  # a topic, never a path to elsewhere
    path = folder / f"{topic}.md" if folder is not None and named else None
    if path is not None and path.is_file():
        return path.read_text(encoding="utf-8")
    if topic == _REFERENCE:
        from relativisticpy.reference import markdown

        return markdown()
    known = ", ".join(name for name, _ in topics())
    raise UsageError(f"there is no help on {topic!r}", hint=f"the topics are {known}")
