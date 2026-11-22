"""The documentation, offline: what ``relativisticpy help`` prints.

The pages of the documentation are Markdown. :func:`topics` lists them and :func:`page`
gives one, from the installed package or, in a source tree, from its ``docs`` folder.
"""

from __future__ import annotations

import re
from pathlib import Path

from relativisticpy.errors import UsageError

__all__ = ["page", "topics"]

_TITLE = re.compile(r"^# (.+)$", re.MULTILINE)


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
    if folder is None:
        return []
    found = []
    for path in sorted(folder.glob("*.md")):
        title = _TITLE.search(path.read_text(encoding="utf-8"))
        found.append((path.stem, title.group(1).strip() if title else path.stem))
    return found


def page(topic: str) -> str:
    """The Markdown of one page, by its topic."""
    folder = _folder()
    path = folder / f"{topic}.md" if folder is not None else None
    if path is None or not path.is_file():
        known = ", ".join(name for name, _ in topics()) or "none are installed"
        raise UsageError(f"there is no help on {topic!r}", hint=f"the topics are {known}")
    return path.read_text(encoding="utf-8")
