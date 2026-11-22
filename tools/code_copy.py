"""An mkdocs hook: copying an example from the docs gives code that runs.

The examples in the docs are Python sessions, with ``>>>`` prompts and the output beneath
them, because the tests run them as doctests. Material's copy button copies what a block
shows, unless the block says otherwise in a ``data-copy`` attribute. This hook gives every
session such an attribute, holding its code alone, without prompts or output, as doctest
reads it. It also highlights every session as one, ``pycon``, whether its fence says
``pycon`` or ``python``.
"""

from __future__ import annotations

import html
import re
from typing import Any

#: A fence of Python whose first line is a prompt: a session, whatever the fence says.
_SESSION_FENCE = re.compile(r"^(?P<indent>[ \t]*)```python[ \t]*\n(?=(?P=indent)>>>)", re.MULTILINE)

#: A highlighted block of Python: its opening, the attributes of its ``<pre>``, its body.
_BLOCK = re.compile(
    r'(?P<open><div class="language-(?:pycon|python) highlight"><pre)(?P<attributes>[^>]*)>'
    r"(?P<body>.*?)</pre>",
    re.DOTALL,
)
_TAG = re.compile(r"<[^>]+>")


def session_code(text: str) -> str | None:
    """The code of a Python session, as doctest reads it: the lines after its ``>>>``
    prompts and the ``...`` lines that continue them, without the prompts; ``None`` if it
    has no prompt."""
    code: list[str] = []
    source = False
    for line in text.split("\n"):
        prompt = line[:3]
        if prompt == ">>>" or (source and prompt == "..."):
            if len(line) > 3 and line[3] != " ":
                source = False  # >>>x is not a prompt, as doctest says
                continue
            code.append(line[4:])
            source = True
        else:
            source = False
    return "\n".join(code) if code else None


def on_page_markdown(markdown: str, **kwargs: Any) -> str:
    """Highlight every Python session as one."""
    return _SESSION_FENCE.sub(r"\g<indent>```pycon\n", markdown)


def on_page_content(html_: str, **kwargs: Any) -> str:
    """Give every block of a Python session the code it copies as."""

    def copy(block: re.Match[str]) -> str:
        code = session_code(html.unescape(_TAG.sub("", block["body"])))
        if code is None:
            return block[0]
        copied = html.escape(code, quote=True)
        return f'{block["open"]}{block["attributes"]} data-copy="{copied}">{block["body"]}</pre>'

    return _BLOCK.sub(copy, html_)
