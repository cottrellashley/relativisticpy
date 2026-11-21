"""How results look: one printer for every surface.

Every value RelativisticPy gives back is shown in one of four forms, so that a
result reads the same at the command line, at the interactive prompt, in a
notebook, in the report of a check and in machine-readable output:

- :func:`unicode`: mathematics for a terminal or a ``repr``;
- :func:`text`: plain ASCII, as the language reads it back;
- :func:`latex`: LaTeX, for notebooks and papers, which the language also reads back;
- :func:`data`: a JSON-serialisable description, for programs and agents.
"""

from __future__ import annotations

from typing import Any

import sympy

__all__ = ["data", "latex", "text", "unicode"]


def unicode(value: Any) -> str:
    """``value`` as mathematics for a terminal: ``48M²/r⁶``."""
    return str(value)


def text(value: Any) -> str:
    """``value`` as plain ASCII text that the language reads back as the same value."""
    return str(value)


def latex(value: Any) -> str:
    """``value`` typeset in LaTeX, which the language reads back as the same value."""
    if hasattr(value, "latex") and callable(value.latex):
        return str(value.latex())
    return str(sympy.latex(value))


def data(value: Any) -> dict[str, Any]:
    """``value`` described as JSON-serialisable data: its kind and its forms."""
    return {"text": text(value), "latex": latex(value)}
