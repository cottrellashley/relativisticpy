r"""Evaluation bars and brackets, read as books write them (``curriculum/ROADMAP.md``,
section 7):

- ``\left. F \right|_{x=a}`` is F at a; ``\left. F \right|_a^b``, ``\left[F\right]_a^b`` and
  ``[F]_{x=a}^{x=b}`` are F(b) − F(a);
- a bar written with ``\left.`` and ``\right|`` evaluates what they enclose, and only that,
  so that ``x + \left. x^2 \right|_{x=2}`` is x + 4; a bar written alone, as ``\Big|``,
  evaluates all that is written before it, and with limits, all of the side of an equation
  it is on;
- where no variable is written, it is that of an integral beside it in a sum, as in
  ``\left[x e^x\right]_0^1 - \int_0^1 e^x\,dx``; else the one symbol what it evaluates is
  written with; else it is refused, with a hint to write it.
"""

from __future__ import annotations

import re

from relativisticpy.language import syntax as ast
from relativisticpy.language.source import Span
from relativisticpy.typing.syntax import _walk

#: The decision of what an evaluation bar or bracket evaluates, and in which variable.
EVALUATION = "evaluation bar"

#: The brackets around what a bar or a bracket evaluates, as written.
_ENCLOSED = re.compile(
    r"(?:\\left\s*(?:\.|\[|\\lbrack|\()|\\[bB]igg?l?\s*(?:\[|\\lbrack)|\[|\\lbrack)\s*"
    r"(?P<inner>.*?)\s*"
    r"(?:\\right\s*(?:\||\]|\\rbrack|\))|\\[bB]igg?r?\s*(?:\]|\\rbrack)|\]|\\rbrack)",
    re.DOTALL,
)


def enclosed(node: ast.Expr) -> str:
    """What a bar or a bracket evaluates, as written, without the brackets around it: x^2 of
    ``\\left. x^2 \\right|``."""
    text = " ".join(node.span.text.split())
    found = _ENCLOSED.fullmatch(text)
    return found.group("inner") if found is not None else text


def integral_variables(sum_: ast.Expr, outside: Span) -> list[ast.Name]:
    """The variables of the integrals a sum writes, each once, but not of those written
    inside what is at ``outside``, or around it."""
    found: dict[str, ast.Name] = {}
    for node in _walk(sum_):
        if not isinstance(node, ast.Integral):
            continue
        if within(node.span, outside) or within(outside, node.span):
            continue
        for bound in node.bounds:
            if isinstance(bound.variable, ast.Name):
                found.setdefault(bound.variable.id, bound.variable)
    return sorted(found.values(), key=lambda name: name.span.start)


def within(inner: Span, outer: Span) -> bool:
    """Whether ``inner`` is written inside ``outer``, in the same source."""
    return inner.source is outer.source and outer.start <= inner.start <= inner.end <= outer.end
