r"""Convolutions, read (design/notation.md, decision 27): an asterisk between two signals.

In signal processing, ``f(t) * g(t)`` and ``x[n] * h[n]`` are convolutions, and elsewhere
``*`` multiplies, in plain text above all. The reader decides by the operands: ``*`` or
``\ast`` between two functions applied at the same variable, or two sequences at points of
the same index, is their convolution, with a note; between anything else it multiplies, as
``2*x`` does, and ``z^*``, a script, is a conjugate. ``\circledast`` is always the circular
convolution.

What a signal is written with is read here: the functions applied in it, and where. Whether
a function or a sequence is one the program has is the lowering's to say.
"""

from __future__ import annotations

from collections.abc import Sequence

from relativisticpy.language import syntax as ast
from relativisticpy.language.source import Span
from relativisticpy.typing.syntax import _walk

#: The decision of what an asterisk between two signals is.
CONVOLUTION = "convolution or product"

#: The asterisks signal processing writes a convolution with.
STARS = ("*", "\\ast", "∗")


def starred(first: ast.Expr, second: ast.Expr) -> str | None:
    """The asterisk written between two factors, as signal processing writes a convolution;
    ``None`` where they are written side by side, or with ``\\cdot`` or ``\\times``."""
    between = Span(first.span.source, first.span.end, second.span.start).text.strip()
    return between if between in STARS else None


def factors(node: ast.Binary) -> list[ast.Expr]:
    """The factors of a product, as written: a product in brackets is one of them, as
    ``\\left(e^{-t} u(t)\\right)`` is in ``\\left(e^{-t} u(t)\\right) * \\left(e^{-2t}
    u(t)\\right)``."""
    found: list[ast.Expr] = [node.right]
    left = node.left
    while isinstance(left, ast.Binary) and left.op == "*" and not _bracketed(left):
        found.insert(0, left.right)
        left = left.left
    found.insert(0, left)
    return found


def _bracketed(node: ast.Binary) -> bool:
    """Whether an operation is written in brackets, which its span then begins with."""
    return node.span.start < node.left.span.start


def product_of(written: Sequence[ast.Expr]) -> ast.Expr:
    """Factors, as the product they are written as."""
    product = written[0]
    for factor in written[1:]:
        span = product.span.to(factor.span)
        product = ast.Binary("*", product, factor, implicit=True, span=span)
    return product


def applied(node: ast.Expr) -> list[ast.Call]:
    """The calls of one argument a signal is written with, as u(t) of ``e^{-t} u(t)``: those
    it is a sum, a product or a quotient of, maybe signed, and none in another function's
    argument, as \\psi(x) is in ``\\operatorname{conj}(\\psi(x))``."""
    found: list[ast.Call] = []
    pending = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, ast.Binary) and current.op in ("+", "-", "*"):
            pending.extend((current.right, current.left))
        elif isinstance(current, ast.Binary) and current.op == "/":
            pending.append(current.left)
        elif isinstance(current, ast.Unary) and current.op in ("-", "+"):
            pending.append(current.operand)
        elif (
            isinstance(current, ast.Call)
            and isinstance(current.func, ast.Name)
            and len(current.args) == 1
        ):
            found.append(current)
    return found


def names_in(node: ast.Expr) -> dict[str, ast.Name]:
    """The names an expression is written with, each with a place it is written."""
    found: dict[str, ast.Name] = {}
    for current in _walk(node):
        if isinstance(current, ast.Name):
            found.setdefault(current.id, current)
    return found
