r"""Sequences, read: square brackets right after a name (design/notation.md, decision 31).

Oppenheim and Schafer write a discrete-time signal as ``x[n]``, a function of an integer.
In LaTeX the parser reads brackets after a name as a product, and the reader decides by
what the name is:

- on the left of ``:=``, ``x[n] := \ldots`` defines a sequence of the label n, and with
  ``=`` so does ``x[n] = \ldots`` of a name that has no meaning yet; one written with its
  own members, as ``y[n] := \frac{1}{2} y[n-1] + \delta[n]``, is a difference equation,
  kept at initial rest;
- a sequence so defined, or a function of one variable, is evaluated at the point in the
  brackets: ``x[3]``, ``x[n - 1]``;
- ``\delta[n]`` and ``u[n]``, where nothing defines δ and u, are the unit impulse and the
  unit step, with a note, where the brackets hold an integer, as books write a point: whole
  numbers and names, in sums, differences and multiples, as ``n - k``; so that
  ``\delta[(-g)^{-1/2}]``, a variation in physics, is read as before;
- in an equation, names nothing defines, written at such points, are unknown sequences,
  as y and x are in the difference equation ``y[n] = x[n] + \frac{1}{2} y[n-1]``;
- brackets right after a list are its element, in a sum over all of it, counted from the
  sum's lower limit (decision 17), and are refused anywhere else; right after the letter
  of a family, ``x_n := \ldots``, they are its member, as its subscript is;
- ``E[X]`` and ``P[A]`` of random variables are an expectation and a probability
  (decision 14);
- a tensor or data is read as before.

``H(e^{j\omega}) := \ldots`` defines the frequency response H as a function of ω, though
it is written at e^{jω}, as books write the discrete-time Fourier transform; called at
e^{jθ}, it is its value at θ.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from typing import Any

from relativisticpy.language import syntax as ast
from relativisticpy.language.source import Span
from relativisticpy.typing.syntax import _walk

#: The decision of what brackets right after a name are.
SAMPLE = "sequence or product"

#: The names a sequence's point may be written with, nothing defining them: a letter, maybe
#: numbered, as n, k and n_0.
_LABEL = re.compile(r"[A-Za-z](?:_\d+)?")


@dataclass(frozen=True)
class Sequence(ast.FunctionDefinition):
    """A sequence defined with brackets, ``x[n] := ...``: the function of its label each of
    its members is, bound under its name. ``recurring`` says whether its value is written
    with its own members, as a difference equation is, which is kept at initial rest; and
    ``shift`` how far the member it defines is from its label, as 1 of ``y[n+1] := ...``."""

    recurring: bool = field(default=False, kw_only=True)
    shift: int = field(default=0, kw_only=True)

    @property
    def label(self) -> str:
        """The label its members are numbered by, n of ``x[n]``."""
        return self.params[0]


@dataclass(frozen=True)
class Response(ast.FunctionDefinition):
    """A frequency response, defined where it is written, at e^{jω}: ``H(e^{j\\omega}) :=
    ...``, a function of ω, its one parameter, called at e^{jθ} with θ."""

    written: str = field(default="", kw_only=True)


@dataclass(frozen=True)
class Sample(ast.Call):
    """A sequence at a point, written with square brackets: ``x[n - 1]``, its name the call's
    function, and the point its one argument."""

    @property
    def point(self) -> ast.Expr:
        return self.args[0]


def bracketed(node: ast.Expr) -> tuple[ast.Expr, tuple[ast.Script, ...]] | None:
    """The point square brackets hold, and the scripts after them, as ``n`` and ``^2`` of
    ``[n]^2``; ``None`` for anything else."""
    scripts: tuple[ast.Script, ...] = ()
    if isinstance(node, ast.Scripted):
        scripts = node.scripts
        node = node.base
    if not (isinstance(node, ast.List) and len(node.items) == 1):
        return None
    if not node.span.text.startswith("["):
        return None
    return node.items[0], scripts


def written_after(
    name: ast.Expr, following: ast.Expr
) -> tuple[ast.Expr, tuple[ast.Script, ...]] | None:
    """The point a name is written at, in square brackets right after it, as n - 1 of
    ``x[n - 1]``, with what is raised on the brackets; ``None`` where the brackets are not
    right after a plain name."""
    if not isinstance(name, ast.Name):
        return None
    found = bracketed(following)
    if found is None:
        return None
    bracket = following.base if isinstance(following, ast.Scripted) else following
    if bracket.span.start != name.span.end:
        return None
    return found


def target(node: ast.Expr) -> tuple[ast.Name, ast.Expr] | None:
    """The name and the point a definition's left side writes, x and n of ``x[n]``; ``None``
    for anything else."""
    if not (isinstance(node, ast.Binary) and node.op == "*"):
        return None
    found = written_after(node.left, node.right)
    if found is None or found[1]:
        return None
    assert isinstance(node.left, ast.Name)
    return node.left, found[0]


def label(point: ast.Expr) -> tuple[str, int] | None:
    """The label a point is written with and how far it is from it: ``("n", -1)`` of
    ``n - 1``, ``("n", 0)`` of ``n``; ``None`` for any other point."""
    if isinstance(point, ast.Name):
        return point.id, 0
    if isinstance(point, ast.Binary) and point.op in ("+", "-"):
        left, right = point.left, point.right
        if isinstance(left, ast.Name) and isinstance(right, ast.Number) and right.text.isdigit():
            shift = int(right.text)
            return left.id, shift if point.op == "+" else -shift
    return None


def integral(point: ast.Expr, integer: Callable[[str], bool | None]) -> bool:
    """Whether a point is written as an integer is, as books write the point of a sequence:
    whole numbers and names, in sums, differences and products. ``integer`` says of a name
    whether it is an integer, by what binds it, ``None`` where nothing does: then a letter,
    maybe numbered, as n or n_0, is one, as a label is."""
    if isinstance(point, ast.Number):
        return point.text.isdigit()
    if isinstance(point, ast.Name):
        known = integer(point.id)
        if known is not None:
            return known
        return not point.verbatim and _LABEL.fullmatch(point.id) is not None
    if isinstance(point, ast.Unary) and point.op in ("-", "+"):
        return integral(point.operand, integer)
    if isinstance(point, ast.Binary) and point.op in ("+", "-", "*"):
        return integral(point.left, integer) and integral(point.right, integer)
    return False


def samples(node: ast.Node) -> list[tuple[ast.Name, ast.Expr]]:
    """Every name an expression writes at a point in square brackets, with the point, in
    the order written."""
    found = []
    for current in _walk(node):
        if isinstance(current, ast.Binary) and current.op == "*":
            before = current.left
            while isinstance(before, ast.Binary) and before.op == "*":
                before = before.right
            written = written_after(before, current.right)
            if written is not None:
                assert isinstance(before, ast.Name)
                found.append((before, written[0]))
    return sorted(found, key=lambda pair: pair[0].span.start)


def _factors(node: ast.Binary) -> list[ast.Expr]:
    """The factors of a product, in order."""
    found: list[ast.Expr] = []
    left: ast.Expr = node
    while isinstance(left, ast.Binary) and left.op == "*":
        found.insert(0, left.right)
        left = left.left
    found.insert(0, left)
    return found


def subscript_form(name: ast.Name, point: ast.Expr, span: Span) -> ast.Expr:
    """``name[point]`` written with the point as its subscript, ``name_{point}``, as the
    parser reads one: a name or a whole number below is an index too."""
    indices = None
    if isinstance(point, ast.Name):
        item = ast.IndexItem(point.id, span=point.span)
        indices = ast.IndexList((item,), span=point.span)
    elif isinstance(point, ast.Number) and point.text.isdigit():
        item = ast.IndexItem(None, point, span=point.span)
        indices = ast.IndexList((item,), span=point.span)
    script = ast.Script("_", point, indices, span=point.span)
    return ast.Scripted(name, (script,), span=span)


def with_subscripts(node: ast.Expr, element: Callable[[ast.Name, ast.Expr], bool]) -> ast.Expr:
    """An expression with each name written at a point in square brackets that ``element``
    says is an element of a list, or a member of a family, written with the point as its
    subscript instead, as ``x[n]`` is ``x_{n}``, which a sum over n then reads as it reads
    ``x_n``. The expression itself where it writes none."""
    marked = {id(name) for name, point in samples(node) if element(name, point)}
    if not marked:
        return node
    result: ast.Expr = _with_subscripts(node, marked)
    return result


def _with_subscripts(node: Any, marked: set[int]) -> Any:
    """A syntax tree with the brackets after each name in ``marked`` made its subscript;
    what it does not change is kept as it is, the same nodes."""
    if isinstance(node, tuple):
        items = tuple(_with_subscripts(item, marked) for item in node)
        return items if any(new is not old for new, old in zip(items, node, strict=True)) else node
    if not isinstance(node, ast.Node):
        return node
    changes = {}
    for item in fields(node):
        old = getattr(node, item.name)
        new = old if item.name == "span" else _with_subscripts(old, marked)
        if new is not old:
            changes[item.name] = new
    rebuilt = replace(node, **changes) if changes else node
    if isinstance(rebuilt, ast.Binary) and rebuilt.op == "*":
        made = merged(rebuilt, lambda name, point, span: id(name) in marked, subscript_form)
        return made if made is not None else rebuilt
    return rebuilt


def merged(
    node: ast.Binary,
    sampled: Callable[[ast.Name, ast.Expr, Span], bool],
    make: Callable[[ast.Name, ast.Expr, Span], ast.Expr] | None = None,
) -> ast.Expr | None:
    """A product with each name and the point after it in square brackets made one
    :class:`Sample`, or what ``make`` makes of them, where ``sampled`` says the name,
    written with the point over the span given, is a sequence there, and what is raised on
    the brackets raised on it; ``None`` where none is."""
    factors = _factors(node)
    found: list[ast.Expr] = []
    changed = False
    position = 0
    while position < len(factors):
        factor = factors[position]
        following = factors[position + 1] if position + 1 < len(factors) else None
        written = written_after(factor, following) if following is not None else None
        if written is not None and following is not None:
            assert isinstance(factor, ast.Name)
            point, scripts = written
            bracket = following.base if isinstance(following, ast.Scripted) else following
            if sampled(factor, point, factor.span.to(bracket.span)):
                over = factor.span.to(bracket.span)
                sample = make(factor, point, over) if make else Sample(factor, (point,), span=over)
                if scripts:
                    found.append(ast.Scripted(sample, scripts, span=factor.span.to(following.span)))
                else:
                    found.append(sample)
                changed = True
                position += 2
                continue
        found.append(factor)
        position += 1
    if not changed:
        return None
    product = found[0]
    for factor in found[1:]:
        product = ast.Binary("*", product, factor, implicit=True, span=product.span.to(factor.span))
    return product


def unit_circle(node: ast.Expr) -> tuple[str, ast.Expr] | None:
    """The imaginary unit and the angle a point on the unit circle is written with, as j and
    ω of ``e^{j\\omega}``, and j and π/2 of ``e^{j\\pi/2}``: e raised to the unit, i or j,
    times the angle, maybe over a number, maybe negated; ``None`` for anything else."""
    exponent: ast.Expr | None
    if isinstance(node, ast.Binary) and node.op == "**":
        base, exponent = node.left, node.right
    elif isinstance(node, ast.Scripted) and len(node.scripts) == 1 and node.scripts[0].kind == "^":
        base, exponent = node.base, node.scripts[0].body
    else:
        return None
    if not (isinstance(base, ast.Name) and base.id == "e") or exponent is None:
        return None
    return _angle(exponent)


def _angle(exponent: ast.Expr) -> tuple[str, ast.Expr] | None:
    """The unit an exponent is written with and what it multiplies, the angle."""
    if isinstance(exponent, ast.Unary) and exponent.op == "-":
        inner = _angle(exponent.operand)
        if inner is None:
            return None
        return inner[0], ast.Unary("-", inner[1], span=exponent.span)
    if isinstance(exponent, ast.Binary) and exponent.op == "/":
        inner = _angle(exponent.left)
        if inner is None:
            return None
        quotient = ast.Binary("/", inner[1], exponent.right, span=exponent.span)
        return inner[0], quotient
    if not (isinstance(exponent, ast.Binary) and exponent.op == "*"):
        return None
    factors = _factors(exponent)
    unit = factors[0]
    if not (isinstance(unit, ast.Name) and unit.id in ("i", "j")) or len(factors) < 2:
        return None
    angle = factors[1]
    for factor in factors[2:]:
        angle = ast.Binary("*", angle, factor, implicit=True, span=angle.span.to(factor.span))
    return unit.id, angle
