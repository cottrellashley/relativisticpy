r"""Typefaces that make names of their own (design/notation.md, decision 9).

Books set a vector bold and its length plain, as mechanics writes ``\mathbf{r}`` and
``r = |\mathbf{r}|``, and a Lagrangian density calligraphic beside a Lagrangian,
``\mathcal{L}`` and L. Where the program defines a name in two typefaces, they are two
names. Where it defines it in one, the other stands for it, as books also set one name in
two, and a note says so. Upright and italic make no name of their own: ``\mathrm{d}x`` is
dx, and ``\mathrm{e}`` is e.

A name defined in a typeface where the plain name is free is bound to the plain name, as
before, so that every layer of the compiler finds it as it finds any other name, and the
binding says the typeface it was ``written`` in. A definition of the name in another
typeface makes the two apart: the one in a typeface is bound to the name as it is written,
``\mathbf{w}``, and each name written so is read as that name, statement by statement,
before anything else reads it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, fields, replace
from typing import Any

from relativisticpy.compiler.scope import Binding, Scope, TensorBinding, VariableBinding
from relativisticpy.language import syntax as ast
from relativisticpy.typing.syntax import _walk

#: The typefaces that make a name of their own, each by the name it is bound to apart and
#: the ways it is written: bold, as books set a vector, calligraphic, script, Fraktur and
#: sans serif.
_TYPEFACES = (
    ("\\mathbf", ("\\mathbf", "\\boldsymbol", "\\bm", "\\pmb", "{\\bf", "\\bf")),
    ("\\mathcal", ("\\mathcal", "{\\cal", "\\cal")),
    ("\\mathscr", ("\\mathscr",)),
    ("\\mathfrak", ("\\mathfrak",)),
    ("\\mathsf", ("\\mathsf",)),
)


#: A name written in a typeface, the command and the name: ``\mathbf{w}``,
#: ``\boldsymbol{\sigma}`` or ``{\bf v}``.
_WRITTEN = re.compile(
    r"\\(?P<command>[A-Za-z]+)\s*\{\s*\\?(?P<name>[A-Za-z]+)\s*\}"
    r"|\{\s*\\(?P<switch>bf|cal)\s+\\?(?P<switched>[A-Za-z]+)\s*\}"
)


def key(node: ast.Name) -> str | None:
    """The name a name set in a typeface of its own is bound to apart from the plain one,
    ``\\mathbf{w}`` of w set bold, however bold is written; ``None`` for a name set plain,
    upright or italic, for one already read as that name, and for a name made of more than
    it is written with, as ``b^(1)`` of ``\\mathbf{b}^{(1)}``."""
    match = _WRITTEN.fullmatch(node.span.text.strip())
    if match is None or (match["name"] or match["switched"]) != node.id:
        return None
    written = f"\\{match['command'] or match['switch']}"
    for command, spellings in _TYPEFACES:
        if written in spellings or f"{{{written}" in spellings:
            return f"{command}{{{node.id}}}"
    return None


def written(binding: Binding | None) -> str | None:
    """The typeface a binding of a plain name was defined in, as ``\\mathbf{w}``, where it
    was; ``None`` for one defined plain, and for anything else."""
    if isinstance(binding, VariableBinding | TensorBinding):
        return binding.written
    return None


@dataclass
class Apart:
    """What :func:`prepare` changed of the scope for a statement, to undo if it fails: the
    bindings it moved, by the plain name they were bound to; and the definitions in a
    typeface whose binding is to say so once it is made, by their plain name, with the
    binding it had before."""

    moved: dict[str, Binding] = field(default_factory=dict)
    styled: dict[str, tuple[str, Binding | None]] = field(default_factory=dict)


def prepare(statement: ast.Statement, scope: Scope) -> tuple[ast.Statement, Apart]:
    """A statement with each name set in a typeface of its own that the program binds
    apart read as that name, as ``\\mathbf{w}`` once ``w`` and ``\\mathbf{w}`` are both
    defined; and the scope ready for what it defines. A plain name it defines, which the
    program defined in a typeface alone, is moved apart first, so that the plain name is
    new, as w is in ``w := 7`` after ``\\mathbf{w} := (1, 2, 3)``; and a name in a typeface
    it defines, where the program defines the plain name otherwise, is defined apart."""
    apart = Apart()
    targets = {id(target) for target in _targets(statement)}
    for target in _targets(statement):
        styled = key(target)
        binding = scope.lookup(target.id)
        if styled is None and (moved := written(binding)) is not None:
            assert binding is not None
            if scope.unbind(target.id) is not None:  # bound at this level: w := 7
                scope.bind(moved, binding)
                apart.moved[target.id] = binding
        elif (
            styled is not None
            and scope.lookup(styled) is None
            and (binding is None or written(binding) == styled)
        ):  # \mathbf{w} := ..., bound to w, which then says how it was written
            apart.styled[target.id] = (styled, binding)

    def renamed(node: ast.Name) -> ast.Name:
        styled = key(node)
        if styled is None:
            return node
        binding = scope.lookup(node.id)
        defined_apart = id(node) in targets and binding is not None and written(binding) != styled
        if scope.lookup(styled) is not None or defined_apart:
            return replace(node, id=styled)
        return node

    return _renaming(statement, renamed), apart


def finish(apart: Apart, scope: Scope) -> None:
    """Say, of each name a statement defined in a typeface and bound to the plain name, the
    typeface it was written in (see :func:`prepare`)."""
    for name, (styled, before) in apart.styled.items():
        binding = scope.lookup(name)
        if binding is not before and isinstance(binding, VariableBinding | TensorBinding):
            scope.bind(name, replace(binding, written=styled))


def undo(apart: Apart, scope: Scope) -> None:
    """Put back what :func:`prepare` moved, as a statement that fails changes nothing."""
    for name, binding in apart.moved.items():
        styled = written(binding)
        if styled is not None and scope.lookup(styled) is binding:
            scope.unbind(styled)
        scope.bind(name, binding)


def stands_for(node: ast.Name, binding: Binding) -> str | None:
    """The note on a name read as the program's name in another typeface, which the
    program defines alone: ``\\mathbf{w}`` read as w, or w as ``\\mathbf{w}``; ``None``
    where it is read as itself."""
    styled, defined = key(node), written(binding)
    if node.id.startswith("\\"):  # read as the name it is bound to apart
        return None
    text = node.span.text
    if styled is None and any(s in text for _, spellings in _TYPEFACES for s in spellings):
        return None  # (\mathcal{L}), or \left. \mathbf{v} \right|: in a typeface, with more
    if styled == defined or not isinstance(binding, VariableBinding | TensorBinding):
        return None
    shown = node.span.text.strip()
    other = defined if defined is not None else node.id
    return (
        f"{shown}, which the program does not define, is read as {other}, which it does: a "
        f"typeface makes a name of its own, so that defining {shown} makes it one"
    )


def _targets(statement: ast.Statement) -> Iterator[ast.Name]:
    """The names a statement may define: that of a definition with one target, and of the
    left side of an equation, as ``r = |\\mathbf{r}|`` defines r, each written plainly or
    with scripts, as a tensor is."""
    if isinstance(statement, ast.Define) and len(statement.targets) == 1:
        target: ast.Expr = statement.targets[0]
    elif isinstance(statement, ast.Evaluate) and isinstance(statement.expr, ast.Equation):
        target = statement.expr.lhs
    else:
        return
    if isinstance(target, ast.Scripted):
        target = target.base
    if isinstance(target, ast.Name):
        yield target


def _renaming(root: ast.Statement, rename: Callable[[ast.Name], ast.Name]) -> ast.Statement:
    """``root`` with each name in it renamed by ``rename``; the same object where none is.
    The tree is walked without recursion, as a sum of many terms is deep."""
    order = list(_walk(root))
    renamed = {id(node): rename(node) for node in order if isinstance(node, ast.Name)}
    if all(renamed[key] is node for key, node in _names(order)):
        return root
    done: dict[int, Any] = dict(renamed)

    def mapped(value: Any) -> Any:
        if isinstance(value, ast.Node):
            return done.get(id(value), value)
        if isinstance(value, tuple):
            items = tuple(mapped(item) for item in value)
            return value if all(a is b for a, b in zip(items, value, strict=True)) else items
        return value

    for node in reversed(order):  # each node after all that are in it
        if isinstance(node, ast.Name):
            continue
        changed = {}
        for each in fields(node):
            value = getattr(node, each.name)
            new = mapped(value)
            if new is not value:
                changed[each.name] = new
        done[id(node)] = replace(node, **changed) if changed else node
    result: ast.Statement = done[id(root)]
    return result


def _names(order: list[ast.Node]) -> Iterator[tuple[int, ast.Name]]:
    return ((id(node), node) for node in order if isinstance(node, ast.Name))
