"""How index notation is written: what the syntax tree says of tensors and their
indices, read without knowing what any name means, and the walk of a tree."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from relativisticpy.compiler.scope import Scope, TensorBinding, VariableBinding
from relativisticpy.errors import IndexNotationError
from relativisticpy.ir import TensorType
from relativisticpy.language import syntax as ast
from relativisticpy.language.parser import split_index_labels
from relativisticpy.language.source import Span


def _walk(node: ast.Node) -> Iterator[ast.Node]:
    """Every node of a syntax tree, without recursion, for expressions of any length."""
    pending: list[object] = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, tuple):
            pending.extend(current)
        elif isinstance(current, ast.Node):
            yield current
            pending.extend(vars(current).values())


def _used(node: ast.Node) -> Iterator[ast.Node]:
    """Every node of an expression as :func:`_walk` gives them, except the tensor that
    ``solve(E, T_{\\mu\\nu})`` is asked to solve for: that names the unknown, and is not a
    tensor the expression uses."""
    pending: list[object] = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, tuple):
            pending.extend(current)
        elif isinstance(current, ast.Node):
            yield current
            if (
                isinstance(current, ast.Call)
                and isinstance(current.func, ast.Name)
                and current.func.id == "solve"
                and len(current.args) == 2
            ):
                pending.extend((current.func, current.args[0]))
            else:
                pending.extend(vars(current).values())


def _operand(node: ast.Expr) -> ast.Expr:
    """``[x + 1]`` used as an operand is a grouping, as LaTeX's brackets are."""
    if isinstance(node, ast.List) and len(node.items) == 1:
        return node.items[0]
    return node


def _signed_terms(node: ast.Expr) -> list[tuple[int, ast.Expr]]:
    """The terms of a sum, with their signs, in order and without recursion."""
    found: list[tuple[int, ast.Expr]] = []
    pending: list[tuple[int, ast.Expr]] = [(1, node)]
    while pending:
        sign, expr = pending.pop()
        if isinstance(expr, ast.Binary) and expr.op in ("+", "-"):
            pending.append((sign if expr.op == "+" else -sign, expr.right))
            pending.append((sign, expr.left))
        else:
            found.append((sign, expr))
    return found


def _product_factors(node: ast.Expr) -> list[ast.Expr]:
    """The factors of a product, in order and without recursion."""
    found: list[ast.Expr] = []
    pending = [node]
    while pending:
        expr = pending.pop()
        if isinstance(expr, ast.Binary) and expr.op == "*":
            pending.append(expr.right)
            pending.append(expr.left)
        else:
            found.append(expr)
    return found


def _zero_literal(node: ast.Expr) -> bool:
    """A written 0, which has whatever free indices the other side of an equation has."""
    node = _operand(node)
    return isinstance(node, ast.Number) and float(node.text) == 0


def _mentions_tensor(node: ast.Expr) -> bool:
    """Whether a tensor with indices is written anywhere in an expression."""
    return any(
        isinstance(current, ast.Scripted) and _tensor_pattern(current) for current in _walk(node)
    )


def _tensor_pattern(node: ast.Scripted) -> tuple[str, list[tuple[str, bool]]] | None:
    """A name with index labels, as a tensor is written: ``T^{\\mu}{}_{\\nu}``."""
    if not isinstance(node.base, ast.Name):
        return None
    entries = []
    for script in node.scripts:
        if script.indices is None or not script.indices.items:
            return None
        for item in script.indices.items:
            if not _plain_label(item) or item.label == "⋯":
                return None
            entries.append((str(item.label), script.kind == "^"))
    return node.base.id, entries


def _tensor_pattern_of_one(node: ast.Scripted) -> tuple[str, bool] | None:
    """``𝟙_a`` or ``1_a``: the ones covector, with its single index label."""
    base = node.base
    ones = (isinstance(base, ast.Number) and base.text == "1") or (
        isinstance(base, ast.Name) and base.id == "𝟙"
    )
    if not ones or len(node.scripts) != 1:
        return None
    script = node.scripts[0]
    items = script.indices.items if script.indices is not None else ()
    if len(items) != 1 or not _plain_label(items[0]):
        return None
    return str(items[0].label), script.kind == "^"


def _plain_label(item: ast.IndexItem) -> bool:
    return (
        item.label is not None
        and item.value is None
        and item.derivative is None
        and item.opens is None
        and item.closes is None
    )


def _rank(node: ast.Scripted) -> int | None:
    """The number of index items, or ``None`` if a script cannot be read as indices."""
    count = 0
    for script in node.scripts:
        if script.indices is None:
            return None
        count += sum(1 for item in script.indices.items if item.derivative is None)
    return count


def _has_derivative(node: ast.Scripted) -> bool:
    """Whether an index is written after a comma or a semicolon, as in ``\\phi_{;\\mu}``, in
    a script that is nothing but indices: ``e^{2\\alpha(t,r)}`` is a power."""
    return any(
        item.derivative is not None
        for script in node.scripts
        if script.indices is not None and script.body is None
        for item in script.indices.items
    )


def _bracket_groups(
    factors: Sequence[Sequence[tuple[ast.IndexItem, bool]]], span: Span, *, closed: bool
) -> list[tuple[str, list[str], bool]]:
    """Bracketed groups of index labels across the index items of ``factors``, each with
    whether it spans more than one factor; with ``closed``, every bracket must close. A
    bracket takes the indices at the level it opens at, so ``T_{(a}{}^{b}{}_{c)}``
    symmetrises a and c alone, and one that closes at the other level means nothing."""
    groups: list[tuple[str, list[str], bool]] = []
    current: tuple[str, list[str], int, bool] | None = None
    for number, entries in enumerate(factors):
        for item, upper in entries:
            if item.opens is not None:
                current = (item.opens, [], number, upper)
            if current is not None and item.label is not None and upper == current[3]:
                current[1].append(item.label)
            if item.closes is not None and current is not None:
                if upper != current[3]:
                    opened, closes = ("upper", "lower") if current[3] else ("lower", "upper")
                    raise IndexNotationError(
                        f"this (anti)symmetrisation opens on {opened} indices and closes on "
                        f"{closes} ones; it is over indices at one level",
                        span=span,
                    )
                groups.append((current[0], current[1], current[2] != number))
                current = None
    if closed and current is not None:
        raise IndexNotationError("an (anti)symmetrisation bracket is never closed", span=span)
    return groups


def _is_minus_one(node: ast.Expr | None) -> bool:
    return (
        isinstance(node, ast.Unary)
        and node.op == "-"
        and isinstance(node.operand, ast.Number)
        and node.operand.text == "1"
    )


def _written(scripts: Sequence[ast.Script]) -> str:
    return "".join(s.span.text for s in scripts)


def _is_literal(value: ast.Expr) -> bool:
    return isinstance(value, ast.List | ast.Matrix)


def _literal_cells(value: ast.Expr, rank: int) -> tuple[tuple[int, ...], list[ast.Expr]] | None:
    """The shape and entries of a nested list or matrix literal."""
    if isinstance(value, ast.Matrix):
        rows = value.rows
        if rank == 1 and (len(rows) == 1 or all(len(row) == 1 for row in rows)):
            cells = [cell for row in rows for cell in row]
            return (len(cells),), cells
        if rank != 2:
            return None
        return (len(rows), len(rows[0])), [cell for row in rows for cell in row]

    def shape_of(node: ast.Expr, depth: int) -> tuple[tuple[int, ...], list[ast.Expr]] | None:
        if depth == 0:
            return ((), [node]) if not isinstance(node, ast.List) else None
        if not isinstance(node, ast.List) or not node.items:
            return None
        parts = [shape_of(item, depth - 1) for item in node.items]
        if any(part is None for part in parts):
            return None
        shapes = {part[0] for part in parts if part is not None}
        if len(shapes) != 1:
            return None
        cells = [cell for part in parts if part is not None for cell in part[1]]
        return (len(node.items), *shapes.pop()), cells

    return shape_of(value, rank)


def _shape_text(shape: tuple[int, ...]) -> str:
    return "×".join(str(n) for n in shape) if shape else "scalar"


def _indexed_name(node: ast.Name, scope: Scope) -> ast.Scripted | None:
    """``y_t`` as ``y_{t}`` when ``y`` is a tensor and ``y_t`` names nothing: data
    notation writes the indices of a tensor without braces."""
    base, underscore, tail = node.id.partition("_")
    binding = scope.lookup(base) if underscore and tail.isalpha() else None
    if not (
        isinstance(binding, TensorBinding)
        or (isinstance(binding, VariableBinding) and isinstance(binding.node.type, TensorType))
    ):
        return None
    span = node.span
    items = tuple(ast.IndexItem(label, span=span) for label in split_index_labels(tail))
    script = ast.Script("_", None, ast.IndexList(items, span=span), span=span)
    return ast.Scripted(ast.Name(base, span=span), (script,), span=span)


_NOTATION_OPERATIONS = frozenset({"solve", "trace", "simplify", "subs", "LHS", "RHS"})
