r"""The unknown of a differential equation, named without its variable, as books name it:
in ``dsolve(\frac{dy}{dt} = r y, y(0) = y_0)``, y is the function y(t) that dsolve solves
for, and in ``t y' + 2y = 4t^2`` the function y(t) of the one other name. A name nothing
defines is otherwise a symbol, whose derivative is 0.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING

from relativisticpy.compiler.builtins import BUILTINS
from relativisticpy.compiler.scope import Scope
from relativisticpy.language import syntax as ast
from relativisticpy.typing.syntax import _walk

if TYPE_CHECKING:
    from relativisticpy.reading.reader import Reader

#: The variable a dot differentiates by: \dot a is da/dt.
_TIME = "t"


def unknown_functions(
    equations: Sequence[ast.Expr], free: Callable[[ast.Name], bool]
) -> dict[str, tuple[str, ast.Name]]:
    """The names a differential equation differentiates, as y in dy/dt, y' or \\dot y,
    where nothing defines them (``free``), each with the variable it is a function of and
    where it is first written: the variable it is differentiated by, or, for a prime, the
    one other name the equations are written with. A name whose variable cannot be told
    is left out."""
    found: dict[str, tuple[str, ast.Name]] = {}
    primed: list[ast.Name] = []
    for node in _walk_all(equations):
        differentiated = _differentiated(node)
        if differentiated is None or not free(differentiated[0]):
            continue
        name, by = differentiated
        if by is None:
            primed.append(name)
        else:
            found.setdefault(name.id, (by, name))
    for name in primed:
        if name.id in found:
            continue
        unknowns = {name.id, *found}
        by = _the_other_name(equations, unknowns, free) or _only({v for v, _ in found.values()})
        if by is not None:
            found[name.id] = (by, name)
    return {name: entry for name, entry in found.items() if entry[0] != name}


def _differentiated(node: ast.Node) -> tuple[ast.Name, str | None] | None:
    """The name a node differentiates, and by what, where it says: y and t of dy/dt, a and
    t of \\dot a, and y and nothing of y'; ``None`` for a node that differentiates no name."""
    if isinstance(node, ast.Derivative) and isinstance(node.expr, ast.Name) and node.variables:
        variable = node.variables[0][0]
        return (node.expr, variable.id) if isinstance(variable, ast.Name) else None
    if isinstance(node, ast.Prime):
        return node.func, None
    if isinstance(node, ast.Accent) and node.accent in ("dot", "ddot"):
        return (node.base, _TIME) if isinstance(node.base, ast.Name) else None
    return None


def _walk_all(equations: Sequence[ast.Expr]) -> Iterator[ast.Node]:
    for equation in equations:
        yield from _walk(equation)


def _the_other_name(
    equations: Sequence[ast.Expr], unknowns: set[str], free: Callable[[ast.Name], bool]
) -> str | None:
    """The one name besides the unknowns that the equations are written with and nothing
    defines, as t in t y' + 2y = 4t^2, but for the functions they call, as \\delta in
    \\delta(t - 5); ``None`` where there are none or several."""
    called = {
        node.func.id
        for node in _walk_all(equations)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    others = {
        node.id
        for node in _walk_all(equations)
        if isinstance(node, ast.Name) and node.id not in unknowns | called and free(node)
    }
    return _only(others)


def _unknown_call(
    node: ast.Node, unknowns: dict[str, tuple[str, ast.Name]], reader: Reader, scope: Scope
) -> bool:
    r"""Whether a node calls a function nothing defines, other than an unknown, as
    \delta(t - 5) does where no delta function is defined."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return False
    name = node.func.id
    return name not in unknowns and name not in BUILTINS and reader._undefined(node.func, scope)


def _only(names: set[str]) -> str | None:
    return next(iter(names)) if len(names) == 1 else None


@contextlib.contextmanager
def read_as_functions(
    reader: Reader, equations: Sequence[ast.Expr], scope: Scope
) -> Iterator[None]:
    """Read the unknowns of a differential equation as functions of their variables while
    ``equations`` are compiled, as a function of the coordinates written once is, and say
    so; as they were afterwards. Only where everything else the equations call is known."""
    session = reader.session
    found = unknown_functions(equations, lambda name: reader._undefined(name, scope))
    if any(_unknown_call(node, found, reader, scope) for node in _walk_all(equations)):
        found = {}  # \delta(t - 5) that nothing defines: what solves it would be unknown too
    saved, factors = dict(session.functions), reader._factors
    for name, (variable, where) in found.items():
        session.functions[name] = (session.symbol(variable),)
        why = f"dsolve solves for it, and it is differentiated by {variable}"
        reader._decide("function named alone", where.span, f"the unknown {name}({variable})", why)
    reader._factors = factors - set(found)  # y in r y is y(t), and y(0) its value at 0
    try:
        yield
    finally:
        session.functions.clear()
        session.functions.update(saved)
        reader._factors = factors
