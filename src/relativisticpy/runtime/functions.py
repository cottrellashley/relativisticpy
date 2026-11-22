"""The functions a program can call by name: SymPy's functions, algebra and calculus,
what takes equations and matrices apart, and the spacetimes.

:mod:`relativisticpy.runtime.registry` registers each under the name the language calls
it by, with how many arguments it takes and whether it applies to each component of a
tensor.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import sympy

from relativisticpy import display, spacetimes
from relativisticpy.diffgeom import Metric
from relativisticpy.errors import EvaluationError
from relativisticpy.tensors import Tensor
from relativisticpy.tensors.equation import TensorEquation

# -- implementations that need more than a SymPy function ----------------------


def _equations(values: Iterable[Any]) -> list[Any]:
    """Expand tensor equations into their component equations."""
    result: list[Any] = []
    for value in values:
        if isinstance(value, TensorEquation):
            result.extend(value.equations())
        elif isinstance(value, list | tuple):
            result.extend(_equations(value))
        else:
            result.append(value)
    return result


def solve(*args: Any) -> Any:
    first, rest = args[0], args[1:]
    if isinstance(first, TensorEquation | list | tuple):
        first = _equations([first])
    return sympy.solve(first, *rest)


def dsolve(*args: Any) -> Any:
    first, rest = args[0], args[1:]
    if isinstance(first, TensorEquation | list | tuple):
        equations = _equations([first])
        first = equations[0] if len(equations) == 1 else equations
    if rest and isinstance(rest[0], sympy.Symbol):  # dsolve(y'' + y, y): y has no variable
        raise EvaluationError(
            f"dsolve solves for a function written with its variable, as in "
            f"dsolve(f''(x) + f(x) = 0, f(x)), not for {display.text(rest[0])}"
        )
    return sympy.dsolve(first, *rest)


def rhs(value: Any) -> Any:
    """The right-hand side of an equation (or of each in a list)."""
    if isinstance(value, list | tuple):
        sides = [rhs(item) for item in value]
        return sides[0] if len(sides) == 1 else sides
    return _equation(value, "RHS").rhs


def lhs(value: Any) -> Any:
    """The left-hand side of an equation (or of each in a list)."""
    if isinstance(value, list | tuple):
        sides = [lhs(item) for item in value]
        return sides[0] if len(sides) == 1 else sides
    return _equation(value, "LHS").lhs


def _equation(value: Any, name: str) -> Any:
    """``value``, which ``name`` takes apart, if it is an equation; an error otherwise."""
    if not hasattr(value, "lhs") or not hasattr(value, "rhs"):
        raise EvaluationError(
            f"{name} takes an equation, as in {name}(x^2 = 4), but {display.text(value)} is not one"
        )
    return value


def subs(expr: Any, *args: Any) -> Any:
    """``subs(expr, old, new)``, or ``subs(expr, old = new, ...)`` with one rule or more,
    each an equation."""
    if args and all(isinstance(rule, sympy.Equality) for rule in args):
        return sympy.sympify(expr).subs([(rule.lhs, rule.rhs) for rule in args])
    if len(args) != 2:
        raise EvaluationError(
            "subs takes an expression and what to put for what, as in subs(x^2, x, 3), or "
            "rules, as in subs(x^2, x = 3)"
        )
    return sympy.sympify(expr).subs(*args)


def symbols(names: Any) -> Any:
    """``symbols('x y')``: the symbols named in a string; a name is its own symbol."""
    if isinstance(names, sympy.Symbol):
        return names
    if not isinstance(names, str):
        raise EvaluationError(
            f"symbols takes names in quotes, as in symbols('x y'), not {display.text(names)}"
        )
    return sympy.symbols(names)


symbols.takes_names = True  # type: ignore[attr-defined]  # text it is given is names, never run


def diag(*entries: Any) -> sympy.ImmutableDenseNDimArray:
    return sympy.ImmutableDenseNDimArray(sympy.diag(*entries).tolist())


def _matrix(value: Any) -> sympy.Matrix:
    if isinstance(value, Tensor):
        value = value.components
    if isinstance(value, sympy.NDimArray):
        value = value.tolist()
    return sympy.Matrix(value)


def _square(value: Any, what: str) -> sympy.Matrix:
    """``value`` as a square matrix, for ``what`` needs one; an error otherwise."""
    matrix = _matrix(value)
    rows, columns = matrix.shape
    if rows != columns:
        flat = isinstance(value, list) and not any(isinstance(row, list) for row in value)
        flat = flat or (isinstance(value, sympy.NDimArray) and value.rank() == 1)
        written = f"a list of {len(matrix)}" if flat else f"a {rows}×{columns} matrix"
        raise EvaluationError(f"{what} needs a square matrix, but this is {written}")
    return matrix


def det(value: Any) -> Any:
    """The determinant of a matrix; a scalar is its own determinant."""
    if isinstance(value, sympy.NDimArray | sympy.MatrixBase | list | Tensor):
        return sympy.simplify(_square(value, "a determinant").det())
    return value


def inverse(value: Any) -> Any:
    matrix = _square(value, "an inverse")
    try:
        inverted = matrix.inv()
    except sympy.matrices.exceptions.NonInvertibleMatrixError:
        raise EvaluationError("this matrix has no inverse: its determinant is zero") from None
    return sympy.ImmutableDenseNDimArray(inverted.tolist())


def transpose(value: Any) -> Any:
    return sympy.ImmutableDenseNDimArray(_matrix(value).T.tolist())


def trace(value: Any) -> Any:
    return _matrix(value).trace()


def latex(value: Any) -> str:
    if hasattr(value, "latex") and callable(value.latex):
        return str(value.latex())
    return str(sympy.latex(value))


def doit(value: Any) -> Any:
    return value.doit() if hasattr(value, "doit") else value


def numerical(value: Any, digits: int = 15) -> Any:
    return sympy.N(value, digits)


def root(value: Any, n: Any) -> Any:
    return sympy.root(value, n)


@dataclass(frozen=True)
class Spacetime:
    """A spacetime called from the language: its factory, given its arguments in order
    (picklable, as a module-level function is)."""

    factory: Callable[..., Metric]

    def __call__(self, *args: Any) -> Metric:
        return self.factory(*args)


def arity(factory: Callable[..., Metric]) -> tuple[int, int | None]:
    """How many arguments a spacetime takes in the language, from its signature: its
    positional parameters, of which those without a default are needed."""
    parameters = list(inspect.signature(factory).parameters.values())
    positional = [p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    needed = sum(1 for p in positional if p.default is p.empty)
    if any(p.kind is p.VAR_POSITIONAL for p in parameters):
        return needed, None
    return needed, len(positional)


# -- the tables --------------------------------------------------------------------

ELEMENTARY = {
    "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan, "cot": sympy.cot,
    "sec": sympy.sec, "csc": sympy.csc, "asin": sympy.asin, "acos": sympy.acos,
    "atan": sympy.atan, "acot": sympy.acot, "asec": sympy.asec, "acsc": sympy.acsc,
    "arcsin": sympy.asin, "arccos": sympy.acos, "arctan": sympy.atan,
    "sinh": sympy.sinh, "cosh": sympy.cosh, "tanh": sympy.tanh, "coth": sympy.coth,
    "sech": sympy.sech, "csch": sympy.csch, "asinh": sympy.asinh, "acosh": sympy.acosh,
    "atanh": sympy.atanh, "acoth": sympy.acoth, "asech": sympy.asech, "acsch": sympy.acsch,
    "exp": sympy.exp, "ln": sympy.log, "sqrt": sympy.sqrt, "abs": sympy.Abs,
    "Abs": sympy.Abs, "sign": sympy.sign, "floor": sympy.floor, "ceiling": sympy.ceiling,
    "factorial": sympy.factorial, "factorial2": sympy.factorial2, "gamma": sympy.gamma,
    "re": sympy.re, "Re": sympy.re,
    "im": sympy.im, "Im": sympy.im, "conjugate": sympy.conjugate, "conj": sympy.conjugate,
    "arg": sympy.arg, "erf": sympy.erf,
}  # fmt: skip

ALGEBRA = {
    "simplify": sympy.simplify, "tsimplify": sympy.simplify, "expand": sympy.expand,
    "factor": sympy.factor, "collect": sympy.collect, "cancel": sympy.cancel,
    "apart": sympy.apart, "together": sympy.together, "trigsimp": sympy.trigsimp,
    "powsimp": sympy.powsimp, "expand_trig": sympy.expand_trig, "nsimplify": sympy.nsimplify,
    "diff": sympy.diff, "integrate": sympy.integrate, "limit": sympy.limit, "lim": sympy.limit,
    "series": sympy.series, "subs": subs, "N": numerical, "numerical": numerical,
    "evalf": numerical, "doit": doit,
}  # fmt: skip

VALUES = {
    "solve": solve, "dsolve": dsolve, "RHS": rhs, "rhs": rhs, "LHS": lhs, "lhs": lhs,
    "Eq": sympy.Eq, "symbols": symbols, "latex": latex, "diag": diag, "det": det,
    "inverse": inverse, "inv": inverse, "transpose": transpose, "trace": trace,
    "log": sympy.log, "root": root, "binomial": sympy.binomial, "atan2": sympy.atan2,
    "min": sympy.Min, "Min": sympy.Min, "max": sympy.Max, "Max": sympy.Max,
    "gcd": sympy.gcd, "lcm": sympy.lcm,
}  # fmt: skip

#: The values that are scalars where their arguments are.
SCALAR_VALUES = frozenset(
    {"log", "root", "binomial", "atan2", "min", "Min", "max", "Max", "gcd", "lcm", "trace"}
)

SPACETIMES: dict[str, Callable[..., Metric]] = {
    "Minkowski": spacetimes.minkowski,
    "Schwarzschild": spacetimes.schwarzschild,
    "ReissnerNordstrom": spacetimes.reissner_nordstrom,
    "Kerr": spacetimes.kerr,
    "KerrNewman": spacetimes.kerr_newman,
    "FLRW": spacetimes.flrw,
    "DeSitter": spacetimes.de_sitter,
    "AntiDeSitter": spacetimes.anti_de_sitter,
}

ARITY: dict[str, tuple[int, int | None]] = {
    "diff": (1, None), "integrate": (1, None), "limit": (3, 4), "lim": (3, 4),
    "series": (1, 4), "subs": (2, 3), "solve": (1, None), "dsolve": (1, None),
    "RHS": (1, 1), "rhs": (1, 1), "LHS": (1, 1), "lhs": (1, 1), "Eq": (2, 2),
    "symbols": (1, 1), "latex": (1, 1), "diag": (1, None), "det": (1, 1),
    "inverse": (1, 1), "inv": (1, 1), "transpose": (1, 1), "trace": (1, 1),
    "log": (1, 2), "root": (2, 2), "binomial": (2, 2), "atan2": (2, 2),
    "min": (1, None), "Min": (1, None), "max": (1, None), "Max": (1, None),
    "gcd": (2, None), "lcm": (2, None), "collect": (2, None), "N": (1, 2),
    "numerical": (1, 2), "evalf": (1, 2),
}  # fmt: skip
