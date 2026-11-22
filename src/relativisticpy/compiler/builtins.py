"""The functions a program can call by name."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import sympy

from relativisticpy import spacetimes
from relativisticpy.diffgeom import Metric
from relativisticpy.tensors import Tensor
from relativisticpy.tensors.equation import TensorEquation


@dataclass(frozen=True)
class Builtin:
    """A function callable from the language.

    A ``componentwise`` builtin given a tensor as its first argument is applied
    to every component, so ``simplify(R_{abcd})`` simplifies each component (in
    parallel, when the workbook has workers). ``scalar`` builtins return a
    scalar for scalar arguments; the rest return arbitrary values.
    """

    name: str
    func: Callable[..., Any]
    componentwise: bool = False
    scalar: bool = False
    min_args: int = 0
    max_args: int | None = None

    def check_arity(self, count: int) -> str | None:
        """An error message if ``count`` arguments are wrong for this function."""
        if count < self.min_args or (self.max_args is not None and count > self.max_args):
            if self.max_args == self.min_args:
                expected = f"{self.min_args}"
            elif self.max_args is None:
                expected = f"at least {self.min_args}"
            else:
                expected = f"{self.min_args} to {self.max_args}"
            plural = "" if expected == "1" else "s"
            return f"{self.name}() takes {expected} argument{plural}, not {count}"
        return None


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
    return sympy.dsolve(first, *rest)


def rhs(value: Any) -> Any:
    """The right-hand side of an equation (or of each in a list)."""
    if isinstance(value, list | tuple):
        sides = [rhs(item) for item in value]
        return sides[0] if len(sides) == 1 else sides
    return value.rhs


def lhs(value: Any) -> Any:
    """The left-hand side of an equation (or of each in a list)."""
    if isinstance(value, list | tuple):
        sides = [lhs(item) for item in value]
        return sides[0] if len(sides) == 1 else sides
    return value.lhs


def subs(expr: Any, *args: Any) -> Any:
    """``subs(expr, old, new)``, or ``subs(expr, {old: new})``."""
    return expr.subs(*args)


def diag(*entries: Any) -> sympy.ImmutableDenseNDimArray:
    return sympy.ImmutableDenseNDimArray(sympy.diag(*entries).tolist())


def _matrix(value: Any) -> sympy.Matrix:
    if isinstance(value, Tensor):
        value = value.components
    if isinstance(value, sympy.NDimArray):
        value = value.tolist()
    return sympy.Matrix(value)


def det(value: Any) -> Any:
    """The determinant of a matrix; a scalar is its own determinant."""
    if isinstance(value, sympy.NDimArray | sympy.MatrixBase | list | Tensor):
        return sympy.simplify(_matrix(value).det())
    return value


def inverse(value: Any) -> Any:
    return sympy.ImmutableDenseNDimArray(_matrix(value).inv().tolist())


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


def _spacetime(factory: Callable[..., Metric]) -> Callable[..., Metric]:
    def make(*args: Any) -> Metric:
        return factory(*args)

    make.__name__ = factory.__name__
    return make


def _arity(factory: Callable[..., Metric]) -> tuple[int, int | None]:
    """How many arguments a spacetime takes in the language, from its signature: its
    positional parameters, of which those without a default are needed."""
    parameters = list(inspect.signature(factory).parameters.values())
    positional = [p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    needed = sum(1 for p in positional if p.default is p.empty)
    if any(p.kind is p.VAR_POSITIONAL for p in parameters):
        return needed, None
    return needed, len(positional)


# -- the table ---------------------------------------------------------------------

_ELEMENTARY = {
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

_ALGEBRA = {
    "simplify": sympy.simplify, "tsimplify": sympy.simplify, "expand": sympy.expand,
    "factor": sympy.factor, "collect": sympy.collect, "cancel": sympy.cancel,
    "apart": sympy.apart, "together": sympy.together, "trigsimp": sympy.trigsimp,
    "powsimp": sympy.powsimp, "expand_trig": sympy.expand_trig, "nsimplify": sympy.nsimplify,
    "diff": sympy.diff, "integrate": sympy.integrate, "limit": sympy.limit, "lim": sympy.limit,
    "series": sympy.series, "subs": subs, "N": numerical, "numerical": numerical,
    "evalf": numerical, "doit": doit,
}  # fmt: skip

_VALUES = {
    "solve": solve, "dsolve": dsolve, "RHS": rhs, "rhs": rhs, "LHS": lhs, "lhs": lhs,
    "Eq": sympy.Eq, "symbols": sympy.symbols, "latex": latex, "diag": diag, "det": det,
    "inverse": inverse, "inv": inverse, "transpose": transpose, "trace": trace,
    "log": sympy.log, "root": root, "binomial": sympy.binomial, "atan2": sympy.atan2,
    "min": sympy.Min, "Min": sympy.Min, "max": sympy.Max, "Max": sympy.Max,
    "gcd": sympy.gcd, "lcm": sympy.lcm,
}  # fmt: skip

_SPACETIMES: dict[str, Callable[..., Metric]] = {
    "Minkowski": spacetimes.minkowski,
    "Schwarzschild": spacetimes.schwarzschild,
    "ReissnerNordstrom": spacetimes.reissner_nordstrom,
    "Kerr": spacetimes.kerr,
    "KerrNewman": spacetimes.kerr_newman,
    "FLRW": spacetimes.flrw,
    "DeSitter": spacetimes.de_sitter,
    "AntiDeSitter": spacetimes.anti_de_sitter,
}

_ARITY: dict[str, tuple[int, int | None]] = {
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


def _table() -> dict[str, Builtin]:
    table: dict[str, Builtin] = {}
    for name, func in _ELEMENTARY.items():
        table[name] = Builtin(name, func, componentwise=True, scalar=True, min_args=1, max_args=1)
    for name, func in _ALGEBRA.items():
        low, high = _ARITY.get(name, (1, 1))
        table[name] = Builtin(
            name, func, componentwise=True, scalar=True, min_args=low, max_args=high
        )
    for name, func in _VALUES.items():
        low, high = _ARITY.get(name, (1, None))
        scalar = name in (
            "log",
            "root",
            "binomial",
            "atan2",
            "min",
            "Min",
            "max",
            "Max",
            "gcd",
            "lcm",
            "trace",
        )
        table[name] = Builtin(name, func, scalar=scalar, min_args=low, max_args=high)
    for name, factory in _SPACETIMES.items():
        low, high = _arity(factory)
        table[name] = Builtin(name, _spacetime(factory), min_args=low, max_args=high)
    return table


#: Every builtin function, by the name the language calls it.
BUILTINS = MappingProxyType(_table())

#: The names of the builtins that are spacetimes, such as ``Schwarzschild``.
SPACETIMES: tuple[str, ...] = tuple(_SPACETIMES)

#: Names with a fixed value: the constants of mathematics.
CONSTANTS = MappingProxyType(
    {
        "pi": sympy.pi,
        "e": sympy.E,
        "i": sympy.I,
        "oo": sympy.oo,
        "infty": sympy.oo,
        "True": sympy.true,
        "False": sympy.false,
    }
)
