"""Kernels: what each IR operation computes, on SymPy values or on numerical arrays,
which :mod:`relativisticpy.tensors.numerical` computes with NumPy."""

from __future__ import annotations

import cmath
import functools
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce, singledispatch
from typing import TYPE_CHECKING, Any

import sympy
from sympy import ImmutableDenseNDimArray, ImmutableMatrix
from sympy.core.function import AppliedUndef
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.polys.monomials import itermonomials

from relativisticpy import display
from relativisticpy.diffgeom import curvature
from relativisticpy.diffgeom.metric import metric_from_line_element
from relativisticpy.errors import EvaluationError, MetricError
from relativisticpy.ir import Node, ops
from relativisticpy.runtime import registry
from relativisticpy.tensors import arrays, linalg, numerical
from relativisticpy.tensors.equation import TensorEquation
from relativisticpy.tensors.simplification import is_zero

if TYPE_CHECKING:
    from relativisticpy.runtime.executor import Executor


@singledispatch
def execute(op: ops.Op, args: list[Any], executor: Executor) -> Any:
    raise NotImplementedError(f"no kernel for {type(op).__name__}")  # pragma: no cover


def refuse_quoted(node: Node, value: Any) -> None:
    """Refuse text written in quotes in a program as a value, as to the compiler, which
    would hand it to SymPy, which reads text as Python."""
    if isinstance(node.op, ops.Const) and node.span is not None and isinstance(value, str):
        raise EvaluationError(
            "text in quotes is not a value to compute with",
            span=node.span,
            hint="write the expression without quotes, as in sin(x) rather than sin('x')",
        )


def refuse_text(op: ops.Op, args: list[Any], nodes: Sequence[Node]) -> None:
    """Refuse text written in a program where a value is computed with: SymPy reads a
    string given to most of its functions as Python, and evaluates it, so that
    ``sin('...')`` could run any program. Text is given only to an operation the registry
    says takes names, as ``symbols('x y')`` does, which never evaluates it. The options
    RelativisticPy itself gives, such as the side of a limit, are not written in the
    program, and so have no span."""
    if isinstance(op, ops.Const):
        return
    if isinstance(op, ops.Call | ops.Map) and registry.lookup(op.name).takes_names:
        return  # it reads its text as names, never as Python
    for value, node in zip(args, nodes, strict=True):
        # A constant the compiler made, with no place in the source, is its own: a name it
        # passes to a kernel, alone or in a tuple. Text from the source has a span.
        own = isinstance(node.op, ops.Const) and node.span is None
        if _has_text(value) and not own:
            raise EvaluationError(
                "text in quotes is not a value to compute with",
                hint="write the expression without quotes, as in sin(x) rather than sin('x')",
            )


def _has_text(value: Any) -> bool:
    if isinstance(value, str):
        return True
    return isinstance(value, list | tuple) and any(_has_text(item) for item in value)


@execute.register(ops.Const)
def _const(op: ops.Const, args: list[Any], executor: Executor) -> Any:
    return op.value


@execute.register(ops.Call)
def _call(op: ops.Call, args: list[Any], executor: Executor) -> Any:
    entry = registry.lookup(op.name)  # what computes the operation the IR names
    if len(args) == 1 and numerical.is_numerical(args[0]) and _numeric(entry, 1) is not None:
        return numerical.apply(args[0], entry.func, numeric=entry.numeric)  # NumPy's sqrt
    return entry.func(*args, **dict(op.attributes))


def _numeric(entry: registry.Implementation, count: int) -> Any:
    """What computes an operation with arrays, given ``count`` arguments, if it has one."""
    if entry.numeric is not None and entry.numeric.takes(count):
        return entry.numeric
    return None


# Arithmetic follows mathematics, not Python: a list is a vector or a matrix, so
# 2 [1, 2] is [2, 4] rather than [1, 2, 1, 2], and matrices multiply as matrices.


@execute.register(ops.Add)
def _add(op: ops.Add, args: list[Any], executor: Executor) -> Any:
    terms = [_arithmetic(arg) for arg in args]
    if numerical.involves(terms):
        terms = [numerical.as_array(term) for term in terms]
    shapes = list(dict.fromkeys(_shape(term) for term in terms))
    if len(shapes) > 1:
        if () in shapes:
            raise EvaluationError("cannot add a scalar and an array")
        raise EvaluationError(f"cannot add arrays of shapes {' and '.join(map(str, shapes))}")
    signed = [term if sign > 0 else -term for sign, term in zip(op.signs, terms, strict=True)]
    if shapes == [()] and all(isinstance(term, sympy.Expr) for term in signed):
        return sympy.Add(*signed)  # one canonical sum, rather than one per term added
    total = signed[0]
    for term in signed[1:]:
        total = total + term
    return numerical.as_array(total) if numerical.involves(terms) else total  # not np.matrix


@execute.register(ops.Mul)
def _mul(op: ops.Mul, args: list[Any], executor: Executor) -> Any:
    values = [_arithmetic(arg) for arg in args]
    if numerical.involves(values) and not any(_shape(value) for value in values):
        values = [numerical.number(value) for value in values]  # numbers, or JAX's
    return reduce(_times, values)


@execute.register(ops.Scale)
def _scale(op: ops.Scale, args: list[Any], executor: Executor) -> Any:
    *factors, components = args
    factor = _mul(ops.Mul(), factors, executor)
    if _shape(factor):
        raise EvaluationError(
            "a tensor can be multiplied by scalars, not by an array",
            hint="to combine it with components, define a tensor and use index notation",
        )
    if numerical.is_numerical(components) or numerical.is_numerical(factor):
        return numerical.number(factor) * numerical.as_array(components)
    return factor * components


@execute.register(ops.Pow)
def _pow(op: ops.Pow, args: list[Any], executor: Executor) -> Any:
    base, exponent = (_arithmetic(arg) for arg in args)
    if numerical.involves([base, exponent]) and not (_shape(base) or _shape(exponent)):
        return numerical.number(base) ** numerical.number(exponent)
    if not (_shape(base) or _shape(exponent)):
        return base**exponent
    square = len(_shape(base)) == 2 and base.shape[0] == base.shape[1]
    if square and not _shape(exponent) and sympy.sympify(exponent).is_integer:
        return arrays.to_array(base.tomatrix() ** exponent)
    raise EvaluationError("only a square matrix can be raised to a power, and only an integer one")


def _arithmetic(value: Any) -> Any:
    """A value as arithmetic sees it: lists and tuples are arrays."""
    if not isinstance(value, list | tuple):
        return value
    try:
        return arrays.to_array(value)
    except ValueError:
        raise EvaluationError(
            "only rectangular lists, such as vectors and matrices, have arithmetic"
        ) from None


def _shape(value: Any) -> tuple[int, ...]:
    """An array's shape; a scalar's is ``()``."""
    if isinstance(value, sympy.NDimArray) or numerical.is_numerical(value):
        return tuple(getattr(value, "shape", ()))
    return ()


def _times(left: Any, right: Any) -> Any:
    if not (_shape(left) and _shape(right)):
        return left * right
    if len(left.shape) == len(right.shape) == 2 and left.shape[1] == right.shape[0]:
        return arrays.to_array(left.tomatrix() * right.tomatrix())
    raise EvaluationError(
        f"cannot multiply arrays of shapes {left.shape} and {right.shape}",
        hint="matrices multiply when their shapes match; for other products use index notation",
    )


@execute.register(ops.Contract)
def _contract(op: ops.Contract, args: list[Any], executor: Executor) -> Any:
    return arrays.contract(list(zip(args, op.labels, strict=True)), op.output, op.sizes)


@execute.register(ops.Permute)
def _permute(op: ops.Permute, args: list[Any], executor: Executor) -> Any:
    return arrays.permute(args[0], op.axes)


@execute.register(ops.Select)
def _select(op: ops.Select, args: list[Any], executor: Executor) -> Any:
    return arrays.select(args[0], dict(op.fixed))


@execute.register(ops.Partial)
def _partial(op: ops.Partial, args: list[Any], executor: Executor) -> Any:
    return arrays.partial(args[0], op.coordinates)


@execute.register(ops.Symmetrize)
def _symmetrize(op: ops.Symmetrize, args: list[Any], executor: Executor) -> Any:
    return arrays.symmetrize(args[0], op.axes, antisymmetric=op.antisymmetric)


@dataclass(frozen=True)
class Positioned:
    """``func`` with the component it is given first put at ``position`` among the other
    arguments, as the x of besselj(nu, x); for ``numeric``, which is given the array
    module before them. Picklable, as :class:`ComponentCall` is."""

    func: Callable[..., Any]
    position: int
    numeric: bool = False

    def __call__(self, *args: Any, **attributes: Any) -> Any:
        before = 1 if self.numeric else 0  # the array module stays first
        lead, (component, *rest) = args[:before], args[before:]
        at = self.position
        return self.func(*lead, *rest[:at], component, *rest[at:], **attributes)


@dataclass(frozen=True)
class ComponentCall:
    """``func(component, *extra, **attributes)``: picklable, so it can run in a worker
    process."""

    func: Callable[..., Any]
    extra: tuple[Any, ...]
    attributes: ops.Attributes = ()

    def __call__(self, component: Any) -> Any:
        return self.func(component, *self.extra, **dict(self.attributes))


@execute.register(ops.Map)
def _map(op: ops.Map, args: list[Any], executor: Executor) -> Any:
    entry = registry.lookup(op.name)  # what computes the operation the IR names
    func: Callable[..., Any] = entry.func
    numeric = _numeric(entry, len(args))
    if op.position:  # the component goes after the arguments before it
        func = Positioned(func, op.position)
        numeric = Positioned(numeric, op.position, numeric=True) if numeric else None
    target, extra = args[0], tuple(args[1:])
    if numerical.is_numerical(target):
        each = functools.partial(func, **dict(op.attributes)) if op.attributes else func
        if numeric is not None and op.attributes:
            numeric = functools.partial(numeric, **dict(op.attributes))
        return numerical.apply(target, each, extra, numeric)
    call = ComponentCall(func, extra, op.attributes)
    if not isinstance(target, sympy.NDimArray):
        return call(target)
    results = executor.map(call, arrays.flatten(target))
    return ImmutableDenseNDimArray(results, target.shape)


@execute.register(ops.MakeArray)
def _make_array(op: ops.MakeArray, args: list[Any], executor: Executor) -> Any:
    cells = [_arithmetic(arg) for arg in args]
    if len(op.shape) == 2 and any(_shape(cell) for cell in cells):
        return _blocks(op.shape, cells)
    return ImmutableDenseNDimArray([sympy.sympify(a) for a in args], op.shape)


def _blocks(shape: tuple[int, ...], cells: list[Any]) -> Any:
    """A matrix written in blocks, as ``\\begin{pmatrix} B & AB \\end{pmatrix}`` and
    ``\\begin{pmatrix} P & A^T \\\\ A & 0 \\end{pmatrix}``: each block a matrix, or a number,
    which is as tall as the blocks of its row and as wide as those of its column, and must
    be 0 where that is more than one entry."""
    rows, columns = shape
    grid = [cells[row * columns : (row + 1) * columns] for row in range(rows)]
    if any(len(_shape(cell)) not in (0, 2) for cell in cells):
        raise EvaluationError(
            "a block of a matrix is a matrix or a number, not a list",
            hint="write a column as one: \\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}",
        )
    heights = [next((cell.shape[0] for cell in row if _shape(cell)), 1) for row in grid]
    widths = [
        next((row[column].shape[1] for row in grid if _shape(row[column])), 1)
        for column in range(columns)
    ]
    numeric = numerical.involves(cells)
    blocks: list[list[Any]] = []
    for row, height in zip(grid, heights, strict=True):
        blocks.append([])
        for cell, width in zip(row, widths, strict=True):
            if _shape(cell) and tuple(cell.shape) != (height, width):
                raise EvaluationError(
                    f"the blocks of a matrix line up, but a {cell.shape[0]}×{cell.shape[1]} "
                    f"block stands where a {height}×{width} one fits"
                )
            if not _shape(cell) and (height, width) != (1, 1) and cell != 0:
                raise EvaluationError(
                    f"the number {display.unicode(cell)} stands for a {height}×{width} block of "
                    "a matrix, which only 0 can"
                )
            blocks[-1].append(_block(cell, height, width, numeric))
    if numeric:
        return numerical.numpy().block(blocks)
    matrix = sympy.Matrix.vstack(*(sympy.Matrix.hstack(*row) for row in blocks))
    return arrays.to_array(matrix)


def _block(cell: Any, height: int, width: int, numeric: bool) -> Any:
    """A block as a matrix: a number fills a block of its size, as 0 does."""
    if numeric:
        np = numerical.numpy()
        if _shape(cell):
            return numerical.dense(numerical.as_array(cell))
        return np.full((height, width), numerical.number(cell))
    if _shape(cell):
        return sympy.Matrix(cell.tolist())
    return sympy.Matrix(height, width, [sympy.sympify(cell)] * (height * width))


@execute.register(ops.ToArray)
def _to_array(op: ops.ToArray, args: list[Any], executor: Executor) -> Any:
    value = args[0]
    expected = f"expected {_size(op.shape)} components"
    if numerical.is_numerical(value):
        if tuple(value.shape) != op.shape:
            raise EvaluationError(f"{expected}, but found {_size(tuple(value.shape))}")
        return value
    if isinstance(value, sympy.NDimArray | sympy.MatrixBase | list | tuple):
        try:
            array = arrays.to_array(value)
        except ValueError:
            raise EvaluationError(f"{expected}, but these are ragged") from None
        if array.shape != op.shape and len(array) == len(ImmutableDenseNDimArray.zeros(*op.shape)):
            array = array.reshape(*op.shape)  # a 1×n matrix for a vector
        if array.shape == op.shape:
            return array
        raise EvaluationError(f"{expected}, but found {_size(tuple(array.shape))}")
    if value == 0:
        return arrays.zeros_of(op.shape)
    raise EvaluationError(f"{expected}, but found the scalar {display.unicode(value)}")


def _size(shape: tuple[int, ...]) -> str:
    """A shape as messages write it: ``3``, ``4×4``."""
    return "×".join(str(n) for n in shape) if shape else "no"


@execute.register(ops.Equation)
def _equation(op: ops.Equation, args: list[Any], executor: Executor) -> Any:
    lhs, rhs = args
    if numerical.involves([lhs, rhs]):  # numbers agree to rounding error, or they differ
        if _shape(lhs) or _shape(rhs):
            return TensorEquation(lhs, rhs)
        if _symbolic(lhs) or _symbolic(rhs):  # a number and a symbol: an equation to solve
            return sympy.Eq(lhs, rhs, evaluate=False)
        return sympy.true if numerical.close(lhs, rhs) else sympy.false  # never Eq(25.0, 24)
    if isinstance(lhs, sympy.NDimArray) or isinstance(rhs, sympy.NDimArray):
        array = lhs if isinstance(lhs, sympy.NDimArray) else rhs
        lhs = lhs if isinstance(lhs, sympy.NDimArray) else arrays.zeros_of(array.shape)
        rhs = rhs if isinstance(rhs, sympy.NDimArray) else arrays.zeros_of(array.shape)
        return TensorEquation(lhs, rhs)
    if isinstance(lhs, sympy.Expr) and isinstance(rhs, sympy.Expr):
        decided = _proven(lhs, rhs)
        if decided is not None:
            return sympy.true if decided else sympy.false
    return sympy.Eq(lhs, rhs)


def _symbolic(value: Any) -> bool:
    """Whether a value has symbols in it, rather than being a number."""
    return isinstance(value, sympy.Basic) and bool(value.free_symbols)


@execute.register(ops.Inverse)
def _inverse(op: ops.Inverse, args: list[Any], executor: Executor) -> Any:
    return linalg.inverse(args[0])


@execute.register(ops.LineElement)
def _line_element(op: ops.LineElement, args: list[Any], executor: Executor) -> Any:
    return ImmutableDenseNDimArray(metric_from_line_element(args[0], op.coordinates).tolist())


@execute.register(ops.Geometry)
def _geometry(op: ops.Geometry, args: list[Any], executor: Executor) -> Any:
    simplify = executor.simplify_map
    coordinates = op.coordinates
    quantity = op.quantity
    if quantity == "kronecker":
        return curvature.kronecker(len(coordinates))
    if quantity == "metric":  # the same error as a Metric built in Python gives
        try:
            matrix = curvature.metric_matrix(args[0])
        except ValueError as error:
            raise MetricError(str(error)) from None
        if matrix.rows != len(coordinates):
            raise MetricError(
                f"a metric in {len(coordinates)} coordinates must be {len(coordinates)}×"
                f"{len(coordinates)}, not {matrix.rows}×{matrix.cols}"
            )
        try:
            curvature.check_metric(matrix)
        except ValueError as error:
            raise MetricError(str(error)) from None
        return ImmutableDenseNDimArray(matrix.tolist())
    matrices = [ImmutableMatrix(a.tolist()) if _is_matrix(a) else a for a in args]
    if quantity == "inverse":
        return _array(curvature.inverse(matrices[0], simplify))
    if quantity == "determinant":
        return curvature.determinant(matrices[0])
    if quantity == "christoffel":
        return curvature.christoffel(matrices[0], matrices[1], coordinates, simplify)
    if quantity == "riemann":
        return curvature.riemann(args[0], coordinates, simplify)
    if quantity == "ricci":
        return curvature.ricci(args[0], coordinates, simplify)
    if quantity == "ricci_scalar":
        return curvature.ricci_scalar(args[0], matrices[1])
    if quantity == "einstein":
        return curvature.einstein(args[0], args[1], matrices[2], simplify)
    if quantity == "riemann_lower":
        return curvature.lower_riemann(args[0], matrices[1])
    if quantity == "weyl":
        return curvature.weyl(args[0], args[1], args[2], matrices[3], simplify)
    if quantity == "kretschmann":
        return curvature.kretschmann(args[0], matrices[1], matrices[2])
    raise NotImplementedError(quantity)  # pragma: no cover


def _is_matrix(value: Any) -> bool:
    return isinstance(value, sympy.NDimArray) and value.rank() == 2


def _array(matrix: ImmutableMatrix) -> ImmutableDenseNDimArray:
    return ImmutableDenseNDimArray(matrix.tolist())


# -- scalar equations ----------------------------------------------------------------

#: Floating-point numbers are equal when they differ by at most this, relative to their size.
TOLERANCE = 1e-12
# Above this many operations a difference is not simplified, as simplify can take minutes.
_SIMPLIFY_LIMIT = 60
_UNEVALUATED = (sympy.Sum, sympy.Product, sympy.Integral, sympy.Limit, sympy.Derivative)
_TRANSCENDENTAL = (TrigonometricFunction, HyperbolicFunction)
_ARRAYS = (sympy.NDimArray, sympy.MatrixBase)


def _proven(lhs: sympy.Expr, rhs: sympy.Expr) -> bool | None:
    """Whether the two sides of a scalar equation are equal: ``True`` when that is proven,
    ``False`` when two numbers differ, and ``None`` when neither is known.

    Sums, products and integrals are carried out first. Exactly, the difference of the
    sides must vanish. Floating-point numbers are equal to a relative :data:`TOLERANCE`,
    so ``0.1 + 0.2 = 0.3`` holds; in an expression, each is first replaced by the simplest
    fraction within it.
    """
    if lhs == rhs:
        return True
    if lhs.atoms(*_ARRAYS) or rhs.atoms(*_ARRAYS):  # a list inside a function, from \pm
        return True if is_zero(lhs - rhs) else None
    lhs, rhs = _carried_out(lhs), _carried_out(rhs)
    if lhs.has(sympy.Float) or rhs.has(sympy.Float):
        if lhs.is_number and rhs.is_number:
            return _close(lhs, rhs)
        lhs, rhs = _rounded(lhs), _rounded(rhs)
    if is_zero(lhs - rhs) or _simplifies_to_zero(lhs, rhs):
        return True
    return None


def _carried_out(expr: sympy.Expr) -> sympy.Expr:
    """``expr`` with its sums, products, integrals, limits and derivatives carried out, in
    the time a closed form is given; as written where SymPy cannot, or not in that time, as
    an integral its heuristics search for ever."""
    from relativisticpy.runtime import core

    if not expr.has(*_UNEVALUATED):
        return expr
    try:
        with core._budget(core._CLOSED_FORM_BUDGET):
            done = expr.doit()
    except (Exception, core._OverBudget):  # the equation is judged as written
        return expr
    return done if isinstance(done, sympy.Expr) else expr


def _close(lhs: sympy.Expr, rhs: sympy.Expr) -> bool | None:
    """Whether two numbers agree to the relative :data:`TOLERANCE` (``None`` if one of
    them is not a finite number)."""
    try:
        left, right = complex(lhs), complex(rhs)
    except (TypeError, ValueError):
        return None
    if not all(cmath.isfinite(value) for value in (left, right)):
        return None
    return abs(left - right) <= TOLERANCE * max(abs(left), abs(right))


def _rounded(expr: sympy.Expr) -> sympy.Expr:
    """``expr`` with each floating-point number replaced by the simplest fraction within
    the :data:`TOLERANCE`, so that exact algebra compares expressions to that tolerance:
    ``1.1 \\times 3`` and ``3.3`` both become 33/10, and an exponent 2.8333... 17/6."""
    fractions = {}
    for number in expr.atoms(sympy.Float):
        if number.is_finite:
            exact = sympy.Rational(number)  # the number's binary value, exactly
            fractions[number] = _simplest_fraction(Fraction(int(exact.p), int(exact.q)))
    return expr.xreplace(fractions)


def _simplest_fraction(exact: Fraction) -> sympy.Rational:
    """The first convergent of the continued fraction of ``exact`` within the
    :data:`TOLERANCE`, relative to it: the fraction with the smallest denominator, near
    enough."""
    # The convergents h/k, from their recurrence h = a h' + h'', k = a k' + k''.
    h, h_before, k, k_before = 1, 0, 0, 1
    rest = abs(exact)
    while True:
        whole = math.floor(rest)
        h, h_before, k, k_before = whole * h + h_before, h, whole * k + k_before, k
        if abs(Fraction(h, k) - abs(exact)) <= TOLERANCE * abs(exact) or rest == whole:
            return sympy.Rational(h if exact >= 0 else -h, k)
        rest = 1 / (rest - whole)


def _simplifies_to_zero(lhs: sympy.Expr, rhs: sympy.Expr) -> bool:
    """A bounded attempt to prove the sides equal where :func:`is_zero` could not: with
    trigonometric and hyperbolic functions written as exponentials, as for
    :math:`\\cosh^2 x - \\sinh^2 x = 1`, then with :func:`sympy.simplify`. Simplifying is
    slow, so it is tried only on a small difference that :func:`is_zero` does not already
    decide, and that numbers do not show to be nonzero."""
    difference = lhs - rhs
    if sympy.count_ops(difference) > _SIMPLIFY_LIMIT:
        return False
    try:
        if difference.has(*_TRANSCENDENTAL) and is_zero(difference.rewrite(sympy.exp)):
            return True
        if not _beyond_algebra(difference) or differs(lhs, rhs, samples=1):
            return False
        return bool(sympy.simplify(difference) == 0)
    except Exception:  # SymPy fails on some expressions: then nothing is proven
        return False


def _beyond_algebra(expr: sympy.Expr) -> bool:
    """Whether ``expr`` is more than a rational function of symbols, exponentials and
    undefined functions and their derivatives, which :func:`is_zero` decides exactly;
    or whether it is a number, which :func:`is_zero` does not try to prove zero."""
    if expr.is_number:
        return True
    for node in sympy.preorder_traversal(expr):
        if isinstance(node, sympy.Pow) and not node.exp.is_Integer:
            return True
        if isinstance(node, sympy.Function) and not isinstance(node, AppliedUndef | sympy.exp):
            return True
    return False


def differs(lhs: sympy.Expr, rhs: sympy.Expr, samples: int = 3) -> bool | None:
    """Whether numbers show that the two sides of an equation not proven to hold differ:
    ``True`` if they do wherever they are tried, ``False`` if they agree somewhere, and
    ``None`` if they cannot be computed as numbers.

    Each symbol is given a positive value, or a whole one if it is an integer or the limit
    of a sum, and each function that nothing defines, such as ``\\alpha(r)``, a random
    polynomial. An identity in every such function agrees; an equation that is not one
    differs, but for a coincidence. The points are the same every time, so that verdicts
    do not change from one run to the next.
    """
    if lhs.atoms(*_ARRAYS) or rhs.atoms(*_ARRAYS):
        return None  # \Gamma(n, \pm x) holds a list: it is not one number
    arities: dict[Any, int] = {}
    for application in lhs.atoms(AppliedUndef) | rhs.atoms(AppliedUndef):
        if arities.setdefault(application.func, len(application.args)) != len(application.args):
            return None  # f(x) and f(x, y): no one polynomial stands for f
    symbols = sorted(lhs.free_symbols | rhs.free_symbols, key=str)
    functions = sorted(arities.items(), key=lambda item: str(item[0]))
    floats = lhs.has(sympy.Float) or rhs.has(sympy.Float)
    bounds = {  # the limits of a sum or a product are whole numbers
        symbol
        for side in (lhs, rhs)
        for operation in side.atoms(sympy.Sum, sympy.Product)
        for limit in operation.limits
        for symbol in sympy.Tuple(*limit[1:]).free_symbols
    }
    rng = random.Random(0)
    verdicts = []
    for _ in range(samples):
        values = {symbol: _value(symbol, rng, symbol in bounds) for symbol in symbols}
        polynomials = {function: _polynomial(arity, rng) for function, arity in functions}
        verdicts.append(_differs_at(lhs, rhs, values, polynomials, floats))
    if any(verdict is False for verdict in verdicts):
        return False
    return True if any(verdicts) else None


def _value(symbol: sympy.Symbol, rng: random.Random, whole: bool) -> sympy.Expr:
    """A value for a symbol, exact so that terms that cancel are seen to."""
    if whole or symbol.is_integer:
        value = sympy.Integer(rng.randrange(5, 60))
    else:
        value = sympy.Rational(rng.randrange(300, 2000), 1009)  # between 0.3 and 2
    return -value if symbol.is_negative or symbol.is_nonpositive else value


def _polynomial(arity: int, rng: random.Random) -> sympy.Lambda:
    """A random polynomial of degree 3 in ``arity`` variables."""
    variables = sympy.symbols(f"v:{arity}")
    monomials = sorted(itermonomials(variables, 3), key=sympy.default_sort_key)
    terms = [sympy.Rational(rng.randrange(-1000, 1000), 997) * m for m in monomials]
    return sympy.Lambda(variables, sympy.Add(*terms))


def _differs_at(
    lhs: sympy.Expr,
    rhs: sympy.Expr,
    values: dict[sympy.Symbol, sympy.Expr],
    polynomials: dict[Any, sympy.Lambda],
    floats: bool,
) -> bool | None:
    """Whether the sides differ at one point (``None`` if they cannot be computed there).
    Exact sides differ when their difference has a significant digit; with floating-point
    numbers, they must also differ by more than the :data:`TOLERANCE`."""

    def concrete(expr: sympy.Expr) -> sympy.Expr:
        return expr.replace(
            lambda e: isinstance(e, AppliedUndef), lambda e: polynomials[e.func](*e.args)
        ).replace(lambda e: isinstance(e, sympy.Derivative | sympy.Subs), lambda e: e.doit())

    try:
        difference = concrete(lhs - rhs)
        if not _computable(difference):
            return None
        value = sympy.N(difference, 30, subs=values)
        if value == 0:
            return False
        parts = [part for part in value.as_real_imag() if part != 0]
        if not all(part.is_Float and part.is_finite for part in parts):
            return None  # not a number, or not a finite one
        if all(part._prec <= 1 for part in parts):
            # No digit is significant: the terms cancel, or an integral did not converge.
            return None if difference.has(sympy.Integral) else False
        if not floats:
            return True
        left, right = (complex(sympy.N(concrete(side), 30, subs=values)) for side in (lhs, rhs))
    except Exception:  # SymPy cannot compute it here
        return None
    return abs(left - right) > TOLERANCE * max(abs(left), abs(right))


def _computable(expr: sympy.Expr) -> bool:
    """Whether numbers for ``expr`` can be trusted, and come quickly: it has no limit, and no
    sum, product or integral to infinity."""
    if expr.has(sympy.Limit):
        return False
    return not any(
        bound.has(sympy.oo, -sympy.oo, sympy.zoo)
        for operation in expr.atoms(sympy.Sum, sympy.Product, sympy.Integral)
        for limit in operation.limits
        for bound in limit[1:]
    )
