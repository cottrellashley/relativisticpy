r"""The core operations: what the compiler emits for notation that is no function of the
language, such as a sum, an evaluation bar or a time derivative, computed by the engines.

Each is registered by :mod:`relativisticpy.runtime.registry` under a name of its own, as
``core.sum`` or ``core.subs``, and is a function of the values of its arguments, then of
its attributes, by name: module-level, so that a worker process can run it.
"""

from __future__ import annotations

import contextlib
import itertools
import math
import re
import signal
import threading
import time
from collections.abc import Iterator, Sequence
from typing import Any

import sympy
from sympy import Symbol
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational

from relativisticpy import assumptions, display, integrations
from relativisticpy.diffgeom.metric import differential
from relativisticpy.errors import EvaluationError, MetricError
from relativisticpy.tensors import Indices, Tensor, arrays, linalg, numerical
from relativisticpy.tensors.equation import TensorEquation
from relativisticpy.tensors.indices import latex_label
from relativisticpy.tensors.simplification import is_zero

# Spaces up to this size get SymPy's identity and ones, which stay exact with symbols.
_SMALL = 16


# -- collections ---------------------------------------------------------------------


def make_list(*items: Any) -> list[Any]:
    return list(items)


def make_tuple(*items: Any) -> tuple[Any, ...]:
    return tuple(items)


def limits(variable: Any, lower: Any, upper: Any) -> tuple[Any, Any, Any]:
    return (variable, lower, upper)


def piecewise(*args: Any) -> Any:
    return sympy.Piecewise(*zip(args[::2], args[1::2], strict=True))


# -- calculus --------------------------------------------------------------------------


#: How long finding the closed form of a sum or a product may take, in seconds: past it,
#: it is kept as written (design/notation.md, decision 4).
_CLOSED_FORM_BUDGET = 5.0


def summation(
    body: Any, variable: Any, lower: Any, upper: Any, *, assuming: tuple[Any, ...] = ()
) -> Any:
    """A sum: carried out where it has a whole number of terms, as \\sum_{n=0}^{2} has;
    otherwise in closed form, where SymPy finds one within a time budget, with the index an
    integer and, where it converges only under conditions, under the bounds the program
    assumes or else the assumption books make (see
    :func:`relativisticpy.assumptions.summed`); else kept as it is written
    (design/notation.md, decision 4)."""
    written = sympy.Sum(body, (variable, lower, upper))
    finite = _finite(written, lower, upper)
    if finite is not written:
        return finite
    try:
        with _budget(_CLOSED_FORM_BUDGET):
            found = assumptions.summed(body, [(variable, lower, upper)], assuming)
    except _OverBudget:
        return written
    return written if isinstance(found, sympy.Basic) and found.has(sympy.Sum) else found


def product(body: Any, variable: Any, lower: Any, upper: Any) -> Any:
    """A product, carried out as a sum is, where SymPy finds its closed form within a time
    budget: see :func:`summation`."""
    written = sympy.Product(body, (variable, lower, upper))
    finite = _finite(written, lower, upper)
    if finite is not written:
        return finite
    try:
        with _budget(_CLOSED_FORM_BUDGET):
            found = written.doit()
    except _OverBudget:
        return written
    return written if isinstance(found, sympy.Basic) and found.has(sympy.Product) else found


class _OverBudget(BaseException):
    """A computation that took longer than it was given, and is kept as written. Not an
    Exception, so that SymPy, which catches those, does not carry on past it."""


@contextlib.contextmanager
def _budget(seconds: float) -> Iterator[None]:
    """Raise :class:`_OverBudget` if the block runs longer than ``seconds``, leaving a time
    limit in force around it as it was, with what remains of it; one that ends first
    decides. Only with ``SIGALRM``, in the main thread; elsewhere the block runs as long
    as it takes."""
    usable = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    outer = signal.getitimer(signal.ITIMER_REAL)[0] if usable else 0.0
    if not usable or (outer and outer <= seconds):
        yield
        return

    def expire(signum: int, frame: Any) -> None:
        raise _OverBudget

    start = time.monotonic()
    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if outer:
            signal.setitimer(signal.ITIMER_REAL, max(outer - (time.monotonic() - start), 1e-6))


def _finite(written: Any, lower: Any, upper: Any) -> Any:
    bounds = [sympy.sympify(bound) for bound in (lower, upper)]
    if all(bound.is_Integer for bound in bounds) and bounds[1] - bounds[0] < 10_000:
        return written.doit()
    return written


def limit(body: Any, variable: Any, point: Any, direction: str) -> Any:
    """A limit (see :func:`_limit`); where SymPy cannot take it for not knowing the sign of
    a parameter, the limit under the assumption books make, which a note says."""
    try:
        return _limit(body, variable, point, direction)
    except NotImplementedError:  # "Result depends on the sign of sign(k)"
        found = assumptions.limited(_limit, body, variable, point, direction)
        if found is None:
            raise
        return found


def _limit(body: Any, variable: Any, point: Any, direction: str) -> Any:
    """A limit, never taken in a case that holds only at points of the variable, as
    ``Eq(k, 0)`` does of a scattering amplitude: a limit is of the values near the point,
    and SymPy would take that case's value at the point itself. One that does not exist,
    as of a step from both sides or of sin(1/h), says so, where SymPy would give a value."""
    if direction == "+-" and point in (sympy.oo, -sympy.oo):
        direction = "-" if point == sympy.oo else "+"
    near = _near(sympy.sympify(body), variable)
    if not near.has(sympy.Piecewise) or point in (sympy.oo, -sympy.oo):
        try:
            found = sympy.limit(near, variable, point, direction)
        except ValueError as error:  # |x|/x at 0: said in RelativisticPy's words
            if "does not exist" not in str(error):
                raise
            said = re.search(r"left hand limit = (.+) and right hand limit = (.+)", str(error))
            why = f": it is {said[1]} from below and {said[2]} from above" if said else ""
            raise ValueError(f"the limit does not exist{why}") from None
        return _existing(found)
    # SymPy takes a case as it holds at the point: each side is approached by a step
    step = sympy.Dummy("h", positive=True)
    values = {
        side: _existing(sympy.limit(near.subs(variable, point + sign * step), step, 0, "+"))
        for side, sign in (("+", 1), ("-", -1))
        if side in direction
    }
    if len(values) == 2 and sympy.simplify(values["-"] - values["+"]) != 0:
        raise ValueError(
            f"the limit does not exist: it is {values['-']} from below and {values['+']} from above"
        )
    return next(iter(values.values()))


def _existing(value: Any) -> Any:
    """A limit SymPy found, unless it only bounds the values, as AccumBounds(-1, 1) does
    of sin(1/h), whose values never settle."""
    if isinstance(value, sympy.AccumBounds):
        raise ValueError(
            f"the limit does not exist: the values keep between {value.min} and {value.max}"
        )
    return value


def _near(expr: Any, variable: Any) -> Any:
    """``expr`` with each piecewise part as it is near any point of ``variable``: without
    the cases that hold only at points, and the first case that holds at every point but
    some taken for all the rest, which no limit sees."""

    def generic(piecewise: Any) -> Any:
        kept = []
        for value, condition in piecewise.args:
            if variable not in condition.free_symbols:
                kept.append((value, condition))
            elif _at_points(condition):
                continue
            elif _but_at_points(condition):
                kept.append((value, sympy.true))
                break
            else:
                kept.append((value, condition))
        if len(kept) == 1 and kept[0][1] is sympy.true:
            return kept[0][0]
        return sympy.Piecewise(*kept) if kept else piecewise

    if not isinstance(expr, sympy.Basic) or not expr.has(sympy.Piecewise):
        return expr
    return expr.replace(lambda node: isinstance(node, sympy.Piecewise), generic)


def _at_points(condition: Any) -> bool:
    """Whether a condition holds only at isolated values, as ``Eq(k, 0)``, or either of
    several such, as ``Eq(k, 0) | Eq(\\theta, 0)`` does."""
    if isinstance(condition, sympy.Eq):
        return True
    if isinstance(condition, sympy.Or):
        return all(_at_points(part) for part in condition.args)
    if isinstance(condition, sympy.And):
        return any(_at_points(part) for part in condition.args)
    return False


def _but_at_points(condition: Any) -> bool:
    """Whether a condition holds everywhere but at isolated values, as ``Ne(x, 0)``."""
    if isinstance(condition, sympy.Ne):
        return True
    if isinstance(condition, sympy.And):
        return all(_but_at_points(part) for part in condition.args)
    if isinstance(condition, sympy.Or):
        return any(_but_at_points(part) for part in condition.args)
    return False


def differentiate(expr: Any, orders: tuple[int, ...], *variables: Any) -> Any:
    return sympy.diff(expr, *((v, n) for v, n in zip(variables, orders, strict=True)))


def derivative_at(body: Any, dummy: Any, order: int, point: Any) -> Any:
    """``f'(p)``. Where the cases of a piecewise function meet at p, as x sin(1/x) and 0 do
    at 0, it is taken from its definition, the limit of the difference quotient, which may
    not exist, rather than case by case, which would give the derivative of the case that
    holds at p alone."""
    body = sympy.sympify(body)
    if not _on_a_seam(body, dummy, point):
        return sympy.diff(body, dummy, order).subs(dummy, point)
    if order != 1:
        raise ValueError(
            f"a derivative of order {order} where the cases of a function meet is not taken yet"
        )
    step = sympy.Dummy("h")
    quotient = (body.subs(dummy, point + step) - body.subs(dummy, point)) / step
    try:
        return limit(quotient, step, 0, "+-")
    except ValueError as error:
        why = str(error).removeprefix("the limit does not exist").lstrip(": ")
        raise ValueError(
            f"the derivative at {point} does not exist: its difference quotient has no limit"
            + (f", as {why}" if why else "")
        ) from None


def _on_a_seam(body: Any, dummy: Any, point: Any) -> bool:
    """Whether the cases of a piecewise part of ``body`` meet at ``point``: a condition on
    ``dummy`` whose two sides are equal there, as x \\neq 0 at 0."""
    for piecewise in body.atoms(sympy.Piecewise):
        for _, condition in piecewise.args:
            for relation in condition.atoms(sympy.core.relational.Relational):
                if dummy in relation.free_symbols:
                    gap = (relation.lhs - relation.rhs).subs(dummy, point)
                    if sympy.simplify(gap) == 0:
                        return True
    return False


def gradient(expr: Any, vector: Any) -> Any:
    return sympy.ImmutableDenseNDimArray([sympy.diff(expr, component) for component in vector])


def time_derivative(expr: Any, order: int) -> Any:
    r"""``\dot{x}``: the derivative with respect to the single variable that the
    functions in ``x`` depend on; a plain symbol ``a`` gives the symbol ``adot``."""
    expr = sympy.sympify(expr)
    applied = expr.atoms(sympy.core.function.AppliedUndef)
    parameters = {argument for function in applied for argument in function.args}
    if len(parameters) == 1:
        return sympy.diff(expr, parameters.pop(), order)
    if isinstance(expr, Symbol):
        return sympy.Symbol(expr.name + ("dot" if order == 1 else "ddot"))
    raise ValueError(
        f"cannot tell what {expr} changes with: write it as a function of one variable, e.g. a(t)"
    )


def substitute(expr: Any, *pairs: Any) -> Any:
    """An evaluation bar, ``x - y |_{x=y, y=x}``: every substitution at once, as a
    change of variables is, rather than one after another."""
    rules = list(zip(pairs[::2], pairs[1::2], strict=True))
    return sympy.sympify(expr).subs(rules, simultaneous=True)


# -- functions and relations -------------------------------------------------------------


def function(*args: Any, name: str) -> Any:
    """``f(x, y)`` of a function no one defined, called ``name``: SymPy's undefined
    function, applied."""
    return sympy.Function(name)(*args)


def less(lhs: Any, rhs: Any, *, assuming: tuple[Any, ...] = ()) -> Any:
    return _decided(sympy.Lt(lhs, rhs), assuming)


def greater(lhs: Any, rhs: Any, *, assuming: tuple[Any, ...] = ()) -> Any:
    return _decided(sympy.Gt(lhs, rhs), assuming)


def less_equal(lhs: Any, rhs: Any, *, assuming: tuple[Any, ...] = ()) -> Any:
    return _decided(sympy.Le(lhs, rhs), assuming)


def greater_equal(lhs: Any, rhs: Any, *, assuming: tuple[Any, ...] = ()) -> Any:
    return _decided(sympy.Ge(lhs, rhs), assuming)


def _decided(relation: Any, bounds: tuple[Any, ...]) -> Any:
    """An inequality, decided where the bounds the program assumes decide it, as x < 2 is
    where 0 < x < 1 is assumed."""
    if not bounds or not isinstance(relation, Relational):
        return relation
    verdict = assumptions.holds(relation, bounds)
    return relation if verdict is None else sympy.true if verdict else sympy.false


def equal(lhs: Any, rhs: Any) -> Any:
    return sympy.Eq(lhs, rhs)


def unequal(lhs: Any, rhs: Any) -> Any:
    return sympy.Ne(lhs, rhs)


def logical_not(value: Any) -> Any:
    return sympy.Not(value)


def logical_and(*values: Any) -> Any:
    return sympy.And(*values)


def logical_or(*values: Any) -> Any:
    return sympy.Or(*values)


def item(whole: Any, index: Any) -> Any:
    """The item of a list or a tuple at ``index``, as of what ``solve`` or ``eig`` give."""
    return whole[index]


# -- the metric -------------------------------------------------------------------------


def where(components: Any, name: str, value: Any) -> Any:
    """A metric's components with a name it is written with given its value, as where
    \\Delta = r^2 - 2Mr + a^2 follows a line element: the symbol, or a function of the
    coordinates the name stands for, as \\alpha does for \\alpha(t, r)."""
    value = sympy.sympify(value)

    def put(component: Any) -> Any:
        component = sympy.sympify(component).replace(
            lambda e: isinstance(e, AppliedUndef) and e.func.__name__ == name, lambda e: value
        )
        return component.xreplace({s: value for s in component.free_symbols if s.name == name})

    return arrays.apply(arrays.to_array(components), put)


def where_function(components: Any, function: tuple[str, tuple[Any, ...]], body: Any) -> Any:
    """A metric's components with a function it is written with, as f is in
    -f(r) dt^2, given its definition: ``body`` in its ``parameters``. Where the function
    is written with other arguments than it is defined with, the definition says what it
    is in terms of the parameters' names, as \\alpha(r) := ... does of \\alpha(t, r)."""
    name, parameters = function
    body = sympy.sympify(body)

    def at(applied: Any) -> Any:
        if len(applied.args) == len(parameters):
            return body.xreplace(dict(zip(parameters, applied.args, strict=True)))
        return body

    def put(component: Any) -> Any:
        return sympy.sympify(component).replace(
            lambda e: isinstance(e, AppliedUndef) and e.func.__name__ == name, at
        )

    return arrays.apply(arrays.to_array(components), put)


def components(metric: Any, *, coordinates: tuple[Symbol, ...]) -> Any:
    """The components of a spacetime's metric in ``coordinates``, the symbols declared for
    the names of its own: a coordinate declared with assumptions, as by Positive := [r],
    takes the place of its namesake."""
    declared = {coordinate.name: coordinate for coordinate in coordinates}
    own = {c: declared.get(c.name, c) for c in metric.coordinates}
    return metric.components.xreplace(own)


def metric_diagnosis(given: Any, *, coordinates: tuple[Symbol, ...]) -> EvaluationError | None:
    """Why components given for a metric in ``coordinates`` are no metric, in the terms of
    what was written: of the wrong size, not symmetric, or degenerate, with the
    differentials no term of it has; ``None`` where none of these is why. The error is the
    definition's, which places it."""
    try:
        components = arrays.to_array(given)
    except Exception:  # what is no array says so itself
        return None
    names = [latex_label(c.name) for c in coordinates]
    listed = _listed(names)
    shape = tuple(components.shape)
    count = len(coordinates)
    if shape != (count, count):
        found = "×".join(map(str, shape)) if shape else "no"
        return EvaluationError(
            f"the metric has {found} components, but there are {count} coordinates",
            notes=[f"the coordinates are {listed}"],
            hint=f"give {count}×{count} components, a row and a column for each coordinate",
        )
    matrix = sympy.Matrix(components.tolist())
    for i, j in itertools.combinations(range(count), 2):
        if sympy.simplify(matrix[i, j] - matrix[j, i]) != 0:
            one, other = names[i], names[j]
            return EvaluationError(
                f"the metric is not symmetric: g_{{{one}{other}}} = "
                f"{display.text(matrix[i, j])} but g_{{{other}{one}}} = "
                f"{display.text(matrix[j, i])}",
                hint="a metric has g_{\\mu\\nu} = g_{\\nu\\mu}",
            )
    if sympy.simplify(matrix.det()) != 0:
        return None
    missing = [
        names[i] for i in range(count) if all(sympy.simplify(entry) == 0 for entry in matrix.row(i))
    ]
    if missing:
        differentials = _listed([f"d{name}" for name in missing], "or")
        return MetricError(
            "the metric is degenerate: its determinant is zero",
            notes=[f"no term of it has {differentials}"],
            hint=f"give it a term with {_listed([f'd{name}' for name in missing])}",
        )
    return MetricError(
        "the metric is degenerate: its determinant is zero",
        notes=["so it has no inverse, and no curvature can be computed from it"],
    )


def _listed(items: Sequence[str], conjunction: str = "and") -> str:
    """``t, r, \\theta and \\phi``."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


def line_element(metric: Any, coordinates: Sequence[Symbol]) -> Any:
    """:math:`g_{\\mu\\nu} dx^\\mu dx^\\nu`, in the differentials of the coordinates."""
    differentials = [differential(c) for c in coordinates]
    count = len(coordinates)
    return sympy.Add(
        *(
            metric[i, j] * differentials[i] * differentials[j]
            for i in range(count)
            for j in range(count)
        )
    )


# -- values for Python ---------------------------------------------------------------------


def tensor(components: Any, *, indices: Indices, coordinates: tuple[Symbol, ...]) -> Tensor:
    """Components wrapped in a :class:`~relativisticpy.tensors.Tensor`, for a function
    that wants one."""
    return Tensor(components, indices, coordinates)


def tensor_equation(
    equation: TensorEquation, *, indices: Indices, coordinates: tuple[Symbol, ...]
) -> TensorEquation:
    """A tensor equation with its indices and coordinates, for a function that wants one."""
    return TensorEquation(equation.lhs, equation.rhs, indices, coordinates)


# -- spaces ---------------------------------------------------------------------------------


def identity(dim: int) -> Any:
    """The identity matrix: SymPy's for a few dimensions, NumPy's for many."""
    if dim <= _SMALL:
        return sympy.ImmutableDenseNDimArray(sympy.eye(dim).tolist())
    return numerical.numpy().eye(dim)


def ones(dim: int) -> Any:
    if dim <= _SMALL:
        return sympy.ImmutableDenseNDimArray([1] * dim)
    return numerical.numpy().ones(dim)


def uniform(dim: int) -> Any:
    """The uniform measure: weight 1/n on each of n elements."""
    if dim <= _SMALL:
        return sympy.ImmutableDenseNDimArray([sympy.Rational(1, dim)] * dim)
    return numerical.numpy().full(dim, 1.0 / dim)


def positions(first: int, dim: int) -> Any:
    """The values an index takes over a space of ``dim`` elements, counted from ``first``,
    as n takes 0, 1, ..., N - 1 in ``x_n := n h``: SymPy's for a few, NumPy's for many."""
    if dim <= _SMALL:
        return sympy.ImmutableDenseNDimArray(list(range(first, first + dim)))
    return numerical.numpy().arange(first, first + dim)


def power(base: Any, exponent: Any) -> Any:
    """A power, taken for each component of a grid, as n^2 is for each n, or
    e^{-2πikn/N} for each k and n."""
    return base**exponent


def space_metric(components: Any, complex_space: bool) -> Any:
    """The components of a space's metric, once found to be one: symmetric (Hermitian, over
    a complex space) and, where they are numbers, positive definite."""
    asymmetric = EvaluationError(
        f"a metric is {'Hermitian' if complex_space else 'symmetric'}, but this one is not"
    )
    indefinite = EvaluationError(
        "a metric of a space is positive definite, but this one is not",
        hint="a covariance is, unless some combination of the data never varies",
    )
    if numerical.is_numerical(components):
        np = numerical.numpy()
        try:
            matrix = np.asarray(numerical.dense(components))
        except TypeError:  # numbers JAX is differentiating, not known yet
            return components
        if not linalg.hermitian(matrix):
            raise asymmetric
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            raise indefinite from None
        return components
    entries = sympy.Matrix(components.tolist())
    if not all(is_zero(entry) for entry in entries - (entries.H if complex_space else entries.T)):
        raise asymmetric
    if all(entry.is_number for entry in entries) and entries.is_positive_definite is False:
        raise indefinite
    return components


def elementwise_product(left: Any, right: Any) -> Any:
    """The elementwise product of lists or matrices of one shape, ``[1, 2] \\odot [3, 4]``,
    which is ``[3, 8]``; a number multiplies every element."""

    def array(value: Any) -> Any:
        listed = isinstance(value, list | tuple | sympy.MatrixBase)
        return arrays.to_array(value) if listed else value

    left, right = array(left), array(right)
    shapes = [tuple(getattr(value, "shape", ())) for value in (left, right)]
    if () in shapes:
        return left * right
    if shapes[0] != shapes[1]:
        raise EvaluationError(
            f"an elementwise product is of arrays of one shape, not {shapes[0]} and {shapes[1]}"
        )
    return linalg.hadamard(left, right)


def conjugated(components: Any, complex_space: bool) -> Any:
    """The complex conjugate of components: of numbers, always (real ones are their own),
    and of SymPy's where they are over a complex space."""
    if numerical.is_numerical(components):
        complex_data = numerical.xp(components).iscomplexobj(components)
        return components.conj() if complex_data else components
    return arrays.apply(components, sympy.conjugate) if complex_space else components


def squared_modulus(components: Any, complex_space: bool) -> Any:
    """:math:`|x|^2` of each component, which is real: :math:`x \\bar x` of complex numbers,
    and of SymPy's over a complex space; :math:`x^2` otherwise."""
    if numerical.is_numerical(components):
        if not numerical.xp(components).iscomplexobj(components):
            return linalg.hadamard(components, components)
        return linalg.hadamard(components, components.conj()).real
    if complex_space:
        return arrays.apply(components, lambda c: sympy.expand(c * sympy.conjugate(c)))
    return linalg.hadamard(components, components)


def real(value: Any) -> Any:
    """The real part of numbers that are real but for rounding, such as the square of a
    complex vector's length; SymPy values as they are."""
    if numerical.is_numerical(value) and numerical.xp(value).iscomplexobj(value):
        return value.real
    return value


# -- linear algebra and statistics ------------------------------------------------------------


def matrix_function(name: str, components: Any, rows: int) -> Any:
    """``exp``, ``sqrt`` or ``log``, called ``name``, of a map."""
    return linalg.function(name, components, rows)


def of_a_map(components: Any, *rest: Any, name: str, position: int, rows: int) -> Any:
    """The function of a matrix called ``name`` of the components of a map, as a matrix whose
    rows are its first ``rows`` axes, given at ``position`` among the ``rest`` of its
    arguments: computed with arrays where they are numbers, and with SymPy otherwise."""
    capability = integrations.find(name)

    def compute(matrix: Any) -> Any:
        arguments = [*rest[:position], matrix, *rest[position:]]
        if numerical.is_numerical(matrix) and capability.numeric is not None:
            return integrations.Numerically(name)(numerical.xp(matrix), *arguments)
        return integrations.Exactly(name)(*arguments)

    return linalg.of_a_map(components, rows, compute, scalar=capability.scalar)


def identity_beside(matrix: Any, *, ones: bool) -> Any:
    """The identity of the size of a square matrix, as I is in A - \\lambda I; beside a
    vector, where it is written \\mathbf{1}, the vector of ones of its shape
    (design/notation.md, 12 and 23)."""
    array = arrays.to_array(matrix)
    shape = tuple(array.shape)
    if len(shape) == 2 and shape[0] == shape[1]:
        if numerical.is_numerical(array):
            return numerical.xp(array).eye(shape[0])
        return sympy.ImmutableDenseNDimArray(sympy.eye(shape[0]).tolist())
    if ones and len(shape) == 2 and 1 in shape:
        return _ones_of(array, shape)
    what = "the identity" if not ones else "the identity, or the vector of ones,"
    raise EvaluationError(
        f"{what} is of the size of the matrix beside it, but a {_size(shape)} matrix is not square",
        hint="write it beside a square matrix, as in A - \\lambda I",
    )


def ones_beside(matrix: Any, *, side: str, row: bool) -> Any:
    """The vector of ones as long as a matrix has ``side``, rows or columns: a row, where
    it is ``\\mathbf{1}^T``, or a column."""
    array = arrays.to_array(matrix)
    shape = tuple(array.shape)
    if len(shape) != 2:
        raise EvaluationError(f"the vector of ones meets a {_size(shape)} array, no matrix")
    length = shape[0] if side == "rows" else shape[1]
    return _ones_of(array, (1, length) if row else (length, 1))


def _ones_of(like: Any, shape: tuple[int, ...]) -> Any:
    """Ones in an array of ``shape``: of numbers where ``like`` is, and exact otherwise."""
    if numerical.is_numerical(like):
        return numerical.xp(like).ones(shape)
    return sympy.ImmutableDenseNDimArray([1] * math.prod(shape), shape)


def entry(matrix: Any, row: int, column: int) -> Any:
    """The entry of a matrix in a row and a column, counted from 0."""
    array = arrays.to_array(matrix)
    shape = tuple(array.shape)
    if len(shape) != 2 or row >= shape[0] or column >= shape[1]:
        raise EvaluationError(
            f"a {_size(shape)} matrix has no entry in row {row + 1} and column {column + 1}"
        )
    value = array[row, column]
    return numerical.number(value) if numerical.is_numerical(array) else value


def _size(shape: tuple[int, ...]) -> str:
    """A shape as it is said, as 2×3."""
    return "×".join(map(str, shape)) or "scalar"


def ledoit_wolf(samples: Any) -> Any:
    """The Ledoit–Wolf covariance of samples in rows, from scikit-learn."""
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            'the Ledoit–Wolf covariance needs scikit-learn: pip install "relativisticpy[data]"'
        ) from error
    data = numerical.dense(numerical.as_array(samples))
    return LedoitWolf(assume_centered=False).fit(data).covariance_
