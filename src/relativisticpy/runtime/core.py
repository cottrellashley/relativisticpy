r"""The core operations: what the compiler emits for notation that is no function of the
language, such as a sum, an evaluation bar or a time derivative, computed by the engines.

Each is registered by :mod:`relativisticpy.runtime.registry` under a name of its own, as
``core.sum`` or ``core.subs``, and is a function of the values of its arguments, then of
its attributes, by name: module-level, so that a worker process can run it.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import sympy
from sympy import Symbol
from sympy.core.function import AppliedUndef

from relativisticpy import display
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


def summation(body: Any, variable: Any, lower: Any, upper: Any) -> Any:
    """A sum: carried out where it has a whole number of terms, as \\sum_{n=0}^{2} has;
    an infinite one, or one to n, is kept as it is written, for doit to carry out."""
    return _finite(sympy.Sum(body, (variable, lower, upper)), lower, upper)


def product(body: Any, variable: Any, lower: Any, upper: Any) -> Any:
    """A product, carried out as a sum is: see :func:`summation`."""
    return _finite(sympy.Product(body, (variable, lower, upper)), lower, upper)


def _finite(written: Any, lower: Any, upper: Any) -> Any:
    bounds = [sympy.sympify(bound) for bound in (lower, upper)]
    if all(bound.is_Integer for bound in bounds) and bounds[1] - bounds[0] < 10_000:
        return written.doit()
    return written


def limit(body: Any, variable: Any, point: Any, direction: str) -> Any:
    if direction == "+-" and point in (sympy.oo, -sympy.oo):
        direction = "-" if point == sympy.oo else "+"
    return sympy.limit(body, variable, point, direction)


def differentiate(expr: Any, orders: tuple[int, ...], *variables: Any) -> Any:
    return sympy.diff(expr, *((v, n) for v, n in zip(variables, orders, strict=True)))


def derivative_at(body: Any, dummy: Any, order: int, point: Any) -> Any:
    return sympy.diff(body, dummy, order).subs(dummy, point)


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


def less(lhs: Any, rhs: Any) -> Any:
    return sympy.Lt(lhs, rhs)


def greater(lhs: Any, rhs: Any) -> Any:
    return sympy.Gt(lhs, rhs)


def less_equal(lhs: Any, rhs: Any) -> Any:
    return sympy.Le(lhs, rhs)


def greater_equal(lhs: Any, rhs: Any) -> Any:
    return sympy.Ge(lhs, rhs)


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
