r"""From symbols to numbers: NumPy functions and geodesics integrated with SciPy.

Symbolic results become fast, vectorised NumPy functions with :func:`lambdify`,
and test particles and light rays follow geodesics with :func:`geodesic`::

    >>> from relativisticpy import spacetimes
    >>> g = spacetimes.schwarzschild()
    >>> x0 = [0, 10, 3.141592653589793 / 2, 0]               # t, r, θ, φ
    >>> u0 = four_velocity(g, x0, [0, 0, (1 / 700) ** 0.5], M=1)   # circular at r = 10M
    >>> orbit = geodesic(g, x0, u0, (0, 2000), M=1)
    >>> print(orbit["r"].min().round(6), orbit["r"].max().round(6))
    10.0 10.0

This module needs NumPy and SciPy: ``pip install "relativisticpy[numeric]"``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import sympy
from sympy import Symbol
from sympy.core.function import AppliedUndef

from relativisticpy.diffgeom import Metric
from relativisticpy.tensors import Tensor

if TYPE_CHECKING:
    import numpy as np


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            'relativisticpy.numeric needs NumPy and SciPy: pip install "relativisticpy[numeric]"'
        ) from error
    return numpy


def _named(values: Mapping[Symbol | str, Any]) -> dict[str, Any]:
    """Values keyed by name, whether they were given by name or by symbol."""
    return {str(key): value for key, value in values.items()}


def _substitute(expr: Any, values: Mapping[str, Any]) -> Any:
    """``expr`` with the values given for its symbols, matched by name, so that a
    symbol with assumptions (a positive ``M``) takes the value given for ``M``."""
    expr = sympy.sympify(expr)
    matches = {s: values[s.name] for s in expr.free_symbols if s.name in values}
    return expr.subs(matches) if matches else expr


def _require_values(expressions: Sequence[Any], variables: Sequence[str] = ()) -> None:
    """Raise ``ValueError`` naming every symbol, other than ``variables``, that has no
    value, and every undefined function, which cannot be given one."""
    missing = sorted({s.name for e in expressions for s in e.free_symbols} - set(variables))
    functions = sorted({f.func.__name__ for e in expressions for f in e.atoms(AppliedUndef)})
    problems = []
    if missing:
        problems.append(f"give a value for {', '.join(missing)}")
    if functions:
        which = "function" if len(functions) == 1 else "functions"
        problems.append(f"replace the undefined {which} {', '.join(functions)} with an expression")
    if problems:
        raise ValueError(", and ".join(problems))


def lambdify(
    value: Any,
    variables: Sequence[Symbol | str],
    parameters: Mapping[Symbol | str, Any] | None = None,
    **named: Any,
) -> Callable[..., Any]:
    """Compile a scalar, a tensor or a metric into a vectorised NumPy function.

    ``variables`` are the function's arguments; every other symbol must be given
    a value, in ``parameters`` or by name::

        >>> import numpy as np
        >>> from relativisticpy import spacetimes
        >>> kretschmann = lambdify(spacetimes.schwarzschild().kretschmann, ["r"], M=1)
        >>> kretschmann(np.array([1.0, 2.0]))
        array([48.  ,  0.75])

    Variables and values are matched to symbols by name. A symbol without a
    value, or an undefined function, is a ``ValueError`` that names it.

    A tensor compiles to a function returning an array of its components'
    shape, with the variables' shape appended when they are arrays.
    """
    numpy = _numpy()
    values = _named({**(parameters or {}), **named})
    names = [str(v) for v in variables]
    symbols = [sympy.Symbol(name) for name in names]

    def prepared(expr: Any) -> Any:  # the values in, the variables as the plain symbols
        expr = _substitute(expr, values)
        return expr.xreplace({s: sympy.Symbol(s.name) for s in expr.free_symbols})

    components = value.components if isinstance(value, Tensor) else value
    if isinstance(components, sympy.NDimArray):
        flat = [prepared(c) for c in components.reshape(len(components))]
        _require_values(flat, names)
        shape = tuple(components.shape)
        compiled = sympy.lambdify(symbols, flat, modules="numpy")

        def tensor_function(*args: Any) -> Any:
            entries = compiled(*args)
            broadcast = numpy.broadcast_arrays(*entries)
            return numpy.stack(broadcast).reshape(shape + broadcast[0].shape)

        return tensor_function
    expr = prepared(components)
    _require_values([expr], names)
    scalar = sympy.lambdify(symbols, expr, modules="numpy")

    def scalar_function(*args: Any) -> Any:
        return numpy.asarray(scalar(*args)) + 0 * sum(numpy.asarray(a) for a in args)

    return scalar_function


def evaluate(value: Any, point: Mapping[Symbol | str, Any] | None = None, **named: Any) -> Any:
    """The numerical value of a scalar or tensor at a point:
    ``evaluate(g.ricci_scalar, r=3, M=1)``. Values are matched to symbols by name,
    and a symbol without one, or an undefined function, is a ``ValueError`` that
    names it."""
    numpy = _numpy()
    values = _named({**(point or {}), **named})
    components = value.components if isinstance(value, Tensor) else value
    if isinstance(components, sympy.NDimArray):
        flat = [_substitute(c, values) for c in components.reshape(len(components))]
        _require_values(flat)
        array = numpy.array([complex(sympy.N(c)) for c in flat]).reshape(components.shape)
        return array.real if not numpy.iscomplex(array).any() else array
    expr = _substitute(components, values)
    _require_values([expr])
    number = complex(sympy.N(expr))
    return number.real if number.imag == 0 else number


def four_velocity(
    metric: Metric,
    position: Sequence[float],
    spatial: Sequence[float],
    *,
    kind: str = "timelike",
    parameters: Mapping[Symbol | str, Any] | None = None,
    **named: Any,
) -> Any:
    r"""Complete a four-velocity :math:`u^\mu` from its spatial components.

    Solves :math:`g_{\mu\nu} u^\mu u^\nu = -1` (``kind="timelike"``, a massive
    particle) or :math:`0` (``kind="null"``, light) for the time component,
    taking the future-directed root.
    """
    norms = {"timelike": -1.0, "null": 0.0}
    if kind not in norms:
        raise ValueError(f"kind must be 'timelike' or 'null', not {kind!r}")
    numpy = _numpy()
    point = {**(parameters or {}), **named, **_point(metric, position)}
    g = numpy.asarray(evaluate(metric, point), dtype=float)
    u = numpy.concatenate([[0.0], numpy.asarray(spatial, dtype=float)])
    norm = norms[kind]
    a = g[0, 0]
    b = 2 * g[0, 1:] @ u[1:]
    c = u[1:] @ g[1:, 1:] @ u[1:] - norm
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        raise ValueError("no real four-velocity has these spatial components")
    roots = [(-b + s * numpy.sqrt(discriminant)) / (2 * a) for s in (1, -1)]
    u[0] = max(roots)
    return u


@dataclass(frozen=True)
class Geodesic:
    """A geodesic: the coordinates and four-velocity along the affine parameter ``tau``."""

    coordinates: tuple[Symbol, ...]
    tau: np.ndarray
    x: np.ndarray  # shape (dim, len(tau))
    u: np.ndarray  # shape (dim, len(tau))
    solution: Any  # SciPy's OdeResult

    def __getitem__(self, coordinate: str | Symbol) -> np.ndarray:
        """The values of one coordinate: ``orbit["r"]``."""
        names = [c.name for c in self.coordinates]
        values: np.ndarray = self.x[names.index(str(coordinate))]
        return values

    @property
    def stopped(self) -> bool:
        """Whether the integration ended at its ``stop`` condition, before the end of its span."""
        return bool(self.solution.status == 1)


def geodesic(
    metric: Metric,
    position: Sequence[float],
    velocity: Sequence[float],
    span: tuple[float, float],
    *,
    parameters: Mapping[Symbol | str, Any] | None = None,
    samples: int = 1000,
    stop: Callable[[np.ndarray, np.ndarray], float] | None = None,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    **named: Any,
) -> Geodesic:
    r"""Integrate the geodesic equation
    :math:`\ddot x^a = -\Gamma^a{}_{bc}\, \dot x^b \dot x^c` from an initial
    position and four-velocity over the affine parameter interval ``span``.

    Parameters of the metric (such as ``M``) are given by name; ``method``,
    ``rtol`` and ``atol`` go to :func:`scipy.integrate.solve_ivp`, which does
    the integration. ``stop``, a function of the position and four-velocity,
    ends the integration where its value crosses zero, for instance as a
    particle nears a horizon: ``stop=lambda x, u: x[1] - 2.1``. The geodesic
    then ends exactly there, and its :attr:`~Geodesic.stopped` is true.
    """
    numpy = _numpy()
    from scipy.integrate import solve_ivp

    values = {**(parameters or {}), **named}
    coordinates = metric.coordinates
    christoffel = lambdify(metric.christoffel, coordinates, values)
    dim = len(coordinates)

    def rates(_: float, state: np.ndarray) -> np.ndarray:
        x, u = state[:dim], state[dim:]
        gamma = christoffel(*x)
        acceleration = -numpy.einsum("abc,b,c->a", gamma, u, u)
        derivative: np.ndarray = numpy.concatenate([u, acceleration])
        return derivative

    events = []
    if stop is not None:

        def crossing(_: float, state: np.ndarray) -> float:
            return float(stop(state[:dim], state[dim:]))

        crossing.terminal = True  # type: ignore[attr-defined]
        events.append(crossing)

    initial = numpy.concatenate([numpy.asarray(position, float), numpy.asarray(velocity, float)])
    times = numpy.linspace(span[0], span[1], samples)
    solution = solve_ivp(
        rates, span, initial, method=method, t_eval=times, events=events, rtol=rtol, atol=atol
    )
    if not solution.success:
        raise RuntimeError(f"the geodesic integration failed: {solution.message}")
    tau, states = solution.t, solution.y
    if solution.status == 1:  # stopped: end exactly where the condition was met
        tau = numpy.append(tau, solution.t_events[0][0])
        states = numpy.column_stack([states, solution.y_events[0][0]])
    return Geodesic(coordinates, tau, states[:dim], states[dim:], solution)


def _point(metric: Metric, position: Sequence[float]) -> dict[Symbol, float]:
    return dict(zip(metric.coordinates, position, strict=True))
