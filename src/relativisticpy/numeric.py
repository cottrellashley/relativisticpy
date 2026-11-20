r"""From symbols to numbers: NumPy functions and geodesics integrated with SciPy.

Symbolic results become fast, vectorised NumPy functions with :func:`lambdify`,
and test particles and light rays follow geodesics with :func:`geodesic`::

    >>> from relativisticpy import spacetimes
    >>> from relativisticpy.numeric import four_velocity, geodesic
    >>> g = spacetimes.schwarzschild()
    >>> x0 = [0, 10, 3.14159 / 2, 0]                        # t, r, θ, φ
    >>> u0 = four_velocity(g, x0, [0, 0, 0.0316], M=1)       # a circular orbit
    >>> orbit = geodesic(g, x0, u0, (0, 2000), M=1)
    >>> orbit["r"].min(), orbit["r"].max()                  # stays at r = 10M
    (9.999..., 10.000...)

This module needs NumPy and SciPy: ``pip install "relativisticpy[numeric]"``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import sympy
from sympy import Symbol

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


def _parameters(values: Mapping[Symbol | str, Any]) -> dict[Symbol, Any]:
    return {sympy.Symbol(k) if isinstance(k, str) else k: v for k, v in values.items()}


def lambdify(
    value: Any,
    variables: Sequence[Symbol | str],
    parameters: Mapping[Symbol | str, Any] | None = None,
    **named: Any,
) -> Callable[..., Any]:
    """Compile a scalar, a tensor or a metric into a vectorised NumPy function.

    ``variables`` are the function's arguments; every other symbol must be given
    a value, in ``parameters`` or by name::

        >>> f = lambdify(g.kretschmann, ["r"], M=1)
        >>> f(np.linspace(3, 10, 5))

    A tensor compiles to a function returning an array of its components'
    shape, with the variables' shape appended when they are arrays.
    """
    numpy = _numpy()
    values = _parameters({**(parameters or {}), **named})
    symbols = [sympy.Symbol(v) if isinstance(v, str) else v for v in variables]
    components = value.components if isinstance(value, Tensor) else value
    if isinstance(components, sympy.NDimArray):
        flat = [sympy.sympify(c).subs(values) for c in components.reshape(len(components))]
        shape = tuple(components.shape)
        compiled = sympy.lambdify(symbols, flat, modules="numpy")

        def tensor_function(*args: Any) -> Any:
            entries = compiled(*args)
            broadcast = numpy.broadcast_arrays(*entries)
            return numpy.stack(broadcast).reshape(shape + broadcast[0].shape)

        return tensor_function
    expr = sympy.sympify(components).subs(values)
    scalar = sympy.lambdify(symbols, expr, modules="numpy")

    def scalar_function(*args: Any) -> Any:
        return numpy.asarray(scalar(*args)) + 0 * sum(numpy.asarray(a) for a in args)

    return scalar_function


def evaluate(value: Any, point: Mapping[Symbol | str, Any] | None = None, **named: Any) -> Any:
    """The numerical value of a scalar or tensor at a point:
    ``evaluate(g.ricci_scalar, r=3, M=1)``."""
    numpy = _numpy()
    values = _parameters({**(point or {}), **named})
    components = value.components if isinstance(value, Tensor) else value
    if isinstance(components, sympy.NDimArray):
        entries = [_number(c, values) for c in components.reshape(len(components))]
        array = numpy.array(entries).reshape(components.shape)
        return array.real if not numpy.iscomplex(array).any() else array
    number = _number(components, values)
    return number.real if number.imag == 0 else number


def _number(expr: Any, values: Mapping[Symbol, Any]) -> complex:
    substituted = sympy.sympify(expr).subs(values)
    missing = substituted.free_symbols
    if missing:
        names = ", ".join(sorted(str(s) for s in missing))
        raise ValueError(f"give a value for {names}")
    return complex(sympy.N(substituted))


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
    numpy = _numpy()
    g = numpy.asarray(
        evaluate(metric, _point(metric, position), **{**(parameters or {}), **named}), dtype=float
    )
    u = numpy.concatenate([[0.0], numpy.asarray(spatial, dtype=float)])
    norm = {"timelike": -1.0, "null": 0.0}[kind]
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


def geodesic(
    metric: Metric,
    position: Sequence[float],
    velocity: Sequence[float],
    span: tuple[float, float],
    *,
    parameters: Mapping[Symbol | str, Any] | None = None,
    samples: int = 1000,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    **named: Any,
) -> Geodesic:
    r"""Integrate the geodesic equation
    :math:`\ddot x^a = -\Gamma^a{}_{bc}\, \dot x^b \dot x^c` from an initial
    position and four-velocity over the affine parameter interval ``span``.

    Parameters of the metric (such as ``M``) are given by name. Extra options
    go to :func:`scipy.integrate.solve_ivp`, which does the integration.
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

    initial = numpy.concatenate([numpy.asarray(position, float), numpy.asarray(velocity, float)])
    times = numpy.linspace(span[0], span[1], samples)
    solution = solve_ivp(rates, span, initial, method=method, t_eval=times, rtol=rtol, atol=atol)
    if not solution.success:
        raise RuntimeError(f"the geodesic integration failed: {solution.message}")
    return Geodesic(coordinates, solution.t, solution.y[:dim], solution.y[dim:], solution)


def _point(metric: Metric, position: Sequence[float]) -> dict[Symbol, float]:
    return dict(zip(metric.coordinates, position, strict=True))
