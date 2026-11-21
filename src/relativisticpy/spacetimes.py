r"""Well-known spacetimes, ready to use.

Each function returns a :class:`~relativisticpy.diffgeom.Metric` in the
coordinates its solution is usually written in. Parameters may be numbers,
SymPy expressions or names; names become symbols, and decimals the exact
fractions they denote, so ``schwarzschild(1.7)`` is a vacuum. Units are geometric
(:math:`G = c = 1`) unless you pass ``G`` and ``c``, which every function takes
after its own parameters::

    >>> schwarzschild().kretschmann
    48*M**2/r**6
    >>> schwarzschild("M", G="G", c="c").kretschmann
    48*G**2*M**2/(c**4*r**6)

In the language each is a function: ``g_{\mu\nu} := Schwarzschild(M)``, or
``Schwarzschild(M, G, c)`` in other units.
"""

from __future__ import annotations

from typing import Any

import sympy
from sympy import Expr, Function, Symbol, cos, sin

from relativisticpy.diffgeom import Metric
from relativisticpy.tensors.simplification import rationalise

__all__ = [
    "anti_de_sitter",
    "de_sitter",
    "flrw",
    "kerr",
    "kerr_newman",
    "minkowski",
    "reissner_nordstrom",
    "schwarzschild",
]

t, r, theta, phi = sympy.symbols("t r theta phi")
x, y, z = sympy.symbols("x y z")


def _value(value: Any) -> Expr:
    """A parameter as SymPy: a name becomes a symbol, and a decimal the exact fraction
    it denotes."""
    return sympy.Symbol(value) if isinstance(value, str) else rationalise(sympy.sympify(value))


def _spherical(f: Expr, *, c: Expr = sympy.S.One) -> Metric:
    """``ds² = -f c² dt² + dr²/f + r² dΩ²``."""
    return Metric(
        sympy.diag(-f * c**2, 1 / f, r**2, r**2 * sin(theta) ** 2), (t, r, theta, phi), check=False
    )


def minkowski(coordinates: str | Symbol = "cartesian", G: Any = 1, c: Any = 1) -> Metric:
    r"""Flat spacetime, in ``"cartesian"`` :math:`(t, x, y, z)` or ``"spherical"`` coordinates.

    ``G`` does not appear in a flat metric; it is taken, as by every spacetime here,
    so that all of them accept the same units."""
    speed = _value(c)
    if str(coordinates) == "cartesian":
        return Metric(sympy.diag(-(speed**2), 1, 1, 1), (t, x, y, z), check=False)
    if str(coordinates) == "spherical":
        return _spherical(sympy.S.One, c=speed)
    raise ValueError(f"unknown coordinates {coordinates!r}: use 'cartesian' or 'spherical'")


def schwarzschild(M: Any = "M", G: Any = 1, c: Any = 1) -> Metric:
    r"""The exterior of a static, spherically symmetric mass :math:`M`.

    .. math::

        ds^2 = -\left(1 - \frac{2GM}{c^2 r}\right) c^2 dt^2
               + \left(1 - \frac{2GM}{c^2 r}\right)^{-1} dr^2 + r^2 d\Omega^2
    """
    mass, gravity, speed = _value(M), _value(G), _value(c)
    return _spherical(1 - 2 * gravity * mass / (speed**2 * r), c=speed)


def reissner_nordstrom(M: Any = "M", Q: Any = "Q", G: Any = 1, c: Any = 1) -> Metric:
    r"""A static, spherically symmetric mass :math:`M` with charge :math:`Q` (Gaussian units)."""
    mass, charge, gravity, speed = _value(M), _value(Q), _value(G), _value(c)
    f = 1 - 2 * gravity * mass / (speed**2 * r) + gravity * charge**2 / (speed**4 * r**2)
    return _spherical(f, c=speed)


def kerr(M: Any = "M", a: Any = "a", G: Any = 1, c: Any = 1) -> Metric:
    r"""A rotating black hole of mass :math:`M` and spin :math:`a = J/Mc`, in Boyer-Lindquist
    coordinates."""
    return kerr_newman(M, a, 0, G, c)


def kerr_newman(M: Any = "M", a: Any = "a", Q: Any = "Q", G: Any = 1, c: Any = 1) -> Metric:
    r"""A rotating black hole of mass :math:`M`, spin :math:`a = J/Mc` and charge :math:`Q`
    (Gaussian units), in Boyer-Lindquist coordinates."""
    mass, spin, charge = _value(M), _value(a), _value(Q)
    gravity, speed = _value(G), _value(c)
    # r_s r - r_Q², from the lengths r_s = 2GM/c² of the mass and r_Q = √G Q/c² of the charge
    source = 2 * gravity * mass * r / speed**2 - gravity * charge**2 / speed**4
    sigma = r**2 + spin**2 * cos(theta) ** 2
    delta = r**2 - 2 * gravity * mass * r / speed**2 + spin**2 + gravity * charge**2 / speed**4
    g = sympy.zeros(4, 4)
    g[0, 0] = -(1 - source / sigma) * speed**2
    g[0, 3] = g[3, 0] = -spin * speed * sin(theta) ** 2 * source / sigma
    g[1, 1] = sigma / delta
    g[2, 2] = sigma
    g[3, 3] = (r**2 + spin**2 + spin**2 * sin(theta) ** 2 * source / sigma) * sin(theta) ** 2
    return Metric(g, (t, r, theta, phi), check=False)


def flrw(k: Any = "k", a: Any = "a", G: Any = 1, c: Any = 1) -> Metric:
    r"""A homogeneous, isotropic universe with spatial curvature :math:`k` and scale
    factor :math:`a(t)`.

    :math:`ds^2 = -c^2 dt^2 + a(t)^2 \left( \frac{dr^2}{1 - k r^2} + r^2 d\Omega^2 \right)`

    A name or symbol for the scale factor makes it a function of time, with what
    is assumed of the symbol; an expression, such as ``t`` (the Milne universe for
    :math:`k = -1`), is taken as it is.

    ``G`` does not appear in the metric, only in the field equations; it is taken so
    that every spacetime accepts the same units.
    """
    curvature, speed, scale = _value(k), _value(c), _value(a)
    if isinstance(scale, Symbol) and scale.name not in ("t", "r", "theta", "phi"):
        assumed = {key: value for key, value in scale.assumptions0.items() if key != "commutative"}
        scale = Function(scale.name, **assumed)(t)
    return Metric(
        sympy.diag(
            -(speed**2),
            scale**2 / (1 - curvature * r**2),
            scale**2 * r**2,
            scale**2 * r**2 * sin(theta) ** 2,
        ),
        (t, r, theta, phi),
        check=False,
    )


def de_sitter(L: Any = "L", G: Any = 1, c: Any = 1) -> Metric:
    r"""The static patch of de Sitter space with curvature radius :math:`L`:
    :math:`f = 1 - r^2/L^2`. ``G`` does not appear, as in :func:`flrw`."""
    return _spherical(1 - r**2 / _value(L) ** 2, c=_value(c))


def anti_de_sitter(L: Any = "L", G: Any = 1, c: Any = 1) -> Metric:
    r"""Global anti-de Sitter space with curvature radius :math:`L`: :math:`f = 1 + r^2/L^2`.
    ``G`` does not appear, as in :func:`flrw`."""
    return _spherical(1 + r**2 / _value(L) ** 2, c=_value(c))
