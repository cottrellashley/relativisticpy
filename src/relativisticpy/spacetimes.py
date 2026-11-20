r"""Well-known spacetimes, ready to use.

Each function returns a :class:`~relativisticpy.diffgeom.Metric` in the
coordinates its solution is usually written in. Parameters may be numbers,
SymPy expressions or names; names become symbols. Units are geometric
(:math:`G = c = 1`) unless you pass ``G`` and ``c``::

    >>> schwarzschild().kretschmann
    48*M**2/r**6
    >>> schwarzschild("M", G="G", c="c").kretschmann
    48*G**2*M**2/(c**4*r**6)

In the language each is a function: ``g_{\mu\nu} := Schwarzschild(M)``.
"""

from __future__ import annotations

from typing import Any

import sympy
from sympy import Expr, Function, cos, sin

from relativisticpy.diffgeom import Metric

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
    return sympy.Symbol(value) if isinstance(value, str) else sympy.sympify(value)


def _spherical(f: Expr, *, c: Expr = sympy.S.One) -> Metric:
    """``ds² = -f c² dt² + dr²/f + r² dΩ²``."""
    return Metric(
        sympy.diag(-f * c**2, 1 / f, r**2, r**2 * sin(theta) ** 2), (t, r, theta, phi), check=False
    )


def minkowski(coordinates: str = "cartesian", *, c: Any = 1) -> Metric:
    r"""Flat spacetime, in ``"cartesian"`` :math:`(t, x, y, z)` or ``"spherical"`` coordinates."""
    speed = _value(c)
    if coordinates == "cartesian":
        return Metric(sympy.diag(-(speed**2), 1, 1, 1), (t, x, y, z), check=False)
    if coordinates == "spherical":
        return _spherical(sympy.S.One, c=speed)
    raise ValueError(f"unknown coordinates {coordinates!r}: use 'cartesian' or 'spherical'")


def schwarzschild(M: Any = "M", *, G: Any = 1, c: Any = 1) -> Metric:
    r"""The exterior of a static, spherically symmetric mass :math:`M`.

    .. math::

        ds^2 = -\left(1 - \frac{2GM}{c^2 r}\right) c^2 dt^2
               + \left(1 - \frac{2GM}{c^2 r}\right)^{-1} dr^2 + r^2 d\Omega^2
    """
    mass, gravity, speed = _value(M), _value(G), _value(c)
    return _spherical(1 - 2 * gravity * mass / (speed**2 * r), c=speed)


def reissner_nordstrom(M: Any = "M", Q: Any = "Q", *, G: Any = 1, c: Any = 1) -> Metric:
    r"""A static, spherically symmetric mass :math:`M` with charge :math:`Q` (Gaussian units)."""
    mass, charge, gravity, speed = _value(M), _value(Q), _value(G), _value(c)
    f = 1 - 2 * gravity * mass / (speed**2 * r) + gravity * charge**2 / (speed**4 * r**2)
    return _spherical(f, c=speed)


def kerr(M: Any = "M", a: Any = "a") -> Metric:
    r"""A rotating black hole of mass :math:`M` and spin :math:`a = J/M`, in Boyer-Lindquist
    coordinates."""
    return kerr_newman(M, a, 0)


def kerr_newman(M: Any = "M", a: Any = "a", Q: Any = "Q") -> Metric:
    r"""A rotating, charged black hole, in Boyer-Lindquist coordinates."""
    mass, spin, charge = _value(M), _value(a), _value(Q)
    sigma = r**2 + spin**2 * cos(theta) ** 2
    delta = r**2 - 2 * mass * r + spin**2 + charge**2
    g = sympy.zeros(4, 4)
    g[0, 0] = -(1 - (2 * mass * r - charge**2) / sigma)
    g[0, 3] = g[3, 0] = -spin * sin(theta) ** 2 * (2 * mass * r - charge**2) / sigma
    g[1, 1] = sigma / delta
    g[2, 2] = sigma
    g[3, 3] = (
        r**2 + spin**2 + spin**2 * sin(theta) ** 2 * (2 * mass * r - charge**2) / sigma
    ) * sin(theta) ** 2
    return Metric(g, (t, r, theta, phi), check=False)


def flrw(k: Any = "k", a: Any = "a", *, c: Any = 1) -> Metric:
    r"""A homogeneous, isotropic universe with spatial curvature :math:`k` and scale
    factor :math:`a(t)`.

    :math:`ds^2 = -c^2 dt^2 + a(t)^2 \left( \frac{dr^2}{1 - k r^2} + r^2 d\Omega^2 \right)`
    """
    curvature, speed = _value(k), _value(c)
    scale = Function(a)(t) if isinstance(a, str) else sympy.sympify(a)
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


def de_sitter(L: Any = "L") -> Metric:
    r"""The static patch of de Sitter space with curvature radius :math:`L`:
    :math:`f = 1 - r^2/L^2`."""
    return _spherical(1 - r**2 / _value(L) ** 2)


def anti_de_sitter(L: Any = "L") -> Metric:
    r"""Global anti-de Sitter space with curvature radius :math:`L`: :math:`f = 1 + r^2/L^2`."""
    return _spherical(1 + r**2 / _value(L) ** 2)
