r"""Spacetimes.

Well-known solutions of Einstein's equation, each a function that gives its metric, as in
``g_{\mu\nu} := Schwarzschild(M)``, in the coordinates it is usually written in, which it
declares (:mod:`relativisticpy.spacetimes`). Units are geometric, G = c = 1, unless ``G``
and ``c`` follow its own parameters.
"""

from __future__ import annotations

import sympy

from relativisticpy import spacetimes
from relativisticpy.integrations import Example, constructor

t, r, theta, M, Q, a, k, L = sympy.symbols("t r theta M Q a k L")


@constructor(
    "Minkowski",
    summary="flat spacetime; `Minkowski(spherical)` in spherical coordinates",
    examples=[
        Example("g_{\\mu\\nu} := Minkowski()\ng_{tt}", -1, "standard"),
        Example("g_{\\mu\\nu} := Minkowski(spherical)\ng_{\\theta\\theta}", r**2,
                "Carroll (2004), section 2.5"),
    ],
)  # fmt: skip
class Minkowski:
    exact = staticmethod(spacetimes.minkowski)


@constructor(
    "Schwarzschild",
    summary="a static black hole of mass M",
    examples=[
        Example("g_{\\mu\\nu} := Schwarzschild(M)\nKretschmann", 48 * M**2 / r**6,
                "Carroll (2004), eq. (5.33)"),
        Example("g_{\\mu\\nu} := Schwarzschild(M)\ng_{tt}", -(1 - 2 * M / r),
                "Carroll (2004), section 5.1"),
    ],
)  # fmt: skip
class Schwarzschild:
    exact = staticmethod(spacetimes.schwarzschild)


@constructor(
    "ReissnerNordstrom",
    summary="a static black hole of mass M and charge Q",
    examples=[
        Example("g_{\\mu\\nu} := ReissnerNordstrom(M, Q)\ng_{tt}",
                -(1 - 2 * M / r + Q**2 / r**2), "Carroll (2004), section 6.5"),
        Example("g_{\\mu\\nu} := ReissnerNordstrom(M, Q)\nRic", 0,
                "standard: the stress tensor of an electromagnetic field is traceless"),
    ],
)  # fmt: skip
class ReissnerNordstrom:
    exact = staticmethod(spacetimes.reissner_nordstrom)


_SIGMA = r**2 + a**2 * sympy.cos(theta) ** 2


@constructor(
    "Kerr",
    summary="a rotating black hole of mass M and spin a = J/Mc, in Boyer–Lindquist coordinates",
    examples=[
        Example("g_{\\mu\\nu} := Kerr(M, a)\ng_{tt}", -(1 - 2 * M * r / _SIGMA),
                "Carroll (2004), section 6.6"),
        Example("g_{\\mu\\nu} := Kerr(M, a)\ng_{rr}", _SIGMA / (r**2 - 2 * M * r + a**2),
                "Carroll (2004), section 6.6"),
    ],
)  # fmt: skip
class Kerr:
    exact = staticmethod(spacetimes.kerr)


@constructor(
    "KerrNewman",
    summary="a rotating black hole of mass M, spin a and charge Q",
    examples=[
        Example("g_{\\mu\\nu} := KerrNewman(M, a, Q)\ng_{tt}",
                -(1 - (2 * M * r - Q**2) / _SIGMA), "Carroll (2004), section 6.6"),
        Example("g_{\\mu\\nu} := KerrNewman(M, a, Q)\ng_{rr}",
                _SIGMA / (r**2 - 2 * M * r + a**2 + Q**2), "Carroll (2004), section 6.6"),
    ],
)  # fmt: skip
class KerrNewman:
    exact = staticmethod(spacetimes.kerr_newman)


@constructor(
    "FLRW",
    summary="a homogeneous, isotropic universe: spatial curvature k, scale factor a(t)",
    examples=[
        Example("g_{\\mu\\nu} := FLRW(k, a)\ng_{rr}",
                sympy.Function("a")(t) ** 2 / (1 - k * r**2), "Carroll (2004), section 8.2"),
        Example("g_{\\mu\\nu} := FLRW(k, a)\ng_{tt}", -1, "Carroll (2004), section 8.2"),
    ],
)  # fmt: skip
class FriedmannLemaitreRobertsonWalker:
    exact = staticmethod(spacetimes.flrw)


@constructor(
    "DeSitter",
    summary="the static patch of de Sitter space, of curvature radius L",
    examples=[
        Example("g_{\\mu\\nu} := DeSitter(L)\ng_{tt}", -(1 - r**2 / L**2),
                "Carroll (2004), section 8.1"),
        Example("g_{\\mu\\nu} := DeSitter(L)\nRic", 12 / L**2, "Carroll (2004), section 8.1"),
    ],
)  # fmt: skip
class DeSitter:
    exact = staticmethod(spacetimes.de_sitter)


@constructor(
    "AntiDeSitter",
    summary="anti-de Sitter space, of curvature radius L",
    examples=[
        Example("g_{\\mu\\nu} := AntiDeSitter(L)\ng_{tt}", -(1 + r**2 / L**2),
                "Carroll (2004), section 8.1"),
        Example("g_{\\mu\\nu} := AntiDeSitter(L)\nRic", -12 / L**2,
                "Carroll (2004), section 8.1"),
    ],
)  # fmt: skip
class AntiDeSitter:
    exact = staticmethod(spacetimes.anti_de_sitter)
