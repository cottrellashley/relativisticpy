r"""Book names no integration provides yet.

A name books write for a function the language does not compute, as ``\zeta(s)`` or
``\operatorname{Res}_{z=0} f``, would otherwise be read as an unknown function or a
symbol, and give an answer that looks right and is not. Each is reserved here, with what
it is and the forms it is written in, so that a program that writes it is refused as not
supported yet, saying what it is. A capability that has the reservation's name retires
it: providing the function is all it takes.
"""

from __future__ import annotations

from relativisticpy.integrations import reserved

beta = reserved("beta", what="the Beta function", written=[r"\mathrm{B}(a, b)", "beta(a, b)"])

zeta = reserved("zeta", what="Riemann's zeta function", written=[r"\zeta(s)", "zeta(s)"])

besselj = reserved(
    "besselj",
    what="the Bessel function of the first kind",
    written=[r"J_{\nu}(x)", "besselj(nu, x)"],
)

legendre = reserved(
    "legendre", what="the Legendre polynomial", written=["P_n(x)", "legendre(n, x)"]
)

hermite = reserved("hermite", what="the Hermite polynomial", written=["H_n(x)", "hermite(n, x)"])

spherical_harmonic = reserved(
    "Ynm",
    what="the spherical harmonic",
    written=[r"Y_l^m(\theta, \phi)", r"Ynm(l, m, \theta, \phi)"],
)

erfc = reserved(
    "erfc",
    what="the complementary error function",
    written=[r"\operatorname{erfc}(x)", r"\operatorname{erfc} x", "erfc(x)"],
)

principal_logarithm = reserved(
    "Log",
    what="the principal branch of the logarithm",
    written=[r"\operatorname{Log}(z)", r"\operatorname{Log} z"],
)

normal_distribution = reserved(
    "normcdf",
    what="the standard normal distribution function",
    written=[r"\Phi(x)", "N(x)"],
)

unit_step = reserved("Heaviside", what="the unit step", written=["Heaviside(x)"])

dirac_delta = reserved("DiracDelta", what="the Dirac delta", written=["DiracDelta(x)"])

residue = reserved(
    "Res",
    what="the residue of a function at a point",
    written=[r"\operatorname{Res}_{z=a} f", r"\operatorname{Res}(f, a)"],
)

trace = reserved(
    "tr",
    what="the trace of a matrix written as an operator",
    written=[r"\operatorname{tr} A", r"\operatorname{tr}(A)", r"\operatorname{Tr} A"],
    instead=r"write trace(A), or contract the indices of a map: A^{a}{}_{a}",
)

rank = reserved(
    "rank",
    what="the rank of a matrix",
    written=[r"\operatorname{rank} A", r"\operatorname{rank}(A)"],
)

levi_civita = reserved(
    "LeviCivita",
    what="the Levi-Civita symbol",
    written=[r"\epsilon_{ijk}"],
    instead=r"define it, as \epsilon_{ij} := [[0, 1], [-1, 0]], to use a symbol of your own",
)
