r"""Real notation, as people write it.

Two guarantees live here. Equations copied from General Relativity texts parse
to the structure a physicist reads in them. And every program from the
original RelativisticPy test suite still parses: that suite is the
specification the language was rebuilt from.
"""

import json
from pathlib import Path

import pytest

from relativisticpy.language import parse, to_sexpr

TEXTBOOK = {
    "Einstein field equations": (
        r"G_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}",
        "(= (+ (G _[mu nu]) (* Lambda (g _[mu nu]))) (* (/ (* (* 8 pi) G) (^ c 4)) (T _[mu nu])))",
    ),
    "Einstein tensor": (
        r"G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu}",
        "(= (G _[mu nu]) (- (R _[mu nu]) (* (* (/ 1 2) R) (g _[mu nu]))))",
    ),
    "Christoffel symbols": (
        r"\Gamma^{\lambda}_{\mu\nu} = \frac{1}{2} g^{\lambda\sigma}"
        r" \left( \partial_\mu g_{\nu\sigma} + \partial_\nu g_{\mu\sigma}"
        r" - \partial_\sigma g_{\mu\nu} \right)",
        "(= (Gamma ^[lambda] _[mu nu]) (* (* (/ 1 2) (g ^[lambda sigma])) "
        "(- (+ (* (∂ _[mu]) (g _[nu sigma])) (* (∂ _[nu]) (g _[mu sigma]))) "
        "(* (∂ _[sigma]) (g _[mu nu])))))",
    ),
    "Riemann tensor": (
        r"R^{\rho}{}_{\sigma\mu\nu} = \partial_\mu \Gamma^{\rho}_{\nu\sigma}"
        r" - \partial_\nu \Gamma^{\rho}_{\mu\sigma}"
        r" + \Gamma^{\rho}_{\mu\lambda} \Gamma^{\lambda}_{\nu\sigma}"
        r" - \Gamma^{\rho}_{\nu\lambda} \Gamma^{\lambda}_{\mu\sigma}",
        "(= (R ^[rho] _[sigma mu nu]) (- (+ (- (* (∂ _[mu]) (Gamma ^[rho] _[nu sigma])) "
        "(* (∂ _[nu]) (Gamma ^[rho] _[mu sigma]))) "
        "(* (Gamma ^[rho] _[mu lambda]) (Gamma ^[lambda] _[nu sigma]))) "
        "(* (Gamma ^[rho] _[nu lambda]) (Gamma ^[lambda] _[mu sigma]))))",
    ),
    "geodesic equation": (
        r"\frac{d^2 x^\mu}{d\tau^2} + \Gamma^\mu_{\alpha\beta} \frac{dx^\alpha}{d\tau}"
        r" \frac{dx^\beta}{d\tau} = 0",
        "(= (+ (d^2/dtau^2 (x ^[mu])) (* (* (Gamma ^[mu] _[alpha beta]) "
        "(d/dtau (x ^[alpha]))) (d/dtau (x ^[beta])))) 0)",
    ),
    "Euler-Lagrange equations": (
        r"\frac{\partial L}{\partial x^\mu}"
        r" - \frac{d}{d\tau} \frac{\partial L}{\partial \dot{x}^\mu} = 0",
        "(= (- (∂/∂(x ^[mu]) L) (* (d/dtau _) (∂/∂((dot x) ^[mu]) L))) 0)",
    ),
    "Schwarzschild line element": (
        r"ds^2 = -\left(1 - \frac{2GM}{c^2 r}\right) c^2 dt^2"
        r" + \left(1 - \frac{2GM}{c^2 r}\right)^{-1} dr^2"
        r" + r^2 \left(d\theta^2 + \sin^2\theta \, d\phi^2\right)",
        "(= (^ ds 2) (+ (+ (* (* (- (- 1 (/ (* 2 GM) (* (^ c 2) r)))) (^ c 2)) (^ dt 2)) "
        "(* (^ (- 1 (/ (* 2 GM) (* (^ c 2) r))) (- 1)) (^ dr 2))) "
        "(* (^ r 2) (+ (* d (^ theta 2)) (* (* (^ (call sin theta) 2) d) (^ phi 2))))))",
    ),
    "FLRW line element": (
        r"ds^2 = -c^2 dt^2 + a(t)^2 \left[ \frac{dr^2}{1 - k r^2} + r^2 d\Omega^2 \right]",
        "(= (^ ds 2) (+ (* (- (^ c 2)) (^ dt 2)) (* (^ (call a t) 2) "
        "(+ (/ (^ dr 2) (- 1 (* k (^ r 2)))) (* (* (^ r 2) d) (^ Omega 2))))))",
    ),
    "Friedmann equation": (
        r"\left(\frac{\dot{a}}{a}\right)^2 = \frac{8\pi G}{3}\rho - \frac{k c^2}{a^2}",
        "(= (^ (/ (dot a) a) 2) (- (* (/ (* (* 8 pi) G) 3) rho) (/ (* k (^ c 2)) (^ a 2))))",
    ),
    "covariant derivative": (
        r"\nabla_\mu V^\nu = \partial_\mu V^\nu + \Gamma^\nu_{\mu\lambda} V^\lambda",
        "(= (* (∇ _[mu]) (V ^[nu])) (+ (* (∂ _[mu]) (V ^[nu])) "
        "(* (Gamma ^[nu] _[mu lambda]) (V ^[lambda]))))",
    ),
    "Kretschmann scalar": (
        r"K = R_{abcd} R^{abcd}",
        "(= K (* (R _[a b c d]) (R ^[a b c d])))",
    ),
    "Faraday tensor": (
        r"F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu",
        "(= (F _[mu nu]) (- (* (∂ _[mu]) (A _[nu])) (* (∂ _[nu]) (A _[mu]))))",
    ),
    "homogeneous Maxwell equations": (
        r"F_{[ab,c]} = 0",
        "(= (F _[<a b , c>]) 0)",
    ),
    "perfect fluid": (
        r"T^{\mu\nu} = (\rho + p) u^\mu u^\nu + p g^{\mu\nu}",
        "(= (T ^[mu nu]) (+ (* (* (+ rho p) (u ^[mu])) (u ^[nu])) (* p (g ^[mu nu]))))",
    ),
    "conservation of energy-momentum": (
        r"T^{\mu\nu}{}_{;\nu} = 0",
        "(= (T ^[mu nu] _[; nu]) 0)",
    ),
    "second Bianchi identity": (
        r"\nabla_{[\lambda} R_{\rho\sigma]\mu\nu} = 0",
        "(= (* (∇ _[<lambda]) (R _[rho sigma> mu nu])) 0)",
    ),
}


@pytest.mark.parametrize(("latex", "reading"), TEXTBOOK.values(), ids=TEXTBOOK.keys())
def test_textbook_equations(latex, reading):
    (statement,) = parse(latex).statements
    assert to_sexpr(statement) == reading


LEGACY = json.loads((Path(__file__).parent / "data" / "legacy_programs.json").read_text())


@pytest.mark.parametrize("program", LEGACY, ids=[p["origin"] for p in LEGACY])
def test_original_programs_still_parse(program):
    parse(program["source"])
