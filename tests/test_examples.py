"""Every example runs, and finds what its text says it finds."""

import runpy
from pathlib import Path

import pytest
from sympy import Function, Symbol, diff

from helpers import same
from relativisticpy import Workbook

EXAMPLES = Path(__file__).parent.parent / "examples"
t = Symbol("t")


def run(name):
    with Workbook() as wb:
        return wb.run_file(EXAMPLES / name)


def test_every_example_is_tested():
    tested = {"schwarzschild.md", "kretschmann.md", "friedmann.md", "geodesics.md", "light.py"}
    assert {path.name for path in EXAMPLES.iterdir()} - {"README.md"} == tested


def test_schwarzschild():
    # B in terms of A, then A, then the constant, then the curvature invariant
    assert [str(result) for result in run("schwarzschild.md")] == [
        "C1/A(r)",
        "C1 + C2/r",
        "[-2*G*M]",
        "48*G**2*M**2/r**6",
    ]


def test_kretschmann():
    *_, kretschmann, difference = run("kretschmann.md")
    assert str(kretschmann) == "48*G**2*M**2/(c**4*r**6)"
    assert difference == 0


def test_friedmann():
    a, rho, p = (Function(name)(t) for name in ("a", "rho", "p"))
    *_, continuity = run("friedmann.md")
    assert same(continuity, diff(rho, t) + 3 * diff(a, t) / a * (rho + p))


def test_geodesics():
    *_, euler_lagrange_is_geodesic = run("geodesics.md")
    assert euler_lagrange_is_geodesic


def test_light(capsys):
    pytest.importorskip("scipy")
    runpy.run_path(str(EXAMPLES / "light.py"), run_name="__main__")
    output = capsys.readouterr().out
    assert "bent by 0.04122 rad" in output
    assert "captured below b = 5.19615M" in output
