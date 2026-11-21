"""Every example runs, and finds what its text says it finds."""

import re
import runpy
from pathlib import Path

import pytest
import sympy
from sympy import Function, Symbol, diff

from helpers import same
from relativisticpy import Workbook
from relativisticpy.documents import strip_comments

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples"
DESIGN = ROOT / "design"
t = Symbol("t")


def run(name):
    with Workbook() as wb:
        return wb.run_file(EXAMPLES / name)


def examples(folder):
    """The examples in a folder: its files but its README, and not what tools leave
    there, such as __pycache__ or .DS_Store."""
    files = {path.name for path in folder.iterdir() if path.is_file()}
    return {name for name in files if not name.startswith(".")} - {"README.md"}


def test_every_example_is_tested():
    tested = {
        "schwarzschild.md",
        "kretschmann.md",
        "friedmann.md",
        "geodesics.md",
        "light.py",
        "spins.md",
        "portfolio.py",
    }
    assert examples(EXAMPLES) == tested


def test_what_tools_leave_among_the_examples_is_not_one(tmp_path):
    for name in ("orbit.md", "README.md", ".DS_Store"):
        (tmp_path / name).write_text("")
    (tmp_path / "__pycache__").mkdir()
    assert examples(tmp_path) == {"orbit.md"}


def test_the_source_distribution_has_what_the_tests_read():
    tomllib = pytest.importorskip("tomllib")
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    include = config["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    # Anchored, a pattern matches only where it says: README.md alone matches every one.
    assert all(pattern.startswith("/") for pattern in include)
    for read in ("/tests", "/examples", "/docs/*.md", "/design/spaces.md", "/benchmarks"):
        assert read in include


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


def test_spins():
    j, omega = Symbol("J"), Symbol("omega")
    spectrum, trace, evolution = run("spins.md")
    assert list(spectrum.components) == [-3 * j, j, j, j]
    assert trace == 0
    phase = sympy.exp(-sympy.I * omega * t / 2)
    assert evolution.components.tolist() == [[phase, 0], [0, 1 / phase]]


def test_portfolio(capsys):
    runpy.run_path(str(EXAMPLES / "portfolio.py"), run_name="__main__")
    output = capsys.readouterr().out
    assert "the tangency portfolio's weights sum to 1.000000" in output
    assert "5 eigenvalues stand above the Marchenko–Pastur edge of 2.09" in output


def test_the_design_of_spaces_shows_the_notation_implemented():
    # Its notation runs as written, given what it says comes from elsewhere: data bound
    # from Python, the correlations and the Pauli matrices.
    np = pytest.importorskip("numpy")
    text = (DESIGN / "spaces.md").read_text(encoding="utf-8")
    _, portfolio, spins, least_squares = (
        strip_comments(block) for block in re.findall(r"```latex\n(.*?)```", text, re.DOTALL)
    )
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0004, 0.01, (60, 4))
    with Workbook() as wb:
        wb["R_{ta}"] = returns
        wb["w^a"] = np.full(4, 0.25)
        wb["Z_{ta}"] = (returns - returns.mean(axis=0)) / returns.std(axis=0)
        wb.exe(r"C_{ab} := \mathbb{E}_t[Z_{ta} Z_{tb}]")
        wb.exe(portfolio)
        assert wb["m"].to_numpy().sum() == pytest.approx(1)
    with Workbook() as wb:
        space, labels, *statements = spins.strip().splitlines()
        wb.exe(f"{space}\n{labels}")
        wb.exe(r"X^i{}_j := [[0, 1], [1, 0]]; Y^i{}_j := [[0, -i], [i, 0]]")
        wb.exe(r"Z^i{}_j := [[1, 0], [0, -1]]")
        *_, spectrum = wb.exe("\n".join(statements))
        assert list(spectrum.components) == [-3 * Symbol("J")] + [Symbol("J")] * 3
    with Workbook() as wb:
        x, y = rng.normal(0, 1, (30, 2)), rng.normal(0, 1, 30)
        wb["X_{tk}"], wb["y_t"] = x, y
        wb.exe(least_squares)
        assert wb["beta"].to_numpy() == pytest.approx(np.linalg.lstsq(x, y, rcond=None)[0])
