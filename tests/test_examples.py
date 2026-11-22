"""Every example runs, finds what its text says it finds, and is shown on the docs'
page of examples as it runs."""

import contextlib
import functools
import importlib.util
import io
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
TOOL = ROOT / "tools" / "examples.py"
t = Symbol("t")


def run(name):
    with Workbook() as wb:
        return wb.run_file(EXAMPLES / name)


@functools.cache
def tool():
    """tools/examples.py, which writes the docs' page of examples."""
    spec = importlib.util.spec_from_file_location("tools_examples", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.cache
def printed(name):
    """What a Python example prints, run once for every test that reads it."""
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return output.getvalue()


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
    tested = ("/tests", "/examples", "/docs/*.md", "/design/spaces.md", "/benchmarks")
    for read in (*tested, "/curriculum"):
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


def test_light():
    pytest.importorskip("scipy")
    output = printed("light.py")
    assert "bent by 0.04122 rad" in output
    assert "captured below b = 5.19615M" in output


def test_spins():
    j, omega = Symbol("J"), Symbol("omega")
    spectrum, trace, evolution = run("spins.md")
    assert list(spectrum.components) == [-3 * j, j, j, j]
    assert trace == 0
    phase = sympy.exp(-sympy.I * omega * t / 2)
    assert evolution.components.tolist() == [[phase, 0], [0, 1 / phase]]


def test_portfolio():
    pytest.importorskip("scipy")
    output = printed("portfolio.py")
    assert "the tangency portfolio's weights sum to 1.000000" in output
    assert "5 eigenvalues stand above the Marchenko–Pastur edge of 2.09" in output


@pytest.mark.skipif(not TOOL.is_file(), reason="the page is written from the source tree")
def test_the_page_of_examples_shows_each_as_it_runs():
    pytest.importorskip("scipy")
    names = [name for name, _ in tool().examples()]
    assert sorted(names) == sorted(examples(EXAMPLES))
    page = tool().page({name: printed(name) for name in names if name.endswith(".py")})
    assert (ROOT / "docs" / "examples.md").read_text(encoding="utf-8") == page, (
        "docs/examples.md is out of date: write it again with uv run python tools/examples.py"
    )


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
