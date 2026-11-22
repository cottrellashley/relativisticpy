"""What notebooks show: RelativisticPy's values display themselves in Jupyter and marimo,
typeset, through relativisticpy.display, and look there as they do everywhere else."""

import sys
import types

import numpy as np
import pytest
from sympy import Array, symbols

from relativisticpy import Tensor, TensorEquation, Workbook, display, spacetimes
from relativisticpy.tensors import Indices

t, x = symbols("t x")


@pytest.fixture(scope="module")
def vacuum():
    wb = Workbook()
    wb.expr(r"g_{\mu\nu} := Schwarzschild(M)")
    return wb


def values(vacuum):
    """One value of every kind RelativisticPy gives back that shows itself."""
    numeric = pytest.importorskip("relativisticpy.numeric")
    metric = spacetimes.schwarzschild()
    position = [0.0, 10.0, np.pi / 2, 0.0]
    velocity = numeric.four_velocity(metric, position, [0, 0, (1 / 700) ** 0.5], M=1)
    return {
        "tensor": vacuum.expr(r"\Gamma^{a}_{bc}"),
        "metric": vacuum.metric,
        "equation that holds": vacuum.expr(r"R_{\mu\nu} = 0"),
        "equation that differs": vacuum.expr(r"R_{\mu\nu} = g_{\mu\nu}"),
        "index expression": Workbook().expr(r"simplify(R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu})"),
        "index equation": Workbook().expr(r"G_{\mu\nu} = 8\pi G T_{\mu\nu}"),
        "geodesic equations": metric.geodesic_equations(),
        "geodesic": numeric.geodesic(metric, position, velocity, (0, 20), M=1, samples=10),
    }


@pytest.fixture(scope="module")
def shown(vacuum):
    return values(vacuum)


KINDS = [
    "tensor",
    "metric",
    "equation that holds",
    "equation that differs",
    "index expression",
    "index equation",
    "geodesic equations",
    "geodesic",
]


@pytest.mark.parametrize("kind", KINDS)
def test_jupyter_typesets_the_latex_of_display(shown, kind):
    value = shown[kind]
    typeset = value._repr_latex_()
    assert typeset == display.jupyter_latex(value)
    assert display.latex(value) in typeset
    assert typeset.startswith("$")
    assert typeset.endswith("$")


def test_a_tag_is_typeset_in_display_mode(shown):
    assert shown["equation that holds"]._repr_latex_() == (
        r"$$ R_{\mu\nu} = 0 \tag*{holds in all 10 components} $$"
    )
    assert shown["tensor"]._repr_latex_().startswith("$\\displaystyle \\begin{aligned}")


@pytest.fixture
def fake_marimo(monkeypatch):
    """A stand-in for marimo, which records what a value asks it to show."""
    module = types.ModuleType("marimo")
    module.md = lambda text: ("md", text)
    module.Html = lambda text: ("Html", text)
    monkeypatch.setitem(sys.modules, "marimo", module)
    return module


@pytest.mark.parametrize("kind", KINDS)
def test_marimo_typesets_the_latex_of_display(shown, fake_marimo, kind):
    value = shown[kind]
    assert value._display_() == ("md", "\\[" + display.latex(value) + "\\]")


def test_numerical_data_is_a_table_in_both(fake_marimo):
    wb = Workbook()
    wb["A_{ij}"] = np.arange(6.0).reshape(2, 3)
    table = wb["A"]
    assert table._repr_html_() == display.html(table) == display.jupyter_html(table)
    assert "<table>" in table._repr_html_()
    assert table._display_() == ("Html", display.html(table))


def test_a_symbolic_tensor_has_no_table():
    eta = Tensor([[-1, 0], [0, 1]], "_{a b}", (t, x), name="eta")
    assert eta._repr_html_() is None
    assert display.jupyter_html(eta) is None
    equation = TensorEquation(Array([1, 2]), Array([1, 2]), Indices("^{a}"), source="v^a = w^a")
    assert display.jupyter_html(equation) is None


def test_ipython_gives_jupyter_the_typeset_value(shown):
    formatters = pytest.importorskip("IPython.core.formatters")
    bundle, _ = formatters.DisplayFormatter().format(shown["tensor"])
    assert bundle["text/latex"] == display.jupyter_latex(shown["tensor"])
    assert bundle["text/plain"] == display.unicode(shown["tensor"])
    wb = Workbook()
    wb["w^a"] = np.array([0.5, 0.25])
    bundle, _ = formatters.DisplayFormatter().format(wb["w"])
    assert "<table>" in bundle["text/html"]


def test_marimo_shows_the_typeset_value(shown):
    pytest.importorskip("marimo", reason="marimo is a tool of the notebook tests, not a dependency")
    from marimo._output.formatting import try_format

    typeset = try_format(shown["tensor"])
    assert typeset.mimetype in ("text/html", "text/markdown")  # HTML either way
    assert "marimo-tex" in typeset.data
    assert r"\Gamma^{t}_{tr}" in typeset.data
    wb = Workbook()
    wb["w^a"] = np.array([0.5, 0.25])
    table = try_format(wb["w"])
    assert table.mimetype == "text/html"
    assert "<table>" in table.data
