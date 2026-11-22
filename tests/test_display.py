"""How results look: the four forms of relativisticpy.display, and that they read back."""

import json
import re
import time

import numpy as np
import pytest
import sympy
from sympy import Derivative, Float, Function, Integer, Rational, symbols

from relativisticpy import Space, Tensor, Workbook, display
from relativisticpy.tensors import Index, Indices

M, r, x, y, theta = symbols("M r x y theta")


def fresh(source):
    return Workbook().expr(source)


def same(value, back):
    """Whether a value read back is the value shown: the same, to rounding error for a
    float, and a list of the same items as an array of them."""
    if isinstance(value, list | tuple | sympy.NDimArray) and isinstance(
        back, list | tuple | sympy.NDimArray
    ):
        value = value.tolist() if isinstance(value, sympy.NDimArray) else list(value)
        back = back.tolist() if isinstance(back, sympy.NDimArray) else list(back)
        pairs = zip(value, back, strict=False)
        return len(value) == len(back) and all(same(a, b) for a, b in pairs)
    if isinstance(value, str) or isinstance(back, str):
        return value == back
    if value == back:
        return True
    if isinstance(value, sympy.Expr) and isinstance(back, sympy.Expr):
        if value.has(Float) != back.has(Float):
            return False
        if value.is_Float and back.is_Float:
            return abs(value - back) <= 1e-15 * abs(value)
        return sympy.simplify(value - back) == 0
    return False


# -- reading back ------------------------------------

#: Results of every kind, as the language computes them from these programs.
CORPUS = [
    # powers, fractions and roots
    "x^2", "x^(-2)", "48 M^2/r^6", "M/(r (r - 2M))", "M (r - 2M)/r^3", "x^(1/3)", "x^(3/2)",
    "sqrt(x + 1)", "1/sqrt(x)", "sqrt(2)", "2^x", "x^(y + 1)", "(x + 1)^2", "-x^2", "-3/4",
    "3x/4", "x/(y z)", "(x + 1)/(x - 1)", "1 - 2M/r", "r^2 sin(theta)^2", "r_s^2",
    "-sin(theta) cos(theta)", "cos(theta)/sin(theta)", "(2M - r) sin(theta)^2", "4/3 pi r^3",
    # functions of coordinates
    "f(x)", "A(r)^2", "A(r) B(r)", r"e^{2 \alpha(r)}", "f(x, y)", r"\gamma(r)", "gamma(x)",
    "x (y + 1)", "tan(x)", "asin(x)", "atan2(y, x)", "log(x)", "exp(x + 1)", "e^(-x)",
    "abs(x + 1)", "factorial(n + 1)", "binomial(n, k)", "floor(x)", "max(x, y)", "erf(x)",
    "conjugate(x)",
    # derivatives
    "a'(t)", "a''(t)", "diff(f(x), x, 4)", "diff(f(x, y), x)", "diff(f(x, y), x, y)",
    "diff(f(x, y), x, 2)", "A'(r)^2", "diff(f(t), t) + diff(f(t), t, 2)",
    # Greek letters and other names
    r"\lambda + \beta x + \zeta + \Gamma + Q + S", r"\alpha_\beta", r"\rho_{\rm vac}",
    r"M_\odot", r"N_\star", r"T_{\mathrm{eff}}", "x_0", "a_{12}", r"\dot a", r"\hat{x}",
    "Kretschmann", "Mr theta", "C1 e^x", r"\hbar",
    # e and i, pi and infinity
    "E + e", "I + i", "3 - 2i", "x + i y", "(1 + i)/2", "e^x", "pi r^2", "oo", "-oo",
    # numbers as written
    "0.3", "0.1 + 0.2", "6.02214076e23", "1.5e-10", "100.0", "2.5 * 2", "0.005", "-1.5",
    "123456789012345678901234567890",
    # sums, integrals and piecewise
    "integrate(f(x), x)", r"\int_0^L f(x) dx", r"\sum_{n=1}^{\infty} 1/n^2",
    r"\prod_{n=1}^{k} n", "{ 1, x >= 0; 0, otherwise }",
    "{ x, x < 0; x^2, x < 1; 1, otherwise }", "limit(f(x), x, 0)", r"\lim_{x \to 0} f(x)",
    # relations and truth values
    "x^2 + y^2 = 1", "x != y", "2x <= 1", "x >= y + 1", "True", "False",
    # containers
    "[1, 2, 3]", "[x, y^2]", "[[1, 2], [3, 4]]", r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}",
    "solve(x^2 = 4, x)",
]  # fmt: skip

#: Where a form does not read back yet, and why: what the parser would need to learn.
GAPS = {
    ("latex", "conjugate(x)"): r"\overline{x} reads as the symbol xbar",
    ("unicode", r"\rho_{\rm vac}"): "ρ_vac reads as the tensor ρ with indices v, a, c",
    ("unicode", r"\dot a"): "combining accents (ȧ, x̂) are not read",
    ("unicode", r"\hat{x}"): "combining accents (ȧ, x̂) are not read",
    ("unicode", "Mr theta"): "beside Unicode, a name of several letters reads as a product",
}


@pytest.mark.parametrize("form", ["text", "latex", "unicode"])
@pytest.mark.parametrize("source", CORPUS)
def test_every_form_reads_back(source, form):
    value = fresh(source)
    shown = getattr(display, form)(value)
    if (form, source) in GAPS:
        pytest.xfail(GAPS[(form, source)])
    assert same(value, fresh(shown)), f"{shown!r} reads back as {fresh(shown)!r}"


def test_text_is_ascii_and_reads_back_beside_latex():
    # A Greek letter is a command, which makes a statement LaTeX: names of several
    # letters then keep together, as \mathrm{Mr}.
    value = sympy.Symbol("Mr") * theta + sympy.Symbol("adot")
    shown = display.text(value)
    assert shown.isascii()
    assert shown == r"adot + \mathrm{Mr}*\theta" or same(value, fresh(shown))
    assert same(value, fresh(shown))


# -- numbers ------------------------------------


def test_floats_are_shown_as_written():  # CLI-29
    assert [display.unicode(fresh(s)) for s in ("0.3", "0.005", "2.0 * 3")] == [
        "0.3",
        "0.005",
        "6.0",
    ]
    assert display.unicode(fresh("6.02214076e23")) == "6.02214076×10²³"
    assert display.text(fresh("6.02214076e23")) == "6.02214076e+23"
    assert display.latex(fresh("6.02214076e23")) == r"6.02214076 \times 10^{23}"
    assert display.text(fresh("1e-05")) == "1.0e-5"  # a float, not 1/100000


@pytest.mark.parametrize("source", ["10000!", "2^{20000}"])
def test_an_integer_too_long_to_read_is_shortened(source):  # ERR-01
    value = fresh(source)
    digits = display.digits(int(value))
    for form in ("unicode", "text", "latex"):
        shown = getattr(display, form)(value)
        assert len(shown) < 120
    assert display.unicode(value).endswith(f"(an integer of {digits:,} digits)")
    assert display.data(value)["digits"] == digits
    assert display.unicode(fresh("2^{100}")) == "1267650600228229401496703205376"


def test_euler_and_the_imaginary_unit_are_e_and_i():  # DOC-21, AGT-19
    assert display.unicode(fresh("E + e")) == "E + e"
    assert display.text(fresh("I + i")) == "I + i"
    assert display.unicode(fresh("e^{i pi/4}")) == "e^(i π/4)"


# -- names and notation ------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"M_\odot", r"M_{\odot}"),
        (r"N_\star", r"N_{\star}"),
        (r"\rho_{\rm vac}", r"\rho_{\mathrm{vac}}"),
        ("True", r"\text{True}"),
        ("x_0", "x_{0}"),
        ("Kretschmann", r"\mathrm{Kretschmann}"),
    ],
)
def test_latex_for_papers(source, expected):  # AGT-15, CLI-30
    assert display.latex(fresh(source)) == expected


def test_derivatives_and_unknown_functions_in_the_language_s_notation():  # DOC-12, CLI-09
    a, t = Function("a"), sympy.Symbol("t")
    f = Function("f")
    assert display.unicode(Derivative(a(t), t)) == "a'(t)"
    assert display.text(Derivative(f(x, y), x, y)) == "diff(f(x, y), x, y)"
    assert display.unicode(Derivative(f(x, y), x, y)) == "∂_x ∂_y f(x, y)"
    assert display.latex(f(x)) == r"f\left(x\right)"
    assert display.unicode(sympy.Ne(sympy.Symbol("Lambda"), 0)) == "Λ ≠ 0"
    assert display.unicode(fresh(r"\sum_{n=1}^{\infty} 1/n^2")) == "∑_{n=1}^∞ 1/n²"


def test_unicode_is_as_a_terminal_shows_mathematics():
    assert display.unicode(48 * M**2 / r**6) == "48M²/r⁶"
    assert display.unicode(M / (r * (r - 2 * M))) == "M/(r (r − 2M))"
    assert display.unicode(sympy.sqrt(x + 1)) == "√(x + 1)"
    assert display.unicode(sympy.pi * r**2) == "π r²"
    assert display.unicode(sympy.oo) == "∞"


def test_strings_are_told_from_values():  # CLI-42
    assert display.unicode("True") == '"True"'
    assert display.text("x + 1") == '"x + 1"'
    assert display.unicode(sympy.true) == "True"
    assert display.latex("hello") == r"\text{hello}"


def test_none_and_booleans():  # CLI-30
    assert [display.text(None), display.latex(None)] == ["None", r"\text{None}"]
    assert display.latex(True) == r"\text{True}"
    assert display.latex(fresh("1 = 1")) == r"\text{True}"


# -- tensors ------------------------------------


@pytest.fixture(scope="module")
def schwarzschild():
    wb = Workbook()
    wb.expr(r"""
        Coordinates := [t, r, \theta, \phi]
        g_{\mu\nu} := Schwarzschild(M)
    """)
    return wb


def test_a_tensor_is_its_name_then_a_line_for_each_component(schwarzschild):  # API-05
    gamma = schwarzschild.expr(r"\Gamma^{a}_{bc}")
    lines = display.unicode(gamma).splitlines()
    assert lines[0] == "Γ^a_{bc} in (t, r, θ, φ): 13 of 64 components non-zero"
    # components equal by an exchange of indices share a line (API-33)
    assert lines[1] == "  Γ^t_{tr} = Γ^t_{rt} = M/(r (r − 2M))"
    assert "  Γ^θ_{φφ} = −sin(θ) cos(θ)" in lines
    assert len(lines) == 10
    assert repr(gamma) == display.unicode(gamma)
    text = display.text(gamma).splitlines()
    assert text[1] == r"  \Gamma^{t}_{tr} = \Gamma^{t}_{rt} = M/(r*(r - 2*M))"


def test_component_lines_read_back(schwarzschild):
    gamma = schwarzschild.expr(r"\Gamma^{a}_{bc}")
    for form in ("unicode", "text"):
        for line in getattr(display, form)(gamma).splitlines()[1:]:
            assert schwarzschild.expr(line.strip()) is sympy.true, line


def test_antisymmetric_components_share_a_line_with_their_sign(schwarzschild):
    riemann = schwarzschild.expr("R_{abcd}")
    lines = display.unicode(riemann).splitlines()
    assert lines[1] == "  R_{trtr} = −R_{trrt} = −R_{rttr} = R_{rtrt} = −2M/r³"
    assert len(lines) == 7


def test_an_all_zero_tensor_is_one_line(schwarzschild):  # API-27
    ricci = schwarzschild.expr(r"R_{\mu\nu}")
    assert display.unicode(ricci) == "R_{μν} = 0"
    assert display.latex(ricci) == r"R_{\mu\nu} = 0"


def test_a_slice_keeps_its_fixed_indices(schwarzschild):  # API-09, DOC-22
    gamma = schwarzschild.expr(r"\Gamma^{a}_{bc}")
    row = gamma["r", "t", :]
    assert (
        display.latex(row).splitlines()[1]
        == r"\Gamma^{r}_{tt} &= \frac{M \left(r - 2 M\right)}{r^{3}}"
    )
    assert "Γ_" not in display.unicode(row)
    # as a workbook shows what was written: \Gamma^{r}_{tb}, b free
    written = Tensor(row.components, "_{b}", row.coordinates, name="Gamma",
                     source=r"\Gamma^{r}_{tb}")  # fmt: skip
    assert display.unicode(written).splitlines()[0].startswith("Γ^r_{tb} in (t, r, θ, φ)")
    assert display.unicode(written).splitlines()[1] == "  Γ^r_{tt} = M (r − 2M)/r³"


def test_an_unnamed_result_is_never_given_a_name(schwarzschild):  # API-26, CLI-10
    t_, x_ = symbols("t x")
    eta = Tensor([[-1, 0], [0, 1]], "_{ab}", (t_, x_), name="eta")
    v = Tensor([2, 3], "^{b}", (t_, x_), name="v")
    shown = display.unicode(eta * v)
    assert shown.splitlines()[1:] == ["  η_{tb} v^b = −2", "  η_{xb} v^b = 3"]
    anonymous = Tensor([1, 2], "_{a}", (t_, x_))
    assert not re.search(r"\bT_", display.unicode(anonymous))
    doubled = 2 * schwarzschild.expr(r"g_{\mu\nu}")
    assert display.unicode(doubled).splitlines()[1] == "  2 g_{tt} = 4M/r − 2"
    written = Tensor(doubled.components, doubled.indices, doubled.coordinates,
                     source=r"2 R_{\mu\nu} + g_{\mu\nu}")  # fmt: skip
    assert display.unicode(written).splitlines()[0].startswith("2 R_{μν} + g_{μν} in")
    assert display.unicode(written).splitlines()[1].startswith("  2 R_{tt} + g_{tt} = ")


def test_a_result_with_an_index_of_its_own_names_components_after_it():
    delta, epsilon = symbols("Delta epsilon")
    root = sympy.sqrt(delta**2 + epsilon**2)
    spectrum = Tensor([-root, root], "_{k}", spaces=[Space("eigenvectors", 2)],
                      source=r"\operatorname{spec}(H^i{}_j)")  # fmt: skip
    assert display.unicode(spectrum).splitlines() == [
        "spec(H^i{}_j) over eigenvectors: 2 components non-zero",
        "  (spec(H^i{}_j))_0 = −√(Δ² + ε²)",
        "  (spec(H^i{}_j))_1 = √(Δ² + ε²)",
    ]


def test_a_name_with_a_subscript_is_set_apart(schwarzschild):  # API-10
    tensor = Tensor([1, x], "_{a}", (sympy.Symbol("t"), x), name="A_0")
    assert display.latex(tensor).splitlines()[1].startswith("{A_{0}}_{t}")
    assert "}_{" not in display.latex(tensor).replace("{A_{0}}_{", "")


def test_a_tensor_over_spaces_of_different_sizes(schwarzschild):  # CLI-4
    wb = Workbook()
    value = wb.expr(r"""
        \mathcal{T} := \mathbb{R}^{3}
        \mathcal{A} := \mathbb{R}^{2}
        t \in \mathcal{T}
        a \in \mathcal{A}
        R_{ta} := [[1, 2], [3, 4], [5, 6]]
        R_{ta}
    """)
    shown = display.unicode(value)
    assert "R_{ta}" in shown
    assert value.nonzero()[(2, 1)] == 6


def test_latex_of_every_rank_lists_components(schwarzschild):  # CLI-30
    metric = schwarzschild.expr(r"g_{\mu\nu}")
    shown = display.latex(metric)
    assert shown.startswith("\\begin{aligned}\n")
    assert r"g_{\theta\theta} &= r^{2}" in shown
    assert "\\\\\n" in shown  # a row to a line


# -- numerical data ------------------------------------


def test_large_data_is_summarised_never_dumped():  # API-18
    wb = Workbook()
    wb["A_{ij}"] = np.random.default_rng(0).normal(size=(500, 500))
    tensor = wb.expr("A_{ij}")
    shown = display.unicode(tensor)
    lines = shown.splitlines()
    assert lines[0].endswith("500 × 500 float64")
    assert "min" in lines[2]
    assert "std" in lines[2]
    assert len(shown) < 1500
    assert "<table>" in tensor._repr_html_()
    assert r"\in \mathbb{R}^{500 \times 500}" in display.latex(tensor)


def test_pandas_labels_are_kept():  # API-18
    pd = pytest.importorskip("pandas")
    wb = Workbook()
    wb["C_{ab}"] = pd.DataFrame(
        [[1.0, 0.3], [0.3, 2.0]],
        index=pd.Index(["AAPL", "MSFT"], name="assets"),
        columns=pd.Index(["AAPL", "MSFT"], name="assets"),
    )
    tensor = wb.expr("C_{ab}")
    shown = display.unicode(tensor)
    assert "a, b ∈ assets (AAPL, MSFT)" in shown
    assert "AAPL  MSFT" in shown
    assert "<th>MSFT</th>" in display.html(tensor)


def test_sparse_and_jax_data():  # API-18
    scipy_sparse = pytest.importorskip("scipy.sparse")
    wb = Workbook()
    wb["L_{ij}"] = scipy_sparse.random(1000, 1000, density=0.001, format="csr", random_state=1)
    shown = display.unicode(wb.expr("L_{ij}"))
    assert "sparse (CSR), 1,000 stored (0.1%)" in shown.splitlines()[0]
    jax = pytest.importorskip("jax")
    with Workbook(backend="jax") as book:
        book["v^a"] = np.array([1.0, 2.0])
        assert "(JAX)" in display.unicode(book.expr("v^a"))
    assert jax is not None


def test_raw_arrays_are_summarised():
    shown = display.unicode(np.arange(12.0).reshape(3, 4))
    assert shown.splitlines()[0] == "3 × 4 float64 (NumPy)"
    assert display.unicode(np.float64(0.25)) == "0.25"


# -- equations ------------------------------------


def test_an_equation_says_whether_it_holds_and_where_not(schwarzschild):  # API-02, DOC-8
    holds = schwarzschild.expr(r"R_{\mu\nu} = 0")
    holds.source = r"R_{\mu\nu} = 0"
    assert display.unicode(holds) == "R_{μν} = 0: holds in all 10 components"
    assert display.latex(holds) == r"R_{\mu\nu} = 0 \tag*{holds in all 10 components}"
    fails = schwarzschild.expr(r"R_{\mu\nu} = g_{\mu\nu}")
    fails.source = r"R_{\mu\nu} = g_{\mu\nu}"
    lines = display.unicode(fails).splitlines()
    assert lines[0] == "R_{μν} = g_{μν}: differs in 4 of 10 components"
    assert lines[1] == "  R_{tt} = 0 but g_{tt} = 2M/r − 1"
    assert r"R_{tt} &= 0 \neq g_{tt} = \frac{2 M}{r} - 1" in display.latex(fails)


def test_an_equation_without_its_source_still_says_where(schwarzschild):
    from relativisticpy.tensors.equation import TensorEquation

    fails = schwarzschild.expr(r"\Gamma^{a}_{bc} = 0")  # a workbook names it as written
    assert display.unicode(fails).splitlines()[0] == "Γ^a_{bc} = 0: differs in 9 of 40 components"
    bare = TensorEquation(fails.lhs, fails.rhs, fails.indices, fails.coordinates, fails.spaces)
    lines = display.unicode(bare).splitlines()
    assert lines[0] == "equation ^a_{bc}: differs in 9 of 40 components"
    assert lines[1] == "  ^t_{tr}: M/(r (r − 2M)) ≠ 0"


def test_a_large_equation_is_shown_in_milliseconds():  # API-3
    wb = Workbook()
    wb["P_{ij}"] = np.random.default_rng(1).normal(size=(500, 500))
    equation = wb.expr("P_{ij} = P_{ji}")
    start = time.perf_counter()
    shown, typeset = repr(equation), display.latex(equation)
    assert time.perf_counter() - start < 1.0
    assert shown.splitlines()[0].startswith("P_{ij} = P_{ji}: differs in 249,500 of 250,000")
    assert len(shown.splitlines()) <= 22
    assert len(typeset) < 5000


# -- data for programs ------------------------------------


def test_data_is_json_with_every_form(schwarzschild):
    values = [
        fresh("48 M^2/r^6"),
        True,
        None,
        "hello",
        [1, x],
        fresh("factorial(100)"),
        schwarzschild.expr(r"\Gamma^{a}_{bc}"),
        schwarzschild.expr(r"R_{\mu\nu} = g_{\mu\nu}"),
        schwarzschild.metric,
        Space("assets", 2, basis=("AAPL", "MSFT")),
        np.eye(2),
        sympy.Eq(25.0, 24),
    ]
    for value in values:
        described = display.data(value)
        json.dumps(described)
        assert {"kind", "text", "unicode", "latex"} <= described.keys()


def test_data_describes_a_tensor_by_its_components(schwarzschild):
    described = display.data(schwarzschild.expr(r"\Gamma^{a}_{bc}"))
    assert described["name"] == "Gamma"
    assert described["indices"]["text"] == "^{a}_{bc}"
    assert described["components"]["t,t,r"]["text"] == "M/(r*(r - 2*M))"
    equation = display.data(schwarzschild.expr(r"R_{\mu\nu} = g_{\mu\nu}"))
    assert (equation["holds"], equation["failing"], equation["components"]) == (False, 4, 10)
    assert equation["failures"][0]["component"] == ["t", "t"]
    assert display.data(sympy.Eq(25.0, 24, evaluate=False))["holds"] is False


def test_other_results_show_themselves_through_hooks():
    class Shown:
        def _display_unicode(self):
            return "ü"

        def _display_data(self):
            return {"kind": "shown"}

    assert display.unicode(Shown()) == "ü"
    assert display.data(Shown())["kind"] == "shown"


def test_indices_as_written():  # API-19
    indices = Indices([Index("mu"), Index("nu")])
    assert (str(indices), indices.unicode(), indices.latex()) == (
        r"_{\mu\nu}",
        "_{μν}",
        r"_{\mu\nu}",
    )
    primed = Indices([Index("mu'"), Index("nu'")])
    assert primed.unicode() == "_{μ′ν′}"
    assert Integer(2) + Rational(1, 2) == Rational(5, 2)
