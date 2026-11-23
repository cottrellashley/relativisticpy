r"""What an agent gets back: requests an agent typically makes, made as it makes them,
through the MCP server, each with its answer computed without RelativisticPy, by SymPy, or
with the refusal it should meet.

The rule they hold the program to: every answer is right, or the program says it cannot
give one. A refusal has a type, a message, a place, and a hint of what to write instead
where there is one; a value kept as written says it is not complete. A request the program
does not meet yet is marked to fail, naming the cause, and the fix of the cause removes the
mark.

The values are read from their text form by SymPy's own parser, never by the program under
test, and compared with what SymPy computes from the same mathematics."""

import re
import sys
from functools import cache
from itertools import product

import pytest
import sympy
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

pytest.importorskip("mcp", reason="the MCP server needs the mcp extra")

from anyio.from_thread import start_blocking_portal
from mcp import Client

from relativisticpy import mcp as server

x, y, k, n, t, r, M, L, c, v = sympy.symbols("x y k n t r M L c v")
theta, phi, omega, beta = sympy.symbols("theta phi omega beta")
a, Y = sympy.Function("a"), sympy.Function("y")

#: The lines of work that own the causes of failures here, as they are named.
TRACK_A = "reading, track A: "
TRACK_B = "reading, track B: "


def xfail(cause):
    return pytest.mark.xfail(strict=True, reason=cause)


# -- the agent's side -------------------------------------------------------------------------


class Agent:
    """A client of the server, kept open, as an agent keeps one: each call gives a tool's
    structured result."""

    def __init__(self, portal, client):
        self.portal, self.client = portal, client

    def call(self, tool, **arguments):
        result = self.portal.call(self.client.call_tool, tool, arguments)
        assert not result.is_error, result.content[0].text
        return result.structured_content

    def compute(self, source, **arguments):
        return self.call("compute", source=source, **arguments)

    def verify(self, claim, context=None):
        return self.call("verify", claim=claim, **({"context": context} if context else {}))


@pytest.fixture(scope="module")
def agent():
    """An agent's connection to a server in memory, open for every request here, so that
    its worker processes start once."""
    built, _ = server._build(server.Settings())
    with start_blocking_portal() as portal:
        connection = portal.wrap_async_context_manager(Client(built, raise_exceptions=True))
        with connection as client:
            yield Agent(portal, client)


def answer(agent, source, **arguments):
    """The one value of a program's last statement, once every statement has run."""
    result = agent.compute(source, **arguments)
    *_, last = result["statements"]
    assert last["error"] is None, last["error"]
    assert result["ok"]
    [value] = last["values"]
    return result, last, value


# -- reading a value as SymPy reads it --------------------------------------------------------

#: The names of the text form that are no symbol, and the Greek letters, which SymPy's
#: parser would take for its functions, as beta and gamma, read as symbols.
NAMES = {
    "e": sympy.E,
    "i": sympy.I,
    "pi": sympy.pi,
    "oo": sympy.oo,
    **{name: sympy.Symbol(name) for name in ("beta", "theta", "phi", "omega", "mu", "nu")},
}
#: a'(t), the derivative of an unknown function, as the text form writes it.
PRIMED = re.compile(r"(\w+)'\((\w+)\)")


def read(text):
    """A value's text form, in the language's plain syntax, read by SymPy's parser."""
    plain = PRIMED.sub(r"Derivative(\1(\2), \2)", re.sub(r"\\([A-Za-z]+)", r"\1", text))
    transformations = (*standard_transformations, convert_xor)
    return parse_expr(plain, local_dict=dict(NAMES), transformations=transformations)


def same(found, expected, *, form=False):
    """Whether a value read is the value expected: equal, or, with ``form``, written alike,
    as an expanded or a factored polynomial must be."""
    if isinstance(expected, sympy.MatrixBase):
        found = sympy.Matrix(found)
        difference = (found - expected).applyfunc(sympy.simplify)
        return found.shape == expected.shape and difference.is_zero_matrix
    if isinstance(expected, set):
        return {sympy.sympify(item) for item in found} == expected
    if isinstance(expected, list):
        return len(found) == len(expected) and all(map(same, found, expected))
    if form:
        return found == expected
    return sympy.simplify(sympy.sympify(found) - expected) == 0


# -- what SymPy computes of a metric, by the formulas of a textbook -------------------------


def christoffel(g, xs):
    """Γ^a_{bc} = g^{ad} (∂_b g_{dc} + ∂_c g_{db} - ∂_d g_{bc}) / 2, by (a, b, c)."""
    inverse, size = g.inv(), range(len(xs))
    found = {}
    for i, b, j in product(size, repeat=3):
        terms = (
            inverse[i, d] * (g[d, b].diff(xs[j]) + g[d, j].diff(xs[b]) - g[b, j].diff(xs[d]))
            for d in size
        )
        found[i, b, j] = sympy.simplify(sum(terms) / 2)
    return found


def riemann(g, xs):
    """R^a_{bcd} = ∂_c Γ^a_{db} - ∂_d Γ^a_{cb} + Γ^a_{ce} Γ^e_{db} - Γ^a_{de} Γ^e_{cb}."""
    gamma, size = christoffel(g, xs), range(len(xs))
    found = {}
    for i, b, j, d in product(size, repeat=4):
        moved = sum(gamma[i, j, e] * gamma[e, d, b] - gamma[i, d, e] * gamma[e, j, b] for e in size)
        changed = gamma[i, d, b].diff(xs[j]) - gamma[i, j, b].diff(xs[d])
        found[i, b, j, d] = sympy.simplify(changed + moved)
    return found


def ricci_scalar(g, xs):
    """R = g^{bd} R^a_{bad}."""
    curvature, inverse, size = riemann(g, xs), g.inv(), range(len(xs))
    terms = (inverse[b, d] * curvature[i, b, i, d] for i, b, d in product(size, repeat=3))
    return sympy.simplify(sum(terms))


def kretschmann(g, xs):
    """R_{abcd} R^{abcd}, of a diagonal metric."""
    curvature, inverse = riemann(g, xs), g.inv()
    terms = (
        g[i, i] * inverse[b, b] * inverse[j, j] * inverse[d, d] * value**2
        for (i, b, j, d), value in curvature.items()
    )
    return sympy.simplify(sum(terms))


@cache
def schwarzschild():
    """Schwarzschild's metric, in units where G = c = 1, with signature (-, +, +, +)."""
    lapse = 1 - 2 * M / r
    return sympy.diag(-lapse, 1 / lapse, r**2, r**2 * sympy.sin(theta) ** 2), (t, r, theta, phi)


@cache
def flat_universe():
    """The spatially flat FLRW metric, ds² = -dt² + a(t)² (dx² + dy² + dz²)."""
    xs = (t, *sympy.symbols("x y z"))
    return sympy.diag(-1, a(t) ** 2, a(t) ** 2, a(t) ** 2), xs


def sphere():
    """The sphere of radius r, ds² = r² (dθ² + sin²θ dφ²)."""
    return sympy.diag(r**2, r**2 * sympy.sin(theta) ** 2), (theta, phi)


SCHWARZSCHILD = r"g_{\mu\nu} := Schwarzschild(M)"
FLRW = r"ds^2 = -dt^2 + a(t)^2 (dx^2 + dy^2 + dz^2)"
MATRIX = r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}"


# -- answers --------------------------------------------------------------------------------

ANSWERS = [
    # algebra
    pytest.param(r"\frac{x^2 - 1}{x - 1}", lambda: (x**2 - 1) / (x - 1), id="latex-fraction"),
    pytest.param("expand((x+1)^3)", lambda: sympy.expand((x + 1) ** 3), id="plain-expand"),
    pytest.param("factor(x^2 - 5x + 6)", lambda: sympy.factor(x**2 - 5 * x + 6), id="plain-factor"),
    pytest.param(
        "solve(x^2 - 5x + 6 = 0, x)",
        lambda: set(sympy.solve(x**2 - 5 * x + 6, x)),
        id="plain-solve",
    ),
    pytest.param(r"2^{100}", lambda: sympy.Integer(2) ** 100, id="latex-large-integer"),
    pytest.param("x² + 2x + 1", lambda: x**2 + 2 * x + 1, id="unicode-polynomial"),
    pytest.param("√2 · √8", lambda: sympy.sqrt(2) * sympy.sqrt(8), id="unicode-roots"),
    pytest.param("(1 + 2i)(3 - i)", lambda: (1 + 2 * sympy.I) * (3 - sympy.I), id="latex-complex"),
    pytest.param(
        "\\gamma := \\frac{1}{\\sqrt{1 - \\beta^2}}\n\\gamma^2 (1 - \\beta^2)",
        lambda: (1 / sympy.sqrt(1 - beta**2)) ** 2 * (1 - beta**2),
        id="latex-lorentz-factor",
    ),
    pytest.param(
        r"\left(xy\right)^2",
        lambda: (x * y) ** 2,
        id="latex-power-of-a-product",
        marks=xfail(TRACK_B + r"\left(xy\right)^2 is read as x y^2, with no note"),
    ),
    # calculus
    pytest.param(
        r"\frac{d}{dx} \sin(x^2)", lambda: sympy.diff(sympy.sin(x**2), x), id="latex-chain-rule"
    ),
    pytest.param("diff(sin(x^2), x)", lambda: sympy.diff(sympy.sin(x**2), x), id="plain-diff"),
    pytest.param(
        "d/dx sin(x)",
        lambda: sympy.diff(sympy.sin(x), x),
        id="plain-d/dx",
        marks=xfail(TRACK_A + "plain d/dx sin(x) is read as the product d sin(x) / dx"),
    ),
    pytest.param(
        "sin²x",
        lambda: sympy.sin(x) ** 2,
        id="unicode-sin²x",
        marks=xfail(TRACK_A + "Unicode sin²x is read as the product s i n² x"),
    ),
    pytest.param(
        "∂/∂x (x² y)",
        lambda: sympy.diff(x**2 * y, x),
        id="unicode-∂/∂x",
        marks=xfail("reading: ∂/∂x is refused, as '\\partial needs an index or a variable'"),
    ),
    pytest.param(
        r"\frac{\partial}{\partial y} (x^2 y^3)",
        lambda: sympy.diff(x**2 * y**3, y),
        id="latex-partial",
    ),
    pytest.param(
        r"\frac{d}{dx} \left(xy\right)^2",
        lambda: sympy.diff((x * y) ** 2, x),
        id="latex-derivative-of-a-power-of-a-product",
        marks=xfail(TRACK_B + r"\left(xy\right)^2 is read as x y^2, so its derivative is y^2"),
    ),
    pytest.param(
        "f(x) := x^2 e^{-x}\nf'(x)",
        lambda: sympy.diff(x**2 * sympy.exp(-x), x),
        id="latex-defined-function-primed",
    ),
    pytest.param(
        r"\int_0^1 x^2 \, dx", lambda: sympy.integrate(x**2, (x, 0, 1)), id="latex-integral"
    ),
    pytest.param(
        r"\int_{-\infty}^{\infty} e^{-x^2} \, dx",
        lambda: sympy.integrate(sympy.exp(-(x**2)), (x, -sympy.oo, sympy.oo)),
        id="latex-gaussian-integral",
    ),
    pytest.param(
        "∫_0^π sin x dx",
        lambda: sympy.integrate(sympy.sin(x), (x, 0, sympy.pi)),
        id="unicode-integral",
        marks=xfail("reading: Unicode sin x, with no brackets, is read as the product s i n x"),
    ),
    pytest.param(
        r"\lim_{x \to 0} \frac{\sin x}{x}",
        lambda: sympy.limit(sympy.sin(x) / x, x, 0),
        id="latex-limit",
    ),
    pytest.param(
        "limit((1 - cos(x))/x^2, x, 0)",
        lambda: sympy.limit((1 - sympy.cos(x)) / x**2, x, 0),
        id="plain-limit",
    ),
    pytest.param(
        r"\sum_{n=1}^{\infty} \frac{1}{n^2}",
        lambda: sympy.summation(1 / n**2, (n, 1, sympy.oo)),
        id="latex-series",
    ),
    pytest.param(r"\sum_{k=0}^{n} k", lambda: sympy.summation(k, (k, 0, n)), id="latex-finite-sum"),
    # linear algebra
    pytest.param(
        rf"\det {MATRIX}", lambda: sympy.Matrix([[1, 2], [3, 4]]).det(), id="latex-determinant"
    ),
    pytest.param(
        "det([[1, 2], [3, 4]])",
        lambda: sympy.Matrix([[1, 2], [3, 4]]).det(),
        id="plain-determinant",
    ),
    pytest.param(
        rf"{MATRIX}^{{-1}}", lambda: sympy.Matrix([[1, 2], [3, 4]]).inv(), id="latex-inverse"
    ),
    pytest.param(
        rf"{MATRIX} \begin{{pmatrix}} 5 \\ 6 \end{{pmatrix}}",
        lambda: sympy.Matrix([[1, 2], [3, 4]]) * sympy.Matrix([5, 6]),
        id="latex-matrix-product",
    ),
    pytest.param(
        f"A := {MATRIX}\n" r"\operatorname{tr} A",
        lambda: sympy.Matrix([[1, 2], [3, 4]]).trace(),
        id="latex-trace",
    ),
    pytest.param(
        "spec([[2, 1], [1, 2]])",
        lambda: set(sympy.Matrix([[2, 1], [1, 2]]).eigenvals()),
        id="plain-eigenvalues",
    ),
    pytest.param(
        "\\vec{a} := (1, 2, 3)\n\\vec{b} := (4, 5, 6)\n\\vec{a} \\times \\vec{b}",
        lambda: list(sympy.Matrix([1, 2, 3]).cross(sympy.Matrix([4, 5, 6]))),
        id="latex-cross-product",
    ),
    # index notation and General Relativity
    pytest.param(
        f"{SCHWARZSCHILD}\n" r"\Gamma^{r}_{tt}",
        lambda: christoffel(*schwarzschild())[1, 0, 0],
        id="latex-christoffel-symbol",
    ),
    pytest.param(
        "g_{mu nu} := Schwarzschild(M)\nGamma^{r}_{t t}",
        lambda: christoffel(*schwarzschild())[1, 0, 0],
        id="plain-christoffel-symbol",
    ),
    pytest.param(
        f"{SCHWARZSCHILD}\nR_{{abcd}} R^{{abcd}}",
        lambda: kretschmann(*schwarzschild()),
        id="latex-kretschmann-scalar",
    ),
    pytest.param(
        f"{SCHWARZSCHILD}\nR", lambda: ricci_scalar(*schwarzschild()), id="latex-ricci-scalar"
    ),
    pytest.param(
        r"ds^2 = r^2 (d\theta^2 + \sin^2\theta \, d\phi^2)" "\nR",
        lambda: ricci_scalar(*sphere()),
        id="latex-line-element-of-a-sphere",
    ),
    pytest.param(
        f"{FLRW}\n" r"\Gamma^{x}_{tx}",
        lambda: christoffel(*flat_universe())[1, 0, 1],
        id="latex-line-element-of-a-universe",
    ),
]

#: The answers that are a form, not only a value: an expansion, a factorisation.
FORMS = {"plain-expand", "plain-factor"}


@pytest.mark.parametrize(("source", "expected"), ANSWERS)
def test_an_answer_is_right_exact_and_complete(agent, request, source, expected):
    result, _, value = answer(agent, source)
    assert (value["complete"], value["reasons"]) == (True, [])
    assert result["complete"]
    assert not value["approximate"]
    named = set(value.get("symbols", []))
    assert not named & {"e", "i"}, "e and i in the text form are Euler's number and i"
    wanted = expected()
    if isinstance(wanted, sympy.Basic) and not wanted.free_symbols:
        assert not named, f"a number holds no symbol: {named}"
    form = request.node.callspec.id in FORMS
    assert same(read(value["text"]), wanted, form=form), value["text"]


def test_an_antiderivative_is_one(agent):
    _, _, value = answer(agent, r"\int x e^x \, dx")
    assert value["complete"]
    assert sympy.simplify(sympy.diff(read(value["text"]), x) - x * sympy.exp(x)) == 0


def test_a_series_is_its_taylor_polynomial_and_order(agent):
    _, _, value = answer(agent, "series(e^x, x, 0, 4)")
    assert read(value["text"]) == sympy.series(sympy.exp(x), x, 0, 4)


# -- differential equations -----------------------------------------------------------------

ODES = [
    pytest.param(
        "dsolve(y'(t) = -k y(t))",
        lambda: sympy.Eq(Y(t).diff(t), -k * Y(t)),
        1,
        None,
        id="plain-decay",
    ),
    pytest.param(
        r"dsolve(\frac{dy}{dt} = -k y)",
        lambda: sympy.Eq(Y(t).diff(t), -k * Y(t)),
        1,
        None,
        id="latex-decay",
    ),
    pytest.param(
        "dsolve(y''(t) + y(t) = 0)",
        lambda: sympy.Eq(Y(t).diff(t, 2) + Y(t), 0),
        2,
        None,
        id="plain-oscillator",
    ),
    pytest.param(
        "dsolve(y'(t) = y(t), y(0) = 1)",
        lambda: sympy.Eq(Y(t).diff(t), Y(t)),
        0,
        {Y(0): 1},
        id="plain-initial-value",
    ),
    pytest.param(
        r"dsolve(y''(t) = -\omega^2 y(t), y(0) = 1, y'(0) = 0)",
        lambda: sympy.Eq(Y(t).diff(t, 2), -(omega**2) * Y(t)),
        0,
        {Y(0): 1, Y(t).diff(t).subs(t, 0): 0},
        id="latex-oscillator-at-rest",
    ),
]


@pytest.mark.parametrize(("source", "ode", "constants", "conditions"), ODES)
def test_an_ode_is_solved(agent, source, ode, constants, conditions):
    _, _, value = answer(agent, source)
    assert value["kind"] == "relation"
    assert value["complete"]
    lhs, rhs = read(value["lhs"]), read(value["rhs"])
    assert lhs == Y(t)
    assert sympy.checkodesol(ode(), sympy.Eq(lhs, rhs)) == (True, 0)
    found = {s for s in rhs.free_symbols if re.fullmatch(r"C\d+", s.name)}
    assert len(found) == constants
    if conditions is not None:
        solved = sympy.dsolve(ode(), Y(t), ics=conditions)
        assert sympy.simplify(rhs - solved.rhs) == 0


# -- numbers that are approximate -----------------------------------------------------------

NUMBERS = [
    pytest.param("0.1 + 0.2", lambda: sympy.Rational(3, 10), id="decimals"),
    pytest.param("N(pi)", lambda: sympy.pi, id="plain-numerical-value"),
    pytest.param(
        r"evalf(\int_0^1 \sin(\sin x) \, dx)",
        lambda: sympy.Integral(sympy.sin(sympy.sin(x)), (x, 0, 1)),
        id="latex-numerical-integral",
    ),
]


@pytest.mark.parametrize(("source", "expected"), NUMBERS)
def test_a_number_computed_in_floating_point_says_it_is_approximate(agent, source, expected):
    _, _, value = answer(agent, source)
    assert (value["complete"], value["approximate"]) == (True, True)
    assert abs(float(read(value["text"])) - float(sympy.N(expected()))) < 1e-12


@xfail("display: _float_text shows a float of five digits with the seventeen of a double")
def test_a_number_of_few_digits_shows_no_more(agent):
    _, _, value = answer(agent, r"evalf(\pi, 5)")
    assert value["approximate"]
    assert value["text"] == str(sympy.N(sympy.pi, 5))


# -- values kept as written -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "reason", "sympy_finds_none"),
    [
        pytest.param(
            r"\int_0^1 \sin(\sin x) \, dx",
            "integral",
            lambda: sympy.integrate(sympy.sin(sympy.sin(x)), (x, 0, 1)).has(sympy.Integral),
            id="latex-integral-with-no-closed-form",
        ),
        pytest.param(
            r"\sum_{n=1}^{\infty} \frac{1}{n^n}",
            "sum",
            lambda: sympy.summation(n**-n, (n, 1, sympy.oo)).has(sympy.Sum),
            id="latex-sum-with-no-closed-form",
        ),
        pytest.param("besselfoo(2.5)", "unknown function", None, id="plain-unknown-function"),
    ],
)
def test_a_value_kept_as_written_says_it_is_no_answer(agent, source, reason, sympy_finds_none):
    result, last, value = answer(agent, source)
    assert (value["complete"], value["reasons"]) == (False, [reason])
    assert not result["complete"]
    assert last["notes"]  # the reason, in words
    if sympy_finds_none is not None:  # kept because there is none to find, not by mistake
        assert sympy_finds_none()


def test_an_integral_is_its_value_or_says_it_is_kept_as_written(agent):
    # √(sinh² x + 1) is cosh x on the reals, whose integral SymPy knows; written so, the
    # integral was given as if it were its value, with no more than a note
    expected = sympy.integrate(sympy.cosh(x), (x, 0, 1))
    _, _, value = answer(agent, r"\int_0^1 \sqrt{\sinh(x)^2+1} dx")
    if value["complete"]:
        assert same(read(value["text"]), expected)
    else:
        assert value["reasons"] == ["integral"]


# -- refusals -------------------------------------------------------------------------------

REFUSALS = [
    pytest.param(r"\frac{1}{2", "ParseError", True, id="latex-unclosed-brace"),
    pytest.param(r"\frca{1}{2}", "ParseError", True, id="latex-misspelt-command"),
    pytest.param("R_{ab}", "CompileError", True, id="latex-curvature-without-a-metric"),
    pytest.param("simplfy(x)", "CompileError", True, id="plain-misspelt-function"),
    pytest.param(r"\infty - \infty", "EvaluationError", True, id="latex-no-value"),
    pytest.param(r"\lim_{x \to \infty} \sin x", "EvaluationError", False, id="latex-no-limit"),
    pytest.param(
        f"{SCHWARZSCHILD}\n" r"\Gamma^r_tt",
        "IndexNotationError",
        True,
        id="latex-scripts-without-braces",
        marks=xfail(r"reading/reader.py: \Gamma^r_tt is refused with no hint of \Gamma^{r}_{tt}"),
    ),
    pytest.param(
        "sum(1/n^2, (n, 1, oo))",
        "CompileError",
        True,
        id="plain-sum-call",
        marks=xfail(
            "lowering/expressions.py: a call of an unknown function given a list is refused "
            r"with no hint to write \sum_{n=1}^{\infty} 1/n^2"
        ),
    ),
]


@pytest.mark.parametrize(("source", "kind", "hinted"), REFUSALS)
def test_a_refusal_says_what_where_and_what_to_write_instead(agent, source, kind, hinted):
    result = agent.compute(source)
    *_, last = result["statements"]
    error = last["error"]
    assert error is not None, last["values"]
    assert last["values"] == []
    assert (error["type"], error["kind"]) == (kind, kind)
    assert error["message"]
    assert None not in (error["line"], error["column"], error["text"])  # where it is
    assert (result["ok"], result["complete"]) == (False, False)
    if hinted:  # what to write instead, where there is something to write
        assert error["hint"]


@pytest.mark.skipif(sys.platform == "win32", reason="a statement is stopped by SIGALRM")
def test_a_statement_over_the_time_limit_is_refused_with_how_to_go_on(agent):
    result = agent.compute("y := 3\nfactor(expand((x+y+z)^60) + 1)", time_limit=1)
    defined, over = result["statements"]
    assert defined["error"] is None
    error = over["error"]
    assert (error["type"], error["message"]) == ("NotComputedError", "over the time limit of 1 s")
    assert error["hint"].startswith("give it longer with time_limit")
    assert not result["ok"]


# -- claims ---------------------------------------------------------------------------------


def judged(lhs, rhs):
    """What SymPy makes of the claim lhs = rhs: holds, or differs by the difference."""
    difference = sympy.simplify(lhs - rhs)
    return ("holds", None) if difference == 0 else ("differs", difference)


EFFECTIVE_POTENTIAL = r"-\frac{M}{r} + \frac{L^2}{2 r^2} - \frac{M L^2}{r^3}"
POTENTIAL = -M / r + L**2 / (2 * r**2) - M * L**2 / r**3

CLAIMS = [
    pytest.param(
        r"\frac{d}{dx} x^3 = 3x^2",
        None,
        lambda: judged(sympy.diff(x**3, x), 3 * x**2),
        id="a-derivative-that-holds",
    ),
    pytest.param(
        r"\frac{d}{dx} x^3 = 2x^2",
        "Reals := [x]",
        lambda: judged(sympy.diff(x**3, x), 2 * x**2),
        id="a-derivative-that-differs",
    ),
    pytest.param(
        r"\sin(2x) = 2 \sin x \cos x",
        None,
        lambda: judged(sympy.sin(2 * x), 2 * sympy.sin(x) * sympy.cos(x)),
        id="a-double-angle",
    ),
    pytest.param(
        r"\int_0^1 x^2 \, dx = \frac{1}{2}",
        None,
        lambda: judged(sympy.integrate(x**2, (x, 0, 1)), sympy.Rational(1, 2)),
        id="an-integral-that-differs",
    ),
    pytest.param(
        r"\Gamma^{r}_{\theta\theta} = 2M - r",
        SCHWARZSCHILD,
        lambda: judged(christoffel(*schwarzschild())[1, 2, 2], 2 * M - r),
        id="a-christoffel-symbol-that-holds",
    ),
    pytest.param(
        r"R_{abcd} R^{abcd} = \frac{12 M^2}{r^6}",
        SCHWARZSCHILD,
        lambda: judged(kretschmann(*schwarzschild()), 12 * M**2 / r**6),
        id="a-curvature-scalar-that-differs",
    ),
    pytest.param(
        r"\Gamma^{x}_{tx} = \frac{\dot a}{a}",
        FLRW,
        lambda: judged(christoffel(*flat_universe())[1, 0, 1], a(t).diff(t) / a(t)),
        id="a-step-in-a-universe",
    ),
    pytest.param(
        rf"\frac{{d}}{{dr}}\left({EFFECTIVE_POTENTIAL}\right) = "
        r"\frac{M}{r^2} - \frac{L^2}{r^3} + \frac{3 M L^2}{r^4}",
        "Reals := [r, M, L]",
        lambda: judged(sympy.diff(POTENTIAL, r), M / r**2 - L**2 / r**3 + 3 * M * L**2 / r**4),
        id="a-step-of-an-orbit-that-holds",
    ),
    pytest.param(
        rf"\frac{{d}}{{dr}}\left({EFFECTIVE_POTENTIAL}\right) = "
        r"\frac{M}{r^2} - \frac{L^2}{r^3} - \frac{3 M L^2}{r^4}",
        "Reals := [r, M, L]",
        lambda: judged(sympy.diff(POTENTIAL, r), M / r**2 - L**2 / r**3 - 3 * M * L**2 / r**4),
        id="a-step-of-an-orbit-with-a-sign-wrong",
    ),
    pytest.param(
        r"\frac{d\gamma}{dv} = \frac{v}{c^2} \gamma^3",
        r"\gamma := \frac{1}{\sqrt{1 - v^2/c^2}}",
        lambda: judged(
            sympy.diff(1 / sympy.sqrt(1 - v**2 / c**2), v),
            v / c**2 * (1 / sympy.sqrt(1 - v**2 / c**2)) ** 3,
        ),
        id="a-step-of-special-relativity",
    ),
]


@pytest.mark.parametrize(("claim", "context", "expected"), CLAIMS)
def test_a_claim_is_judged_as_sympy_judges_it(agent, claim, context, expected):
    verdict, difference = expected()
    result = agent.verify(claim, context)
    [link] = result["links"]
    assert (result["verdict"], link["verdict"]) == (verdict, verdict)
    assert link["error"] is None
    if difference is not None:
        assert same(read(link["difference"]["text"]), difference)


@pytest.mark.parametrize(
    ("claim", "verdict", "kind"),
    [
        pytest.param(
            r"\pi \approx 3.14",
            "not decided",
            "NotComputedError",
            id="approximately",
            marks=xfail(
                "lowering/expressions.py: a relation read but not computed, as ≈, is refused "
                "with no hint, as to write the difference of its sides"
            ),
        ),
        pytest.param(r"e^x \geq 1 + x", "not decided", "UndecidedError", id="an-inequality"),
        pytest.param("x^2 + 1", "error", "UsageError", id="an-expression"),
        pytest.param(r"\frac{1}{2 = x", "error", "ParseError", id="a-claim-unread"),
    ],
)
def test_a_claim_not_decided_or_refused_says_why_and_what_to_write(agent, claim, verdict, kind):
    result = agent.verify(claim)
    [link] = result["links"]
    assert (result["verdict"], link["verdict"]) == (verdict, verdict)
    assert link["error"]["type"] == kind
    assert link["error"]["message"]
    assert link["error"]["line"] is not None
    assert link["error"]["hint"]


def test_a_claim_about_what_nothing_defines_relates_and_says_by_how_much(agent):
    result = agent.verify(r"\frac{d}{dx} \sin^2 x = 2\cos x")
    [link] = result["links"]
    assert result["verdict"] == "relates"
    expected = sympy.diff(sympy.sin(x) ** 2, x) - 2 * sympy.cos(x)
    assert same(read(link["difference"]["text"]), expected)


def test_a_chain_is_judged_link_by_link(agent):
    result = agent.verify("(x+1)^2 = x^2 + 2x + 1 = x^2 + 1", "Reals := [x]")
    first, second = result["links"]
    assert [first["verdict"], second["verdict"]] == [
        judged((x + 1) ** 2, x**2 + 2 * x + 1)[0],
        judged(x**2 + 2 * x + 1, x**2 + 1)[0],
    ]
    assert result["verdict"] == "differs"
    assert same(read(second["difference"]["text"]), 2 * x)


def test_a_claim_with_an_integral_is_never_judged_wrong_for_want_of_its_value(agent):
    # √(sinh² x + 1) = cosh x: the claim holds, or is not decided, never differs
    result = agent.verify(r"\int_0^1 \sqrt{\sinh(x)^2+1} dx = \sinh(1)")
    assert judged(sympy.integrate(sympy.cosh(x), (x, 0, 1)), sympy.sinh(1))[0] == "holds"
    assert result["verdict"] in ("holds", "not decided")


# -- sessions -------------------------------------------------------------------------------


def test_a_session_keeps_a_metric_from_one_call_to_the_next(agent):
    agent.call("reset_session", session="agent-suite")
    defined = agent.compute(SCHWARZSCHILD, session="agent-suite")
    later = agent.compute("R_{abcd} R^{abcd}", session="agent-suite")
    agent.call("reset_session", session="agent-suite")
    forgotten = agent.compute("R_{abcd} R^{abcd}", session="agent-suite")
    agent.call("reset_session", session="agent-suite")
    assert (defined["ok"], defined["session"]) == (True, "agent-suite")
    [value] = later["statements"][0]["values"]
    assert same(read(value["text"]), kretschmann(*schwarzschild()))
    assert forgotten["statements"][0]["error"]["type"] == "CompileError"
