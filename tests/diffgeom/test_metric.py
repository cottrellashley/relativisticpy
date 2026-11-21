"""The Metric class: geometry in Python."""

import inspect

import pytest
import sympy
from sympy import Array, Function, Matrix, Rational, Symbol, cos, diag, sin, sqrt, symbols
from sympy.core.function import AppliedUndef

from relativisticpy import Metric, Tensor, Workbook, spacetimes
from relativisticpy.diffgeom import differential, metric_from_line_element

t, r, theta, phi, M, a, Q, k, L = symbols("t r theta phi M a Q k L")


def test_construction_checks_the_components():
    with pytest.raises(ValueError, match="symmetric"):
        Metric([[1, 2], [0, 1]], "xy")
    with pytest.raises(ValueError, match="non-degenerate"):
        Metric([[1, 1], [1, 1]], "xy")
    with pytest.raises(ValueError, match="needs 2 coordinates"):
        Metric([[1, 0], [0, 1]], "xyz")
    with pytest.raises(ValueError, match="square"):
        Metric([[1, 0, 0], [0, 1, 0]], "xy")


def test_a_metric_is_a_tensor():
    g = Metric(diag(-1, 1), "tx")
    assert isinstance(g, Tensor)
    assert str(g.indices) == "_{mu}_{nu}"
    assert repr(g) == "Metric([[-1, 0], [0, 1]])"
    assert g.inverse.components == Array([[-1, 0], [0, 1]])


def test_the_round_sphere():
    R = Symbol("R", positive=True)
    sphere = Metric(diag(R**2, R**2 * sin(theta) ** 2), (theta, phi))
    assert sphere.ricci_scalar == 2 / R**2
    assert sphere.christoffel[0, 1, 1] == -sin(theta) * cos(theta)
    assert sphere.christoffel[1, 0, 1] == cos(theta) / sin(theta)
    with pytest.raises(ValueError, match="three or more dimensions"):
        sphere.weyl  # noqa: B018


def test_schwarzschild():
    g = spacetimes.schwarzschild()
    assert g.coordinates == (t, r, theta, phi)
    assert g.ricci.components == Array.zeros(4, 4)
    assert g.ricci_scalar == 0
    assert g.einstein.components == Array.zeros(4, 4)
    assert g.kretschmann == 48 * M**2 / r**6
    assert sympy.simplify(g.determinant + r**4 * sin(theta) ** 2) == 0
    assert g.riemann[0, 1, 0, 1] == 2 * M / (r**2 * (r - 2 * M))


def test_weyl_squared_equals_kretschmann_in_vacuum():
    g = spacetimes.schwarzschild()
    weyl = g.weyl
    from relativisticpy.tensors import contract

    upper = Tensor(g.inverse_matrix.tolist(), "^{a e}")
    squared = contract(
        weyl.relabel("_{a b c d}"),
        weyl.relabel("_{e f h i}"),
        upper,
        upper.relabel("^{b f}"),
        upper.relabel("^{c h}"),
        upper.relabel("^{d i}"),
    )
    assert sympy.simplify(squared - g.kretschmann) == 0


def test_flrw_is_conformally_flat_and_gives_friedmann():
    g = spacetimes.flrw()
    scale = Function("a")(t)
    assert set(g.weyl.components.reshape(256).tolist()) == {0}
    assert sympy.simplify(g.einstein[0, 0] - 3 * (k + scale.diff(t) ** 2) / scale**2) == 0


def test_a_named_scale_factor_is_a_function_of_time():
    scale = Function("a")(t)
    assert spacetimes.flrw(k, Symbol("a")).matrix[1, 1] == scale**2 / (1 - k * r**2)
    assert spacetimes.flrw(-1, t).matrix[2, 2] == t**2 * r**2  # an expression is kept
    (positive,) = spacetimes.flrw(k, Symbol("b", positive=True)).matrix[2, 2].atoms(AppliedUndef)
    assert positive.is_positive
    wb = Workbook()
    wb.exe(r"g_{\mu\nu} := FLRW(k, a)")
    assert sympy.simplify(wb.expr(r"G^{t}{}_{t}") + 3 * (k + scale.diff(t) ** 2) / scale**2) == 0


def test_reissner_nordstrom_is_traceless():
    g = spacetimes.reissner_nordstrom()
    assert g.ricci_scalar == 0
    assert sympy.simplify(g.einstein[0, 0] - Q**2 * (r**2 - 2 * M * r + Q**2) / r**6) == 0


def test_de_sitter_has_constant_curvature():
    assert spacetimes.de_sitter().ricci_scalar == 12 / L**2
    assert spacetimes.anti_de_sitter().ricci_scalar == -12 / L**2


def test_minkowski():
    assert spacetimes.minkowski().riemann.components == Array.zeros(4, 4, 4, 4)
    assert spacetimes.minkowski("spherical").ricci_scalar == 0
    assert spacetimes.minkowski(Symbol("spherical")).coordinates == (t, r, theta, phi)  # a name
    with pytest.raises(ValueError, match="unknown coordinates"):
        spacetimes.minkowski("cylindrical")


def test_parameters_in_any_units():
    G, c = symbols("G c")
    g = spacetimes.schwarzschild("M", G="G", c="c")
    assert g.kretschmann == 48 * G**2 * M**2 / (c**4 * r**6)
    assert spacetimes.schwarzschild(2).kretschmann == 48 * 4 / r**6


def test_every_spacetime_takes_G_and_c():
    G, c = symbols("G c", positive=True)
    for name in spacetimes.__all__:
        assert list(inspect.signature(getattr(spacetimes, name)).parameters)[-2:] == ["G", "c"]
    # in any units, Kerr-Newman is the geometric metric with M → GM/c², Q² → GQ²/c⁴, t → ct
    geometric = spacetimes.kerr_newman().matrix.subs({M: G * M / c**2, Q: sqrt(G) * Q / c**2})
    stretch = diag(c, 1, 1, 1)
    difference = spacetimes.kerr_newman(M, a, Q, G, c).matrix - stretch * geometric * stretch
    assert difference.applyfunc(sympy.simplify) == sympy.zeros(4, 4)
    assert spacetimes.kerr(M, a, G, c).matrix == spacetimes.kerr_newman(M, a, 0, G, c).matrix
    assert spacetimes.de_sitter(L, G, c).ricci_scalar == 12 / L**2
    assert spacetimes.de_sitter(L, G, c).matrix[0, 0] == -(1 - r**2 / L**2) * c**2
    assert spacetimes.anti_de_sitter(L, G, c).matrix[0, 0] == -(1 + r**2 / L**2) * c**2
    assert spacetimes.flrw(k, "a", G, c).matrix[0, 0] == -(c**2)
    assert spacetimes.minkowski("spherical", G, c).matrix[0, 0] == -(c**2)


def test_units_in_the_language():
    wb = Workbook()
    wb.exe(r"g_{\mu\nu} := Schwarzschild(M, G, c)")
    G, c = symbols("G c")
    assert sympy.simplify(wb.expr("g_{tt}") + (1 - 2 * G * M / (c**2 * r)) * c**2) == 0


def test_decimal_parameters_are_exact():
    g = spacetimes.schwarzschild(1.7)
    assert g.matrix[0, 0] == -(1 - Rational(17, 5) / r)
    assert g.ricci.components == Array.zeros(4, 4)
    assert g.kretschmann == 48 * Rational(17, 10) ** 2 / r**6


def test_decimal_components_are_exact():
    f = 1 - 3.4 / r
    g = Metric(diag(-f, 1 / f, r**2, r**2 * sin(theta) ** 2), (t, r, theta, phi))
    assert g.ricci.components == Array.zeros(4, 4)


def test_decimals_in_the_language_are_exact():
    wb = Workbook()
    wb.exe(r"g_{\mu\nu} := Schwarzschild(1.7)")
    assert wb.expr(r"R_{\mu\nu} = 0")
    wb = Workbook()
    wb.exe(r"ds^2 = -(1 - 3.4/r) dt^2 + \frac{dr^2}{1 - 3.4/r} + r^2 d\Omega^2")
    assert wb.expr(r"R_{\mu\nu} = 0")
    wb = Workbook()  # given by its inverse, which is inverted exactly
    wb.exe(r"""
        Coordinates := [v, r, \theta, \phi]
        g^{\mu\nu} := [[0, 0.3, 0, 0], [0.3, 1 - 3.4/r, 0, 0],
                       [0, 0, 1/r^2, 0], [0, 0, 0, 1/(r^2 \sin^2\theta)]]
    """)
    assert wb.expr(r"R_{\mu\nu} = 0")


@pytest.mark.slow
def test_kerr_is_ricci_flat():
    g = spacetimes.kerr()
    assert g.ricci.components == Array.zeros(4, 4)


def test_line_elements_round_trip():
    g = spacetimes.schwarzschild()
    rebuilt = metric_from_line_element(g.line_element, g.coordinates)
    assert (rebuilt - g.matrix).applyfunc(sympy.simplify) == sympy.zeros(4, 4)
    again = Metric.from_line_element(g.line_element, g.coordinates)
    assert again.ricci_scalar == 0


def test_off_diagonal_line_element():
    dt, dphi = differential(t), differential(phi)
    g = metric_from_line_element(-(dt**2) + 2 * a * dt * dphi + dphi**2, (t, phi))
    assert g == Matrix([[-1, a], [a, 1]])


def test_a_line_element_must_be_quadratic():
    with pytest.raises(ValueError, match="quadratic form"):
        metric_from_line_element(differential(t) ** 3, (t,))


def test_differential_symbols_typeset_well():
    assert sympy.latex(differential(theta)) == r"d\theta"


def test_geodesic_equations():
    equations = spacetimes.schwarzschild().geodesic_equations()
    tau = Symbol("tau")
    th, ph = Function("theta")(tau), Function("phi")(tau)
    theta_equation = equations[2]
    assert theta_equation.rhs == 0
    assert theta_equation.lhs.has(th.diff(tau, 2))
    assert theta_equation.lhs.has(ph.diff(tau))
    assert len(equations) == 4
