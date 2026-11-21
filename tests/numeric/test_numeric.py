"""Numbers from symbols: NumPy functions and geodesics integrated with SciPy."""

import pytest
from sympy import Symbol, diag, sin, symbols

numpy = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from relativisticpy import Metric, Tensor, spacetimes  # noqa: E402
from relativisticpy.numeric import evaluate, four_velocity, geodesic, lambdify  # noqa: E402

SCHWARZSCHILD = spacetimes.schwarzschild()


def test_lambdify_a_scalar_is_vectorised():
    kretschmann = lambdify(SCHWARZSCHILD.kretschmann, ["r"], M=1)
    radii = numpy.array([2.0, 3.0, 10.0])
    assert numpy.allclose(kretschmann(radii), 48 / radii**6)


def test_lambdify_a_constant_still_broadcasts():
    constant = lambdify(SCHWARZSCHILD.ricci_scalar, ["r"], M=1)
    assert constant(numpy.array([2.0, 3.0])).shape == (2,)


def test_lambdify_a_tensor():
    metric = lambdify(SCHWARZSCHILD, ["t", "r", "theta", "phi"], M=1)
    values = metric(0.0, 4.0, numpy.pi / 2, 0.0)
    assert values.shape == (4, 4)
    assert numpy.isclose(values[0, 0], -(1 - 2 / 4))
    assert numpy.isclose(values[3, 3], 16.0)


def test_lambdify_a_tensor_over_a_grid():
    metric = lambdify(SCHWARZSCHILD, ["t", "r", "theta", "phi"], M=1)
    radii = numpy.linspace(3, 10, 7)
    assert metric(0.0, radii, numpy.pi / 2, 0.0).shape == (4, 4, 7)


def test_evaluate():
    assert evaluate(SCHWARZSCHILD.kretschmann, r=2, M=1) == pytest.approx(0.75)
    gamma = evaluate(SCHWARZSCHILD.christoffel, r=10, theta=numpy.pi / 2, M=1)
    assert gamma[1, 0, 0] == pytest.approx((1 / 100) * (1 - 2 / 10))
    assert evaluate(Tensor([1, 2], "_a")).tolist() == [1, 2]


def test_evaluate_names_what_is_missing():
    with pytest.raises(ValueError, match="give a value for theta"):
        evaluate(SCHWARZSCHILD.christoffel, r=10, M=1)


def test_lambdify_and_geodesic_name_what_is_missing():
    with pytest.raises(ValueError, match="give a value for M"):
        lambdify(SCHWARZSCHILD.kretschmann, ["r"])
    with pytest.raises(ValueError, match="give a value for M"):
        lambdify(SCHWARZSCHILD.christoffel, SCHWARZSCHILD.coordinates)
    with pytest.raises(ValueError, match="give a value for M"):
        geodesic(SCHWARZSCHILD, [0, 10, numpy.pi / 2, 0], [1.2, 0, 0, 0.01], (0, 1))


def test_an_undefined_function_is_named():
    universe = spacetimes.flrw()  # its scale factor a(t) has no value
    missing = "give a value for k, and replace the undefined function a with an expression"
    with pytest.raises(ValueError, match=missing):
        evaluate(universe.einstein, t=1, r=0.5, theta=1)
    with pytest.raises(ValueError, match="replace the undefined function a with an expression"):
        lambdify(universe.christoffel, universe.coordinates, k=0)
    t = Symbol("t")
    assert evaluate(spacetimes.flrw(0, t**2).ricci_scalar, t=1) == pytest.approx(36)


def test_values_are_matched_by_name():
    position = [0.0, 10.0, numpy.pi / 2, 0.0]
    by_name = four_velocity(SCHWARZSCHILD, position, [0, 0, 0.01], M=1)
    by_symbol = four_velocity(SCHWARZSCHILD, position, [0, 0, 0.01], parameters={Symbol("M"): 1})
    assert numpy.allclose(by_name, by_symbol)
    M, r = symbols("M r", positive=True)  # as declared in the language: Positive := [M, r]
    f = 1 - 2 * M / r
    g = Metric(diag(-f, 1 / f, r**2, r**2 * sin(Symbol("theta")) ** 2), ["t", r, "theta", "phi"])
    assert evaluate(g.kretschmann, r=2, M=1) == pytest.approx(0.75)
    assert lambdify(g.kretschmann, ["r"], M=1)(numpy.array([2.0])) == pytest.approx([0.75])
    assert numpy.allclose(four_velocity(g, position, [0, 0, 0.01], M=1), by_name)


def test_four_velocity_is_normalised():
    position = [0.0, 10.0, numpy.pi / 2, 0.0]
    u = four_velocity(SCHWARZSCHILD, position, [0.1, 0.0, 0.01], M=1)
    g = evaluate(SCHWARZSCHILD, dict(zip(SCHWARZSCHILD.coordinates, position, strict=True)), M=1)
    assert u @ g @ u == pytest.approx(-1)
    assert u[0] > 0
    light = four_velocity(SCHWARZSCHILD, position, [1.0, 0.0, 0.0], kind="null", M=1)
    assert light @ g @ light == pytest.approx(0, abs=1e-12)


def test_four_velocity_rejects_the_impossible():
    with pytest.raises(ValueError, match="no real four-velocity"):
        four_velocity(SCHWARZSCHILD, [0, 1.5, numpy.pi / 2, 0], [0, 0, 0], M=1)


def test_a_circular_orbit_stays_circular():
    radius = 10.0
    position = [0.0, radius, numpy.pi / 2, 0.0]
    angular = numpy.sqrt(1 / (radius**2 * (radius - 3)))
    orbit = geodesic(
        SCHWARZSCHILD,
        position,
        four_velocity(SCHWARZSCHILD, position, [0, 0, angular], M=1),
        (0, 1000),
        M=1,
    )
    assert numpy.allclose(orbit["r"], radius, atol=1e-6)
    assert orbit["phi"][-1] > 2 * numpy.pi  # it really orbits
    assert orbit.x.shape == (4, 1000)


def test_the_norm_is_conserved_along_an_eccentric_orbit():
    position = [0.0, 20.0, numpy.pi / 2, 0.0]
    u0 = four_velocity(SCHWARZSCHILD, position, [0.0, 0.0, 0.011], M=1)  # below circular: bound
    orbit = geodesic(SCHWARZSCHILD, position, u0, (0, 500), M=1, samples=50)
    metric = lambdify(SCHWARZSCHILD, SCHWARZSCHILD.coordinates, M=1)
    norms = [orbit.u[:, i] @ metric(*orbit.x[:, i]) @ orbit.u[:, i] for i in range(50)]
    assert numpy.allclose(norms, -1, atol=1e-8)
    assert orbit["r"].min() < 20 - 1  # it is eccentric
    assert orbit["r"].max() == pytest.approx(20, abs=1e-6)  # starting at apoapsis


def test_radial_infall_from_rest_speeds_up():
    position = [0.0, 10.0, numpy.pi / 2, 0.0]
    fall = geodesic(
        SCHWARZSCHILD,
        position,
        four_velocity(SCHWARZSCHILD, position, [0, 0, 0], M=1),
        (0, 20),
        M=1,
    )
    assert fall["r"][-1] < 10
    assert numpy.all(numpy.diff(fall.u[1]) < 0)  # dr/dτ ever more negative


def test_a_geodesic_can_stop_where_a_condition_is_met():
    position = [0.0, 10.0, numpy.pi / 2, 0.0]
    u0 = four_velocity(SCHWARZSCHILD, position, [0, 0, 0], M=1)
    fall = geodesic(SCHWARZSCHILD, position, u0, (0, 100), M=1, stop=lambda x, u: x[1] - 3)
    assert fall.stopped
    assert fall["r"][-1] == pytest.approx(3)
    assert numpy.all(fall["r"][:-1] > 3)
    assert not geodesic(
        SCHWARZSCHILD, position, u0, (0, 1), M=1, stop=lambda x, u: x[1] - 3
    ).stopped


def _photon(b, r0=1000.0):
    """Light from radius r0, aimed inwards with impact parameter b (M = 1)."""
    position = [0.0, r0, numpy.pi / 2, 0.0]
    inward = -numpy.sqrt(1 - (1 - 2 / r0) * b**2 / r0**2)
    u0 = four_velocity(SCHWARZSCHILD, position, [inward, 0, b / r0**2], kind="null", M=1)
    return geodesic(SCHWARZSCHILD, position, u0, (0, 2.5 * r0), M=1, stop=lambda x, u: x[1] - 2.5)


def test_light_closer_than_the_photon_sphere_is_captured():
    critical = 3 * numpy.sqrt(3)  # 3√3 M
    assert _photon(0.99 * critical).stopped
    escaped = _photon(1.01 * critical)
    assert not escaped.stopped
    assert escaped["r"].min() > 3  # it passed outside the photon sphere


def test_integration_failure_is_reported(monkeypatch):
    import scipy.integrate

    class Failed:
        success, message = False, "step size too small"

    monkeypatch.setattr(scipy.integrate, "solve_ivp", lambda *args, **kwargs: Failed())
    position = [0.0, 3.0, numpy.pi / 2, 0.0]
    u0 = four_velocity(SCHWARZSCHILD, position, [-1.0, 0, 0], M=1)
    with pytest.raises(RuntimeError, match="geodesic integration failed: step size too small"):
        geodesic(SCHWARZSCHILD, position, u0, (0, 50), M=1)
