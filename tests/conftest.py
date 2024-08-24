import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# conftest.py
import pytest

from relativisticpy.algebras import Idx
from relativisticpy.diffgeom import CoordinatePatch, Patch, Manifold
from relativisticpy.symengine import SymbolArray, sin, symbols, cos, Function
from relativisticpy.diffgeom.metric import Metric, MetricIndices


# We provide defined Metric Tensor objects for testing of all metric dependent classes suchs as:
# Christoffel symbols, Riemann tensor, Ricci tensor, etc.


@pytest.fixture(scope="module")
def schwarzschild_metric():
    t, r, theta, phi = symbols('t r theta phi')
    G, M = symbols('G M')
    # Metric definition similar to before
    g_tt = -(1 - 2 * G * M / r)
    g_rr = 1 / (1 - 2 * G * M / r)
    g_theta_theta = r ** 2
    g_phi_phi = r ** 2 * sin(theta) ** 2

    components = SymbolArray([
        [g_tt, 0, 0, 0],
        [0, g_rr, 0, 0],
        [0, 0, g_theta_theta, 0],
        [0, 0, 0, g_phi_phi]
    ])

    coordinate_patch = CoordinatePatch(
        patch=Patch(
            name='schwarzschild',
            manifold=Manifold('blackhole', 4)
        ),
        symbols=SymbolArray([t, r, theta, phi])
    )

    indices = MetricIndices(Idx('a'), Idx('b'), coord_patch=coordinate_patch)
    return Metric(indices, components)


@pytest.fixture(scope="module")
def kerr_metric():
    # Define the symbols
    t, r, theta, phi, a = symbols('t r theta phi a')
    G, M = symbols('G M')

    # Define the metric components
    rho_squared = r**2 + a**2 * cos(theta)**2
    delta = r**2 - 2 * G * M * r + a**2

    g_tt = -(1 - 2 * G * M * r / rho_squared)
    g_tphi = -2 * a * G * M * r * sin(theta)**2 / rho_squared
    g_rr = rho_squared / delta
    g_thetatheta = rho_squared
    g_phiphi = (r**2 + a**2 + 2 * G * M * r * a**2 * sin(theta)**2 / rho_squared) * sin(theta)**2

    # Create the metric components
    components = SymbolArray([
        [g_tt, 0, 0, g_tphi],
        [0, g_rr, 0, 0],
        [0, 0, g_thetatheta, 0],
        [g_tphi, 0, 0, g_phiphi]
    ])

    # Define the coordinate patch
    coordinate_patch = CoordinatePatch(
        patch=Patch(
            name='kerr',
            manifold=Manifold('rotating_blackhole', 4)
        ),
        symbols=SymbolArray([t, r, theta, phi])
    )

    # Put it all together
    indices = MetricIndices(Idx('a'), Idx('b'), coord_patch=coordinate_patch)
    return Metric(indices, components)


@pytest.fixture(scope="module")
def frw_metric():
    t, r, theta, phi = symbols('t r theta phi')  # Define the symbols
    k = symbols('k')  # Spatial curvature constant
    a = Function('a')(t) # 'a' is the scale factor, assumed to be a function of time t

    g_tt = -1
    g_rr = a**2 / (1 - k * r**2)
    g_thetatheta = a**2 * r**2
    g_phiphi = a**2 * r**2 * sin(theta)**2

    components = SymbolArray([
        [g_tt, 0, 0, 0],
        [0, g_rr, 0, 0],
        [0, 0, g_thetatheta, 0],
        [0, 0, 0, g_phiphi]
    ])

    coordinate_patch = CoordinatePatch(
        patch=Patch(
            name='frw',
            manifold=Manifold('universe', 4)
        ),
        symbols=SymbolArray([t, r, theta, phi])
    )

    indices = MetricIndices(Idx('a'), Idx('b'), coord_patch=coordinate_patch)
    return Metric(indices, components)


@pytest.fixture(scope="module")
def charged_bh_metric():
    t, r, theta, phi = symbols('t r theta phi')
    G, M, Q = symbols('G M Q')  # 'Q' is the charge of the black hole

    # Metric definition similar to before
    g_tt = -(1 - 2 * G * M / r + G * Q**2 / r**2)
    g_rr = 1 / (1 - 2 * G * M / r + G * Q**2 / r**2)
    g_thetatheta = r**2
    g_phiphi = r**2 * sin(theta)**2

    components = SymbolArray([
        [g_tt, 0, 0, 0],
        [0, g_rr, 0, 0],
        [0, 0, g_thetatheta, 0],
        [0, 0, 0, g_phiphi]
    ])

    coordinate_patch = CoordinatePatch(
        patch=Patch(
            name='charged_bh',
            manifold=Manifold('charged_blackhole', 4)
        ),
        symbols=SymbolArray([t, r, theta, phi])
    )

    indices = MetricIndices(Idx('a'), Idx('b'), coord_patch=coordinate_patch)
    return Metric(indices, components)


@pytest.fixture(scope="module")
def rindler_metric():
    t, x, y, z = symbols('t x y z')

    g_tt = -x**2
    g_xx = 1
    g_yy = 1
    g_zz = 1

    components = SymbolArray([
        [g_tt, 0, 0, 0],
        [0, g_xx, 0, 0],
        [0, 0, g_yy, 0],
        [0, 0, 0, g_zz]
    ])

    coordinate_patch = CoordinatePatch(
        patch=Patch(
            name='rindler',
            manifold=Manifold('accelerating_frame', 4)
        ),
        symbols=SymbolArray([t, x, y, z])
    )

    indices = MetricIndices(Idx('a'), Idx('b'), coord_patch=coordinate_patch)
    return Metric(indices, components)


@pytest.fixture(scope="module")
def sphere_metric():
    theta, phi = symbols('theta phi')  # Angular coordinates on the sphere

    # The radius 'R' could be a symbol if it varies, or a numeric value
    R = symbols('R')

    g_thetatheta = R**2
    g_phiphi = R**2 * sin(theta)**2

    components = SymbolArray([
        [g_thetatheta, 0],
        [0, g_phiphi]
    ])

    coordinate_patch = CoordinatePatch(
        patch=Patch(
            name='sphere',
            manifold=Manifold('2d_sphere', 2)
        ),
        symbols=SymbolArray([theta, phi])
    )

    indices = MetricIndices(Idx('a'), Idx('b'), coord_patch=coordinate_patch)
    return Metric(indices, components)


@pytest.fixture(scope="module")
def saddle_metric():
    x, y = symbols('x y')  # Rectangular coordinates on the hyperbolic plane

    # The constant 'a' affects the curvature, a^2 is often set to 1
    a = symbols('a')

    g_xx = a**2 / (1 + x**2 + y**2)**2
    g_yy = a**2 / (1 + x**2 + y**2)**2

    components = SymbolArray([
        [g_xx, 0],
        [0, g_yy]
    ])

    coordinate_patch = CoordinatePatch(
        patch=Patch(
            name='saddle',
            manifold=Manifold('hyperbolic_plane', 2)
        ),
        symbols=SymbolArray([x, y])
    )

    indices = MetricIndices(Idx('a'), Idx('b'), coord_patch=coordinate_patch)
    return Metric(indices, components)


@pytest.fixture(scope="module")
def flat_surface_metric():
    x, y = symbols('x y')  # Cartesian coordinates on the flat surface

    g_xx = 1
    g_yy = 1

    components = SymbolArray([
        [g_xx, 0],
        [0, g_yy]
    ])

    coordinate_patch = CoordinatePatch(
        patch=Patch(
            name='flat_surface',
            manifold=Manifold('euclidean_plane', 2)
        ),
        symbols=SymbolArray([x, y])
    )

    indices = MetricIndices(Idx('a'), Idx('b'), coord_patch=coordinate_patch)
    return Metric(indices, components)


