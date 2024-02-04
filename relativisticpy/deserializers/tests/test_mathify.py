from relativisticpy.symengine import diff, SymbolArray, Symbol, symbols, sin, cos
from relativisticpy.deserializers.tensors import tensor_from_string
from relativisticpy.deserializers.mathify import Mathify
import pytest

@pytest.fixture
def polar_coordinates_setup():
    r, t, theta, phi = Symbol('r'), Symbol('t'), Symbol('theta'), Symbol('phi')
    return t, r, theta, phi

@pytest.fixture
def cartesian_coordinates_setup():
    t, x, y, z = Symbol('t'), Symbol('x'), Symbol('y'), Symbol('z')
    return t, x, y, z

@pytest.fixture
def schild_setup():
    r, t, theta, phi, G = Symbol('r'), Symbol('t'), Symbol('theta'), Symbol('phi'), Symbol('G')
    array = SymbolArray([
    [
        -1+(G/r), 0, 0, 0
    ],
    [
        0, 1/(1-G/r), 0, 0
    ],
    [
        0, 0, r**2, 0
    ],
    [
        0, 0, 0, r**2*sin(theta)**2
    ]])
    return array

def test_mathify_diff(polar_coordinates_setup):
    t, r, theta, phi = polar_coordinates_setup
    assert Mathify('diff(r + t, t)') == diff(r + t, t)
    assert Mathify('diff(r**2 + t, t)') == diff(r**2 + t, t)
    assert Mathify('diff(sin(r)*t + cos(t), t)') == diff(sin(r)*t + cos(t), t)

def test_mathify_array_building(schild_setup):
    schilc_array = schild_setup
    assert Mathify('[[-1+(G/r), 0, 0, 0],[0, 1/(1-G/r), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2*sin(theta)**2]]') == schilc_array