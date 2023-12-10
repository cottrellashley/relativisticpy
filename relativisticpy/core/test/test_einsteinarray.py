# Assuming Idx is imported from your module
from relativisticpy.core.indices import Idx, Indices
from relativisticpy.symengine import Symbol, SymbolArray
import pytest

@pytest.fixture
def symbols_setup():
    r = Symbol("r")
    t = Symbol("t")
    theta = Symbol("theta")
    phi = Symbol("phi")
    basis_2D = SymbolArray([t, r])
    basis_4D = SymbolArray([t, r, theta, phi])
    return t, r, theta, phi, basis_2D, basis_4D

def test_negation():
    assert -Idx('a', order=1, values=5, covariant=True) == Idx('a', order=1, values=5, covariant=False)