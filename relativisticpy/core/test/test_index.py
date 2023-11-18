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

def test_equality():
    assert Idx('a') != Idx('b')
    assert Idx('a') != -Idx('a')
    assert Idx('a') == Idx('a')
    assert Idx('a', order=1, values=5, covariant=True) == Idx('a', order=1, values=5, covariant=True)
    assert Idx('a', order=1, values=5, covariant=True) == Idx('a', order=2, values=5, covariant=True)
    assert Idx('a', order=1, values=5, covariant=True) == Idx('a', order=2, values=6, covariant=True)

def test_is_contracted_with():
    assert Idx('a', covariant=True).is_contracted_with(Idx('a', covariant=False))
    assert not Idx('a', covariant=True).is_contracted_with(Idx('a', covariant=True))
    assert not Idx('b', covariant=True).is_contracted_with(Idx('a', covariant=False))

def test_iteration(symbols_setup):
    _, _, _, _, basis_2D, basis_4D = symbols_setup

    a_const = Idx('a', order=1, values=1, covariant=True)
    a_values = Idx('a', order=1, values=[1,2], covariant=True)
    a_running = Idx('a', order=1, covariant=True)

    a_values.basis = basis_2D
    a_const.basis = basis_2D
    a_running.basis = basis_2D

    assert [i for i in a_running] == [0, 1]
    assert [i for i in a_const] == [1]
    assert [i for i in a_values] == [1,2]

    # Swithc the basis to higher dimention => only running index should increases iter
    a_const.basis = basis_4D
    a_running.basis = basis_4D
    a_values.basis = basis_4D

    assert [i for i in a_running] == [0, 1, 2, 3]
    assert [i for i in a_const] == [1]
    assert [i for i in a_values] == [1,2]


def test_len(symbols_setup):
    _, _, _, _, basis_2D, basis_4D = symbols_setup

    a_const = Idx('a', order=1, values=1, covariant=True)
    a_values = Idx('a', order=1, values=[1,2], covariant=True)
    a_running = Idx('a', order=1, covariant=True)

    a_values.basis = basis_2D
    a_const.basis = basis_2D
    a_running.basis = basis_2D

    assert len(a_values) == 2
    assert len(a_const) == 2
    assert len(a_running) == 2

    # Swithc the basis to higher dimention => only running index should increases iter
    a_const.basis = basis_4D
    a_running.basis = basis_4D
    a_values.basis = basis_4D

    assert len(a_values) == 4
    assert len(a_const) == 4
    assert len(a_running) == 4

def test_str():
    a = Idx('a')
    b = Idx('b')
    c = Idx('c', covariant=False)
    mu = Idx('mu', covariant=False)
    nu = Idx('nu', order=12313, covariant=False)

    # Changes with how user wants to represent index object.
    # In the future he might want to use strings representations like latex or other representations.
    assert str(a) == '_{a}'
    assert str(b) == '_{b}'
    assert str(c) == '^{c}'
    assert str(nu) == '^{nu}'
    assert str(mu) == '^{mu}'

@pytest.mark.skip(reason="Test not implemented.")
def test_is_identical_to(): pass

@pytest.mark.skip(reason="Test not implemented.")
def test_is_summed_wrt_indices(): pass

@pytest.mark.skip(reason="Test not implemented.")
def test_get_summed_location(): pass

@pytest.mark.skip(reason="Test not implemented.")
def test_get_repeated_location(): pass

@pytest.mark.skip(reason="Test not implemented.")
def test_get_summed_locations(): pass # Repeated method <- remove

@pytest.mark.skip(reason="Test not implemented.")
def test_get_repeated_locations(): pass # Repeated method <- remove
