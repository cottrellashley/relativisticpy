# Assuming Idx is imported from your module
from relativisticpy.core.indices import Idx
import pytest

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

@pytest.mark.skip(reason="Test not implemented.")
def test_iteration(): pass

@pytest.mark.skip(reason="Test not implemented.")
def test_len(): pass

@pytest.mark.skip(reason="Test not implemented.")
def test_repr(): pass

@pytest.mark.skip(reason="Test not implemented.")
def test_str(): pass

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
