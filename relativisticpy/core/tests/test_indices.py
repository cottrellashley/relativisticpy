import pytest
from relativisticpy.core.indices import Idx, Indices

@pytest.fixture
def indices_setup():
    idx1 = Idx('a', order=1, values=5, covariant=True)
    idx2 = Idx('b', order=2, values=4, covariant=True)
    indices = Indices(idx1, idx2)
    return indices, idx1, idx2

def test_len(indices_setup):
    indices, _, _ = indices_setup
    assert len(indices) == 2

def test_eq(indices_setup):
    indices, idx1, idx2 = indices_setup
    other_indices = Indices(idx1, idx2)
    assert indices == other_indices

def test_mul(indices_setup):
    indices1, idx1, idx2 = indices_setup
    indices2 = Indices(idx1, -idx2)
    product_indices = indices1 * indices2
    assert isinstance(product_indices, Indices)

def test_iteration(indices_setup):
    indices, _, _ = indices_setup
    for idx in indices.indices:
        assert isinstance(idx, Idx)

def test_iteration(indices_setup):
    indices, _, _ = indices_setup
    for idx in indices:
        assert isinstance(idx, tuple)

def test_getitem(indices_setup):
    indices, idx1, _ = indices_setup
    assert indices[idx1] == [idx for idx in indices.indices if idx.symbol == idx1.symbol and idx.covariant == idx1.covariant]
