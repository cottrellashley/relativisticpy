from relativisticpy.core import EinsteinArray, Metric, MetricIndices, Indices, Idx
from relativisticpy.symengine import SymbolArray, Symbol
import pytest

def test_indices_from_string_init():
    # Object to test
    generated_indices = Indices.from_string('_{a}_{b}', '[t, r]')

    indices = Indices(Idx('a'), Idx('b'))
    t = Symbol('t')
    r = Symbol('r')
    theta = Symbol('theta')
    basis = SymbolArray([t, r])
    indices.basis = basis

    assert generated_indices == indices
    assert generated_indices.anyrunnig == indices.anyrunnig
    assert generated_indices.indices == indices.indices
    assert generated_indices.basis == indices.basis
    assert generated_indices.dimention == indices.dimention
    assert str(generated_indices) == str(indices)
    assert generated_indices.rank == indices.rank
    assert generated_indices.shape == indices.shape
    assert generated_indices.scalar == indices.scalar
    assert [i for i in generated_indices] == [i for i in indices]

    # Object to test
    generated_indices = Indices.from_string('^{a}_{b}^{c}_{d}_{e}', '[t, r, theta]')

    indices = Indices(-Idx('a'), Idx('b'), -Idx('c'), Idx('d'), Idx('e'))
    basis = SymbolArray([t, r, theta])
    indices.basis = basis

    assert generated_indices == indices
    assert generated_indices.anyrunnig == indices.anyrunnig
    assert generated_indices.indices == indices.indices
    assert generated_indices.basis == indices.basis
    assert generated_indices.dimention == indices.dimention
    assert str(generated_indices) == str(indices)
    assert generated_indices.rank == indices.rank
    assert generated_indices.shape == indices.shape
    assert generated_indices.scalar == indices.scalar
    assert [i for i in generated_indices] == [i for i in indices]

    # Object to test
    generated_indices = Indices.from_string('^{a:1}_{b:0}^{c}_{d:0}_{e}', '[t, r, theta]')

    indices = Indices(-Idx('a', values=1), Idx('b', values=0), -Idx('c'), Idx('d', values=0), Idx('e'))
    basis = SymbolArray([t, r, theta])
    indices.basis = basis

    assert generated_indices == indices
    assert generated_indices.anyrunnig == indices.anyrunnig
    assert generated_indices.indices == indices.indices
    assert generated_indices.basis == indices.basis
    assert generated_indices.dimention == indices.dimention
    assert str(generated_indices) == str(indices)
    assert generated_indices.rank == indices.rank
    assert generated_indices.shape == indices.shape
    assert generated_indices.scalar == indices.scalar
    assert [i for i in generated_indices] == [i for i in indices]

    # Object to test
    generated_indices = Indices.from_string('^{a:1}_{b:0}^{c}_{d:0}_{e}', '[t, r, theta]')

    indices = Indices(-Idx('a', values=9), Idx('b', values=1), -Idx('c'), Idx('d', values=0), Idx('e'))
    basis = SymbolArray([t, r, theta])
    indices.basis = basis

    assert generated_indices == indices
    assert generated_indices.anyrunnig == indices.anyrunnig
    assert generated_indices.indices == indices.indices
    assert generated_indices.basis == indices.basis
    assert generated_indices.dimention == indices.dimention
    assert str(generated_indices) == str(indices)
    assert generated_indices.rank == indices.rank
    assert generated_indices.shape == indices.shape
    assert generated_indices.scalar == indices.scalar
    assert [i for i in generated_indices] != [i for i in indices]