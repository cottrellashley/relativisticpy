from relativisticpy.core import MultiIndexObject, Metric, MetricIndices, Indices, Idx
from relativisticpy.symengine import diff, SymbolArray, Symbol, symbols
from relativisticpy.deserializers import tensor_from_string
import pytest

def test_multi_index_object():
    assert -Idx('a', order=1, values=5, covariant=True) == Idx('a', order=1, values=5, covariant=False)