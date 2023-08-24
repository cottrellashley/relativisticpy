"""
=======
RelativisticPy
=======
Python library for Tensor Manipulation in General Relativity.
"""

from sympy import NDimArray, Symbol, MutableDenseNDimArray
from relativisticpy.core.indices import Indices, Idx
from relativisticpy.core.multi_index_obj import MultiIndexObject
from relativisticpy.core.metric import Metric, MetricIndices
from relativisticpy.gr.derivative import Derivative
from itertools import product, combinations