"""
Core module for relativisticpy. 
The core module currently contains the core logic of algebra rules for the einstein summation convention of multi-indexed arrays.
"""

from relativisticpy.core.einsum_convention import einstein_convention
from relativisticpy.core.einsteinarray import EinsteinArray
from relativisticpy.core.indices import Idx, Indices
from relativisticpy.core.metric import Metric, MetricIndices
from relativisticpy.core.tensor_equality_types import TensorEqualityType
