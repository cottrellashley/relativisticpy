from relativisticpy.core import MultiIndexObject, Indices, Metric
from typing import Union
from abc import ABC, abstractmethod
from relativisticpy.providers import SymbolArray

class GeometricObject(MultiIndexObject):
    """
        GrTensors can all be defined from either the Connection or the Metric Tensor.
        (Although the metric cannot be determined from a connection, the Christoffell connection can be determined from the
        condition of setting Covarient derivative of the metric to be zero.) The GR tensor inheriting this object will all
        be tensors which can be defined from the connection and/or the metric tensor. 
    """

    def __init__(self, indices: Indices, metric: Metric, components = None, basis = None):
        # Gr Tensors are all associated to a metric or a connection => both are defining properties of a Manifold
        self.components = GeometricObject.compute_components()
        self.metric = metric
        super().__init__(indices, components, basis)

    def compute_components(self) -> SymbolArray: return NotImplementedError()