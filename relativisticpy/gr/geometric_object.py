from relativisticpy.core import MultiIndexObject, Indices, Metric
from typing import Union

class GeometricObject(MultiIndexObject):
    """
        GrTensors can all be defined from either the Connection or the Metric Tensor.
        (Although the metric cannot be determined from a connection, the Christoffell connection can be determined from the
        condition of setting Covarient derivative of the metric to be zero.) The GR tensor inheriting this object will all
        be tensors which can be defined from the connection and/or the metric tensor. 
    """
    _get_comps = lambda args : None

    def __init__(self, indices: Indices, metric: Metric, comps = None, basis = None):
        # Gr Tensors are all associated to a metric or a connection => both are defining properties of a Manifold
        self.comps = GeometricObject._get_comps()
        self.metric = metric
        super().__init__(indices, comps, basis)

    def __setitem__(self, key, value):
        return super().__setitem__(key, value)
