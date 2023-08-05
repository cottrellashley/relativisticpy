from core import MultiIndexObject, Indices
from typing import Union
from gr import Metric, Connection

class GrTensor(MultiIndexObject):
    """
    GrTensors can all be defined from either the Connection or the Metric Tensor.
    (Although the metric cannot be determined from a connection, the Christoffell connection can be determined from the
    condition of setting Covarient derivative of the metric to be zero.) The GR tensor inheriting this object will all
    be tensors which can be defined from the connection and/or the metric tensor. 
    """
    _get_comps = lambda args : None

    def __init__(self, indices: Indices, metric: Metric, comps = None, basis = None):
        # Gr Tensors are all associated to a metric or a connection => both are defining properties of a Manifold
        self.comps = GrTensor._get_comps()
        self.metric = metric
        super().__init__(indices, comps, basis)

    def __getitem__(self, idcs: Indices):
        # This should be implemented as follows:
        # 1. If the indices cov and contravarient indices structure matches the self.indices, then just return the current components
        # 2. If the indices does not match the self.indices, we must then perform a summation with the metric tensor in order to return the components
        #    which represent the indices structure given by the input parameter
        res = self.comps
        deltas = self.indices.covariance_delta(idcs) # { 'raise': [0,1], 'lower': [3] }
        for delta in deltas:
            res = getattr(self.metric, delta[0])(res, delta[1])
        return res[idcs.__index__()]

    def __setitem__(self, key, value):
        return super().__setitem__(key, value)
