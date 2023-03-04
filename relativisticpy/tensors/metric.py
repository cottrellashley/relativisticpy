import sympy as smp

from relativisticpy.indices.indices import Indices
from relativisticpy.base_tensor.gr_tensor import GrTensor
from relativisticpy.base_tensor.context import TensorContext

# The metric tensor with lower indices, such as g_{ab}, is called the "covariant metric tensor" or simply the "metric tensor".
# The metric tensor with raised indices, such as g^{ab}, is called the "contravariant metric tensor" or the "inverse metric tensor".
# In general, the covariant and contravariant metric tensors are related by matrix inversion. 
# That is, if g_{ab} is the covariant metric tensor, then g^{ab} is the contravariant metric tensor, and the two are related by:
# g^{ab} = (g_{cd})^{-1}
# where (g_{cd})^{-1} is the inverse of the matrix of g_{cd}.
class Metric(GrTensor):

    def __init__(self, comp, ind, base, signature = None):
        GrTensor.__init__(self, 
                           components = comp,
                           indices = ind,
                           basis = base,
                           context=TensorContext(is_metric_tensor=True))

        expected_shape = (self.dimention,) * 2
        if self.shape != expected_shape:
            raise ValueError(f"Invalid metric: The shape should be {expected_shape}.")

        expected_ranks = [(0,2), (2,0)]
        if self.rank not in expected_ranks:
            raise ValueError(f"Invalid metric: The rank should be of (2,0) or (0,2).")

        self.signature = signature

    def get_metric(self):
        if self.rank == (0, 2):
            comp = self.components
            ind = self.indices
        else:
            comp = smp.MutableDenseNDimArray(self.components.tomatrix().inv())
            ind = Indices(tuple([-j for j in self.indices.indices]), self.basis)

        return Metric(comp, ind, self.basis, self.signature)
        
    def get_inverse(self):
        if self.rank == (2, 0):
            comp = self.components
            ind = self.indices
        else:
            comp = smp.MutableDenseNDimArray(self.components.tomatrix().inv())
            ind = Indices(tuple([-j for j in self.indices.indices]), self.basis)

        return Metric(comp, ind, self.basis, self.signature)

class MetricWork(GrTensor):

    def __init__(self, metric : Metric, indices):
        self.metric = metric
        self.comp = self.metric.get_metric() if indices.count('_') == 2 else self.metric.get_inverse()
        GrTensor.__init__(self ,
                        components=self.comp.components,
                        indices= indices,
                        basis=self.metric.basis
        )