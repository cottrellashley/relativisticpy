from relativisticpy.base_tensor.gr_tensor import GrTensor
from relativisticpy.indices.indices import Indices
from relativisticpy.tensors.metric import Metric
from relativisticpy.shared.computations import GrComputations
from relativisticpy.indices.representations import IndicesRepresentationA


class Riemann(GrTensor):
    
    def __init__(self, metric : Metric, indices : Indices):
        self.metric = metric
        calc_comp = GrComputations(self.metric.get_metric().components , self.metric.basis)
        ind_structure = [0 if i.comp_type == 'covariant' else 1 for i in IndicesRepresentationA(indices, self.metric.basis).indices]
        if ind_structure not in [[0,0,0,0],[1,0,0,0]]:
            raise ValueError("Currectly do not support initializing indices for Riemann of this type.")
        components = calc_comp.Riemann0000() if ind_structure == [0,0,0,0] else calc_comp.Riemann1000()
        GrTensor.__init__(self,
                                components  =   components,
                                indices     =   indices,
                                basis       =   self.metric.basis
                            )