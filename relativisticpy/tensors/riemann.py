from relativisticpy.deserializers.representations import IndicesRepresentationA
from relativisticpy.indices.indices import Indices
from relativisticpy.shared.functions import riemann1000, riemann0000, riemann1100, riemann1110, riemann1111
from relativisticpy.tensors.core.tensor import GrTensor
from relativisticpy.tensors.metric import Metric


class Riemann(GrTensor):
    
    def __init__(self, metric : Metric, indices : Indices):
        self.metric = metric

        get_components = lambda x : {
            '0000' : riemann0000,
            '1000' : riemann1000
        }.get(x)

        ind_structure = [0 if i.comp_type == 'covariant' else 1 for i in IndicesRepresentationA(indices, self.metric.basis).indices]
        if ind_structure not in [[0,0,0,0],[1,0,0,0]]:
            raise ValueError("Currectly do not support initializing indices for Riemann of this type.")
        components = get_components('0000' if ind_structure == [0,0,0,0] else '1000')(self.metric.get_metric().components , self.metric.basis)
        GrTensor.__init__(self,
                                components  =   components,
                                indices     =   indices,
                                basis       =   self.metric.basis
                            )