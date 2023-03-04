from relativisticpy.base_tensor.gr_tensor import GrTensor
from relativisticpy.shared.computations import GrComputations
from relativisticpy.tensors.metric import Metric

class Ricci(GrTensor):
    
    def __init__(self, metric : Metric, indices):
        self.metric = metric
        components = GrComputations(self.metric.get_metric().components , self.metric.basis)
        GrTensor.__init__(self,
                            components  =   components.Ricci(),
                            indices     =   indices,
                            basis       =   self.metric.basis
                        )