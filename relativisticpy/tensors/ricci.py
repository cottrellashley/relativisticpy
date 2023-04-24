
from relativisticpy.shared.functions import ricci
from relativisticpy.tensors.core.tensor import GrTensor
from relativisticpy.tensors.metric import Metric


class Ricci(GrTensor):
    
    def __init__(self, metric : Metric, indices):
        self.metric = metric
        components = ricci(self.metric.get_metric().components , self.metric.basis)
        GrTensor.__init__(self,
                            components  =   components,
                            indices     =   indices,
                            basis       =   self.metric.basis
                        )