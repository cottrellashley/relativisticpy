from relativisticpy.tensors.core.tensor import GrTensor
from relativisticpy.tensors.metric import Metric
from relativisticpy.shared.functions import christoffel


class Gamma(GrTensor):

    def __init__(self, metric : Metric, indices):
        self.metric = metric
        components = christoffel(self.metric.get_metric().components , self.metric.basis)
        GrTensor.__init__(self,
                            components  =   components,
                            indices     =   indices,
                            basis       =   self.metric.basis
                        )