from itertools import product
from relativisticpy.core import Metric, Indices, einstein_convention
from relativisticpy.gr.connection import Connection
from relativisticpy.gr.tensors.geometric import GeometricObject
from relativisticpy.symengine import SymbolArray, simplify


@einstein_convention
class MetricScalar(GeometricObject):
    def __init__(self, metric: Metric, basis: SymbolArray):
        super().__init__(symbols=metric, indices=Indices(), basis=basis)

    def from_metric(self, metric: Metric):
        N = metric.dimention
        g = metric.components
        ig = metric.inv.components
        A = float()
        for i, j in product( range(N), range(N) ):
            A += (
                ig[i, j] * g[i, j]
            )
        return simplify(A)
