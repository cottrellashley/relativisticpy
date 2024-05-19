from itertools import product
from relativisticpy.algebras import Indices, Tensor
from relativisticpy.diffgeom.metric import Metric
from relativisticpy.diffgeom.connection import LeviCivitaConnection
from relativisticpy.symengine import SymbolArray, simplify


class MetricScalar(Tensor):
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
