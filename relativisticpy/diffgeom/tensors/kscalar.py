from itertools import product
from relativisticpy.algebras import Indices, Tensor
from relativisticpy.diffgeom.metric import Metric
from relativisticpy.diffgeom.connection import LeviCivitaConnection
from relativisticpy.symengine import SymbolArray, simplify

from relativisticpy.diffgeom.tensors.riemann import Riemann

class KScalar(Tensor):
    def __init__(self, metric: Metric, basis: SymbolArray):
        super().__init__(symbols=metric, indices=Indices(), basis=basis)

    def from_metric(self, metric: Metric):
        N = metric.dimention
        R = Riemann.riemann0000_components_from_metric(metric)
        ig = metric.inv.components
        A = float()
        for i, j, k, p, d, n, s, t in product(
            range(N),
            range(N),
            range(N),
            range(N),
            range(N),
            range(N),
            range(N),
            range(N),
        ):
            A += (
                ig[i, j]
                * ig[k, p]
                * ig[d, n]
                * ig[s, t]
                * R[i, k, d, s]
                * R[j, p, d, s]
            )
        return simplify(A)

    def from_connection(self, connection: LeviCivitaConnection) -> SymbolArray:
        raise NotImplementedError(
            "KScalar from connection has not been implemented yet."
        )
