from itertools import product
from relativisticpy.core import Metric, Indices, einstein_convention
from relativisticpy.gr.connection import Connection
from relativisticpy.gr.tensors.geometric import GeometricObject
from relativisticpy.symengine import SymbolArray, simplify

from relativisticpy.gr.tensors.riemann import Riemann

@einstein_convention
class KScalar(GeometricObject):
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

    def from_connection(self, connection: Connection) -> SymbolArray:
        raise NotImplementedError(
            "KScalar from connection has not been implemented yet."
        )
