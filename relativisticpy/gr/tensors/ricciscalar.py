from itertools import product
from relativisticpy.core import Metric, Indices, einstein_convention
from relativisticpy.gr.connection import Connection
from relativisticpy.gr.tensors.geometric import GeometricObject
from relativisticpy.gr.tensors.ricci import Ricci
from relativisticpy.symengine import SymbolArray, simplify, Rational, zeros, diff


@einstein_convention
class RicciScalar(GeometricObject):
    def __init__(self, metric: Metric, basis: SymbolArray):
        super().__init__(symbols=metric, indices=Indices(), basis=basis)

    def from_metric(self, metric: Metric):
        N = metric.dimention
        R = self.__ricci_components_from_metric(metric)
        ig = metric.inv.components
        A = float()
        for i, j in product(range(N), range(N)):
            A += ig[i, j] * R[i, j]
        return simplify(A)

    def __ricci_components_from_metric(self, metric: Metric) -> SymbolArray:
        N = metric.dimention
        wrt = metric.basis
        Gamma = Connection.from_metric(metric)
        A = SymbolArray(zeros(N**2), (N, N))
        for j, p, i, d in product(range(N), range(N), range(N), range(N)):
            A[j, p] += Rational(1, N) * (
                diff(Gamma[i, p, j], wrt[i]) - diff(Gamma[i, i, j], wrt[p])
            ) + (Gamma[i, i, d] * Gamma[d, p, j] - Gamma[i, p, d] * Gamma[d, i, j])
        return simplify(A)
