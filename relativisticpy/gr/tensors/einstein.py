# Standard Library
from itertools import product

# External Modules
from relativisticpy.core import Indices, Metric, einstein_convention
from relativisticpy.symengine import SymbolArray, Rational, zeros, diff, simplify

# This Module
from relativisticpy.gr.connection import Connection
from relativisticpy.gr.tensors.geometric import GeometricObject


@einstein_convention
class EinsteinTensor(GeometricObject):
    def __init__(self, indices: Indices, arg, basis: SymbolArray = None):
        super().__init__(indices=indices, symbols=arg, basis=basis)

    def from_metric(self, metric: Metric) -> SymbolArray:
        Ricci = self.__ricci_components_from_metric(metric)
        g = metric.components
        ig = metric.inv.components
        N = metric.dimention
        A = SymbolArray(zeros(N**2), (N, N))
        for i, j, l, k in product(range(N), range(N), range(N), range(N)):
            A[i, j] += Ricci[i, j] - g[i, j] * ig[l, k] * Ricci[l, k]
        return simplify(A)

    def __ricci_components_from_connection(self, connection: Connection) -> SymbolArray:
        N = connection.dimention
        wrt = connection.basis
        Gamma = connection.components
        A = SymbolArray(zeros(N**2), (N, N))
        for j, p, i, d in product(range(N), range(N), range(N), range(N)):
            A[j, p] += Rational(1, N) * (
                diff(Gamma[i, p, j], wrt[i]) - diff(Gamma[i, i, j], wrt[p])
            ) + (Gamma[i, i, d] * Gamma[d, p, j] - Gamma[i, p, d] * Gamma[d, i, j])
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
