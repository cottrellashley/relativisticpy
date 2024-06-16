# Standard Library
from itertools import product

# External Modules
from relativisticpy.diffgeom.tensor import Tensor
from relativisticpy.diffgeom.manifold import CoordIndices
from relativisticpy.diffgeom import Metric
from relativisticpy.symengine import SymbolArray, Rational, zeros, diff, simplify

# This Module
from relativisticpy.diffgeom import LeviCivitaConnection

class EinsteinTensor(Tensor):
    # TODO: Implement the Einstein Tensor with new base classes.
    def __init__(self, indices: CoordIndices, arg):
        super().__init__(indices=indices, symbols=arg)

    def from_metric(self, metric: Metric) -> SymbolArray:
        Ricci = self.__ricci_components_from_metric(metric)
        g = metric.components
        ig = metric.inv.components
        N = metric.dimention
        A = SymbolArray(zeros(N**2), (N, N))
        for i, j, l, k in product(range(N), range(N), range(N), range(N)):
            A[i, j] += Ricci[i, j] - g[i, j] * ig[l, k] * Ricci[l, k]
        return simplify(A)

    def __ricci_components_from_connection(self, connection: LeviCivitaConnection) -> SymbolArray:
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
        Gamma = LeviCivitaConnection.from_metric(metric)
        A = SymbolArray(zeros(N**2), (N, N))
        for j, p, i, d in product(range(N), range(N), range(N), range(N)):
            A[j, p] += Rational(1, N) * (
                diff(Gamma[i, p, j], wrt[i]) - diff(Gamma[i, i, j], wrt[p])
            ) + (Gamma[i, i, d] * Gamma[d, p, j] - Gamma[i, p, d] * Gamma[d, i, j])
        return simplify(A)
