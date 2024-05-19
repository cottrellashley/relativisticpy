# Standard Library
from itertools import product

# External Modules
from relativisticpy.algebras import Indices, Tensor
from relativisticpy.diffgeom.metric import Metric
from relativisticpy.symengine import SymbolArray, Rational, zeros, diff, simplify

# This Module
from relativisticpy.diffgeom.connection import LeviCivitaConnection
from relativisticpy.diffgeom.geotensor import GrTensor
from relativisticpy.diffgeom.tensors.riemann import Riemann

class Ricci(Tensor):
    def __init__(self, indices: Indices, components: SymbolArray, metric: Metric):
        super().__init__(indices=indices, components=components, metric=metric)

    @property
    def args(self): return [self.indices, self.components, self.metric]

    @staticmethod
    def components_from_metric(metric: Metric) -> SymbolArray:
        N = metric.dimention
        wrt_array = metric.coordinate_patch.symbols
        lev_civ_comps = LeviCivitaConnection.componens_from_metric(metric)
        A = SymbolArray(zeros(N**2), (N, N))
        for j, p, i, d in product(range(N), range(N), range(N), range(N)):
            A[j, p] += Rational(1, N) * (
                diff(lev_civ_comps[i, p, j], wrt_array[i])
                - diff(lev_civ_comps[i, i, j], wrt_array[p])
            ) + (
                lev_civ_comps[i, i, d] * lev_civ_comps[d, p, j]
                - lev_civ_comps[i, p, d] * lev_civ_comps[d, i, j]
            )
        return simplify(A)

    @staticmethod
    def components_from_connection(connection: LeviCivitaConnection) -> SymbolArray:
        N = connection.dimention
        wrt = connection.metric.coordinate_patch.symbols
        Gamma = connection.components
        A = SymbolArray(zeros(N**2), (N, N))
        for j, p, i, d in product(range(N), range(N), range(N), range(N)):
            A[j, p] += Rational(1, N) * (
                diff(Gamma[i, p, j], wrt[i])
                - diff(Gamma[i, i, j], wrt[p])
            ) + (
                Gamma[i, i, d] * Gamma[d, p, j]
                - Gamma[i, p, d] * Gamma[d, i, j]
            )
        return simplify(A)

    def from_connection(self, connection: LeviCivitaConnection) -> SymbolArray:
        return 

    @classmethod
    def from_metric(cls, metric: Metric, indices: Indices) -> SymbolArray:
        comps = cls.components_from_metric(metric)
        if indices:
            return cls(indices=indices, components=comps, metric=metric)
