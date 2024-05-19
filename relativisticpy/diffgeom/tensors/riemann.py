# Standard Library
from typing import Union
from itertools import product

# External Modules
from relativisticpy.algebras import Indices, Tensor
from relativisticpy.diffgeom.metric import Metric
from relativisticpy.symengine import SymbolArray, Rational, zeros, diff, simplify

# This Module
from relativisticpy.diffgeom.connection import LeviCivitaConnection

# TODO: Add mechanism which detects self-contractions and map any self contraction to relevant tensor i.e. Riemann^{a}_{b}_{a}_{c} == Ricci_{b}_{c}

class Riemann(Tensor):
    def __init__(self, indices: Indices, arg, basis: SymbolArray = None):
        super().__init__(symbols=arg, indices=indices, basis=basis)

    def from_components(components: SymbolArray) -> SymbolArray:
        return components

    def from_metric(self, metric: Metric, index_structure: int = None) -> SymbolArray:
        if index_structure == 0000:
            return Riemann.riemann0000_components_from_metric(metric)

        # Simplify object on before returning.
        return Riemann.riemann1000_components_from_metric(metric)

    @classmethod
    def riemann1000_components_from_metric(cls, metric: Metric):
        N = metric.dimention
        wrt = metric.basis
        C = LeviCivitaConnection.from_metric(metric)
        A = SymbolArray(zeros(N**4), (N, N, N, N))
        for i, j, k, p, d in product(range(N), range(N), range(N), range(N), range(N)):
            A[i, j, k, p] += Rational(1, N) * (
                diff(C[i, p, j], wrt[k]) - diff(C[i, k, j], wrt[p])
            ) + (C[i, k, d] * C[d, p, j] - C[i, p, d] * C[d, k, j])
        return simplify(A)

    @classmethod
    def riemann0000_components_from_metric(cls, metric: Metric):
        dim = metric.dimention
        metric_comps = metric.components
        rie_comps = Riemann.riemann1000_components_from_metric(metric)
        A = SymbolArray(zeros(dim**4), (dim, dim, dim, dim))
        for i, j, k, p, d in product(
            range(dim), range(dim), range(dim), range(dim), range(dim)
        ):
            A[i, j, k, p] += metric_comps[i, d] * rie_comps[d, j, k, p]
        return simplify(A)

    def from_connection(connection: LeviCivitaConnection) -> SymbolArray:
        # Setup Relevant quantities for computation
        N = connection.dimention
        wrt = connection.basis
        C = connection.components
        A = SymbolArray(zeros(N**4), (N, N, N, N))

        # Perform computation
        for i, j, k, p, d in product(range(N), range(N), range(N), range(N), range(N)):
            A[i, j, k, p] += Rational(1, N) * (
                diff(C[i, p, j], wrt[k]) - diff(C[i, k, j], wrt[p])
            ) + (C[i, k, d] * C[d, p, j] - C[i, p, d] * C[d, k, j])

        # Simplify object on before returning.
        return simplify(A)
