# Standard Library
from itertools import product

# External Modules
from relativisticpy.diffgeom.manifold import CoordIndices
from relativisticpy.diffgeom.tensor import Tensor
from relativisticpy.diffgeom.metric import Metric
from relativisticpy.symengine import SymbolArray, Rational, zeros, diff, simplify

# This Module
from relativisticpy.diffgeom.connection import LeviCivitaConnection
from relativisticpy.diffgeom.tensors.riemann import Riemann

class Ricci(Tensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_equation(cls, indices: CoordIndices, *args, **kwargs) -> 'LeviCivitaConnection':
        "Dynamic constructor for the inheriting classes."
        components = None
        metric = None

        # Categorize positional arguments
        for arg in args:
            if isinstance(arg, SymbolArray):
                components = arg
            elif isinstance(arg, Metric):
                metric = arg
            elif isinstance(arg, LeviCivitaConnection):
                connection = arg
            elif isinstance(arg, Riemann):
                riemann = arg

        # Categorize keyword arguments
        for _, value in kwargs.items():
            if isinstance(value, SymbolArray):
                components = arg
            elif isinstance(value, Metric):
                metric = value
            elif isinstance(value, LeviCivitaConnection):
                connection = value
            elif isinstance(value, Riemann):
                riemann = value

        # Always compute least expensive comutations first.
        if components is None:
            if components is not None:
                components = cls.components_from_connection(connection)
            elif metric is not None:
                components = cls.components_from_metric(metric)
            else:
                raise TypeError("Components or metric is required.")

        return cls(indices, components)

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

    @staticmethod
    def components_from_riemann(riemann: Riemann) -> SymbolArray:
        N = riemann.dimention
        wrt = riemann.basis
        R = riemann.components
        A = SymbolArray(zeros(N**2), (N, N))
        for j, p, i, d in product(range(N), range(N), range(N), range(N)):
            A[j, p] += Rational(1, N) * (
                diff(R[i, p, j, i], wrt[i])
                - diff(R[i, i, j, p], wrt[p])
            ) + (
                R[i, i, d, p] * R[d, p, j, i]
                - R[i, p, j, d] * R[d, i, i, j]
            )
        return simplify(A)