from itertools import product
from relativisticpy.algebras import Indices
from relativisticpy.diffgeom.tensor import Tensor
from relativisticpy.diffgeom.manifold import CoordIndices
from relativisticpy.diffgeom.metric import Metric
from relativisticpy.diffgeom.connection import LeviCivitaConnection
from relativisticpy.diffgeom.tensors.ricci import Ricci
from relativisticpy.symengine import SymbolArray, simplify, Rational, zeros, diff

class RicciScalar(Tensor):
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

        # Categorize keyword arguments
        for _, value in kwargs.items():
            if isinstance(value, SymbolArray):
                components = arg
            elif isinstance(value, Metric):
                metric = value
            elif isinstance(value, LeviCivitaConnection):
                connection = value

        # Always compute least expensive comutations first.
        if components is None:
            if components is not None:
                components = cls.components_from_connection(connection)
            elif metric is not None:
                components = cls.components_from_metric(metric)
            else:
                raise TypeError("Components or metric is required.")

        return cls(indices, components)

    @staticmethod
    def components_from_metric(metric: Metric):
        N = metric.dimention
        R = Ricci.components_from_metric(metric)
        ig = metric.uu_components
        A = float()
        for i, j in product(range(N), range(N)):
            A += ig[i, j] * R[i, j]
        return simplify(A)
    
    @staticmethod
    def components_from_connection(connection: LeviCivitaConnection):
        metric = connection.metric
        R = Ricci.components_from_connection(connection)
        ig = metric.uu_components
        A = float()
        for i, j in product(range(connection.dimention), range(connection.dimention)):
            A += ig[i, j] * R[i, j]
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
