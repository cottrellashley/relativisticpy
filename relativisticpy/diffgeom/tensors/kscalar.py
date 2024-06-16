from itertools import product
from relativisticpy.diffgeom.manifold import CoordIndices
from relativisticpy.diffgeom.tensor import Tensor
from relativisticpy.diffgeom.metric import Metric
from relativisticpy.diffgeom.connection import LeviCivitaConnection
from relativisticpy.symengine import SymbolArray, simplify
from relativisticpy.diffgeom.tensors.riemann import Riemann

class KScalar(Tensor):
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

        if components is None:
            if metric is not None:
                components = cls.components_from_metric(metric)
            elif connection is not None:
                components = cls.components_from_connection(connection)
            else:
                raise TypeError("Components or metric is required.")

        return cls(indices, components)

    @staticmethod
    def components_from_metric(metric: Metric):
        N = metric.dimention
        R = Riemann.riemann0000_components_from_metric(metric)
        ig = metric.uu_components
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

    @staticmethod
    def components_from_connection(connection: LeviCivitaConnection) -> SymbolArray:
        raise NotImplementedError(
            "KScalar from connection has not been implemented yet."
        )
