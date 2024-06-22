from itertools import product
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
    def from_equation(cls, indices: CoordIndices, *args, **kwargs) -> 'RicciScalar':
        "Dynamic constructor for the inheriting classes."
        components = None
        metric = None
        connection = None

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
                components = value
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
        dim = metric.dimention
        ric = Ricci.components_from_metric(metric)
        inv_metric = metric.uu_components
        skeleton = float()
        for i, j in product(range(dim), range(dim)):
            skeleton += inv_metric[i, j] * ric[i, j]
        return simplify(skeleton)

    @staticmethod
    def components_from_connection(connection: LeviCivitaConnection):
        metric = connection.metric
        ric = Ricci.components_from_connection(connection)
        ig = metric.uu_components
        skeleton = float()
        for i, j in product(range(connection.dimention), range(connection.dimention)):
            skeleton += ig[i, j] * ric[i, j]
        return simplify(skeleton)
