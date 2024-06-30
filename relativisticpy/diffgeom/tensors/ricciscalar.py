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
    def component_equations(cls):
        return [
            (SymbolArray, lambda arg: arg),
            (Metric, cls.components_from_metric),
            (LeviCivitaConnection, cls.components_from_connection),
        ]

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
