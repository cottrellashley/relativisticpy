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
    def component_equations(cls):
        return [
            (Metric, cls.components_from_metric),
            (LeviCivitaConnection, cls.components_from_connection),
        ]

    @staticmethod
    def components_from_metric(metric: Metric):
        dim = metric.dimention
        riemann = Riemann.llll_components_from_metric(metric)
        inv_metric = metric.uu_components
        skeleton = float()
        for i, j, k, p, d, n, s, t in product(
            range(dim),
            range(dim),
            range(dim),
            range(dim),
            range(dim),
            range(dim),
            range(dim),
            range(dim),
        ):
            skeleton += (
                inv_metric[i, j]
                * inv_metric[k, p]
                * inv_metric[d, n]
                * inv_metric[s, t]
                * riemann[i, k, d, s]
                * riemann[j, p, d, s]
            )
        return simplify(skeleton)

    @staticmethod
    def components_from_connection(connection: LeviCivitaConnection) -> SymbolArray:
        raise NotImplementedError(
            "KScalar from connection has not been implemented yet."
        )
