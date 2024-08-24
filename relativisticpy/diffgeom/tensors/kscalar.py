from functools import singledispatchmethod
from itertools import product

from loguru import logger

from relativisticpy.algebras import Indices, EinsumArray
from relativisticpy.diffgeom.manifold import CoordIndices
from relativisticpy.diffgeom.tensor import Tensor
from relativisticpy.diffgeom.metric import Metric
from relativisticpy.diffgeom.connection import LeviCivitaConnection
from relativisticpy.symengine import SymbolArray, simplify
from relativisticpy.diffgeom.tensors.riemann import Riemann


class KScalar(Tensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @singledispatchmethod
    @classmethod
    def _new(cls, operand, indices):
        logger.debug(f"[KScalar] Handling init: {operand.__class__.__name__}")

    @_new.register
    @classmethod
    def _(cls, operand: Metric, indices: Indices) -> Tensor:
        components = cls.components_from_metric(operand)
        return cls(indices, components)

    @_new.register
    @classmethod
    def _(cls, operand: LeviCivitaConnection, indices: Indices) -> Tensor:
        components = cls.components_from_connection(operand)
        return cls(indices, components)

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
