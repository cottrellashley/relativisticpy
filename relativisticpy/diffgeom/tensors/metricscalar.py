from functools import singledispatchmethod
from itertools import product

from loguru import logger

from relativisticpy.algebras import Indices, EinsumArray
from relativisticpy.diffgeom.tensor import Tensor
from relativisticpy.diffgeom.metric import Metric
from relativisticpy.symengine import simplify


class MetricScalar(Tensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @singledispatchmethod
    @classmethod
    def _new(cls, operand, indices):
        logger.debug(f"[MetricScalar] Handling init: {operand.__class__.__name__}")

    @_new.register
    @classmethod
    def _(cls, operand: Metric, indices: Indices) -> Tensor:
        components = cls.components_from_metric(operand)
        return cls(indices, components)

    @staticmethod
    def components_from_metric(metric: Metric):
        dim = metric.dimention
        g = metric.components
        inv_g = metric.uu_components
        skeleton = float()
        for i, j in product(range(dim), range(dim)):
            skeleton += (
                    inv_g[i, j] * g[i, j]
            )
        return simplify(skeleton)
