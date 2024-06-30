from itertools import product
from relativisticpy.diffgeom.tensor import Tensor
from relativisticpy.diffgeom.metric import Metric
from relativisticpy.symengine import simplify


class MetricScalar(Tensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def component_equations(cls):
        return [
            (Metric, cls.components_from_metric),
        ]

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
