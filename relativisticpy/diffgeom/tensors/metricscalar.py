from itertools import product
from relativisticpy.diffgeom.manifold import CoordIndices
from relativisticpy.diffgeom.tensor import Tensor
from relativisticpy.diffgeom.metric import Metric
from relativisticpy.diffgeom.connection import LeviCivitaConnection
from relativisticpy.symengine import SymbolArray, simplify


class MetricScalar(Tensor):
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

        # Categorize keyword arguments
        for _, value in kwargs.items():
            if isinstance(value, SymbolArray):
                components = arg
            elif isinstance(value, Metric):
                metric = value

        if components is None:
            if metric is not None:
                components = cls.components_from_metric(metric)
            else:
                raise TypeError("Components or metric is required.")

        return cls(indices, components)

    @staticmethod
    def components_from_metric(metric: Metric):
        N = metric.dimention
        g = metric.components
        ig = metric.uu_components
        A = float()
        for i, j in product( range(N), range(N) ):
            A += (
                ig[i, j] * g[i, j]
            )
        return simplify(A)
