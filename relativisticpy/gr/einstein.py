# Standard Library
from itertools import product

# External Modules
from relativisticpy.diffgeom.tensor import Tensor
from relativisticpy.diffgeom import Metric, Ricci
from relativisticpy.symengine import SymbolArray, Rational, zeros, diff, simplify

# This Module
from relativisticpy.diffgeom import LeviCivitaConnection


class EinsteinTensor(Tensor):
    # TODO: Implement the Einstein Tensor with new base classes.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def component_equations(cls):
        return [
            (SymbolArray, lambda arg: arg),
            (Metric, cls.components_from_metric)
        ]

    @staticmethod
    def components_from_metric(metric: Metric) -> SymbolArray:
        ric = Ricci.components_from_metric(metric)
        g = metric.ll_components
        ig = metric.uu_components
        dim = metric.dimention
        skeleton = SymbolArray(zeros(dim ** 2), (dim, dim))
        for i, j, l, k in product(range(dim), range(dim), range(dim), range(dim)):
            skeleton[i, j] += ric[i, j] - g[i, j] * ig[l, k] * ric[l, k]
        return simplify(skeleton)