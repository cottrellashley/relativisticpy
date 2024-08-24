# Standard Library
from functools import singledispatchmethod
from itertools import product
from typing import Union, Any

# External Modules
from relativisticpy.algebras import EinsumArray, Indices
from relativisticpy.diffgeom.manifold import CoordIndices
from relativisticpy.diffgeom.metric import Metric
from relativisticpy.symengine import SymbolArray, Rational, diff, simplify, Basic
from loguru import logger

class LeviCivitaConnection(EinsumArray):
    """
        The Levi-Civita Connection is a connection that is compatible with the metric but not a tensor.
        Here we represent it as a multi-index object with three indices, dependent on the metric.

    - **Type:** Not a tensor.
    - **Role:** Represent the connection coefficients that define how vectors change as they are parallel transported. Derived from the metric tensor.
    - **Properties:** These are components of the Levi-Civita connection, not true tensors because they do not transform covariantly.
    """

    def __init__(self, indices: CoordIndices, components: SymbolArray, metric: Metric = None):
        logger.debug(f"Creating Levi-Civita Connection with indices {indices} and components {components}")

        if indices.rank != (1, 2):
            raise Exception("The Levi-Civita Connection must have be of rank (1, 2)")

        if len(components.shape) != 3 and all(number == components.shape[0] for number in components.shape):
            raise Exception(
                f"The Levi-Civita Connection must have shape ({components.shape[0]}, {components.shape[0]}, {components.shape[0]}). Got {components.shape} instead.")

        super().__init__(indices=indices, components=components)
        self.metric = metric

    @singledispatchmethod
    @classmethod
    def _new(cls, operand, indices):
        logger.debug(f"[MetricScalar] Handling init: {operand.__class__.__name__}")

    @_new.register
    @classmethod
    def _(cls, operand: Metric, indices):
        components = cls.components_from_metric(operand)
        return cls(indices, components)

    @classmethod
    def components_constructor(cls, dependent_object: Any) -> SymbolArray:
        if isinstance(dependent_object, EinsumArray):
            return dependent_object.components
        elif isinstance(dependent_object, SymbolArray):
            return dependent_object
        else:
            raise ValueError("Invalid argument type for components constructor.")

    @staticmethod
    def components_from_metric(metric: Metric) -> SymbolArray:
        dim = metric.dimention
        empty = SymbolArray.zeros(dim, dim, dim)
        g = metric.ll_components
        ig = metric.uu_components
        wrt = metric.indices.basis
        for i, j, k, d in product(range(dim), range(dim), range(dim), range(dim)):
            empty[i, j, k] += (
                    Rational(1, 2) * (ig[i, d]) * (
                        diff(g[k, d], wrt[j]) + diff(g[d, j], wrt[k]) - diff(g[j, k], wrt[d]))
            )
        return simplify(empty)

    @classmethod
    def from_metric(cls, indices: CoordIndices, metric: Metric) -> 'LeviCivitaConnection':
        return cls(indices=indices, components=cls.components_from_metric(metric), metric=metric)

    def __add__(self, other: EinsumArray):
        return self.add(other, EinsumArray)

    def __sub__(self, other: EinsumArray):
        return self.sub(other, EinsumArray)

    def __mul__(self, other: EinsumArray):
        if isinstance(other, (int, float, Basic)):
            return self.mul(other, type(self))
        elif isinstance(other, EinsumArray):
            # type(other) will most often be GrTensor. This is temporary until we have a better solution. When
            # connections multiply with tensors, the result is not necessarily a tensor. We make it so to not lose
            # tensor functionality.
            return self.mul(other, EinsumArray)
        else:
            raise TypeError(f"Expected int, float, or EinsumArray, got {type(other).__name__}")

    def __truediv__(self, other: Union[int, float, Basic]):
        return self.div(other, type(self))

    def __pow__(self, other: Union[int, float, Basic]):
        return self.pow(other, type(self))
