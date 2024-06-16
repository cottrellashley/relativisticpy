# Standard Library
from itertools import product

# External Modules
from relativisticpy.algebras import EinsumArray
from relativisticpy.diffgeom.manifold import CoordIndices
from relativisticpy.diffgeom.metric import Metric
from relativisticpy.symengine import SymbolArray, Rational, diff, simplify, Basic

class LeviCivitaConnection(EinsumArray):
    """
        The Levi-Civita Connection is a connection that is compatible with the metric but not a tensor.
        Here we represent it as a multi-index object with three indices, dependent on the metric.
    """

    def __init__(self, indices: CoordIndices, components: SymbolArray, metric: Metric = None):

        if len(indices.indices) != 3:
            raise Exception("The Levi-Civita Connection must have exactly three indices.")
        
        for idx in indices.indices: # Only the first index can be contravariant
            if not (idx.order == 0) == idx.contravariant:
                raise Exception("The Levi-Civita Connection must have (contravariant, covariant, covariant) indices in that order.")
        
        if len(components.shape) != 3 and all(number == components.shape[0] for number in components.shape):
            raise Exception(f"The Levi-Civita Connection must have shape ({components.shape[0]}, {components.shape[0]}, {components.shape[0]}).")
        
        super().__init__(indices=indices, components=components)
        self.metric = metric


    @classmethod
    def from_tensors(cls, indices: CoordIndices, *args, **kwargs) -> 'LeviCivitaConnection':
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
        
        if indices is None:
            raise TypeError("Indices are required.")
        if metric is None:
            return cls(indices, components)
        else:
            return cls.componens_from_metric(indices, metric)

    @property
    def args(self): return [self.indices, self.components, self.metric]

    @staticmethod
    def componens_from_metric(metric: Metric) -> SymbolArray:
        D = metric.dimention
        empty = SymbolArray.zeros(D, D, D)
        g = metric.ll_components
        ig = metric.uu_components
        wrt = metric.basis
        for i, j, k, d in product(range(D), range(D), range(D), range(D)):
            empty[i, j, k] += (
                Rational(1, 2) * (ig[i, d]) * (diff(g[k, d], wrt[j]) + diff(g[d, j], wrt[k]) - diff(g[j, k], wrt[d]))
            )
        return simplify(empty)
    
    @classmethod
    def from_metric(cls, indices: CoordIndices, metric: Metric) -> 'LeviCivitaConnection':
        return cls(indices=indices, components=cls.componens_from_metric(metric), metric=metric)
    
    def __add__(self, other: EinsumArray): return self.add(other, EinsumArray)
    def __sub__(self, other: EinsumArray): return self.sub(other, EinsumArray)

    def __mul__(self, other: EinsumArray):
        if isinstance(other, int, float, Basic):
            return self.mul(other, type(self))
        elif isinstance(other, EinsumArray):
            # type(other) will most often be GrTensor. This is temporary until we have a better solution.
            # When connections multiply with tensors, the result is not necessarily a tensor. We make it so to not lose tensor functionality.
            return self.mul(other, type(other)) 
        else:
            raise TypeError(f"Expected int, float, or EinsumArray, got {type(other).__name__}")
        
    def __truediv__(self, other: EinsumArray): return self.div(other, type(self))
    def __pow__(self, other: EinsumArray): return self.pow(other, type(self))
