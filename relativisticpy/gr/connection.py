# Standard Library
from itertools import product
from typing import Optional

# External Modules
from relativisticpy.core import Idx, Indices, MultiIndexObject, einstein_convention, deserialisable_tensor
from relativisticpy.providers import SymbolArray, Rational, diff, simplify

# This Module
from relativisticpy.gr.metric import Metric

@einstein_convention
@deserialisable_tensor
class Connection(MultiIndexObject):
    _cls_idcs = Indices # On Descerialization, this is the class that will be __init__ for indices.

    @classmethod
    def from_metric(metric: Metric) -> SymbolArray:
        D = metric.dimention
        empty = SymbolArray.zeros(D, D, D)
        g = metric._.components
        ig = metric.inv.components
        wrt = metric.basis
        for (i, j, k, d) in product(range(D), range(D), range(D), range(D)): empty[i, j, k] += Rational(1, 2)*(ig[d,i])*(diff(g[k,d],wrt[j]) + diff(g[d,j],wrt[k]) - diff(g[j,k],wrt[d]))
        return simplify(empty)

    def __init__(self, indices: Indices, components: Optional[MultiIndexObject] = None, basis: Optional[MultiIndexObject] = None, metric: Metric = None):
        self.__default_comp = indices.zeros_array()
        self.__default_basis = Idx.default_basis
        super().__init__(
                            indices     =   indices,
                            components  =   components if components != None else Connection.from_metric(metric) if metric != None else self.__default_comp,
                            basis       =   basis if basis != None else metric.basis if metric != None else self.__default_basis
                        )
