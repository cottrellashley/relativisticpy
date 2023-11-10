# Standard Library
from itertools import product
from typing import Optional

# External Modules
from relativisticpy.core import Idx, Indices, MultiIndexObject, einstein_convention, Metric
from relativisticpy.symengine import SymbolArray, Rational, diff, simplify
from relativisticpy.deserializers import tensor_from_string

@einstein_convention
class Connection(MultiIndexObject):

    @classmethod
    def from_string(cls, indices_str, comp_str, basis_str):
        return tensor_from_string(Idx, Indices, Connection, indices_str, comp_str, basis_str)

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
