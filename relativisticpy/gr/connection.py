# Standard
from itertools import product
from operator import itemgetter
from typing import Union, Optional

# External
from sympy import MutableDenseNDimArray, Rational, diff, simplify, zeros

# Internal 
#   - core
from relativisticpy.core.helpers import transpose_list, empty_sympy_array
from relativisticpy.core.decorators import einstein_convention
from relativisticpy.core.multi_index_obj import MultiIndexObject
from relativisticpy.core.indices import Idx, Indices
#   - gr
from relativisticpy.gr.metric import Metric, MetricIndices
from relativisticpy.gr.derivative import Derivative

@einstein_convention
class Connection(MultiIndexObject):
    # Idea: if user initiates this object without components => set all zeros => This will allow users to then call __setitem__ object, mapping the zeros into the resulting expression.

    def _compute_comps(metric: Metric):
        D = metric.dimention
        empty = empty_sympy_array(D, (D, D, D))
        g = metric._.components
        ig = metric.inv.components
        wrt = metric.basis
        for (i, j, k, d) in product(range(D), range(D), range(D), range(D)): empty[i, j, k] += Rational(1, 2)*(ig[d,i])*(diff(g[k,d],wrt[j]) + diff(g[d,j],wrt[k]) - diff(g[j,k],wrt[d]))
        return simplify(empty)

    def __init__(self, indices: Indices, metric : Optional[Union[Metric, MultiIndexObject]] = None):
        self.__default_comp = indices.zeros_array()
        self.__default_basis = Idx.default_basis
        self.metric = metric
        super().__init__(
                            components  =   Connection._compute_comps(metric) if metric != None else self.__default_comp,
                            indices     =   indices,
                            basis       =   metric.basis if metric != None else self.__default_basis
                        )
