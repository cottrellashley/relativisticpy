# Standard Library

from itertools import product
from typing import Optional, Union

# External Modules
from relativisticpy.core import Idx, Indices, EinsteinArray, einstein_convention, Metric
from relativisticpy.symengine import SymbolArray, Rational, diff, simplify


@einstein_convention
class Connection(EinsteinArray):

    def __init__(
        self,
        indices: Indices,
        symbols: Union[Metric, SymbolArray],
        basis: SymbolArray = None,
    ):
        if symbols == None:
            raise Exception("The argument entered was invalid.")

        components = symbols

        if isinstance(symbols, Metric):
            self._metric = symbols  # Only property which lives at this level
            components = self._from_metric(symbols)
            basis = symbols.basis

        super().__init__(indices=indices, components=components, basis=basis)

    @classmethod
    def init_from_metric(cls, metric: Metric) -> 'Connection':
        return cls(indices=Indices(-Idx('a'), Idx('b'), Idx('c')), symbols=metric)
    
    def _from_metric(self, metric: Metric) -> SymbolArray:
        D = metric.dimention
        empty = SymbolArray.zeros(D, D, D)
        g = metric._.components
        ig = metric.inv.components
        wrt = metric.basis
        for i, j, k, d in product(range(D), range(D), range(D), range(D)):
            empty[i, j, k] += (
                Rational(1, 2)
                * (ig[d, i])
                * (
                    diff(g[k, d], wrt[j])
                    + diff(g[d, j], wrt[k])
                    - diff(g[j, k], wrt[d])
                )
            )
        return simplify(empty)
    
    def from_metric(metric: Metric) -> SymbolArray:
        D = metric.dimention
        empty = SymbolArray.zeros(D, D, D)
        g = metric._.components
        ig = metric.inv.components
        wrt = metric.basis
        for i, j, k, d in product(range(D), range(D), range(D), range(D)):
            empty[i, j, k] += (
                Rational(1, 2)
                * (ig[d, i])
                * (
                    diff(g[k, d], wrt[j])
                    + diff(g[d, j], wrt[k])
                    - diff(g[j, k], wrt[d])
                )
            )
        return simplify(empty)