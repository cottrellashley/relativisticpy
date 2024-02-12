""" 
 Base Metric dependent Geometrical Tensors: Ricci - Riemann - 
"""

from typing import Union

from relativisticpy.gr.connection import Connection

from relativisticpy.symengine import SymbolArray
from relativisticpy.core import EinsteinArray, Indices, Metric


class GeometricObject(EinsteinArray):
    """
    Base class for any geometrical tensor. Computes the compoents of the Tensor based on which object is provided.
    Implementation of the computation is on the child tensor classed.
    """

    def __init__(
        self,
        symbols: Union[Metric, Connection, SymbolArray],
        indices: Indices,
        basis: SymbolArray = None,
    ):
        if symbols == None:
            raise ValueError("The argument entered was invalid.")

        components = symbols

        if isinstance(symbols, Metric):
            self._metric = symbols  # Only property which lives at this level
            components = self.from_metric(symbols)
            basis = symbols.basis

        elif isinstance(symbols, Connection):
            self._metric = symbols.metric
            components = self.from_connection(symbols)
            basis = symbols.basis
        
        super().__init__(indices=indices, components=components, basis=basis)

    @property
    def metric(self):
        return self._metric if self._metric != None else None

    @metric.setter
    def metric(self, metric: Metric) -> None:
        self._metric = metric

    def from_metric(self, metric: Metric) -> SymbolArray:
        return metric.components

    def from_connection(self, connection: Connection) -> SymbolArray:
        return connection.components

    def from_components(self, components: SymbolArray) -> SymbolArray:
        return components

    def is_valid(self) -> bool:
        pass
