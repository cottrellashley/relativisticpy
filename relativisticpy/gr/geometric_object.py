from relativisticpy.core import MultiIndexObject, Indices, Metric
from typing import Union
from abc import ABC, abstractmethod
from relativisticpy.core.exceptions import ArgumentException
from relativisticpy.gr.connection import Connection
from relativisticpy.utils import SymbolArray

class GeometricObject(MultiIndexObject):
    """
        Base class for any geometrical tensor. Computes the compoents of the Tensor based on which object is provided. 
        Implementation of the computation is on the child tensor classed.
    """

    def __init__(self, indices: Indices, symbols: Union[Metric, Connection, SymbolArray], basis: SymbolArray = None):

        if symbols == None:
            raise ArgumentException('The argument entered was invalid.')

        components = symbols

        if isinstance(symbols, Metric): 
            self._metric = symbols # Only property which lives at this level
            components = self.from_metric(symbols)
            basis = symbols.basis

        elif isinstance(symbols, Connection): 
            components = self.from_connection(symbols)
            basis = symbols

        super().__init__(indices=indices, components=components, basis=basis)

    @property
    def metric(self): return self._metric if self._metric != None else None

    @metric.setter
    def metric(self, metric: Metric) -> None: self._metric = metric

    def from_metric(self, metric: Metric): pass
    def from_connection(self, connection: Connection): pass
    def from_components(self, components: SymbolArray): pass
    def is_valid(self) -> bool: pass