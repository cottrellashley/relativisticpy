from relativisticpy.core import MultiIndexObject, Indices, Metric
from typing import Union
from abc import ABC, abstractmethod
from relativisticpy.core.exceptions import ArgumentException
from relativisticpy.gr.connection import Connection
from relativisticpy.providers import SymbolArray

class GeometricObject(MultiIndexObject):
    """
        Base class for any geometrical tensor. Computes the compoents of the Tensor based on which object is provided. 
        Implementation of the computation is on the child tensor classed.
    """

    def __init__(self, indices: Indices, symbols: Union[Metric, Connection, SymbolArray], basis: SymbolArray = None):
        if isinstance(symbols, Metric): components = self.from_metric(symbols)
        elif isinstance(symbols, Connection): components = self.from_connection(symbols)
        elif isinstance(symbols, SymbolArray): components = self.from_components(symbols)
        else: raise ArgumentException('The argument entered was invalid.')
        super().__init__(indices, components, basis)

    def from_metric(self, metric: Metric): pass
    def from_connection(self, connection: Connection): pass
    def from_components(self, components: SymbolArray): pass
    def is_valid(self) -> bool: pass