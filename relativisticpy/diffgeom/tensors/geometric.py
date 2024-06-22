""" 
 Base Metric dependent Geometrical Tensors: Ricci - Riemann - 
"""

from typing import Union, Protocol
from relativisticpy.diffgeom.connection import LeviCivitaConnection

from relativisticpy.symengine import SymbolArray
from relativisticpy.algebras import Indices, Idx, EinsumArray
from relativisticpy.diffgeom.metric import Metric

class GeometricTensor(Protocol):
    """
    Base class for any geometrical tensor. Computes the compoents of the Tensor based on which object is provided.
    Implementation of the computation is on the child tensor classed.
    """

    @classmethod
    def from_metric(cls, metric: Metric) -> 'GeometricTensor':
        ...

    @classmethod
    def from_connection(cls, connection: LeviCivitaConnection) -> 'GeometricTensor':
        ...
