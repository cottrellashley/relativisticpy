""" 
 Base Metric dependent Geometrical Tensors: Ricci - Riemann - 
"""


from relativisticpy.diffgeom.connection import LeviCivitaConnection
from relativisticpy.diffgeom.tensor import Tensor

from relativisticpy.symengine import SymbolArray
from relativisticpy.algebras import Indices, Idx, EinsumArray
from relativisticpy.diffgeom.metric import Metric

class GeometricTensor(Tensor):
    """
    Base class for any geometrical tensor. Computes the compoents of the Tensor based on which object is provided.
    Implementation of the computation is on the child tensor classed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)