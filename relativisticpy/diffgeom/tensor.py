from typing import Protocol
from relativisticpy.algebras import Indices, EinsumArray
from relativisticpy.symengine import SymbolArray

class TensorIndices(Indices):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def cooridnate_transform(self, *args, **kwargs):
        ...

    def coordinate_patch(self, *args, **kwargs):
        ...

class Tensor(EinsumArray):
    def __init__(self, indices: TensorIndices, components: SymbolArray):
        ...
