from dataclasses import dataclass
from sympy import MutableDenseNDimArray
from relativisticpy.indices.data_structure import TensorIndicesObject

@dataclass
class BaseTensor:

    components : MutableDenseNDimArray

    indices : TensorIndicesObject

    basis : MutableDenseNDimArray

    rank : tuple[int]

    dimention : int

    shape : tuple[int]

    scalar : bool

    def get_specified_components(self):
        return self.components[self.indices.__index__()]



