from dataclasses import dataclass
from sympy import MutableDenseNDimArray
from src.relativisticpy.indices.data_structure import TensorIndicesObject

@dataclass
class BaseTensor:

    components : MutableDenseNDimArray

    indices : TensorIndicesObject

    basis : MutableDenseNDimArray

    rank : tuple[int]

    dimention : int

    shape : tuple[int]

    scalar : bool
