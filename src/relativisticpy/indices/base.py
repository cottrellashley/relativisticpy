from dataclasses import dataclass
from src.relativisticpy.index.index import Index
from sympy import MutableDenseNDimArray

@dataclass
class BaseTensorIndices:

    indices         : tuple[Index]
    "List of Index objects, representing the indices of parent Tensor."

    basis           : MutableDenseNDimArray
    "List of Index objects, representing the indices of parent Tensor."

    dimention       : int
    "Dimention of parent Tensor."

    rank            : tuple
    "Rank of tensor, represented as typle."

    scalar          : bool
    "Boolean representing whether Parent tensor is a scalar"

    shape           : tuple
    "Shape of the Parent tensor: tuple of the dimention of index list."

    valid           : bool
    "Given all properties, is resulting parent Tesnor a valid Tesnor."

    parent_tensor   : str = None
    "The name of the tensor to which this indices object belongs to, represented as a string."
