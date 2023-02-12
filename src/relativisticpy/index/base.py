from typing import Union
from dataclasses import dataclass
from sympy import MutableDenseNDimArray

@dataclass
class BaseIndex:
    symbol          : str
    "The Symbol / id of the index."
    
    order           : int 
    "The order which the index sits in the tensor."

    running         : bool
    "True -> User has not specified specific index. False -> User has specified a number to index"

    basis           : MutableDenseNDimArray
    "The Sympy object basis of the tensor. (Useful for derivative operation.)"

    dimention       : int
    "Dimention of parent Tensor."

    comp_type       : str
    "Type of component, Covarient vs Contravariant."

    values          : Union[list, int] = None
    "Values of index, list of numbers if running, else integer."

    metric_parent   : bool = False
    "Represents whether the parent of this index, the metric tensor."

    parent          : str = None
    "The id / name of the parent indices object which this index sits in."

@dataclass
class IndexContext:

    summed_index    : BaseIndex = None
    "The index to which this index is being summed with."

    repeated_index  : BaseIndex = None
    "The index to which this index is the same as (mostly so we know which components to add/subtract)."

    child_index     : tuple[BaseIndex] = None
    "The index or indices to which this index is a child to within an expression."
