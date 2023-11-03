from dataclasses import dataclass
from typing import Callable, Dict, List
from relativisticpy.core import Indices, MultiIndexObject
from relativisticpy.core.tensor_equality_types import TensorEqualityType
from relativisticpy.providers.regex import extract_tensor_symbol, extract_tensor_indices

@dataclass
class TensorReference:
    """ Tensor Cache Helper class which represents strings representations of tensors. """
    tstring : str

    @property
    def indices(self) -> Indices: return Indices.from_string(self.indices_repr)

    @property
    def symbol(self) -> str: return extract_tensor_symbol(self.tstring)

    @property
    def indices_repr(self) -> str: return extract_tensor_indices(self.tstring)

    @property
    def repr(self) -> str: return str(self)

    @property
    def id(self) -> str: return self._tid()
    def _tid(self): return self.symbol

    def __str__(self) -> str: return self.tstring
    def __hash__(self) -> int: return hash(self.id)
    def __eq__(self, other) -> bool: return self.id == other.id if isinstance(other, TensorReference) else False

class TensorList(list):
    """ Helper class for Cache to more easily store and retriev tensor objects. """

    def __init__(self, *tensors: MultiIndexObject):
        self.tensors : List[MultiIndexObject] = list(tensors)

    def iterfind(self, *funcs: Callable):
        filtered_tensors = self.tensors
        for func in funcs:
            filtered_tensors = [tensor for tensor in filtered_tensors if func(tensor.indices)]
            if not filtered_tensors:
                return None
        return filtered_tensors if filtered_tensors else None

    def find(self, func):
        for tensor in self.tensors:
            if func(tensor.indices):
                return tensor
        return None

    def add(self, tensor):
        self.tensors.append(tensor)

    def any(self, func: Callable): 
        return any([func(tensor.indices) for tensor in self.tensors])

    def all(self, func: Callable): 
        return all([func(tensor.indices) for tensor in self.tensors])

class RelPyCache:
    """ 
        Cache specific to RelativisticPy Cache.
        This class does not know how to initiate new tensors, but it does know how to manage cached tensors to return them in the most efficient manner.
    """

    def __init__(self):
        self.store : Dict[str, any] = {}
        self.tensors : Dict[str, TensorList] = {}

    ##### OTHER CACHE METHODS #######

    def set_variable(self, name, value):
        self.store[name] = value

    def get_variable(self, name):
        return self.store.get(name, None)

    def has_variable(self, name):
        return name in self.store
    
    ##### TENSOR CACHE METHODS #######

    def set_tensor(self, tensor_string: TensorReference, tensor: MultiIndexObject):
        if self.has_variable(tensor_string.id):
            self.tensors[tensor_string.id].add(tensor)
        self.tensors[tensor_string.id] = TensorList(tensor)

    def get_tensors(self, tensor_symbol_id: str):
        if tensor_symbol_id in self.tensors:
            return self.tensors[tensor_symbol_id]
        return None

    def has_tensor(self, tref: TensorReference):
        tref.id in self.tensors

    def match_existing_tensor_with_equality_type(self, equality_type: TensorEqualityType, tref: TensorReference):
    
        if not self.has_variable(tref.id):
            return None

        cached_tensors : TensorList[MultiIndexObject] = self.tensors[tref.id]
        return cached_tensors.find(tref.indices.get_equality_type_callback(equality_type))


    def get_tensor_instance(self, tref: TensorReference):
    
        if not self.has_variable(tref.id):
            return None

        cached_tensors : TensorList[MultiIndexObject] = self.tensors[tref.id]
        return cached_tensors.find(tref.indices.symbol_order_rank_eq)

    def get_tensor_with_eq_indices(self, tref: TensorReference):
    
        if not self.has_variable(tref.id):
            return None

        cached_tensors : TensorList[MultiIndexObject] = self.tensors[tref.id]
        return cached_tensors.find(tref.indices.symbol_eq)

    def get_tensor_with_eq_rank(self, tref: TensorReference):

        if not self.has_variable(tref.id):
            return None

        cached_tensors : TensorList[MultiIndexObject] = self.tensors[tref.id]
        return cached_tensors.find(tref.indices.rank_eq)

        
