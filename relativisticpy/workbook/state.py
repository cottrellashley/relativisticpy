from dataclasses import dataclass
from typing import Callable, Dict, List

from relativisticpy.core import Indices, MultiIndexObject, TensorEqualityType
from relativisticpy.utils import extract_tensor_symbol, extract_tensor_indices

from relativisticpy.workbook.constants import WorkbookConstants

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

class TensorList:
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

    def get(self, idx: int):
        return self.tensors[idx]

    def any(self, func: Callable): 
        return any([func(tensor.indices) for tensor in self.tensors])

    def all(self, func: Callable): 
        return all([func(tensor.indices) for tensor in self.tensors])

class WorkbookState:
    """ 
        A Cache Object specific to handle RelativisticPy Workbooks.
        This class does not know how to initiate new tensors, but it does know how to manage cached tensors to return them in the most efficient manner.
    """

    def __init__(self):
        self.store : Dict[str, any] = {}
        self.tensors : Dict[str, TensorList] = {}

        # Set Default Symbol definitions
        self.store[WorkbookConstants.METRICSYMBOL.value] = 'G'
        self.store[WorkbookConstants.RICCISYMBOL.value] = 'Ric'
        self.store[WorkbookConstants.RIEMANNSYMBOL.value] = 'R'
        self.store[WorkbookConstants.DERIVATIVESYMBOL.value] = 'd'
        self.store[WorkbookConstants.COVDERIVATIVESYMBOL.value] = 'D'

        self.__metric_symbol = 'g'
        self.__ricci_symbol = 'Ric'
        self.__reimann_symbol = 'R'
        self.__derivative_symbol = 'd'
        self.__cov_derivative_symbol = 'D'

    @property 
    def metric_symbol(self) -> str: return self.__metric_symbol

    @metric_symbol.setter
    def metric_symbol(self, value: str) -> None: self.__metric_symbol = value

    @property 
    def ricci_symbol(self) -> str: return self.__ricci_symbol

    @ricci_symbol.setter
    def ricci_symbol(self, value: str) -> None: self.__ricci_symbol = value

    @property 
    def reimann_symbol(self) -> str: return self.__reimann_symbol

    @reimann_symbol.setter
    def reimann_symbol(self, value: str) -> None: self.__reimann_symbol = value

    @property 
    def derivative_symbol(self) -> str: return self.__derivative_symbol

    @derivative_symbol.setter
    def derivative_symbol(self, value: str) -> None: self.__derivative_symbol = value

    @property 
    def cov_derivative_symbol(self) -> str: return self.__cov_derivative_symbol

    @cov_derivative_symbol.setter
    def cov_derivative_symbol(self, value: str) -> None: self.__cov_derivative_symbol = value

    ##### OTHER CACHE METHODS #######

    def set_variable(self, name, value):
        self.store[name] = value

    def get_variable(self, name):
        return self.store.get(name, None)

    def has_variable(self, name):
        return name in self.store
    
    ##### TENSOR CACHE METHODS #######

    def has_metric(self) -> bool:
        return self.store[WorkbookConstants.METRICSYMBOL.value] in self.tensors
    
    def get_metric(self):
        return self.tensors[self.store[WorkbookConstants.METRICSYMBOL.value]].get(0)

    def set_coordinates(self, coordinates: MultiIndexObject):
        self.coordinates = coordinates

    def set_tensor(self, tensor_string: TensorReference, tensor: MultiIndexObject):
        if self.has_tensor(tensor_string.id):
            self.tensors[tensor_string.id].add(tensor)
        self.tensors[tensor_string.id] = TensorList(tensor)

    def get_tensors(self, tensor_symbol_id: str):
        if tensor_symbol_id in self.tensors:
            return self.tensors[tensor_symbol_id].get(0)
        return None

    def has_tensor(self, tensor_id: str):
        return tensor_id in self.tensors

    def match_on_tensors(self, equality_type: TensorEqualityType, tref: TensorReference):
    
        if not self.has_tensor(tref.id):
            return None

        cached_tensors : TensorList[MultiIndexObject] = self.tensors[tref.id]
        callback : Callable = tref.indices.get_equality_type_callback(equality_type)
        tensor_matched = cached_tensors.find(callback)
        return tensor_matched
