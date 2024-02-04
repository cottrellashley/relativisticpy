from dataclasses import dataclass
from typing import Callable, Dict, List

from relativisticpy.gr import MetricScalar, RicciScalar
from relativisticpy.core import Indices, MetricIndices, EinsteinArray, TensorEqualityType, Idx
from relativisticpy.utils import extract_tensor_symbol, extract_tensor_indices

from relativisticpy.workbook.constants import WorkbookConstants

from relativisticpy.parsers.types.gr_nodes import TensorNode

class TensorReference:
    """Tensor Cache Helper class which represents strings representations of tensors."""

    def __init__(self, tensor: TensorNode):
        self.__descerialized_indices = tensor.indices.indices
        self.tensor = tensor
        self.__is_metric = False
        self.__symbol = tensor.identifier

    @property
    def indices(self) -> Indices:
        if self.is_metric:
            test = MetricIndices(*[Idx(symbol=idx.identifier, values=idx.values) if idx.covariant else -Idx(symbol=idx.identifier, values=idx.values) for idx in self.__descerialized_indices])
            return test
        else:
            test1 = Indices(*[Idx(symbol=idx.identifier, values=idx.values) if idx.covariant else -Idx(symbol=idx.identifier, values=idx.values) for idx in self.__descerialized_indices])
            return test1

    @property
    def is_calling_tensor_subcomponent(self) -> bool: return any([idx.values != None for idx in self.__descerialized_indices])

    @property
    def symbol(self) -> str: self.tensor.identifier

    @property
    def repr(self) -> str:
        return str(self.tensor)

    @property
    def id(self) -> str:
        return self._tid()

    @property
    def is_metric(self) -> bool:
        return self.__is_metric
    
    @is_metric.setter
    def is_metric(self, value: bool) -> None:
        self.__is_metric = value

    def _tid(self):
        return self.__symbol

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        return self.id == other.id if isinstance(other, TensorReference) else False


class TensorList:
    """Helper class for Cache to more easily store and retriev tensor objects."""

    def __init__(self, *tensors: EinsteinArray):
        self.tensors: List[EinsteinArray] = list(tensors)

    def iterfind(self, *funcs: Callable):
        filtered_tensors = self.tensors
        for func in funcs:
            filtered_tensors = [
                tensor for tensor in filtered_tensors if func(tensor.indices)
            ]
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
        self.store: Dict[str, any] = {}
        self.tensors: Dict[str, TensorList] = {}

        # Set Default Symbol definitions
        self.store[WorkbookConstants.METRICSYMBOL.value] = "g"
        self.store[WorkbookConstants.EINSTEINTENSORSYMBOL.value] = "Ein"
        self.store[WorkbookConstants.RICCISYMBOL.value] = "Ric"
        self.store[WorkbookConstants.RIEMANNSYMBOL.value] = "R"
        self.store[WorkbookConstants.CONNECTION.value] = "C"
        self.store[WorkbookConstants.DERIVATIVESYMBOL.value] = "d"
        self.store[WorkbookConstants.COVDERIVATIVESYMBOL.value] = "D"

        self.__metric_symbol = "g"
        self.einstein_tensor_symbol = 'G'
        self.__connection_symbol = "C"
        self.__ricci_symbol = "Ric"
        self.__reimann_symbol = "R"
        self.__derivative_symbol = "d"
        self.__cov_derivative_symbol = "D"

        self.ricci_scalar = None
        self.metric_scalar = None

    @property
    def metric_symbol(self) -> str:
        return self.__metric_symbol

    @metric_symbol.setter
    def metric_symbol(self, value: str) -> None:
        self.__metric_symbol = value

    @property
    def connection_symbol(self) -> str:
        return self.__connection_symbol

    @connection_symbol.setter
    def connection_symbol(self, value: str) -> None:
        self.__connection_symbol = value

    @property
    def ricci_symbol(self) -> str:
        return self.__ricci_symbol

    @ricci_symbol.setter
    def ricci_symbol(self, value: str) -> None:
        self.__ricci_symbol = value

    @property
    def reimann_symbol(self) -> str:
        return self.__reimann_symbol

    @reimann_symbol.setter
    def reimann_symbol(self, value: str) -> None:
        self.__reimann_symbol = value

    @property
    def derivative_symbol(self) -> str:
        return self.__derivative_symbol

    @derivative_symbol.setter
    def derivative_symbol(self, value: str) -> None:
        self.__derivative_symbol = value

    @property
    def cov_derivative_symbol(self) -> str:
        return self.__cov_derivative_symbol

    @cov_derivative_symbol.setter
    def cov_derivative_symbol(self, value: str) -> None:
        self.__cov_derivative_symbol = value

    ##### OTHER CACHE METHODS #######

    def set_variable(self, name, value):
        self.store[name] = value

    def get_variable(self, name):
        return self.store.get(name, None)

    def has_variable(self, name):
        return name in self.store
    
    def clear_all(self):
        self.store: Dict[str, any] = {}
        self.tensors: Dict[str, TensorList] = {}

    ##### TENSOR CACHE METHODS #######

    def has_metric(self) -> bool:
        return self.__metric_symbol in self.tensors

    def get_metric(self):
        return self.tensors[self.__metric_symbol].get(0)

    def set_coordinates(self, coordinates: EinsteinArray):
        self.coordinates = coordinates

    def set_metric_scalar(self):
        if self.has_metric():
            self.metric_scalar = MetricScalar(self.get_metric(), self.coordinates)

    def set_ricci_scalar(self):
        if self.has_metric():
            self.ricci_scalar = RicciScalar(self.get_metric(), self.coordinates)

    def set_tensor(self, tensor_string: TensorReference, tensor: EinsteinArray):
        if self.has_tensor(tensor_string.id):
            self.tensors[tensor_string.id].add(tensor)
        self.tensors[tensor_string.id] = TensorList(tensor)

    def get_tensors(self, tensor_symbol_id: str):
        if tensor_symbol_id in self.tensors:
            return self.tensors[tensor_symbol_id].get(0)
        return None

    def has_tensor(self, tensor_id: str):
        return tensor_id in self.tensors

    def match_on_tensors(
        self, equality_type: TensorEqualityType, tref: TensorReference
    ):
        if not self.has_tensor(tref.id):
            return None

        cached_tensors: TensorList[EinsteinArray] = self.tensors[tref.id]
        callback: Callable = tref.indices.get_equality_type_callback(equality_type)
        tensor_matched = cached_tensors.find(callback)
        return tensor_matched
