from relativisticpy.interpreter.protocols import Tensor
from typing import Union, Dict, List, Callable

class TensorList:
    """Helper class for Cache to more easily store and retriev tensor objects."""

    def __init__(self, *tensors: Tensor):
        self.tensors: List[Tensor] = list(tensors)

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

class Scope:
    # In-build and replaceable variables
    MetricSymbol = "MetricSymbol"
    EinsteinTensorSymbol = "EinsteinTensorSymbol"
    ConnectionSymbol = "ConnectionSymbol"
    RicciSymbol = "RicciSymbol"
    RiemannSymbol = "RiemannSymbol"
    DerivativeSymbol = "DerivativeSymbol"
    CovariantDerivativeSymbol = "CovariantDerivativeSymbol"

    BUILT_IN_VARS = {
        "MetricSymbol" : "g",
        "EinsteinTensorSymbol" : "G",
        "ConnectionSymbol" : "C",
        "RicciSymbol" : "Ric",
        "RiemannSymbol" : "R",
        "DerivativeSymbol" : "d",
        "CovariantDerivativeSymbol" : "D"
    }

    Coordinates = "Coordinates"

    def __init__(self):
        self.variables = {}
        self.function_variables = {}
        self.tensor_variables : Dict[str, TensorList] = {}
        self.variables.update(Scope.BUILT_IN_VARS)

    def check_variable(self, var: str) -> bool:
        " Returns True if variable exists in this current scope. "
        return var in self.variables
    
    def check_function(self, function: str) -> bool:
        " Returns True if variable exists in this current scope. "
        return function in self.function_variables
    
    def check_tensor(self, tensor_key: str) -> bool:
        " Returns True if variable exists in this current scope. "
        return tensor_key in self.tensor_variables
    
    def set_tensor_symbol(self, symbol: str) -> None:
        return NotImplementedError("Still not implemented user setted tensor symbols.")
    
    @property
    def has_metric(self) -> bool: return self.variables["MetricSymbol"] in self.tensor_variables
    @property
    def metric_tensor(self): return self.tensor_variables[self.variables["MetricSymbol"]].get(0)

    def set_tensor(self, tensor_id: str, tensor: Tensor):
        if self.has_tensor(tensor_id):
            self.tensor_variables[tensor_id].add(tensor)
        self.tensor_variables[tensor_id] = TensorList(tensor)

    def get_tensors(self, tensor_symbol_id: str):
        if tensor_symbol_id in self.tensor_variables:
            return self.tensor_variables[tensor_symbol_id].get(0)
        return None

    def has_tensor(self, tensor_id: str):
        return tensor_id in self.tensor_variables

    def match_on_tensors(
        self, callback: Callable, tensor_node
    ):
        if not self.has_tensor(tensor_node.identifier):
            return None

        cached_tensors: TensorList[Tensor] = self.tensor_variables[tensor_node.identifier]
        tensor_matched = cached_tensors.find(callback)
        return tensor_matched

    def reset(self):
        self.variables = {}
        self.function_variables = {}
        self.tensor_variables : Dict[str, TensorList] = {}
        self.variables.update(Scope.BUILT_IN_VARS)

class ScopedState:

    def __init__(self):
        self.stack = [Scope()]
        self.tensor_factory = None

    def reset(self):
        del self.stack
        self.stack = []
        self.stack.append(Scope())

    @property
    def is_global(self): return len(self.stack) == 1

    def push_scope(self) -> None:
        "Enter a new function call (or scope) by adding a new dictionary to the stack"
        self.stack.append(Scope())

    def pop_scope(self) -> None:
        "Exit the current scope by popping the top dictionary from the stack"
        if len(self.stack) > 1:
            self.stack.pop()
        else:
            raise Exception("Cannot pop the global scope")

    ############################################################
    ################# VARIABLE STATE FUNCTIONS #################
    ############################################################

    def set_variable(self, name, value):
        "Set a variable in the current scope (the top dictionary on the stack)"
        self.stack[-1].variables[name] = value

    def get_variable(self, name) -> Union[any, None]:
        "Search for a variable in the stack, starting from the top"
        for scope in reversed(self.stack):
            if scope.check_variable(name):
                return scope.variables[name]
        return None
    
    def has_variable(self, name) -> Union[any, None]:
        "Search for a variable in the stack, starting from the top"
        for scope in reversed(self.stack):
            if scope.check_variable(name):
                return True
        return False

    ############################################################
    ################## TENSOR STATE OPERATIONS #################
    ############################################################

    def set_tensor(self, name, value):
        "Set a variable in the current scope (the top dictionary on the stack)"
        self.stack[-1].set_tensor(name, value)

    def get_tensor(self, name) -> Union[any, None]:
        "Search for a variable in the stack, starting from the top"
        for scope in reversed(self.stack):
            if scope.check_tensor(name):
                return scope.tensor_variables[name]
        return None
    
    def has_tensor(self, name) -> Union[any, None]:
        "Search for a variable in the stack, starting from the top"
        for scope in reversed(self.stack):
            if scope.check_tensor(name):
                return True
        return False

    @property
    def metric_tensor(self):
        "Search for a variable in the stack, starting from the top"
        for scope in reversed(self.stack):
            if scope.has_metric:
                return scope.metric_tensor
        return False

    ############################################################
    ################# FUNCTION STATE FUNCTIONS #################
    ############################################################

    def set_function(self, name, value):
        "Set a variable in the current scope (the top dictionary on the stack)"
        self.stack[-1].function_variables[name] = value

    def get_function(self, name) -> Union[any, None]:
        "Search for a variable in the stack, starting from the top"
        for scope in reversed(self.stack):
            if scope.check_function(name):
                return scope.function_variables[name]
        return None
    
    def has_function(self, name) -> Union[any, None]:
        "Search for a variable in the stack, starting from the top"
        for scope in reversed(self.stack):
            if scope.check_function(name):
                return True
        return False

    @property
    def current_scope(self) -> Scope:
        "Return the current scope"
        return self.stack[-1]
    
    @property
    def global_scope(self) -> Scope:
        "Return the global scope"
        return self.stack[0]
