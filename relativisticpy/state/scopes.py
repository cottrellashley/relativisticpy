import copy

from relativisticpy.algebras import Indices
from relativisticpy.diffgeom import Metric, LeviCivitaConnection, CovDerivative
from relativisticpy.diffgeom.tensors.metricscalar import MetricScalar
from relativisticpy.diffgeom.tensor import Tensor
from relativisticpy.diffgeom.tensors.ricci import Ricci
from relativisticpy.diffgeom.tensors.ricciscalar import RicciScalar
from relativisticpy.diffgeom.tensors.riemann import Riemann
from relativisticpy.gr.einstein import EinsteinTensor
from typing import Union, Dict, List, Callable, Type
from loguru import logger

from relativisticpy.symengine.sympy import SymbolArray


class TensorList:
    """Helper class for Cache to more easily store and retrieve tensor objects."""

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
        "MetricSymbol": "g",
        "EinsteinTensorSymbol": "G",
        "ConnectionSymbol": "C",
        "RicciSymbol": "Ric",
        "RiemannSymbol": "R",
        "DerivativeSymbol": "d",
        "CovariantDerivativeSymbol": "D"
    }

    Coordinates = "Coordinates"

    def __init__(self):
        self.variables = {}
        self.function_variables = {}
        self.tensor_variables: Dict[str, TensorList] = {}
        self.variables.update(Scope.BUILT_IN_VARS)

    def check_variable(self, var: str) -> bool:
        """ Returns True if variable exists in this current scope. """
        return var in self.variables

    def check_function(self, function: str) -> bool:
        """ Returns True if variable exists in this current scope. """
        return function in self.function_variables

    def check_tensor(self, tensor_key: str) -> bool:
        """ Returns True if variable exists in this current scope. """
        return tensor_key in self.tensor_variables

    def set_tensor_symbol(self, symbol: str) -> NotImplementedError:
        raise NotImplementedError("Still not implemented user set tensor symbols.")

    @property
    def has_metric(self) -> bool:
        return self.variables["MetricSymbol"] in self.tensor_variables

    @property
    def metric_tensor(self):
        tensors = self.tensor_variables[self.variables["MetricSymbol"]]
        logger.debug(f"Found {len(tensors.tensors)} Metric Tensor Objects in scope.")
        logger.debug(f"Returning {tensors.get(-1)}.")
        return tensors.get(-1)

    def set_tensor(self, tensor_id: str, tensor: Tensor):
        if self.has_tensor(tensor_id):
            self.tensor_variables[tensor_id].add(tensor)
        self.tensor_variables[tensor_id] = TensorList(tensor)

    def get_tensors(self, tensor_symbol_id: str):
        if tensor_symbol_id in self.tensor_variables:
            return self.tensor_variables[tensor_symbol_id].get(0)
        return None

    def new_tensor(self, indices: Indices, tensor_symbol_id: str) -> Tensor:
        return self.tensor_factory(indices, components, basis)

    def has_tensor(self, tensor_id: str):
        return tensor_id in self.tensor_variables

    def match_on_tensors(
            self, callback: Callable, tensor_node
    ):
        if not self.has_tensor(tensor_node.identifier):
            return None

        cached_tensors: TensorList = self.tensor_variables[tensor_node.identifier]
        tensor_matched = cached_tensors.find(callback)
        return tensor_matched

    def _match_on_tensors(
            self, callback: Callable, identifier: str
    ):
        if not self.has_tensor(identifier):
            return None

        cached_tensors: TensorList = self.tensor_variables[identifier]
        tensor_matched = cached_tensors.find(callback)
        return tensor_matched

    def reset(self):
        self.variables = {}
        self.function_variables = {}
        self.tensor_variables: Dict[str, TensorList] = {}
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
    def is_global(self):
        return len(self.stack) == 1

    def push_scope(self) -> None:
        """Enter a new function call (or scope) by adding a new dictionary to the stack"""
        self.stack.append(Scope())

    def pop_scope(self) -> None:
        """Exit the current scope by popping the top dictionary from the stack"""
        if len(self.stack) > 1:
            self.stack.pop()
        else:
            raise Exception("Cannot pop the global scope")

    ############################################################
    ################# VARIABLE STATE FUNCTIONS #################
    ############################################################

    def set_variable(self, name, value):
        """Set a variable in the current scope (the top dictionary on the stack)"""
        self.stack[-1].variables[name] = value

    def get_variable(self, name) -> Union[any, None]:
        """Search for a variable in the stack, starting from the top"""
        for scope in reversed(self.stack):
            if scope.check_variable(name):
                return scope.variables[name]
        return None

    def has_variable(self, name) -> Union[any, None]:
        """Search for a variable in the stack, starting from the top"""
        for scope in reversed(self.stack):
            if scope.check_variable(name):
                return True
        return False

    ############################################################
    ################## TENSOR STATE OPERATIONS #################
    ############################################################

    def set_tensor(self, name, value):
        """Set a variable in the current scope (the top dictionary on the stack)"""
        logger.debug(f"Caching new tensor '{name}' with components '{value.components}'")
        self.stack[-1].set_tensor(name, copy.copy(value))

    def get_tensor(self, name) -> Union[any, None]:
        """Search for a variable in the stack, starting from the top"""
        for scope in reversed(self.stack):
            if scope.check_tensor(name):
                return copy.copy(scope.tensor_variables[name])
        return None

    def has_tensor(self, name) -> Union[any, None]:
        """Search for a variable in the stack, starting from the top"""
        for scope in reversed(self.stack):
            if scope.check_tensor(name):
                return True
        return False

    @property
    def metric_tensor(self):
        """Search for a variable in the stack, starting from the top"""
        for scope in reversed(self.stack):
            if scope.has_metric:
                logger.debug(f"Found metric tensor in scope with components: '{scope.metric_tensor.components}'")
                return copy.copy(scope.metric_tensor)
        return False

    ############################################################
    ################# FUNCTION STATE FUNCTIONS #################
    ############################################################

    def set_function(self, name, value):
        """Set a variable in the current scope (the top dictionary on the stack)"""
        self.stack[-1].function_variables[name] = value

    def get_function(self, name) -> Union[any, None]:
        """Search for a variable in the stack, starting from the top"""
        for scope in reversed(self.stack):
            if scope.check_function(name):
                return scope.function_variables[name]
        return None

    def has_function(self, name) -> Union[any, None]:
        """Search for a variable in the stack, starting from the top"""
        for scope in reversed(self.stack):
            if scope.check_function(name):
                return True
        return False

    @property
    def current_scope(self) -> Scope:
        """Return the current scope"""
        return self.stack[-1]

    @property
    def global_scope(self) -> Scope:
        """Return the global scope"""
        return self.stack[0]

    def cache_new_tensor(self, identifier: str, indices: Indices, components: SymbolArray) -> None:
        cls = self.get_tensor_cls(identifier)
        new_tensor = cls.from_equation(indices, components)
        self.set_tensor(identifier, new_tensor)

    def init_tensor(self, indices: Indices, identifier: str, sub_components_called: bool):
        logger.debug(f"Attempting to initiating '{identifier}{indices}' by equation from cached tensors.")

        if self.get_variable(Scope.MetricSymbol) == identifier:
            logger.debug(f"     Identified '{identifier}{indices}' to be a Metric. Initializing Metric from Metric.")
            metric = Metric.from_metric(self.metric_tensor, indices)
            logger.debug(f"     Computed components of '{identifier}{indices}' to be: '{metric.components}'")
            self.set_tensor(identifier, metric)
            return metric.subcomponents if sub_components_called else metric

        if not self.has_tensor(identifier):  # If not stated => skip to generate immediately.
            # last thing we do is generate a new instance of the tensor - Since the Interpreter Modules is not
            # responsible for implementations we make it generic.
            cls: Type[Tensor] = self.get_tensor_cls(identifier)
            new_tensor = cls.from_equation(indices, self.metric_tensor)
            self.set_tensor(identifier, new_tensor)
            return new_tensor.subcomponents if sub_components_called else new_tensor

        is_same_indices = self.current_scope._match_on_tensors(indices.symbol_eq, identifier)

        if is_same_indices is not None:
            # _{a}_{b}_{c} == _{a}_{b}_{c}
            tensor = self.current_scope._match_on_tensors(indices.symbol_order_rank_eq, identifier)

            if tensor is None:
                #  => state does not have an instance.
                # Reset tensor to point to tensor we know not to be Null
                tensor = is_same_indices

                # Handling any changes in order of indices.
                diff_order = tensor.indices.get_reshape(indices)
                if diff_order is None:
                    # No order changes => just init new instance with new indices.
                    new_tensor: Tensor = type(tensor).from_equation(indices, tensor)
                    return new_tensor.subcomponents if sub_components_called else new_tensor  # TODO: subcomponent
                    # check should be done in the tensor class.

                new_tensor = tensor.reshape(indices)
                return new_tensor.subcomponents if sub_components_called else new_tensor

            # We need to generate the new subcomponents if user is calling new subcomponents
            tensor: Tensor = type(tensor).from_equation(indices, tensor)
            if sub_components_called and indices.anyrunnig:
                return tensor.subcomponents

            return tensor.subcomponents if sub_components_called else tensor

        has_same_rank = self.current_scope._match_on_tensors(indices.rank_eq, identifier)
        if has_same_rank is not None:
            cls: Type[Tensor] = self.get_tensor_cls(identifier)
            new_tensor: Tensor = cls.from_equation(indices, has_same_rank)
            self.set_tensor(identifier, new_tensor)
            return new_tensor.subcomponents if sub_components_called else new_tensor

        # last thing we do is generate a new instance of the tensor - Since the Interpreter Modules is not reponsible
        # for implementations we make it generic.
        cls: Type[Tensor] = self.get_tensor_cls(identifier)
        new_tensor = cls.from_equation(indices, self.metric_tensor)
        self.set_tensor(identifier, new_tensor)
        return new_tensor.subcomponents if sub_components_called else new_tensor

    def get_tensor_cls(self, tensor_key: str, is_scalar: bool = False):
        types_map = {
            self.get_variable(Scope.EinsteinTensorSymbol): EinsteinTensor,
            self.get_variable(Scope.RiemannSymbol): Riemann,
            self.get_variable(Scope.CovariantDerivativeSymbol): CovDerivative,
            self.get_variable(Scope.ConnectionSymbol): LeviCivitaConnection,
        }
        if is_scalar:
            types_map.update({
                self.get_variable(Scope.RicciSymbol): RicciScalar,
                self.get_variable(Scope.MetricSymbol): MetricScalar,
            })
        else:
            types_map.update({
                self.get_variable(Scope.RicciSymbol): Ricci,
                self.get_variable(Scope.MetricSymbol): Metric,
            })
        return types_map[tensor_key] if tensor_key in types_map else Tensor