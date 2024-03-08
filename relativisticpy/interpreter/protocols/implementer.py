from typing import Protocol, Any, Type
from relativisticpy.interpreter.protocols.tensor import Tensor, Indices
from relativisticpy.interpreter.protocols.state import State
from relativisticpy.interpreter.protocols.nodes import TreeNodes
from relativisticpy.interpreter.protocols.symbolic import Symbolic, Equation, Expression, SymbolArray

# Protocol typing class allows for very flexible and powerful type hints, 
# enabling you to specify exactly what behavior is required from an object, 
# without specifying the exact class of the object. 
# This is especially useful in cases where you want to allow multiple different classes to be used, as long as they provide certain methods.

class Implementer(Protocol):
    """ 
        Contains all methods required to implement language nodes. \n
        All external classes which are implementing the Nodes generated by the interpreter module MUST match this protocol. Implementing all it's methods. 
        \n
        Pattern: \n
            - When we add the 'Node.callback' method we always add it from this Class: Implementer.method_name which will return a string with the method name.
        
    """

    @property
    def state(self) -> State:
        ...
    
    @state.setter
    def state(self, state: State) -> None:
        ...

    def init_tensor(self, node: TreeNodes) -> Tensor:
        "Based on the state of the Tensor node and the sate - we will initialize a tensor."
        ...

    def init_indices(self, node: TreeNodes) -> Indices:
        "Based on the state of the Tensor node and the sate - we will initialize the indices of a tensor."
        ...
    
    def init_metric_indices(self, node: TreeNodes) -> Indices:
        "Based on the state of the Tensor node and the sate -  we will initialize a metric indices."
        ...

    def init_metric_tensor(self, indices: Indices, components: SymbolArray, basis: SymbolArray) -> Tensor:
        "Based on the state of the Tensor node and the sate - we will initialize the indices of a tensor."
        ...

    def init_einstein_array(self, indices: Indices, components: SymbolArray, basis: SymbolArray) -> Tensor:
        "Based on the state of the Tensor node and the sate - we will initialize the indices of a tensor."
        ...

    def init_ricci_scalar(self, node: TreeNodes) -> Tensor:
        "Based on the state of the Tensor node and the sate - we will initialize the indices of a tensor."
        ...

    def init_metric_scalar(self, node: TreeNodes) -> Tensor:
        "Based on the state of the Tensor node and the sate - we will initialize the indices of a tensor."
        ...

    def init_tensor_derivative(self, node: TreeNodes) -> Tensor:
        "Based on the state of the Tensor node and the sate - we will initialize the indices of a tensor."
        ...

    def metric_dependent_types(self, tensor_key: str) -> Type[Tensor]:
        "Based on the state of the Tensor node and the sate - we will initialize the indices of a tensor."
        ...

    # Implementation Methods
    def tensor(self, node: TreeNodes) -> Tensor:
        """ Implementes TensorNode when components already in state. """
        ...

    def tensor_assignment(self, node: TreeNodes) -> Tensor:
        """ Implementes TensorNode when components are being defined by user, then set in state. """
        ...

    def definition(self, node: TreeNodes) -> None:
        ...

    def factorial(self, node: TreeNodes) -> None:
        ...

    def absolute(self, node: TreeNodes) -> None:
        ...

    def ln(self, node: TreeNodes) -> None:
        ...

    def assignment(self, node: TreeNodes) -> None:
        ...

    def clear(self, node: TreeNodes) -> None:
        ...

    def function_def(self, node: TreeNodes) -> None:
        ...

    def symbol_definition(self, node: TreeNodes) -> None:
        ...

    def define(self, node: TreeNodes) -> None:
        ...

    def coordinate_definition(self, node: TreeNodes) -> None:
        ...

    def metric_symbol_definition(self, node: TreeNodes) -> None:
        " Allows user to defined the name of the metric tensor. Will set the value in state and return None. "
        ...

    def infinitesimal(self, node: TreeNodes) -> None:
        " Represents an infinitessimal derivative.  "
        ...

    def equation(self, node: TreeNodes) -> None:
        ...

    def sub(self, node: TreeNodes) -> Expression:
        ...

    def add(self, node: TreeNodes) -> Expression:
        ...

    def neg(self, node: TreeNodes) -> Expression:
        ...

    def pos(self, node: TreeNodes) -> Expression:
        ...

    def mul(self, node: TreeNodes) -> Expression:
        ...

    def div(self, node: TreeNodes) -> Expression:
        ...

    def pow(self, node: TreeNodes) -> Expression:
        ...

    def int(self, node: TreeNodes) -> int:
        ...

    def float(self, node: TreeNodes) -> float:
        ...

    def subs(self, node: TreeNodes) -> Expression:
        ...

    def lim(self, node: TreeNodes) -> Expression:
        ...
    
    def sqrt(self, node: TreeNodes) -> Expression:
        ...

    def expand(self, node: TreeNodes) -> Expression:
        ...
    
    def func_derivative(self, node: TreeNodes) -> Expression:
        ...

    def diff(self, node: TreeNodes) -> Expression:
        ...

    def integrate(self, node: TreeNodes) -> Expression:
        ...

    def simplify(self, node: TreeNodes) -> Expression:
        ...

    def latex(self, node: TreeNodes) -> str:
        ...

    def solve(self, node: TreeNodes) -> Equation:
        ...

    def numerical(self, node: TreeNodes):
        ...

    def exp(self, node: TreeNodes):
        ...

    def dsolve(self, node: TreeNodes):
        ...

    def sin(self, node: TreeNodes):
        ...

    def cos(self, node: TreeNodes):
        ...

    def tan(self, node: TreeNodes):
        ...

    def asin(self, node: TreeNodes):
        ...

    def acos(self, node: TreeNodes):
        ...

    def atan(self, node: TreeNodes):
        ...

    def sinh(self, node: TreeNodes):
        ...

    def cosh(self, node: TreeNodes):
        ...

    def tanh(self, node: TreeNodes):
        ...

    def asinh(self, node: TreeNodes):
        ...

    def acosh(self, node: TreeNodes):
        ...

    def atanh(self, node: TreeNodes):
        ...

    def array(self, node: TreeNodes):
        ...

    def tsimplify(self, node: TreeNodes):
        ...

    def constant(self, node: TreeNodes):
        ...

    def call(self, node: TreeNodes):
        ...
        
    def symbolfunc(self, node: TreeNodes):
        ...

    def symbol(self, node: TreeNodes):
        ...

    def diag(self, node: TreeNodes):
        ...

    def print_(self, node: TreeNodes):
        ...

    def RHS(self, node: TreeNodes):
        ...

    def LHS(self, node: TreeNodes):
        ...

    def sum(self, node: TreeNodes):
        ...

    def dosum(self, node: TreeNodes):
        ...

    def prod(self, node: TreeNodes):
        ...

    def doprod(self, node: TreeNodes):
        ...

    def not_(self, node: TreeNodes):
        ...

    def and_(self, node: TreeNodes):
        ...

    def or_(self, node: TreeNodes):
        ...

    def eqequal(self, node: TreeNodes):
        ...

    def less(self, node: TreeNodes):
        ...

    def greater(self, node: TreeNodes):
        ...

    def lessequal(self, node: TreeNodes):
        ...

    def greaterequal(self, node: TreeNodes):
        ...