from abc import ABC, abstractproperty
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Union

from relativisticpy.parsers.types.base import AstNode, NodeType
from .position import Position


######## EXPORSTS #########

class ITensor(ABC):
     @abstractproperty
     def indices(self):
          pass
     
     @abstractproperty
     def identifier(self):
          pass


##### LEAF TENSOR NODE DEFINITION

@dataclass
class _Index:
     identifier: str
     covariant: bool
     values: Union[int, List] = None


class _Indices:
    def __init__(self, indices: List[_Index]):
        self.indices = indices

    def __str__(self) -> str:
        return ''.join([idx.identifier.join(['_{', '}']) if idx.covariant else idx.identifier.join(['^{', '}']) for idx in self.indices])


class TensorNode(
    AstNode
):  # This is the type which the callable handling the AstNode of type TokenType
    """
    This is the object passed in as argument which the callable handling the AstNode of type TokenType.TENSOR.

    Object representing the Tensor Node.

    When executed, the tensor node should represent a Tensor object. 
    The components for the tensor can be soured from three different places, and we need to track this:
        1. Derived from metric (this assumed the metric is already defined in the cache of the Executor Object.)
            We can then compute the components from knowing the object being called 
            (Riemann, Connection, Metric, Ricci, etc... All Metric dependent can be computed without the user expliciply defining the components.)
        2. User defines the components in Array/Matrix form. This looks like:
            T_{mu nu} := [ [1, 0, ...], [0, 1, ...], ... ]
        3. User defines the components from a tensor expression. (The tensor expression must then all be defined, but thats not for this node to need to know.)
            T_{mu nu} := A_{mu nu} - A_{nu mu}
    """

    def __init__(self, position: Position):
        super().__init__(NodeType.TENSOR, position, args=[])
        self._data_type = 'none'
        self._indices = []
        self.identifier = ""

        # The following are all set during build of object.
        self.start_position = None
        self.end_position = None

        # Used for user defining components. Whether its defined as array/matrix or tensor expression will be determined later and will set the comp_type.
        self.component_ast : AstNode = None
        self.component_ast_result = None

    ######### Standard Node Getter Setter Definitions #########
        
    @property
    def callback(self) -> str:
        return "tensor" if self.component_ast == None else 'tensor_assignment'

    @property
    def data_type(self) -> str:
        return "tensor" if self.component_ast == None else "none"
    
    @data_type.setter
    def data_type(self, value) -> None:
        self._data_type = value

    @property
    def is_leaf(self) -> bool:
        return self.component_ast == None and all([idx.values == None for idx in self.indices.indices])

    ######### Specific Tensor Node Property definitions #########

    @property
    def indices(self):
        return _Indices(self._indices)
    
    def __str__(self) -> str:
        return self.identifier + str(self.indices)

    def new_index(self, identifier: str, covariant: bool, values=None):
        self._indices.append(_Index(identifier, covariant, values))
    
    @property
    def sub_components_called(self) -> bool:
        if self._indices == []:
            return False
        return any([idx.values != None for idx in self.indices.indices])

    @property
    def components_definition_type(self):
        if self.component_ast == None:
            return 'none'
        return 'array' if self.component_ast.data_type == 'array' else 'tensor'

    def execute_node(self, executor: Callable) -> None:
        for idx in self.indices.indices:
            if idx.values != None:
                idx.values = executor(idx.values)
        if self.component_ast != None:
            self.component_ast_result = executor(self.component_ast)


class Function(
    AstNode
):  # Functions which user defines to be called again at a later stage in computation.
    """
    This is the object passed in as argument which the callable handling the AstNode of type TokenType.FUNCTION or TokenType.FUNCTION_DEF.

    >    def function(args: Function):
    >       ---> code to handle the Function object. <----

    OR if the hanlder callback is the function definiton node:

    >    def function_def(args: Function):
    >        --- > code to handle the Function object. <----
    """

    # simply add name of built-in functions here.
    # At execution, it the callback function called to handle the node will be the built in insteaf of generic 'function' callback
    BUILT_INS = (
                    "diff", "simplify", "integrate", "expand", 
                    "diag", "lim", "solve", "dsolve", "subs", 
                    "LHS", "RHS", "tsimplify", "sum", "dosum",
                    "sqrt", "func_derivative",
                    "prod", "doprod",
                    "sin", "cos", "tan", 
                    "asin", "atan", "acos", 
                    "cosh", "sinh", "tanh", 
                    "acosh", "asinh", "atanh"
                 )

    def __init__(self):
        self._identifier = ""
        self.type = (NodeType.FUNCTION,)
        self.data_type = "function"
        self._data_type = None

        self._arguments = []
        self._is_called = False
        self._exe_tree = None

    def execute_node(self, executor: Callable) -> None:
        for i, arg in enumerate(self.args):
            self.args[i] = executor(arg)

    @property
    def data_type(self) -> str:
        if (
            self.callback == "function"
        ):  # WE KNOW THIS NODE IS ONLY CONVERTER TO FUNCTION SYMBOL IF CALLBACK IS 'function'
            self._data_type = "function"
            return self._data_type
        elif (
            self._data_type != None
        ):  # We have infered data type of the return value of the function at the Semantic Analyzer phase.
            return self._data_type
        return None

    @data_type.setter
    def data_type(self, value: str) -> None:
        self._data_type = value

    @property
    def is_called(self) -> bool:
        return self._is_called
    
    @property
    def is_leaf(self):
        return self._exe_tree == None and self.args == None

    @is_called.setter
    def is_called(self, is_being_called: bool) -> None:
        if self._exe_tree == None and is_being_called:
            raise AttributeError(
                f"Cannot set to call a function property {self._is_called} when the function has no tree defined to execute."
            )
        self._is_called = is_being_called

    @property
    def is_callable(self) -> bool:
        return self._exe_tree != None

    @property
    def args(self) -> List[AstNode]:
        return self._arguments

    @property
    def identifier(self) -> str:
        return self._identifier

    # Very important property -> determines the callback i.e. what will be called from the Class implementing this node at runtime.
    @property
    def callback(self) -> str:
        if self._identifier == "":
            raise AttributeError(
                "Cannot set a callback when we do not have a function identifier yet."
            )
        if self.is_callable:
            return "function_call"
        if any([self.identifier == builtin_func for builtin_func in self.BUILT_INS]):
            return self.identifier
        return "function"
    
    @identifier.setter
    def identifier(self, funtion_identifier_value: str) -> None:
        if self._identifier != "":
            raise AttributeError(
                f"Cannot change the function Identifyer at class {type(self)}"
            )
        self._identifier = funtion_identifier_value

    @property
    def executable(self) -> AstNode:
        return self._exe_tree

    @executable.setter
    def executable(self, func_scoped_executable_tree: str) -> None:
        if self._exe_tree != None:
            raise AttributeError(
                f"Cannot change the function Executable Tree at class {type(self)}"
            )
        self._exe_tree = func_scoped_executable_tree

    def new_argument(self, arg: AstNode) -> None:
        if isinstance(arg, AstNode):
            arg.parent = self
        self._arguments.append(arg)

    def get_return_type(self, semantic_analyzer_type_traverser: Callable):
        return semantic_analyzer_type_traverser(self.executable)

    def set_position(self, position):
        self.position = position


# ID := EXPR | ARRAY
class Definition(AstNode):
    PRE_DEFINED = {
                        "Coordinates" : "coordinate_definition",
                        "MetricSymbol" :  "metric_symbol_definition"
                    }

    def __init__(self, position, args):
        super().__init__(type=NodeType.DEFINITION, position=position, args=args)
        self.data_type = 'none'

    @property
    def is_leaf(self) -> bool: return False

    def execute_node(self, executor: Callable): self.args[1] = executor(self.args[1])

    @property
    def callback(self): return Definition.PRE_DEFINED[self.args[0]] if self.args[0] in Definition.PRE_DEFINED else "definition"
        