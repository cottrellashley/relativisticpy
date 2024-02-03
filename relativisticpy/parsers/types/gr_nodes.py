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

    >    def tensor(args: Tensor):
    >        code to handle the Tensor object.
    """

    def __init__(self):
        self.gr_node = "LEAF"
        self.type = NodeType.TENSOR
        self._indices = []
        self._identifier = ""
        self.args = []
        self.start_position = None
        self.end_position = None
        self.computed_expr = None
        self.computed_comps = None


    ######### Standard Node Getter Setter Definitions #########
        
    @property
    def callback(self) -> str:
        if not hasattr(self, "_callback"):
            return "tensor"
        return self._callback
    
    @callback.setter
    def callback(self, value):
        self._callback = value

    @property
    def data_type(self) -> str:
        return "tensor"
    
    @property
    def is_leaf(self) -> bool:
        return not (self.defined_from_components or self.defined_from_tensor_expr)

    ######### Specific Tensor Node Property definitions #########

    @property
    def indices(self):
        return _Indices(self._indices)
    
    def __str__(self) -> str:
        return self.identifier + str(self.indices)

    def new_index(self, identifier: str, covariant: bool, values=None):
        self._indices.append(_Index(identifier, covariant, values))

    @property
    def defined_from_tensor_expr(self) -> bool:
        return self.__tensor_expr_ast != None

    @property
    def defined_from_components(self) -> bool:
        return self.__tensor_components_ast != None
    
    @property
    def sub_components_called(self) -> bool:
        return any([idx.values != None for idx in self.indices.indices])

    @property
    def tensor_expr_ast(self) -> Union[AstNode, None]:
        if hasattr(self, "__tensor_expr_ast"):
            return None
        return self.__tensor_expr_ast

    @tensor_expr_ast.setter
    def tensor_expr_ast(self, tensor_expr_ast) -> None:
        self.__tensor_expr_ast = tensor_expr_ast

    @property
    def tensor_components_ast(self) -> Union[AstNode, None]:
        if hasattr(self, "__tensor_components_ast"):
            return None
        return self.__tensor_components_ast

    @tensor_components_ast.setter
    def tensor_components_ast(self, array_ast) -> None:
        self.__tensor_components_ast = array_ast

    @property
    def identifier(self) -> str:
        return self._identifier

    @identifier.setter
    def identifier(self, tensor_identifier_value: str) -> None:
        self._identifier = tensor_identifier_value


    def execute_node(self, executor: Callable):
        if self.defined_from_components:
            self.computed_comps = executor(self.tensor_components_ast)
        elif self.defined_from_tensor_expr:
            self.computed_expr = executor(self.tensor_expr_ast)


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

    BUILT_INS = ("diff", "simplify", "integrate")

    def __init__(self):
        self.gr_node = "LEAF"  # Either a call, a symbol object or definiton object.... all of which are leaf nodes.
        self._identifier = ""
        self.type = (NodeType.FUNCTION,)
        self.data_type = "function"
        self.__data_type = None

        self._arguments = []
        self.__is_called = False
        self._exe_tree = None

    @property
    def data_type(self) -> str:
        if (
            self.callback == "function"
        ):  # WE KNOW THIS NODE IS ONLY CONVERTER TO FUNCTION SYMBOL IF CALLBACK IS 'function'
            self.__data_type = "function"
            return self.__data_type
        elif (
            self.__data_type != None
        ):  # We have infered data type of the return value of the function at the Semantic Analyzer phase.
            return self.__data_type
        return None

    @data_type.setter
    def data_type(self, value: str) -> None:
        self.__data_type = value

    @property
    def is_called(self) -> bool:
        return self.__is_called
    
    @property
    def is_leaf(self):
        return self._exe_tree == None

    @is_called.setter
    def is_called(self, is_being_called: bool) -> None:
        if self._exe_tree == None and is_being_called:
            raise AttributeError(
                f"Cannot set to call a function property {self.__is_called} when the function has no tree defined to execute."
            )
        self.__is_called = is_being_called

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

    def execute_node(self, executor: Callable):
        pass


# ID := EXPR | ARRAY
class Definition(AstNode):
    PRE_DEFINED = ("Coordinates", "MetricSymbol")

    def __init__(self, position, args):
        super().__init__(type=NodeType.DEFINITION, position=position, args=args)
        self.data_type = 'none'

    @property
    def is_leaf(self) -> bool: return False

    def execute_node(self, executor: Callable):
        self.args[1] = executor(self.args[1])

    @property
    def callback(self):
        if self.args[0] == "Coordinates":
            return "coordinate_definition"
        else:
            return "definition"
        