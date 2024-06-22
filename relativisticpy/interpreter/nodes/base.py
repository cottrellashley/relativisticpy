from dataclasses import dataclass
from abc import ABC, abstractproperty

from enum import Enum
from typing import Callable, List, Union, Type
from .position import Position
from relativisticpy.interpreter.state.scopes import ScopedState, Scope
from relativisticpy.interpreter.protocols import Node, Implementer, Tensor, Indices
from relativisticpy.interpreter.shared.errors import IndicesError


class NodeType(Enum):
    """An enumeration of node types used by a parser."""

    OR = "BINARY OP: Boolean expression"
    ID = "UNARY OP: Value getter | Symbol creator"  # If variable not created in memory, build a symbol.
    CONSTANT = 'constant'

    # Since we never use the value of these 'constants' we give a description.
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    POW = "^"
    AND = "and"  # Positive operator '+'

    INT = "int"
    POS = "+"
    NEG = "-"
    NOT = "not"

    LESS = "<"

    FLOAT = "float"  # A floating-point number
    ARRAY = "array"  # Array object '[elements]'
    EQUALS = "="  # The `=` symbol for assignment or comparison

    LESSEQUAL = "<="
    GREATER = ">"

    GREATEREQUAL = ">="

    ASSIGNMENT = "="
    DEFINITION = ":="

    EQEQUAL = "=="
    NOTEQUAL = "!="

    EXPONENTIATION1 = "^"  # The `**` symbol for exponentiation
    EXPONENTIATION2 = "**"
    DEFINITION_ID = "definition_id"
    ASSIGNMENT_ID = "assignment_id"

    TENSOR = "tensor"
    FUNCTION = "function"  # A function name
    CALL = 'call'
    INFINITESIMAL = "infinitesimal"
    DERIVATIVE = "derivative"
    FACTORIAL = 'factorial'
    ABSOLUTE = 'absolute'

    FUNCTION_DEF = "FUNCTION_DEF"

    PRINT = "PRINT"

    SYMBOL = "symbol"  # symbol object
    SYMBOLFUNC = "symbolfunc"  # function as a symbol .i.e undefined non-callable object

    TENSOR_EXPR_ASSIGNMENT = "TENSOR_EXPR_ASSIGNMENT"
    TENSOR_COMPONENT_ASSIGNMENT = "TENSOR_COMPONENT_ASSIGNMENT"
    TENSOR_COMPONENT_DEFINITION = "TENSOR_COMPONENT_DEFINITION"


class AstNode(Node):
    """Nested Node type. The whole object represents the Abstract Syntax Tree."""

    def __init__(
            self,
            type: NodeType,  # swtich name to node_type instead!
            position: Position,
            callback: str = None,
            args: List["AstNode"] = None,
    ):
        self.type = type
        self.position = position
        self._callback = callback
        self.data_type = None

        self.args = args
        self.children = args
        self.parent = None

        for child in self.children:
            if isinstance(child, AstNode):
                child.parent = self

    def execute_node(self, implementer: Implementer):
        """
        Executes node by traversing and executing all child nodes first, then getting the node implementer callback and passing the args to it.

        Args:
            implementer (Implementer): The object implementing the Node execution. Implementer contains all methods called as callbacks for the nodes. Implementer also contains state of the executing AST Node tree.
        
        Returns:
            Any: The return value of the node execution. This can be anything depending on the implementation of the node.
        
        Raises:
            NotImplementedError: If the Implementer object does not have the callback method required to execute the node.
        """
        self_exe: Callable = self.get_executor(implementer)

        for i, child in enumerate(self.children):
            self.args[i] = child.execute_node(implementer)

        return self_exe(self)

    def analyze_node(self, implementer: Implementer):
        """ Defined how this node is analyzed by Semantic Analyzer. Note: For most nodes this is same implementation
        as execute_node."""
        self_exe: Callable = self.get_executor(implementer)

        for child in self.children:
            child.analyze_node(implementer)

        self_exe(self)

    def get_executor(self, implementer: Implementer) -> Callable:
        if not hasattr(implementer, self.callback):
            raise NotImplementedError(
                f"""The object {type(implementer).__name__} does not have the method '{self.callback}()' required to 
                implement the Node: {type(self).__name__}. """)
        else:
            return getattr(implementer, self.callback)

    def get_state(self, implementer: Implementer) -> ScopedState:
        if not hasattr(implementer, 'state'):
            raise NotImplementedError(
                f"There exists no {type(implementer).__name__}.state property required to implement the Node: {type(self).__name__}. ")
        else:
            return implementer.state

    @property
    def callback(self):
        return self._callback

    @callback.setter
    def callback(self, value):
        self._callback = value

    @property
    def is_root(self) -> bool:
        return self.parent == None

    def remove_child(self, node_rmv):
        self.children = [child for child in self.children if child is not node_rmv]

    def get_level(self):
        level = 0
        p = self.parent
        while p:
            level += 1
            p: "AstNode" = p.parent
        return level

    def print_tree(self, level=-1):
        spaces = " " * self.get_level() * 3
        prefix = spaces + "|__" if self.parent else ""
        if len(self.children) == 1:
            print(prefix + str(self.children[0]))
        else:
            print(prefix + self.type.value)
        if level != 0:
            for child in self.children:
                if isinstance(child, AstNode):
                    child.print_tree(level - 1)


class UnaryNode(AstNode):
    def __init__(
            self,
            type: NodeType,
            position: Position,
            callback: str,
            args: List["AstNode"],
    ):
        if len(args) != 1:
            raise ValueError("UnaryNode requires exactly one arguments")
        super().__init__(type, position, callback, args)

    @property
    def callback(self):
        return self._callback

    @callback.setter
    def callback(self, value):
        self._callback = value

    @property
    def operand(self) -> AstNode:
        return self.args[0]


class BinaryNode(AstNode):
    def __init__(
            self, type: NodeType, position: Position, callback: str, args: List["AstNode"]
    ):
        if len(args) != 2:
            raise ValueError("BinaryNode requires exactly two arguments")
        super().__init__(type, position, callback, args)

    @property
    def is_leaf(self) -> bool: return False

    @property
    def callback(self):
        return self._callback

    @callback.setter
    def callback(self, value):
        self._callback = value

    @property
    def left_child(self) -> AstNode:
        return self.args[0]

    @property
    def right_child(self) -> AstNode:
        return self.args[1]


class ArrayNode(AstNode):
    def __init__(
            self,
            position: Position,
            args: List["AstNode"],
    ):
        super().__init__(type=NodeType.ARRAY, position=position, callback=Implementer.array.__name__, args=args)
        self.data_type = 'array'

    @property
    def is_leaf(self) -> bool: return False

    @property
    def shape(self):
        "Compute the shape of the array"
        pass


# arith_expr = arith_expr
class AssignmentNode(BinaryNode):
    """ Arithmatic expression equality builds an object Eq() representing an equation. """

    def __init__(
            self,
            position: Position,
            args: List["AstNode"],
    ):
        super().__init__(type=NodeType.ASSIGNMENT, position=position, callback=Implementer.assignment.__name__,
                         args=args)
        self.data_type = 'none'

    @property
    def is_leaf(self) -> bool: return False

    def execute_node(self, implementor: Implementer):
        self.args[1] = self.children[1].execute_node(implementor)


# - expr
class NegNode(UnaryNode):
    def __init__(
            self,
            position: Position,
            args: List["AstNode"],
    ):
        super().__init__(NodeType.NEG, position, Implementer.neg.__name__, args)

    @property
    def is_leaf(self) -> bool: return False


# + expr
class PosNode(UnaryNode):
    def __init__(
            self,
            position: Position,
            args: List["AstNode"],
    ):
        super().__init__(NodeType.POS, position, Implementer.pos.__name__, args)

    @property
    def is_leaf(self) -> bool: return False


# not | !
class NotNode(UnaryNode):
    def __init__(
            self,
            position: Position,
            args: List["AstNode"],
    ):
        super().__init__(NodeType.NOT, position, Implementer.not_.__name__, args)

    @property
    def is_leaf(self) -> bool: return False


# print
class PrintNode(UnaryNode):
    def __init__(
            self,
            position: Position,
            args: List["AstNode"],
    ):
        super().__init__(NodeType.PRINT, position, Implementer.print_.__name__, args)

    @property
    def is_leaf(self) -> bool: return False


# d expr | d() | d{}
class Infinitesimal(UnaryNode):
    def __init__(
            self,
            position: Position,
            args: List["AstNode"],
    ):
        super().__init__(NodeType.INFINITESIMAL, position, Implementer.infinitesimal.__name__, args)
        self.diff_order: int = None
        self.expression = None
        self.is_partial: bool = None
        self.diff_order_as_int: int = None

    @property
    def is_leaf(self) -> bool: return False

    def execute_node(self, implementer: Implementer) -> None:
        state: ScopedState = self.get_state(implementer)

        # Execute Node
        state.set_variable("".join(self.args[0]), self.args[1].execute_node(implementer))

    def analyze_node(self, implementer: Implementer):
        """ Defined how this node is analyzed by Semantic Analyzer. Note: For most nodes this is same implementation as execute_node. """
        state: ScopedState = self.get_state(implementer)
        # Execute Node
        state.set_variable("".join(self.args[0]), self.args[1].analyze_node(implementer))


# ID :=
class Definition(AstNode):

    def __init__(self, position, args):
        super().__init__(type=NodeType.DEFINITION, position=position, args=args)
        self.data_type = 'none'

    @property
    def is_leaf(self) -> bool:
        return False

    def execute_node(self, implementer: Implementer) -> None:
        state: ScopedState = self.get_state(implementer)

        # Execute Node
        if "".join(self.args[0]) in Scope.BUILT_IN_VARS:
            state.set_variable("".join(self.args[0]), "".join(self.args[1].args))
        elif "".join(self.args[0]) == "Constants":
            var = self.args[1].execute_node(implementer)
            constants = [implementer.symbol_str(str(sym), constant=True) for sym in var]
            [state.set_variable(str(constant), constant) for constant in constants]
        elif "".join(self.args[0]) == "Reals":
            var = self.args[1].execute_node(implementer)
            constants = [implementer.symbol_str(str(sym), real=True) for sym in var]
            [state.set_variable(str(constant), constant) for constant in constants]
        elif "".join(self.args[0]) == "Integers":
            var = self.args[1].execute_node(implementer)
            constants = [implementer.symbol_str(str(sym), integer=True) for sym in var]
            [state.set_variable(str(constant), constant) for constant in constants]
        elif "".join(self.args[0]) == "Complex":
            var = self.args[1].execute_node(implementer)
            constants = [implementer.symbol_str(str(sym), complex=True) for sym in var]
            [state.set_variable(str(constant), constant) for constant in constants]
        else:
            state.set_variable("".join(self.args[0]), self.args[1].execute_node(implementer))

    def analyze_node(self, implementer: Implementer):
        """ Defined how this node is analyzed by Semantic Analyzer. Note: For most nodes this is same implementation as execute_node. """
        state: ScopedState = self.get_state(implementer)
        # Execute Node
        state.set_variable("".join(self.args[0]), self.args[1].analyze_node(implementer))


class Def(AstNode):
    """ Node for the definition of a Function. """

    def __init__(
            self,
            identifier: str,
            body: AstNode,
            position: Position,
            args: List["AstNode"] = None
    ):
        super().__init__(NodeType.FUNCTION_DEF, position, Implementer.function_def.__name__, args)
        self.data_type = "none"
        self.identifier = identifier
        self.str_args = ["".join(arg.args) for arg in args] if len(args) != 0 else []
        self.body = body

    def execute_node(self, implementer: Implementer) -> None:
        state: ScopedState = self.get_state(implementer)
        # Execute Node
        state.set_function(self.identifier, self)

    @property
    def is_leaf(self): return False


class Call(AstNode):
    """
    Node for the call of a Function. Defaults to returning a Funbol, unless user actually defines a exe
    expression.
    """
    BUILT_INS = (
        Implementer.diff.__name__,
        Implementer.simplify.__name__,
        Implementer.integrate.__name__,
        Implementer.expand.__name__,
        Implementer.diag.__name__,
        Implementer.lim.__name__,
        Implementer.exp.__name__,
        Implementer.solve.__name__,
        Implementer.dsolve.__name__,
        Implementer.subs.__name__,
        Implementer.LHS.__name__,
        Implementer.RHS.__name__,
        Implementer.tsimplify.__name__,
        Implementer.sum.__name__,
        Implementer.dosum.__name__,
        Implementer.clear.__name__,
        Implementer.sqrt.__name__,
        Implementer.func_derivative.__name__,
        Implementer.prod.__name__,
        Implementer.doprod.__name__,
        Implementer.sin.__name__,
        Implementer.cos.__name__,
        Implementer.tan.__name__,
        Implementer.asin.__name__,
        Implementer.atan.__name__,
        Implementer.acos.__name__,
        Implementer.cosh.__name__,
        Implementer.sinh.__name__,
        Implementer.tanh.__name__,
        Implementer.acosh.__name__,
        Implementer.asinh.__name__,
        Implementer.atanh.__name__,
        Implementer.ln.__name__
    )

    def __init__(
            self,
            identifier: str,
            position: Position,
            args: List["AstNode"] = None
    ):
        self.args = args
        self.children = args
        self.position = position
        self.type = NodeType.CALL
        self.data_type = "undef"  # syntactic analysis will infer the data type of the return value of the function.
        self.identifier = identifier  # The name of the function being called.
        self.is_built_in: bool = self.identifier in self.BUILT_INS  # Is the function being called a built-in function?
        self.call_return = None

    def execute_node(self, implementer: Implementer) -> None:
        state = self.get_state(implementer)

        # Does the function we are calling exist in the function stack in our current scope?
        func_def: Def = state.get_function(self.identifier)

        if func_def:
            # First push a new scope in the state.
            state.push_scope()

            # We now set the arguments to the object passed into the call. i.e. f(x, t) called as -> f( n**2 + 9 , 10 ) => set: 'x' :  n**2 + 9 and 't' : 10
            for i, arg in enumerate(self.children):
                state.set_variable(func_def.str_args[i], arg.execute_node(implementer))

            # Finally we execute the body of the function.
            if isinstance(func_def.body, AstNode):
                call_return_obj = func_def.body.execute_node(implementer)
            elif isinstance(func_def.body, list):
                return_objs = []
                for statement in func_def.body:
                    obj = statement.execute_node(implementer)
                    if obj != None:
                        return_objs.append(obj)
                call_return_obj = return_objs[-1] if len(return_objs) != 0 else None

            state.pop_scope()
            return call_return_obj
        else:
            for i, arg in enumerate(self.children):
                self_exe: Callable = self.get_executor(implementer)

                for i, child in enumerate(self.children):
                    self.args[i] = child.execute_node(implementer)

                return self_exe(self)

    @property
    def is_leaf(self):
        return False

    @property
    def callback(self) -> str:
        if self.identifier in self.BUILT_INS:
            return self.identifier
        elif self.call_return == None:
            return Implementer.symbolfunc.__name__
        return Implementer.call.__name__

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
        return ''.join(
            [idx.identifier.join(['_{', '}']) if idx.covariant else idx.identifier.join(['^{', '}']) for idx in
             self.indices])


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

    def __init__(self, position: Position, source_code: str = None):
        self.type = NodeType.TENSOR
        self.position = position
        self._data_type = 'none'
        self._indices = []
        self.identifier = ""
        self.source_code = source_code

        # The following are all set during build of object.
        self.start_position = None
        self.end_position = None

        # Used for user defining components. Whether its defined as array/matrix or tensor expression will be determined later and will set the comp_type.
        self.component_ast: AstNode = None
        self.component_ast_result = None

    ######### Standard Node Getter Setter Definitions #########

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

    def execute_node(self, implementer: Implementer) -> None:
        # TODO: Maze of IF / ELIF / ELSE statements => There is a lot of improvements to be made in terms of
        #  simplicity / simplification / elegance.
        state: ScopedState = self.get_state(implementer)
        # First we descerialize indices numbers if they were set.
        for idx in self.indices.indices:  # TODO: Can indices be its own node?
            symbol = implementer.symbol_str(idx.identifier)

            if idx.values != None:
                idx.values = idx.values.execute_node(implementer)  # '1' : str -> 1 : int

            if symbol in state.get_variable(Scope.Coordinates):  # _{a:r} i.e. for cases when the used wants the
                # component by calling coordinate symbol.
                if isinstance(idx.values, int):
                    raise IndicesError(self.position,  # TODO <--
                                       f"Tensor sub-component is already being called by using index as the "
                                       f"Coordinate variable '{idx.identifier}', cannot index again with method ': "
                                       f"<int>'",
                                       self.source_code)
                idx.values = list(state.get_variable(Scope.Coordinates)).index(symbol)

        if self.identifier == state.get_variable(Scope.DerivativeSymbol):
            return implementer.init_tensor_derivative(self)

        # If these components of the tensor were set => We do not return anything. We only cache the object
        elif self.component_ast is not None:
            tensor_component = self.component_ast.execute_node(implementer)

            if state.get_variable(Scope.MetricSymbol) == self.identifier and self.components_definition_type == 'array':
                # This is the metric tensor
                tensor_indices = implementer.init_indices(self)
                new_tensor = implementer.init_metric_tensor(tensor_indices, tensor_component)
            elif self.components_definition_type == 'array':
                # This is an einstein array.
                tensor_indices = implementer.init_indices(self)
                new_tensor = implementer.init_einstein_array(tensor_indices, tensor_component)
            elif self.components_definition_type == 'tensor':
                # This is an einstein array defined from other tensors.
                tensor_indices = implementer.init_indices(self)
                einstein_array_obj: Tensor = tensor_component
                new_tensor = einstein_array_obj.reshape(tensor_indices)  # WARNING: This could cause issues.

            state.set_tensor(self.identifier, new_tensor)
        else:
            return self.tensor_cache_retrieval(implementer)

    def tensor_cache_retrieval(self, implementer: Implementer):
        tensor_indices: Indices = implementer.init_indices(self)

        state: ScopedState = self.get_state(implementer)

        # Place this within the init_tensor_indices method.
        tensor_indices.basis = (
            state.get_variable(Scope.Coordinates)
        )  # Error handling needed => if no coordinates defined cannot continue

        if not state.has_tensor(self.identifier):  # If not stated => skip to generate immediately.
            # last thing we do is generate a new instance of the tensor - Since the Interpreter Modules is not
            # reponsible for implementations we make it generic.
            if self.identifier == state.get_variable(Scope.MetricSymbol):
                metric = implementer.init_metric_tensor(tensor_indices,
                                                        state.metric_tensor[tensor_indices.get_non_running()])
                return metric.subcomponents if self.sub_components_called else metric

            elif self.identifier == state.get_variable(Scope.ConnectionSymbol):
                connection = implementer.connection_cls.from_equation(tensor_indices, state.metric_tensor)
                return connection.subcomponents if self.sub_components_called else connection

            else:
                cls: Type[Tensor] = implementer.metric_dependent_types(self.identifier)
                new_tensor = cls.from_equation(tensor_indices, state.metric_tensor)
                state.set_tensor(self.identifier, new_tensor)
                return new_tensor.subcomponents if self.sub_components_called else new_tensor

        is_same_indices = state.current_scope.match_on_tensors(tensor_indices.symbol_eq, self)

        if is_same_indices is not None:
            tensor = state.current_scope.match_on_tensors(tensor_indices.symbol_order_rank_eq, self)

            if tensor is None:
                #  => state does not have an instance.
                # Reset tensor to point to tensor we know not to be Null
                tensor = is_same_indices
                tensor_indices = implementer.init_indices(self)

                # Handling any changes in order of indices.
                diff_order = tensor.indices.get_reshape(tensor_indices)
                if diff_order is None:
                    # No order changes => just init new instance with new indices.
                    new_tensor: Tensor = type(tensor).from_equation(tensor_indices, tensor)
                    return new_tensor.subcomponents if self.sub_components_called else new_tensor # TODO: subcomponent check should be done in the tensor class.

                new_tensor = tensor.reshape(tensor_indices)
                return new_tensor.subcomponents if self.sub_components_called else new_tensor

            # We need to generate the new subcomponents if user is calling new subcomponents
            tensor: Tensor = type(tensor).from_equation(tensor_indices, tensor)
            if self.sub_components_called and implementer.init_indices(self).anyrunnig:
                return tensor.subcomponents

            return (
                tensor[implementer.init_indices(self)] if implementer.init_indices(
                    self).anyrunnig or self.sub_components_called else tensor
            )

        has_same_rank = state.current_scope.match_on_tensors(tensor_indices.rank_eq, self)
        if has_same_rank != None:
            has_same_rank.indices = tensor_indices
            new_tensor: Tensor = type(has_same_rank)(
                *has_same_rank.args
            )
            state.set_tensor(self.identifier, new_tensor)
            return new_tensor.subcomponents if self.sub_components_called else new_tensor

        # last thing we do is generate a new instance of the tensor - Since the Interpreter Modules is not reponsible for implementations we make it generic.
        if self.identifier == state.get_variable(Scope.MetricSymbol):
            metric: Tensor = implementer.init_metric_tensor(tensor_indices,
                                                            state.metric_tensor[tensor_indices.get_non_running()])
            return metric.subcomponents if self.sub_components_called else metric
        else:
            cls: Type[Tensor] = implementer.metric_dependent_types(self.identifier)
            new_tensor = cls.from_equation(tensor_indices, state.metric_tensor)
            state.set_tensor(self.identifier, new_tensor)
            return new_tensor.subcomponents if self.sub_components_called else new_tensor

    def analyze_node(self, implementer: Implementer):
        """ Defined how this node is analyzed by Semantic Analyzer. Note: For most nodes this is same implementation as execute_node. """
        if hasattr(implementer, "tensor") and self.component_ast == None:
            self_exe: Callable = implementer.tensor
        elif hasattr(implementer, "tensor_assignment"):
            self_exe: Callable = implementer.tensor_assignment
        else:
            raise NotImplementedError(
                f"The object {type(implementer).__name__} has not implemented the TensorNode methods. ")

        for idx in self.indices.indices:
            if idx.values != None:
                idx.values.analyze_node(implementer)
        if self.component_ast != None:
            self.component_ast.analyze_node(implementer)

        self_exe(self)


######################################################################################################################
###################################################   LEAF NODES   ###################################################
######################################################################################################################

# CONSTANTS: pi - e - infinity
class ConstantNode(UnaryNode):
    def __init__(
            self,
            position: Position,
            args: List["AstNode"],
    ):
        super().__init__(NodeType.CONSTANT, position, Implementer.constant.__name__, args)
        self.data_type = 'constant'

    @property
    def is_leaf(self) -> bool: return True

    def execute_node(self, implementer: Implementer): return implementer.constant(self)

    def analyze_node(self, implementer: Implementer): implementer.constant(self)


# INT
class IntNode(UnaryNode):
    def __init__(
            self,
            position: Position,
            args: List["AstNode"],
    ):
        super().__init__(NodeType.INT, position, Implementer.int.__name__, args)
        self.data_type = 'int'

    @property
    def is_leaf(self) -> bool: return True

    def execute_node(self, implementer: Implementer): return implementer.int(self)

    def analyze_node(self, implementer: Implementer): implementer.int(self)


# FLOAT
class FloatNode(UnaryNode):
    def __init__(
            self,
            position: Position,
            args: List["AstNode"],
    ):
        super().__init__(NodeType.FLOAT, position, Implementer.float.__name__, args)
        self.data_type = 'float'

    @property
    def is_leaf(self) -> bool: return True

    def execute_node(self, implementer: Implementer): return implementer.float(self)

    def analyze_node(self, implementer: Implementer): implementer.float(self)


# SYMBOL | ID
class SymbolNode(UnaryNode):
    def __init__(
            self,
            position: Position,
            args: List["AstNode"],
    ):
        super().__init__(NodeType.SYMBOL, position, Implementer.symbol.__name__, args)
        self.data_type = 'symbol'
        self.var_key = "".join(args)

    @property
    def is_leaf(self) -> bool:
        return True

    def execute_node(self, implementer: Implementer):
        state: ScopedState = self.get_state(implementer)

        # Is the symbol really just a scalar tensor ?
        if self.var_key == state.get_variable(Scope.MetricSymbol):
            metric_scalar = implementer.init_metric_scalar(self)
            state.set_variable(self.var_key, metric_scalar)
            return metric_scalar

        elif self.var_key == state.get_variable(Scope.RicciSymbol):
            ricci_scalar = implementer.init_ricci_scalar(self)
            state.set_variable(self.var_key, ricci_scalar)
            return ricci_scalar

        elif state.has_variable(self.var_key):
            return state.get_variable(self.var_key)

        else:
            return implementer.symbol(self)

    def analyze_node(self, implementer: Implementer):
        executor: Callable = self.get_executor(implementer)
        executor(self)
