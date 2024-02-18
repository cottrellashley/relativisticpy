from abc import ABC, abstractproperty
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Union
from .position import Position
from relativisticpy.parsers.scope.state import ScopedState

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

    FUNCTION_DEF = "FUNCTION_DEF"

    PRINT = "PRINT"

    SYMBOL = "symbol" # symbol object
    SYMBOLFUNC = "symbolfunc" # function as a symbol .i.e undefined non-callable object

    TENSOR_EXPR_ASSIGNMENT = "TENSOR_EXPR_ASSIGNMENT"
    TENSOR_COMPONENT_ASSIGNMENT = "TENSOR_COMPONENT_ASSIGNMENT"
    TENSOR_COMPONENT_DEFINITION = "TENSOR_COMPONENT_DEFINITION"



class AstNode:
    """Nested Node type. The whole object represents the Abstract Syntax Tree."""

    def __init__(
        self,
        type: NodeType, # swtich name to node_type instead!
        position: Position,
        callback: str = None,
        args: List["AstNode"] = None,
    ):
        self.type = type
        self.position = position
        self._callback = callback
        self.parent = None
        self.args = args
        self.data_type = None

        for arg in self.args:
            if isinstance(arg, AstNode):
                arg.parent = self

    def execute_node(self, executor: Callable, state : ScopedState = None): 
        for i, arg in enumerate(self.args):
            self.args[i] = executor(arg)

    def analyze_node(self, analyzer: Callable, state : ScopedState = None): 
        """ Defined how this node is analyzed by Semantic Analyzer. Note: For most nodes this is same implementation as execute_node. """
        self.execute_node(analyzer, state)

    @property
    def is_root(self) -> bool:
        return self.parent == None

    @property
    def callback(self):
        return self._callback
          
    @callback.setter
    def callback(self, value):
        self._callback = value

    def remove_arg(self, node_rmv):
        self.args = [node for node in self.args if node is not node_rmv]

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
        if len(self.args) == 1:
            print(prefix + str(self.args[0]))
        else:
            print(prefix + self.type.value)
        if level != 0:
            for child in self.args:
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
    def operand(self):
        return self.args[0]
    
    def execute_node(self, executor: Callable, state = None): 
        self.args[0] = executor(self.args[0])


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
        super().__init__(type=NodeType.ARRAY, position=position, callback='array', args=args)
        self.data_type = 'array'

    @property
    def is_leaf(self) -> bool: return False

    @property
    def callback(self): return "array"

    @property
    def shape(self):
        "Compute the shape of the array"
        pass

    def execute_node(self, executor: Callable, state = None): 
        for i, arg in enumerate(self.args):
            self.args[i] = executor(arg)

# arith_expr = arith_expr
class AssignmentNode(BinaryNode):
    """ Arithmatic expression equality builds an object Eq() representing an equation. """
    def __init__(
        self,
        position: Position,
        args: List["AstNode"],
    ):
        super().__init__(type=NodeType.ASSIGNMENT, position=position, callback='assignment', args=args)
        self.data_type = 'none'

    @property
    def is_leaf(self) -> bool: return False

    @property
    def callback(self): return "assignment"


    def execute_node(self, executor: Callable, state = None): 
        self.args[1] = executor(self.args[1])

# INT
class IntNode(UnaryNode):
    def __init__(
        self,
        position: Position,
        args: List["AstNode"],
    ):
        super().__init__(NodeType.INT, position, 'int', args)
        self.data_type = 'int'


    @property
    def is_leaf(self) -> bool: return True

# FLOAT
class FloatNode(UnaryNode):
    def __init__(
        self,
        position: Position,
        args: List["AstNode"],
    ):
        super().__init__(NodeType.FLOAT, position, 'float', args)
        self.data_type = 'float'


    @property
    def is_leaf(self) -> bool: return True

# SYMBOL | ID
class SymbolNode(UnaryNode):
    def __init__(
        self,
        position: Position,
        args: List["AstNode"],
    ):
        super().__init__(NodeType.SYMBOL, position, 'symbol', args)
        self.data_type = 'symbol'

    @property
    def is_leaf(self) -> bool: return True

# - expr
class NegNode(UnaryNode):
    def __init__(
        self,
        position: Position,
        args: List["AstNode"],
    ):
        super().__init__(NodeType.NEG, position, 'neg', args)

    @property
    def is_leaf(self) -> bool: return False

# + expr
class PosNode(UnaryNode):
    def __init__(
        self,
        position: Position,
        args: List["AstNode"],
    ):
        super().__init__(NodeType.POS, position, 'pos', args)

    @property
    def is_leaf(self) -> bool: return False

# not | !
class NotNode(UnaryNode):
    def __init__(
        self,
        position: Position,
        args: List["AstNode"],
    ):
        super().__init__(NodeType.NOT, position, 'not_', args)

    @property
    def is_leaf(self) -> bool: return False

# print
class PrintNode(UnaryNode):
    def __init__(
        self,
        position: Position,
        args: List["AstNode"],
    ):
        super().__init__(NodeType.PRINT, position, 'print_', args)

    @property
    def is_leaf(self) -> bool: return False

# d expr | d() | d{}
class Infinitesimal(UnaryNode):
    def __init__(
        self,
        position: Position,
        args: List["AstNode"],
    ):
        super().__init__(NodeType.INFINITESIMAL, position, 'infinitesimal', args)
        self.diff_order : int = None
        self.expression = None
        self.is_partial : bool = None
        self.diff_order_as_int : int = None

    @property
    def is_leaf(self) -> bool: return False

# ID :=
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

    def execute_node(self, executor: Callable, scope: ScopedState = None):
        scope.set_variable("".join(self.args[0]), executor(self.args[1]))

    @property
    def callback(self): return Definition.PRE_DEFINED[self.args[0]] if self.args[0] in Definition.PRE_DEFINED else "definition"

class Def(AstNode):
    """ Node for the definition of a Function. """
    def __init__(
        self,
        identifier: str,
        body: AstNode,
        position: Position,
        args: List["AstNode"] = None
    ):
        super().__init__(NodeType.FUNCTION_DEF, position, 'function_def', args)
        self.data_type = "none"
        self.identifier = identifier
        self.str_args = []
        self.body = body

    def execute_node(self, executor: Callable, state: ScopedState) -> None:
        state.set_function(self.identifier, self)

    @property
    def is_leaf(self): return False

class Call(AstNode):
    """ Node for the call of a Function. Defaults to returing a Funbol, unless user actually defines a exe expression. """
    BUILT_INS = (
                    "diff", "simplify", "integrate", "expand", 
                    "diag", "lim", "solve", "dsolve", "subs", 
                    "LHS", "RHS", "tsimplify", "sum", "dosum","clear",
                    "sqrt", "func_derivative",
                    "prod", "doprod",
                    "sin", "cos", "tan", 
                    "asin", "atan", "acos", 
                    "cosh", "sinh", "tanh", 
                    "acosh", "asinh", "atanh"
                 )
    def __init__(
        self,
        identifier: str,
        position: Position,
        args: List["AstNode"] = None
    ):
        self.args = args
        self.position = position
        self.type = NodeType.CALL
        self.data_type = "undef"
        self.identifier = identifier
        self.is_built_in : bool = self.identifier in self.BUILT_INS
        self.call_return = None

    def execute_node(self, executor: Callable, state: ScopedState) -> None:
        # Does the function we are calling exist in the function stack in our current scope?
        func_def : Def = state.get_function(self.identifier)

        if func_def:
            # First push a new scope in the state.
            state.push_scope()

            # We now set the arguments to the object passed into the call. i.e. f(x, t) called as -> f( n**2 + 9 , 10 ) => set: 'x' :  n**2 + 9 and 't' : 10
            for i, arg in enumerate(self.args):
                state.set_variable(func_def.str_args[i], executor(arg))

            # Finally we execute the body of the function.
            if isinstance(func_def.body, AstNode):
                self.call_return = executor(func_def.body)
            elif isinstance(func_def.body, list):
                return_objs = []
                for statement in func_def.body:
                    obj = executor(statement)
                    if obj != None:
                        return_objs.append(obj)
                self.call_return = return_objs[-1] if len(return_objs) != 0 else None
                    

            state.pop_scope()
        else:
            for i, arg in enumerate(self.args):
                self.args[i] = executor(arg)

    @property
    def is_leaf(self): return False

    @property
    def callback(self) -> str:
        if self.identifier in self.BUILT_INS:
            return self.identifier
        elif self.call_return == None:
            return 'symbolfunc'
        return "call"

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