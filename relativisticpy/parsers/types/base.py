from abc import ABC, abstractproperty
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Union
from .position import Position


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
    INFINITESIMAL = "infinitesimal"

    FUNCTION_DEF = "FUNCTION_DEF"

    PRINT = "PRINT"

    SYMBOL = "symbol"

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

    def execute_node(self, executor: Callable): 
        for i, arg in enumerate(self.args):
            self.args[i] = executor(arg)

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
    
    def execute_node(self, executor: Callable):
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

    def execute_node(self, executor: Callable): 
        for i, arg in enumerate(self.args):
            self.args[i] = executor(arg)


class AssignmentNode(BinaryNode):
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


    def execute_node(self, executor: Callable): 
        self.args[1] = executor(self.args[1])


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

class NegNode(UnaryNode):
    def __init__(
        self,
        position: Position,
        args: List["AstNode"],
    ):
        super().__init__(NodeType.NEG, position, 'neg', args)

    @property
    def is_leaf(self) -> bool: return False

class PosNode(UnaryNode):
    def __init__(
        self,
        position: Position,
        args: List["AstNode"],
    ):
        super().__init__(NodeType.POS, position, 'pos', args)

    @property
    def is_leaf(self) -> bool: return False

class NotNode(UnaryNode):
    def __init__(
        self,
        position: Position,
        args: List["AstNode"],
    ):
        super().__init__(NodeType.NOT, position, 'not_', args)

    @property
    def is_leaf(self) -> bool: return False

class PrintNode(UnaryNode):
    def __init__(
        self,
        position: Position,
        args: List["AstNode"],
    ):
        super().__init__(NodeType.PRINT, position, 'print_', args)

    @property
    def is_leaf(self) -> bool: return False

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
