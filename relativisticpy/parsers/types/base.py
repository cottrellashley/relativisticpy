from abc import ABC, abstractproperty
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Union
from .position import Position


class NodeType(Enum):
    """An enumeration of node types used by a parser."""

    OR = "BINARY OP: Boolean expression"
    ID = "UNARY OP: Value getter | Symbol creator" # If variable not created in memory, build a symbol.

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

    FUNCTION_DEF = "FUNCTION_DEF"

    PRINT = "PRINT"

    SYMBOL = "symbol"

    TENSOR_EXPR_ASSIGNMENT = "TENSOR_EXPR_ASSIGNMENT"
    TENSOR_COMPONENT_ASSIGNMENT = "TENSOR_COMPONENT_ASSIGNMENT"
    TENSOR_COMPONENT_DEFINITION = "TENSOR_COMPONENT_DEFINITION"


class AstNode:
    """ Nested Node type. The whole object represents the Abstract Syntax Tree. """

    def __init__(self, type: NodeType, position: Position, callback: str = None, inferenced_type: str = None, args: List["AstNode"] = None):
        self.type = type
        self.position = position
        self.callback = callback
        self.inferenced_type = inferenced_type
        self.parent = None
        self.args = args
        for arg in self.args:
            if isinstance(arg, AstNode):
                arg.parent = self
            

    def remove_arg(self, node_rmv):
        self.args = [node for node in self.args if node is not node_rmv]

    def get_level(self):
        level = 0
        p = self.parent
        while p:
            level += 1
            p : "AstNode" = p.parent
        return level

    def print_tree(self, level=-1):
        spaces = ' ' * self.get_level() * 3
        prefix = spaces + "|__" if self.parent else ""
        if len(self.args) == 1:
            print(prefix + str(self.args[0]))
        else:
            print(prefix + self.type.value)
        if level != 0:
            for child in self.args:
                if isinstance(child, AstNode):
                    child.print_tree(level-1)



class UnaryNode(AstNode):
     def __init__(self, type: NodeType, position: Position, callback: str, inferenced_type: str, args: List["AstNode"]):
          if len(args) != 1:
               raise ValueError("UnaryNode requires exactly one arguments")
          super().__init__(type, position, callback, inferenced_type, args)
        
     @property
     def operand(self):
          return self.args[0]

class BinaryNode(AstNode):
     
     def __init__(self, type: NodeType, position: Position, callback: str, args: List["AstNode"]):
        if len(args) != 2:
             raise ValueError("BinaryNode requires exactly two arguments")
        super().__init__(type, position, callback, None, args)

     @property
     def left_child(self) -> AstNode:
          return self.args[0]
     
     @property
     def right_child(self) -> AstNode:
          return self.args[1]

class ArrayNode(AstNode):

     def __inti__(self, position: Position, callback: str, inferenced_type: str, args: List["AstNode"]):
          super().__init__(NodeType.ARRAY, position, callback, inferenced_type, args)

     @property
     def shape(self):
         "Compute the shape of the array"
         pass
     