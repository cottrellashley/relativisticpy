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
    ADD = "BINARY OP: Addition"
    SUB = "BINARY OP: Subtraction"
    MUL = "BINARY OP: Multiplication"
    DIV = "BINARY OP: Divition"
    POW = "BINARY OP: Exponentiation"
    AND = "BINARY OP: Boolean expression"  # Positive operator '+'

    INT = "UNARY OP: Integer cast"
    POS = "UNARY OP: Positve map"
    NEG = "UNARY OP: Negation map"
    NOT = "UNARY OP: Boolean NOT"

    LESS = "BINARY OP: Comparative less than"

    FLOAT = "UNARY OP: Float cast"  # A floating-point number
    ARRAY = "MULTI OP: Array builder"  # Array object '[elements]'
    EQUALS = "BINARY OP: Create assignment"  # The `=` symbol for assignment or comparison

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


@dataclass
class AstNode:
    """ Nested Node type. The whole object represents the Abstract Syntax Tree. """

    type: NodeType
    "Type of the object in node. (This can sometimes be the same as the token type in the node.)"

    position: Position
    "Position in which the node value is within the script/code."

    callback: str
    "Name of the method to be called by interpreter object to implement node."

    inferenced_type: str
    "The type which the callback will return when executed. This is an abstraction 'type' as in (tensor, int, float, array, symbol, symbol_expr, etc... )"

    args: List["AstNode"] # Could be our own data structure which wraps AstNode list. I.e. so we can call node.children for example.
    "Arguments of this node. AKA: Child nodes." 
