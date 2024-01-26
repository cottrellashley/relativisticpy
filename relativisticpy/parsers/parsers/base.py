from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Union

from typing import Iterable
from relativisticpy.parsers.lexers.base import Token
from relativisticpy.parsers.shared.models.error import Error
from relativisticpy.parsers.shared.models.position import Position

class NodeType(Enum):
    """An enumeration of node types used by a parser."""

    ADD = "+"  # Addition operator
    SUB = "-"  # Subtraction operator
    MUL = "*"  # Multiplication operator
    DIV = "/"  # Division operator
    POW = "POW"
    INT = "int"
    POS = "positive"  # Positive operator '+'
    NEG = "negative"
    NOT = "NOT"
    FLOAT = "float"  # A floating-point number


    LESS = "<"
    LESSEQUAL = "<="
    GREATER = ">"
    GREATEREQUAL = ">="
    ASSIGNMENT = "="
    DEFINITION = ":="
    EQEQUAL = "=="
    NOTEQUAL = "!="

    EQUALS = "="  # The `=` symbol for assignment or comparison
    EXPONENTIATION1 = "^"  # The `**` symbol for exponentiation
    EXPONENTIATION2 = "**"
    OBJECT = "object"  # A variable name
    VARIABLEKEY = "variable_key"  # A variable name

    FUNCTION = "function"  # A function name
    FUNCTION_DEF = "FUNCTION_DEF"

    ARRAY = "array"  # Array object '[elements]'
    AND = "&"  # Positive operator '+'
    OR = "|"  # Positive operator '+'

    TENSOR_EXPR_ASSIGNMENT = "TENSOR_EXPR_ASSIGNMENT"
    TENSOR_COMPONENT_ASSIGNMENT = "TENSOR_COMPONENT_ASSIGNMENT"
    TENSOR_COMPONENT_DEFINITION = "TENSOR_COMPONENT_DEFINITION"

    TENSOR_KEY = "tensor_key"
    PRINT = "PRINT"
    SYMBOL_DEFINITION = "symbol_definition"
    SYMBOL_KEY = "symbol_key"
    SYMBOL = "symbol"
    TENSOR = "tensor"
    ID = "ID"


@dataclass
class AstNode:
    """ Nested Node type. The whole object represents the Abstract Syntax Tree. """

    type: NodeType
    "Type of the object in node. (This can sometimes be the same as the token type in the node.)"

    position: Position
    "Position in which the node value is within the script/code."

    callback: str
    "Name of the method to be called by interpreter object to implement node."

    args: List["AstNode"] # Could be our own data structure which wraps AstNode list. I.e. so we can call node.children for example.
    "Arguments of this node. AKA: Child nodes." 


class Iterator:
    def __init__(self, object: Iterable):
        self.object = object
        if isinstance(self.object, Iterable):
            self.iterable_object = iter(object)
            self.current_item_location = -1

    def advance(self):
        self.current_item = next(self.iterable_object, None)
        if self.current_item != None:
            self.current_item_location += 1

    def peek(self, n: int = 1, default=None):
        i = self.current_item_location
        if i + n < len(self.object):
            return self.object[i + n]
        else:
            return default

    def __len__(self):
        return len(self.object)

    def current(self):
        return self.current_item

class NodeProvider :

    def new_node(self, node: str, type: str, value: str, position: Tuple[Position, Position], callback: str, args:  List[AstNode]):
        return AstNode(node, type, value, position, callback, args)


# We can add very specific methods here - for the only purpose of making the parser files more readable
class BaseParser(ABC):
    """ Base class all parsers. """

    def __init__(self, tokens: List[Token]):
        self.__tokens = Iterator(tokens)
        self.__node_provider_instance = NodeProvider()
        self.__tokens.advance()

    @property
    def iter_tokens(self) -> Iterable: return self.__tokens
    @property
    def current_token(self) -> Token: return self.__tokens.current()
    @property
    def node_provider(self) -> NodeProvider: return self.__node_provider_instance

    def peek(self, n: int, default: any = None) -> Union[Token, None]: return self.__tokens.peek(n, default)
    def advance_token(self) -> None: self.__tokens.advance()
    
    @abstractmethod
    def parse() -> AstNode:
        pass

    def raise_error(self, error: Error):
        # 1. concatinate all tokens to recreate the string intup
        # 2. generate an error message, include the section of the string in which the error was generated
        # 3. return the error object
        raise Exception(error.message)
    
    ############
    # The following methods are only here to make the actual parser much easier to read.

    ###################################
    ## SIMPLE ARITHMETIC NODES
    ###################################
    def new_add_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.ADD,
            position=position,
            callback='add',
            args=args
        )
    
    def new_sub_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.SUB,
            position=position,
            callback='sub',
            args=args
        )
    
    def new_mul_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.MUL,
            position=position,
            callback='mul',
            args=args
        )
    
    def new_div_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.DIV,
            position=position,
            callback='div',
            args=args
        )
    
    def new_pow_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.POW,
            position=position,
            callback='pow',
            args=args
        )
    
    ###################################
    ### SIMPLE DATA TYPE NODES 
    ###################################

    def new_int_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.INT,
            position=position,
            callback='int',
            args=args
        )
    
    def new_float_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.FLOAT,
            position=position,
            callback='float',
            args=args
        )
    
    def new_symbol_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.SYMBOL,
            position=position,
            callback='sym',
            args=args
        )
    
    def new_neg_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.NEG,
            position=position,
            callback='neg',
            args=args
        )

    def new_pos_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.POS,
            position=position,
            callback='pos',
            args=args
        )
    
    def new_array_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.ARRAY,
            position=position,
            callback='array',
            args=args
        )
    
    def new_function_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.FUNCTION,
            position=position,
            callback='function',
            args=args
        )
    
    def new_tensor_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.TENSOR,
            position=position,
            callback='tensor',
            args=args
        )
    
    def new_not_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.NOT,
            position=position,
            callback='not',
            args=args
        )
    
    def new_and_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.AND,
            position=position,
            callback='and',
            args=args
        )
    
    def new_or_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.OR,
            position=position,
            callback='or',
            args=args
        )
    
    def new_eqequal_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.EQEQUAL,
            position=position,
            callback='eqequal',
            args=args
        )
    
    def new_less_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.LESS,
            position=position,
            callback='less',
            args=args
        )
    
    def new_greater_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.GREATER,
            position=position,
            callback='greater',
            args=args
        )
    
    def new_lessequal_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.LESSEQUAL,
            position=position,
            callback='lessequal',
            args=args
        )
    
    def new_greaterequal_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.GREATEREQUAL,
            position=position,
            callback='greaterequal',
            args=args
        )
    
    def new_assignment_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.ASSIGNMENT,
            position=position,
            callback='assignment',
            args=args
        )
    
    def new_definition_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.DEFINITION,
            position=position,
            callback='definition',
            args=args
        )
    
    def new_tensor_comp_assignment_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.TENSOR_COMPONENT_ASSIGNMENT,
            position=position,
            callback='tensor_component_assignment',
            args=args
        )
    
    def new_tensor_expr_assignment_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.TENSOR_EXPR_ASSIGNMENT,
            position=position,
            callback='tensor_expr_assignment',
            args=args
        )
    
    def new_function_def_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.FUNCTION_DEF,
            position=position,
            callback='function_def',
            args=args
        )
    
    def new_print_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.PRINT,
            position=position,
            callback='print',
            args=args
        )
    
    def new_tensor_comp_definition_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.TENSOR_COMPONENT_DEFINITION,
            position=position,
            callback='tensor_definition',
            args=args
        )