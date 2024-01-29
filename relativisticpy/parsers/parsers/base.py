from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Union

from typing import Iterable
from relativisticpy.parsers.lexers.base import LexerResult, Token
from relativisticpy.parsers.shared.errors import IllegalCharacterError, IllegalSyntaxError
from relativisticpy.parsers.shared.iterator import Iterator

from relativisticpy.parsers.types.gr_nodes import AstNode, Definition, Function, NodeType
from relativisticpy.parsers.types.position import Position


# 

@dataclass
class ParserResult:
    code: str
    ast_tree: List[AstNode]

# We can add very specific methods here - for the only purpose of making the parser files more readable
class BaseParser(ABC):
    """ Base class all parsers. """

    def __init__(self, lexer_result: LexerResult):
        self.raw_code = lexer_result.code
        self.__tokens = Iterator(lexer_result.tokens)
        self.__tokens.advance()

    @property
    def iter_tokens(self) -> Iterable: return self.__tokens
    @property
    def current_token(self) -> Token: return self.__tokens.current()

    def peek(self, n: int, default: any = None) -> Union[Token, None]: return self.__tokens.peek(n, default)
    def advance_token(self) -> None: self.__tokens.advance()
    
    @abstractmethod
    def parse() -> ParserResult: pass


    def invalid_syntax_error(self, details: str, pos_start: Position, pos_end: Position, raw_code: str):
        return IllegalSyntaxError(
            pos_start,
            pos_end,
            details,
            raw_code
        )
    
    def illegal_character_error(self, details: str, pos_start: Position, pos_end: Position, raw_code: str):
        return IllegalCharacterError(
            pos_start,
            pos_end,
            details,
            raw_code
        )
    
    ########################## !!!!!!! NOTE !!!!!!! ############################
    # The following methods are only here to make the actual parser much easier to read.
    # From a code perspective, the base class should never know the implementation of classese
    # which inherit it. 

    ###################################
    ## SIMPLE ARITHMETIC NODES
    ###################################
    def new_add_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.ADD,
            position=position,
            callback='add',
            inferenced_type=None,
            args=args
        )
    
    def new_sub_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.SUB,
            position=position,
            callback='sub',
            inferenced_type=None,
            args=args
        )
    
    def new_mul_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.MUL,
            position=position,
            callback='mul',
            inferenced_type=None,
            args=args
        )
    
    def new_div_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.DIV,
            position=position,
            callback='div',
            inferenced_type=None,
            args=args
        )
    
    def new_pow_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.POW,
            position=position,
            callback='pow',
            inferenced_type=None,
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
            inferenced_type='int',
            args=args
        )
    
    def new_float_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.FLOAT,
            position=position,
            callback='float',
            inferenced_type='float',
            args=args
        )
    
    def new_symbol_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.SYMBOL,
            position=position,
            callback='symbol',
            inferenced_type='symbol',
            args=args
        )
    
    def new_neg_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.NEG,
            position=position,
            callback='neg',
            inferenced_type=None,
            args=args
        )

    def new_pos_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.POS,
            position=position,
            callback='pos',
            inferenced_type=None,
            args=args
        )
    
    def new_array_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.ARRAY,
            position=position,
            callback='array',
            inferenced_type='array',
            args=args
        )
    
    def new_tensor_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.TENSOR,
            position=position,
            callback='tensor',
            inferenced_type='tensor',
            args=args
        )
    
    def new_not_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.NOT,
            position=position,
            callback='not_',
            inferenced_type=None,
            args=args
        )
    
    def new_and_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.AND,
            position=position,
            callback='and_',
            inferenced_type=None,
            args=args
        )
    
    def new_or_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.OR,
            position=position,
            callback='or',
            inferenced_type=None,
            args=args
        )
    
    def new_eqequal_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.EQEQUAL,
            position=position,
            callback='eqequal',
            inferenced_type=None,
            args=args
        )
    
    def new_less_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.LESS,
            position=position,
            callback='less',
            inferenced_type=None,
            args=args
        )
    
    def new_greater_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.GREATER,
            position=position,
            callback='greater',
            inferenced_type=None,
            args=args
        )
    
    def new_lessequal_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.LESSEQUAL,
            position=position,
            callback='lessequal',
            inferenced_type=None,
            args=args
        )
    
    def new_greaterequal_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.GREATEREQUAL,
            position=position,
            callback='greaterequal',
            inferenced_type=None,
            args=args
        )
    
    def new_assignment_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.ASSIGNMENT,
            position=position,
            callback='assignment',
            inferenced_type=None,
            args=args
        )
    
    def new_definition_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return Definition(
            position=position,
            args=args
        )
    
    def new_tensor_comp_assignment_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.TENSOR_COMPONENT_ASSIGNMENT,
            position=position,
            callback='tensor_component_assignment',
            inferenced_type=None,
            args=args
        )
    
    def new_tensor_expr_assignment_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.TENSOR_EXPR_ASSIGNMENT,
            position=position,
            callback='tensor_expr_assignment',
            inferenced_type=None,
            args=args
        )
    
    def new_function_def_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.FUNCTION_DEF,
            position=position,
            callback='function_def',
            inferenced_type=None,
            args=args
        )
    
    def new_print_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.PRINT,
            position=position,
            callback='print_',
            inferenced_type=None,
            args=args
        )
    
    def new_tensor_comp_definition_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.TENSOR_COMPONENT_DEFINITION,
            position=position,
            callback='tensor_definition',
            inferenced_type=None,
            args=args
        )