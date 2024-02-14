from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Union

from typing import Iterable
from relativisticpy.parsers.lexers.base import LexerResult, Token, TokenType
from relativisticpy.parsers.shared.errors import IllegalCharacterError, IllegalSyntaxError
from relativisticpy.parsers.shared.iterator import Iterator

from relativisticpy.parsers.types.base import AstNode, UnaryNode, BinaryNode, ArrayNode, IntNode, FloatNode, SymbolNode, NegNode, PosNode, NotNode, PrintNode, AssignmentNode
from relativisticpy.parsers.types.gr_nodes import Definition, Function, NodeType, TensorNode
from relativisticpy.parsers.types.position import Position

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
    def peek_type(self, n: int) -> TokenType: return self.__tokens.peek(n, Token(TokenType.NONE)).type
    def peek_prev_token(self, ignore_NEWLINE: bool = False) -> Token:

        n = -1
        while self.peek(n, '').type == TokenType.NEWLINE and ignore_NEWLINE:
            n -= 1

        return self.peek(n, '')
    def advance_token(self) -> None: self.__tokens.advance()
    
    @abstractmethod
    def parse() -> ParserResult: pass

    def ignore_newlines(self):
        while ( self.current_token != None and self.current_token.type == TokenType.NEWLINE ): 
            self.advance_token()

    def confirm_tok_value(self, value: any, expected_value: any):
        if value != expected_value:
            return self.invalid_syntax_error(
                f"Value Error: {value} \n Expecting value: {expected_value}",
                self.current_token.start_position.copy(),
                self.current_token.end_position.copy(),
                self.raw_code,
            )

    def confirm_syntax(self, current_type: TokenType, expected_type: Union[TokenType, List[TokenType]]):
        ##### <<< NEEDS TO BE MODIFILED TO TAKE IN A TOKEN, NOT TOKENTYPE and THEN PERFORM A NULL CHECK.
        # The TOKEN should never be null when this function is called but I've has a few errors cause because of this.

        if isinstance(expected_type, list):
            if current_type not in expected_type:
                exp_str = " OR ".join([i.value for i in expected_type])
                return self.invalid_syntax_error(
                    f"Syntax Error with object: {current_type.value} \n Expecting one of the following: {exp_str}",
                    self.current_token.start_position.copy(),
                    self.current_token.end_position.copy(),
                    self.raw_code,
                )
        else:
            if current_type != expected_type:
                return self.invalid_syntax_error(
                    f"Syntax Error with object: {current_type.value} \n Expecting object: {expected_type.value}",
                    self.current_token.start_position.copy(),
                    self.current_token.end_position.copy(),
                    self.raw_code,
                )



    def invalid_syntax_error(self, details: str, pos_start: Position, pos_end: Position, raw_code: str):
        return IllegalSyntaxError(
            pos_start,
            pos_end,
            details,
            raw_code
        )
    
    def illegal_character_error(self, details: str):
        return IllegalCharacterError(
            self.current_token.start_position.copy(),
            self.current_token.end_position.copy(),
            details,
            self.raw_code
        )
    
    ########################## !!!!!!! NOTE !!!!!!! ############################
    # The following methods are only here to make the actual parser much easier to read.
    # From a code perspective, the base class should never know the implementation of classese
    # which inherit it. 

    def new_add_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.ADD,
            position=position,
            callback='add',
            args=args
        )
    
    def new_sub_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.SUB,
            position=position,
            callback='sub',
            args=args
        )
    
    def new_mul_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.MUL,
            position=position,
            callback='mul',
            args=args
        )
    
    def new_div_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.DIV,
            position=position,
            callback='div',
            args=args
        )
    
    def new_pow_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.POW,
            position=position,
            callback='pow',
            args=args
        )
    
    def new_constant_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return UnaryNode(
            type=NodeType.CONSTANT,
            position=position,
            callback='constant',
            args=args
        )
    
    ###################################
    ### SIMPLE DATA TYPE NODES 
    ###################################

    def new_int_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return IntNode(
            position=position,
            args=args
        )
    
    def new_float_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return FloatNode(
            position=position,
            args=args
        )
    
    def new_symbol_node(self, position: Position, args: List[AstNode]) -> AstNode:
        symbol_node = SymbolNode(
            position=position,
            args=args
        )
        symbol_node.data_type = 'symbol' # Technically we don't actually know if this is a symbol yet or not, as it could be a pointer to another object stored
        return symbol_node
    
    def new_neg_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return NegNode(
            position=position,
            args=args
        )

    def new_pos_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return PosNode(
            position=position,
            args=args
        )
    
    def new_array_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return ArrayNode(
            position=position,
            args=args
        )

    def new_not_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return NotNode(
            position=position,
            args=args
        )
    
    def new_and_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.AND,
            position=position,
            callback='and_',
            args=args
        )
    
    def new_or_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.OR,
            position=position,
            callback='or',
            args=args
        )
    
    def new_eqequal_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.EQEQUAL,
            position=position,
            callback='eqequal',
            args=args
        )
    
    def new_less_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.LESS,
            position=position,
            callback='less',
            args=args
        )
    
    def new_greater_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.GREATER,
            position=position,
            callback='greater',
            args=args
        )
    
    def new_lessequal_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.LESSEQUAL,
            position=position,
            callback='lessequal',
            args=args
        )
    
    def new_greaterequal_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.GREATEREQUAL,
            position=position,
            callback='greaterequal',
            args=args
        )
    
    def new_assignment_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AssignmentNode(
            position=position,
            args=args
        )
    
    def new_definition_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return Definition(
            position=position,
            args=args
        )
    
    def new_function_def_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.FUNCTION_DEF,
            position=position,
            callback='function_def',
            args=args
        )
    
    def new_print_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return PrintNode(
            position=position,
            args=args
        )
    