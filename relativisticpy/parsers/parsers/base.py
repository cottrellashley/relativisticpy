from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Union

from typing import Iterable
from relativisticpy.parsers.lexers.base import LexerResult, Token, TokenType
from relativisticpy.parsers.shared.errors import IllegalCharacterError, IllegalSyntaxError
from relativisticpy.parsers.shared.iterator import Iterator

from relativisticpy.parsers.types.gr_nodes import AstNode, Definition, Function, NodeType, Tensor
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
    def peek_prev_token(self, ignore_NEWLINE: bool = False) -> Token:

        n = -1
        while self.peek(n, '').type == TokenType.NEWLINE and ignore_NEWLINE:
            n -= 1

        return self.peek(n, '')
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