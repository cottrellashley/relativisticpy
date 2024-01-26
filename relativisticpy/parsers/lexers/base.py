from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, Dict
from relativisticpy.parsers.shared.interfaces.tokens import ITokenProvider
from relativisticpy.parsers.shared.models.error import Error
from typing import Iterable
from relativisticpy.parsers.shared.models.position import Position
from relativisticpy.parsers.shared.models.token import Token
from dataclasses import dataclass

def enum__contains(cls):
    """ Enum.has_value(value) -> True or False -> if Enum contains that value. """
    @classmethod
    def has_value(cls, value): return any(value == item.value for item in cls)
    cls.has_value = has_value
    return cls

class TokenType(Enum):
    """An enumeration of token types used by a the lexer to genrate tokens."""

    NONE = "NONE"
    NEWLINE = "\n"

    # Single Charater Tokens
    PLUS = "+"
    MINUS = "-"
    STAR = "*" 
    SLASH = "/" 
    BACKSLASH = "\\"
    EQUAL = "=" 
    LPAR = "(" 
    RPAR = ")" 
    COMMA = "," 
    LSQB = "["
    RSQB = "]" 
    DOT = "."
    CIRCUMFLEX = "^" 
    UNDER = "_" 
    EXCLAMATION = "!" 
    PERCENT = "%" 
    AMPER = "&" 
    COLON = ":" 
    SEMI = ";" 
    LESS = "<" 
    GREATER = ">" 
    AT = "@" 
    LBRACE = "{"
    RBRACE = "}"  
    VBAR = "|"
    TILDE = "~"

    # Double Character Tokens
    NOTEQUAL = "!="
    PERCENTEQUAL = "%="
    AMPEREQUAL = "&="
    AMPERAMPER = "&&"
    DOUBLESTAR = "**"
    STAREQUAL = "*="
    PLUSEQUAL = "+="
    MINEQUAL = "-="
    RARROW = ">>"
    DOUBLESLASH = "//"
    SLASHEQUAL = "/="
    COLONEQUAL = ":="
    LEFTSHIFT = "<<"
    LESSEQUAL = "<="
    EQEQUAL = "=="
    GREATEREQUAL = ">="
    RIGHTSHIFT = ">>"
    ATEQUAL = "@="
    CIRCUMFLEXEQUAL = "^="
    VBAREQUAL = "|="
    VBARVBAR = "||"

    # Tripple Character Tokens
    DOUBLESTAREQUAL = "**="
    ELLIPSIS = "..."
    DOUBLESLASHEQUAL = "//="
    LEFTSHIFTEQUAL = "<<="
    RIGHTSHIFTEQUAL = ">>="

    FLOAT = "FLOAT"
    INT = "INT"
    TENSORID = "TENSORID"
    FUNCTIONID = "FUNCTIONID"
    LATEXID = "LATEXID"
    ID = "ID"
    STRING = "STRING"
    BOOL = "BOOL"

    # KEYWORDS
    NOT = 'not'
    AND = 'and'
    OR = 'or'
    PRINT = 'print'

    @classmethod
    def KEYWORDS(cls):
        return {
                'not': cls.NOT, 
                'and': cls.AND, 
                'or': cls.OR,
                'print': cls.PRINT
                }


    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)

class Characters(Enum):
    """An object containing the set of supported characters, keyd on a set name."""

    WHITESPACE = " \t"
    NEWLINE = '\n'
    DELINIMATORS = ";" + NEWLINE
    COMMENT = "#"
    DIGITS = "0987654321"
    LOWERCASECHARACTERS = "abcdefghijklmnopqrstuvwxyz"
    UPPERCASECHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    NONLETTERCHARACTERS = "_{}:^;"
    OPERATIONS = "*-+=[](){}^/|&!~><:;.,@%_"
    LETTERS = LOWERCASECHARACTERS + UPPERCASECHARACTERS
    CHARACTERS = LOWERCASECHARACTERS + UPPERCASECHARACTERS + NONLETTERCHARACTERS
    IDENTIFIERCHARS = ( LOWERCASECHARACTERS + UPPERCASECHARACTERS + DIGITS )


@dataclass
class Token:
    type: TokenType = None
    value: str = None
    start_position: Position = None
    end_position: Position = None


class TokenProvider(ITokenProvider):
    def __init__(self):
        self.tokens = []

    def new_token(self, type: TokenType, value: str, start_pos: Position, end_pos: Position) -> None:
        self.tokens.append(Token(type, value, start_pos, end_pos))

    def get_tokens(self):
        return self.tokens

    def new_single_operation_token(self, c1: str, start_pos: Position, end_pos: Position) -> None:
        if self.single_match_exists(c1):
            token_type : TokenType = self.singles[c1]
            self.tokens.append(Token(token_type, token_type.value, start_pos, end_pos))

    def new_double_operation_token(self, c1: str, c2: str, start_pos: Position, end_pos: Position) -> None:
        if self.double_match_exists(c1, c2):
            token_type : TokenType = self.doubles[c1][c2]
            self.tokens.append(Token(token_type, token_type.value, start_pos, end_pos))

    def new_tripple_operation_token(self, c1: str, c2: str, c3: str, start_pos: Position, end_pos: Position) -> None:
        if self.tripple_match_exists(c1, c2, c3):
            token_type : TokenType = self.tripples[c1][c2][c3]
            self.tokens.append(Token(token_type, token_type.value, start_pos, end_pos))

    def single_match_exists(self, c1: str) -> bool:
        try:
            self.singles[c1]
            return True
        except:
            return False

    def double_match_exists(self, c1: str, c2: str) -> bool:
        try:
            self.doubles[c1][c2]
            return True
        except:
            return False

    def tripple_match_exists(self, c1: str, c2: str, c3: str) -> bool:
        try:
            self.tripples[c1][c2][c3]
            return True
        except:
            return False

    @property
    def singles(self):
        return {
            "*": TokenType.STAR,
            "-": TokenType.MINUS,
            "+": TokenType.PLUS,
            "=": TokenType.EQUAL, 
            "[": TokenType.LSQB,
            "]": TokenType.RSQB,
            "(": TokenType.LPAR,
            ")": TokenType.RPAR, 
            "{": TokenType.LBRACE, 
            "}": TokenType.RBRACE, 
            "^": TokenType.CIRCUMFLEX, 
            "/": TokenType.SLASH,
            "|": TokenType.VBAR,
            "&": TokenType.AMPER, 
            "!": TokenType.EXCLAMATION,
            "~": TokenType.TILDE, 
            ">": TokenType.GREATER, 
            "<": TokenType.LESS, 
            ":": TokenType.COLON, 
            ".": TokenType.DOT,
            ",": TokenType.COMMA, 
            ";": TokenType.SEMI,
            "@": TokenType.AT, 
            "%": TokenType.PERCENT,
            ":": TokenType.COLON,
            "_": TokenType.UNDER,
            "{": TokenType.LBRACE,
            "}": TokenType.RBRACE
        }

    @property
    def doubles(self):
        return {
            "!": {"=": TokenType.NOTEQUAL},
            "%": {"=": TokenType.PERCENTEQUAL},
            "&": {"=": TokenType.AMPEREQUAL},
            "+": {"=": TokenType.PLUSEQUAL},
            ":": {"=": TokenType.COLONEQUAL},
            "=": {"=": TokenType.EQEQUAL},
            "|": {"|": TokenType.VBARVBAR},
            "@": {"=": TokenType.ATEQUAL},
            "^": {"=": TokenType.CIRCUMFLEXEQUAL},
            "|": {"=": TokenType.VBAREQUAL},
            "*": {
                "*": TokenType.DOUBLESTAR,
                "=": TokenType.STAREQUAL,
            },
            ">": {
                "=": TokenType.GREATEREQUAL,
                ">": TokenType.RIGHTSHIFT,
            },
            "/": {
                "/": TokenType.DOUBLESLASH,
                "=": TokenType.SLASHEQUAL,
            },
            "-": {
                "=": TokenType.MINEQUAL,
                ">": TokenType.RARROW,
            },
            "<": {
                "=": TokenType.LESSEQUAL,
                "<": TokenType.LEFTSHIFTEQUAL,
                ">": TokenType.NOTEQUAL,
            }
        }
    
    @property
    def tripples(self):
        return {
            "*": {"*": {"=": TokenType.DOUBLESTAREQUAL}},
            ".": {".": {".": TokenType.ELLIPSIS}},
            "/": {"/": {"=": TokenType.DOUBLESLASHEQUAL}},
            "<": {"<": {"=": TokenType.LEFTSHIFTEQUAL}},
            ">": {">": {"=": TokenType.RIGHTSHIFTEQUAL}}
        }


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



class BaseLexer(ABC):
    "Base Lexer all lexer types within this module should inherit."

    def __init__(self, string: str):
        self.__characters = Iterator(string)
        self.__token_provider_instance = TokenProvider()
        self.character = 0
        self.line = 0
        self.__characters.advance()
        

    @abstractmethod
    def tokenize(): pass

    @property
    def token_provider(self) -> TokenProvider: return self.__token_provider_instance

    def current_char(self) -> str: return self.__characters.current()
    def advance_char(self) -> None:
        self.character += 1

        # Iterate the character and postion
        if self.current_char() == '\n':
            self.line += 1
            self.character = 0
        
        self.__characters.advance()

    def peek_char(self, n: int, default: any = None) -> Union[str, None]: return self.__characters.peek(n, default)
    def raise_error(self, error: Error): raise Exception(error.message)
    def current_pos(self) -> Position: return Position(self.line, self.character)

