from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Union, Dict
from relativisticpy.parsers.shared.errors import Error
from relativisticpy.parsers.shared.iterator import Iterator

from dataclasses import dataclass

from relativisticpy.parsers.types.position import Position


def enum__contains(cls):
    """Enum.has_value(value) -> True or False -> if Enum contains that value."""

    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)

    cls.has_value = has_value
    return cls


class TokenType(Enum):
    """An enumeration of token types used by a the lexer to genrate tokens."""

    NONE = "NONE"
    NEWLINE = "\n"
    BACKSLASH = "\\"
    DOUBLEBACKSLASH = "\\\\"

    ##########################################
    ############    OPERATIONS    ############ 
    ##########################################

    # Single Charater Tokens
    PLUS = "+"
    MINUS = "-"
    STAR = "*"
    SLASH = "/"
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
    RARROW = "->"
    LARROW = "<-"
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

    ##########################################
    ############       TYPES      ############ 
    ##########################################

    FLOAT = "FLOAT"
    INT = "INT"
    ID = "ID"
    STRING = "STRING"
    BOOL = "BOOL"

    ##########################################
    ############     KEYWORDS     ############ 
    ##########################################

    NOT = "not"
    AND = "and"
    OR = "or"
    PRINT = "print"

    @classmethod
    def KEYWORDS(cls):
        return {"not": cls.NOT, "and": cls.AND, "or": cls.OR, "print": cls.PRINT}

    ##########################################
    ############   Latex Symbols  ############ 
    ##########################################

    SYMBOL = "SYMBOL"
    @classmethod
    def LATEX_SYMBOLS(cls):
        return [
            "alpha",
            "Alpha",
            "beta",
            "Beta",
            "gamma",
            "Gamma",
            "delta",
            "Delta",
            "epsilon",
            "Epsilon",
            "zeta",
            "Zeta",
            "eta",
            "Eta",
            "theta",
            "Theta",
            "iota",
            "Iota",
            "kappa",
            "Kappa",
            "lambda",
            "Lambda",
            "mu",
            "Mu",
            "nu",
            "Nu",
            "xi",
            "Xi",
            "omicron",
            "Omicron",
            "pi",
            "Pi",
            "rho",
            "Rho",
            "sigma",
            "Sigma",
            "tau",
            "Tau",
            "upsilon",
            "Upsilon",
            "phi",
            "Phi",
            "chi",
            "Chi",
            "psi",
            "Psi",
            "omega",
            "Omega",
            "infty",
            "e"
        ]

    ##########################################
    ############ Latex Operations ############ 
    ##########################################

    LATEXNEWLINE = "newline"
    SUM = "sum"
    LIMIT = "lim"
    FRAC = "frac"
    BEGIN = "begin"
    END = "end"
    DOSUM = "dosum"
    TO = "to"
    RIGHTARROW = "rightarrow"
    LEFTARROW = "leftarrow"
    PROD = "prod"
    DOPROD = "doprod"
    EQUIVALENT = "equiv"

    @classmethod
    def LATEX_OPERATIONS(cls):
        return {
            cls.SUM.value: cls.SUM,
            cls.DOSUM.value: cls.DOSUM,
            cls.PROD.value: cls.PROD,
            cls.DOPROD.value: cls.DOPROD,
            cls.LIMIT.value: cls.LIMIT,
            cls.FRAC.value: cls.FRAC,
            cls.BEGIN.value: cls.BEGIN,
            cls.END.value: cls.END,
            cls.TO.value: cls.RARROW,
            cls.RIGHTARROW.value: cls.RARROW,
            cls.LEFTARROW.value: cls.LARROW,
            cls.LATEXNEWLINE.value: cls.NEWLINE,
            cls.EQUIVALENT.value: cls.COLONEQUAL
        }


class Characters(Enum):
    """An object containing the sets of supported characters for RelativisticPy. Key'd on set name."""

    WHITESPACE = " \t"
    NEWLINE = "\n"
    DELINIMATORS = ";" + NEWLINE
    COMMENT = "#"

    OPERATIONS = "*-+=[](){}^/|&!~><:;.,@%_"

    DIGITS = "0987654321"
    NUMBER = DIGITS + "."

    LOWER = "abcdefghijklmnopqrstuvwxyz"
    UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    LETTERS = LOWER + UPPER
    IDS = LETTERS + DIGITS

@dataclass
class Token:
    type: TokenType = None
    value: str = None
    start_position: Position = None
    end_position: Position = None


class TokenProvider:
    """  """

    def __init__(self):
        self.tokens = []

    def new_token(
        self, type: TokenType, value: str, start_pos: Position, end_pos: Position
    ) -> None:
        self.tokens.append(Token(type, value, start_pos, end_pos))

    def get_tokens(self):
        return self.tokens

    def new_single_operation_token(
        self, c1: str, start_pos: Position, end_pos: Position
    ) -> None:
        if self.single_match_exists(c1):
            token_type: TokenType = self.singles()[c1]
            self.tokens.append(Token(token_type, token_type.value, start_pos, end_pos))

    def new_double_operation_token(
        self, c1: str, c2: str, start_pos: Position, end_pos: Position
    ) -> None:
        if self.double_match_exists(c1, c2):
            token_type: TokenType = self.doubles()[c1][c2]
            self.tokens.append(Token(token_type, token_type.value, start_pos, end_pos))

    def new_tripple_operation_token(
        self, c1: str, c2: str, c3: str, start_pos: Position, end_pos: Position
    ) -> None:
        if self.tripple_match_exists(c1, c2, c3):
            token_type: TokenType = self.tripples()[c1][c2][c3]
            self.tokens.append(Token(token_type, token_type.value, start_pos, end_pos))

    def single_match_exists(self, c1: str) -> bool:
        try:
            self.singles()[c1]
            return True
        except:
            return False

    def double_match_exists(self, c1: str, c2: str) -> bool:
        try:
            self.doubles()[c1][c2]
            return True
        except:
            return False

    def tripple_match_exists(self, c1: str, c2: str, c3: str) -> bool:
        try:
            self.tripples()[c1][c2][c3]
            return True
        except:
            return False

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
            "}": TokenType.RBRACE,
        }

    def doubles(self):
        return {
            "!": {"=": TokenType.NOTEQUAL},
            "%": {"=": TokenType.PERCENTEQUAL},
            "&": {"=": TokenType.AMPEREQUAL},
            "+": {"=": TokenType.PLUSEQUAL},
            ":": {"=": TokenType.COLONEQUAL},
            "=": {"=": TokenType.EQEQUAL},
            "@": {"=": TokenType.ATEQUAL},
            "^": {"=": TokenType.CIRCUMFLEXEQUAL},
            "|": {"=": TokenType.VBAREQUAL, "|": TokenType.VBARVBAR},
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
                "-": TokenType.LARROW,
                "=": TokenType.LESSEQUAL,
                "<": TokenType.LEFTSHIFTEQUAL,
                ">": TokenType.NOTEQUAL,
            },
        }

    def tripples(self):
        return {
            "*": {"*": {"=": TokenType.DOUBLESTAREQUAL}},
            ".": {".": {".": TokenType.ELLIPSIS}},
            "/": {"/": {"=": TokenType.DOUBLESLASHEQUAL}},
            "<": {"<": {"=": TokenType.LEFTSHIFTEQUAL}},
            ">": {">": {"=": TokenType.RIGHTSHIFTEQUAL}},
        }


@dataclass
class LexerResult:
    code: str
    tokens: List[Token]


class BaseLexer(ABC):
    "Base Lexer all lexer types within this module should inherit."

    def __init__(self, string: str):
        self.raw_code = string
        self.__characters = Iterator(string)
        self.__token_provider_instance = TokenProvider()
        self.character = 0
        self.line = 1
        self.__characters.advance()

    @abstractmethod
    def tokenize() -> LexerResult:
        pass

    @property
    def token_provider(self) -> TokenProvider:
        return self.__token_provider_instance

    def current_char(self) -> str:
        return self.__characters.current()

    def advance_char(self) -> None:
        self.character += 1

        # Iterate the character and postion
        if self.current_char() == "\n":
            self.line += 1
            self.character = 0

        self.__characters.advance()

    def peek_char(self, n: int, default: any = None) -> Union[str, None]:
        return self.__characters.peek(n, default)

    def raise_error(self, error: Error):
        raise Exception(error.message)

    def current_pos(self) -> Position:
        return Position(self.line, self.character)
