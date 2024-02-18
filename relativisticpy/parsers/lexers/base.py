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

    @classmethod
    def SINGLES(self):
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
            "'": TokenType.APOSTROPHE,
        }
    
    @classmethod
    def DOUBLES(self):
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

    @classmethod
    def TRIPPLES(self):
        return {
            "*": {"*": {"=": TokenType.DOUBLESTAREQUAL}},
            ".": {".": {".": TokenType.ELLIPSIS}},
            "/": {"/": {"=": TokenType.DOUBLESLASHEQUAL}},
            "<": {"<": {"=": TokenType.LEFTSHIFTEQUAL}},
            ">": {">": {"=": TokenType.RIGHTSHIFTEQUAL}},
        }

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
    APOSTROPHE = "'"

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

    ID = "ID"
    INT = "INT"
    BOOL = "BOOL"
    FLOAT = "FLOAT"
    SYMBOL = "SYMBOL"
    STRING = "STRING"

    @classmethod
    def Keywords(cls):
        return {
            # Keywords mappings
            cls.NOT.value: cls.NOT, 
            cls.AND.value: cls.AND, 
            cls.OR.value: cls.OR, 
            cls.PRINT.value: cls.PRINT, 
            cls.INFINITESIMAL.value: cls.INFINITESIMAL,
            cls.PI.value : cls.CONSTANT,
            cls.E.value : cls.CONSTANT,
            cls.INFTY.value : cls.CONSTANT,
            cls.OO.value : cls.CONSTANT
        }

    ##########################################
    ############     KEYWORDS     ############ 
    ##########################################
    NOT = "not"
    AND = "and"
    OR = "or"
    PRINT = "print"
    INFINITESIMAL = 'd'
    PI = 'pi'
    E = 'e'
    OO = 'oo'
    INFTY = 'infty'
    CONSTANT = 'constant'

    @classmethod
    def LaTeX(cls):
        return {
            # LaTeX operations
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
            cls.EQUIVALENT.value: cls.COLONEQUAL,
            cls.DERIVATIVE.value: cls.DERIVATIVE,
            cls.PDERIVATIVE.value: cls.PDERIVATIVE,
            cls.PARTIAL.value: cls.PARTIAL,
            cls.INTEGRATE.value : cls.INTEGRATE,
            cls.INFTY.value : cls.CONSTANT,
            cls.SQRT.value : cls.SQRT,
            # LaTeX symbols are all mapped to cls.SYMBOL
            cls.alpha.value: cls.SYMBOL,
            cls.Alpha.value: cls.SYMBOL,
            cls.beta.value: cls.SYMBOL,
            cls.Beta.value: cls.SYMBOL,
            cls.gamma.value: cls.SYMBOL,
            cls.Gamma.value: cls.SYMBOL,
            cls.delta.value: cls.SYMBOL,
            cls.Delta.value: cls.SYMBOL,
            cls.epsilon.value: cls.SYMBOL,
            cls.Epsilon.value: cls.SYMBOL,
            cls.zeta.value: cls.SYMBOL,
            cls.Zeta.value: cls.SYMBOL,
            cls.eta.value: cls.SYMBOL,
            cls.Eta.value: cls.SYMBOL,
            cls.theta.value: cls.SYMBOL,
            cls.Theta.value: cls.SYMBOL,
            cls.iota.value: cls.SYMBOL,
            cls.Iota.value: cls.SYMBOL,
            cls.kappa.value: cls.SYMBOL,
            cls.Kappa.value: cls.SYMBOL,
            cls.lambda_.value: cls.SYMBOL,  # Using lambda_ because 'lambda' is a reserved keyword in Python
            cls.Lambda.value: cls.SYMBOL,
            cls.mu.value: cls.SYMBOL,
            cls.Mu.value: cls.SYMBOL,
            cls.nu.value: cls.SYMBOL,
            cls.Nu.value: cls.SYMBOL,
            cls.xi.value: cls.SYMBOL,
            cls.Xi.value: cls.SYMBOL,
            cls.omicron.value: cls.SYMBOL,
            cls.Omicron.value: cls.SYMBOL,
            cls.pi.value: cls.SYMBOL,
            cls.Pi.value: cls.SYMBOL,
            cls.rho.value: cls.SYMBOL,
            cls.Rho.value: cls.SYMBOL,
            cls.sigma.value: cls.SYMBOL,
            cls.Sigma.value: cls.SYMBOL,
            cls.tau.value: cls.SYMBOL,
            cls.Tau.value: cls.SYMBOL,
            cls.upsilon.value: cls.SYMBOL,
            cls.Upsilon.value: cls.SYMBOL,
            cls.phi.value: cls.SYMBOL,
            cls.Phi.value: cls.SYMBOL,
            cls.chi.value: cls.SYMBOL,
            cls.Chi.value: cls.SYMBOL,
            cls.psi.value: cls.SYMBOL,
            cls.Psi.value: cls.SYMBOL,
            cls.omega.value: cls.SYMBOL,
            cls.Omega.value: cls.SYMBOL
            }

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
    PDERIVATIVE = "pdv"
    DERIVATIVE = "dv"
    INTEGRATE = "int"
    PARTIAL = "partial"
    SQRT = "sqrt"

    ##########################################
    ############   Latex Symbols  ############ 
    ##########################################
    alpha = "alpha"
    Alpha = "Alpha"
    beta = "beta"
    Beta = "Beta"
    gamma = "gamma"
    Gamma = "Gamma"
    delta = "delta"
    Delta = "Delta"
    epsilon = "epsilon"
    Epsilon = "Epsilon"
    zeta = "zeta"
    Zeta = "Zeta"
    eta = "eta"
    Eta = "Eta"
    theta = "theta"
    Theta = "Theta"
    iota = "iota"
    Iota = "Iota"
    kappa = "kappa"
    Kappa = "Kappa"
    lambda_ = "lambda"  # 'lambda' is a reserved keyword in Python, so we use a trailing underscore
    Lambda = "Lambda"
    mu = "mu"
    Mu = "Mu"
    nu = "nu"
    Nu = "Nu"
    xi = "xi"
    Xi = "Xi"
    omicron = "omicron"
    Omicron = "Omicron"
    pi = "pi"
    Pi = "Pi"
    rho = "rho"
    Rho = "Rho"
    sigma = "sigma"
    Sigma = "Sigma"
    tau = "tau"
    Tau = "Tau"
    upsilon = "upsilon"
    Upsilon = "Upsilon"
    phi = "phi"
    Phi = "Phi"
    chi = "chi"
    Chi = "Chi"
    psi = "psi"
    Psi = "Psi"
    omega = "omega"
    Omega = "Omega"
    infty = "infty"


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


class _TokenProvider:
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
            token_type: TokenType = TokenType.SINGLES()[c1]
            self.tokens.append(Token(token_type, token_type.value, start_pos, end_pos))

    def new_double_operation_token(
        self, c1: str, c2: str, start_pos: Position, end_pos: Position
    ) -> None:
        if self.double_match_exists(c1, c2):
            token_type: TokenType = TokenType.DOUBLES()[c1][c2]
            self.tokens.append(Token(token_type, token_type.value, start_pos, end_pos))

    def new_tripple_operation_token(
        self, c1: str, c2: str, c3: str, start_pos: Position, end_pos: Position
    ) -> None:
        if self.tripple_match_exists(c1, c2, c3):
            token_type: TokenType = TokenType.TRIPPLES()[c1][c2][c3]
            self.tokens.append(Token(token_type, token_type.value, start_pos, end_pos))

    def single_match_exists(self, c1: str) -> bool:
        try:
            TokenType.SINGLES()[c1]
            return True
        except:
            return False

    def double_match_exists(self, c1: str, c2: str) -> bool:
        try:
            TokenType.DOUBLES()[c1][c2]
            return True
        except:
            return False

    def tripple_match_exists(self, c1: str, c2: str, c3: str) -> bool:
        try:
            TokenType.TRIPPLES()[c1][c2][c3]
            return True
        except:
            return False

@dataclass
class LexerResult:
    code: str
    tokens: List[Token]


class BaseLexer(ABC):
    "Base Lexer all lexer types within this module should inherit."

    def __init__(self, string: str):
        self.raw_code = string
        self.__characters = Iterator(string)
        self.__token_provider_instance = _TokenProvider()
        self.character = 0
        self.line = 1
        self.__characters.advance()

    @abstractmethod
    def tokenize() -> LexerResult:
        pass

    @property
    def token_provider(self) -> _TokenProvider:
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
