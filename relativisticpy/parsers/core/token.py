from relativisticpy.parsers.shared.constants import TokenType
from relativisticpy.parsers.shared.interfaces.tokens import ITokenProvider
from relativisticpy.parsers.shared.models.token import Token


class TokenProvider(ITokenProvider):
    def __init__(self):
        self.tokens = []

    def new_token(self, type, value) -> None:
        token = Token(type, value)
        self.tokens.append(token)

    def get_tokens(self):
        return self.tokens

    def new_single_operation_token(self, c1) -> None:
        if self.single_match_exists(c1):
            self.tokens.append(self.singles()[c1])

    def new_double_operation_token(self, c1, c2) -> None:
        if self.double_match_exists(c1, c2):
            self.tokens.append(self.doubles()[c1][c2])

    def new_tripple_operation_token(self, c1, c2, c3) -> None:
        if self.tripple_match_exists(c1, c2, c3):
            self.tokens.append(self.tripples()[c1][c2][c3])

    def singles(self):
        return {
            "*": Token(TokenType.STAR, "*"),
            "-": Token(TokenType.MINUS, "-"),
            "+": Token(TokenType.PLUS, "+"),
            "=": Token(TokenType.EQUAL, "="),
            "[": Token(TokenType.LSQB, "["),
            "]": Token(TokenType.RSQB, "]"),
            "(": Token(TokenType.LPAR, "("),
            ")": Token(TokenType.RPAR, ")"),
            "{": Token(TokenType.LBRACE, "{"),
            "}": Token(TokenType.RBRACE, "}"),
            "^": Token(TokenType.CIRCUMFLEX, "^"),
            "/": Token(TokenType.SLASH, "/"),
            "|": Token(TokenType.VBAR, "|"),
            "&": Token(TokenType.AMPER, "&"),
            "!": Token(TokenType.EXCLAMATION, "!"),
            "~": Token(TokenType.TILDE, "~"),
            ">": Token(TokenType.GREATER, ">"),
            "<": Token(TokenType.LESS, "<"),
            ":": Token(TokenType.COLON, ":"),
            ".": Token(TokenType.DOT, "."),
            ",": Token(TokenType.COMMA, ","),
            ";": Token(TokenType.SEMI, ";"),
            "@": Token(TokenType.AT, "@"),
            "%": Token(TokenType.PERCENT, "%"),
        }

    def doubles(self):
        return {
            "!": {"=": Token(TokenType.NOTEQUAL, "!=")},
            "%": {"=": Token(TokenType.PERCENTEQUAL, "%=")},
            "&": {"=": Token(TokenType.AMPEREQUAL, "&=")},
            "+": {"=": Token(TokenType.PLUSEQUAL, "+=")},
            ":": {"=": Token(TokenType.COLONEQUAL, ":=")},
            "=": {"=": Token(TokenType.EQEQUAL, "==")},
            "|": {"|": Token(TokenType.VBARVBAR, "||")},
            "@": {"=": Token(TokenType.ATEQUAL, "@=")},
            "^": {"=": Token(TokenType.CIRCUMFLEXEQUAL, "^=")},
            "|": {"=": Token(TokenType.VBAREQUAL, "|=")},
            "*": {
                "*": Token(TokenType.DOUBLESTAR, "**"),
                "=": Token(TokenType.STAREQUAL, "*="),
            },
            ">": {
                "=": Token(TokenType.GREATEREQUAL, ">="),
                ">": Token(TokenType.RIGHTSHIFT, ">>"),
            },
            "/": {
                "/": Token(TokenType.DOUBLESLASH, "//"),
                "=": Token(TokenType.SLASHEQUAL, "/="),
            },
            "-": {
                "=": Token(TokenType.MINEQUAL, "-="),
                ">": Token(TokenType.RARROW, "->"),
            },
            "<": {
                "=": Token(TokenType.LESSEQUAL, "<="),
                "<": Token(TokenType.LEFTSHIFTEQUAL, "<<"),
                ">": Token(TokenType.NOTEQUAL, "<>"),
            },
        }

    def tripples(self):
        return {
            "*": {"*": {"=": Token(TokenType.DOUBLESTAREQUAL, "**=")}},
            ".": {".": {".": Token(TokenType.ELLIPSIS, "...")}},
            "/": {"/": {"=": Token(TokenType.DOUBLESLASHEQUAL, "//=")}},
            "<": {"<": {"=": Token(TokenType.LEFTSHIFTEQUAL, "<<=")}},
            ">": {">": {"=": Token(TokenType.RIGHTSHIFTEQUAL, ">>=")}},
        }

    def single_match_exists(self, c1):
        try:
            self.singles()[c1]
            return True
        except:
            return False

    def double_match_exists(self, c1, c2):
        try:
            self.doubles()[c1][c2]
            return True
        except:
            return False

    def tripple_match_exists(self, c1, c2, c3):
        try:
            self.tripples()[c1][c2][c3]
            return True
        except:
            return False
