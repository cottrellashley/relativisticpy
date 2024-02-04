from enum import Enum


class NodeKeys(Enum):
    Node = "node"
    Handler = "handler"
    Arguments = "args"


class TokenSinglets(Enum):
    PLUS = "+"  # Addition operator
    MINUS = "-"  # Subtraction operator
    MULTIPLY = "*"  # Multiplication operator
    DIVIDE = "/"  # Division operator
    EQUALS = "="  # The `=` symbol for assignment or comparison
    OPEN_BRACE = "("  # Left parenthesis
    CLOSED_BRACE = ")"  # Right parenthesis
    COMMA = ","  # The `,` symbol for function arguments or tensor indices
    OPEN_SQUARE_BRACE = "["  # Open square brackets '['
    CLOSED_SQUARE_BRACE = "]"  # Closed square brackets ']'
    DOT = "."


class NodeType(Enum):
    """An enumeration of node types used by a parser."""

    LESS = "<"
    LESSEQUAL = "<="
    GREATER = ">"
    GREATEREQUAL = ">="
    ASSIGNMENT = ":="
    EQEQUAL = "=="
    NOTEQUAL = "!="
    PLUS = "+"  # Addition operator
    MINUS = "-"  # Subtraction operator
    MULTIPLY = "*"  # Multiplication operator
    DIVIDE = "/"  # Division operator
    EQUALS = "="  # The `=` symbol for assignment or comparison
    EXPONENTIATION1 = "^"  # The `**` symbol for exponentiation
    EXPONENTIATION2 = "**"
    OBJECT = "object"  # A variable name
    VARIABLEKEY = "variable_key"  # A variable name
    FLOAT = "float"  # A floating-point number
    INTEGER = "integer"  # An integer number
    FUNCTION = "function"  # A function name
    ARRAY = "array"  # Array object '[elements]'
    POSITIVE = "positive"  # Positive operator '+'
    AND = "&"  # Positive operator '+'
    OR = "|"  # Positive operator '+'
    NEGATIVE = "negative"
    TENSOR_INIT = "tensor_init"
    TENSOR_KEY = "tensor_key"
    SYMBOL_DEFINITION = "symbol_definition"
    SYMBOL_KEY = "symbol_key"
    SYMBOL = "symbol"
    TENSOR = "tensor"
    ID = "ID"


class TokenType(Enum):
    """An enumeration of token types used by a the lexer to genrate tokens."""

    NONE = "NONE"

    # Single Charater Tokens
    PLUS = "PLUS"
    MINUS = "MINUS"
    STAR = "STAR"
    SLASH = "SLASH"
    EQUAL = "EQUAL"
    LPAR = "LPAR"
    RPAR = "RPAR"
    COMMA = "COMMA"
    LSQB = "LSQB"
    RSQB = "RSQB"
    DOT = "DOT"
    OPEN_CURLY_BRACE = "OPEN_CURLY_BRACE"
    CLOASED_CURLY_BRACE = "CLOASED_CURLY_BRACE"
    CIRCUMFLEX = "CIRCUMFLEX"
    EXCLAMATION = "EXCLAMATION"
    PERCENT = "PERCENT"
    AMPER = "AMPER"
    COLON = "COLON"
    SEMI = "SEMI"
    LESS = "LESS"
    GREATER = "GREATER"
    AT = "AT"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    VBAR = "VBAR"
    TILDE = "TILDE"

    # Double Character Tokens
    NOTEQUAL = "NOTEQUAL"
    PERCENTEQUAL = "PERCENTEQUAL"
    AMPEREQUAL = "AMPEREQUAL"
    DOUBLESTAR = "DOUBLESTAR"
    STAREQUAL = "STAREQUAL"
    PLUSEQUAL = "PLUSEQUAL"
    MINEQUAL = "MINEQUAL"
    RARROW = "RARROW"
    DOUBLESLASH = "DOUBLESLASH"
    SLASHEQUAL = "SLASHEQUAL"
    COLONEQUAL = "COLONEQUAL"
    LEFTSHIFT = "LEFTSHIFT"
    LESSEQUAL = "LESSEQUAL"
    EQEQUAL = "EQEQUAL"
    GREATEREQUAL = "GREATEREQUAL"
    RIGHTSHIFT = "RIGHTSHIFT"
    ATEQUAL = "ATEQUAL"
    CIRCUMFLEXEQUAL = "CIRCUMFLEXEQUAL"
    VBAREQUAL = "VBAREQUAL"
    VBARVBAR = "VBARVBAR"

    # Tripple Character Tokens
    DOUBLESTAREQUAL = "DOUBLESTAREQUAL"
    ELLIPSIS = "ELLIPSIS"
    DOUBLESLASHEQUAL = "DOUBLESLASHEQUAL"
    LEFTSHIFTEQUAL = "LEFTSHIFTEQUAL"
    RIGHTSHIFTEQUAL = "RIGHTSHIFTEQUAL"

    # Arbitarily Sized Tokens
    OBJECT = "OBJECT"
    FLOAT = "FLOAT"
    INTEGER = "INTEGER"
    FUNCTION = "FUNCTION"
    ARRAY = "ARRAY"

    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)


class Characters(Enum):
    """An object containing the set of supported characters, keyd on a set name."""

    WHITESPACE = " \t"
    DIGITS = "0987654321"
    LOWERCASECHARACTERS = "abcdefghijklmnopqrstuvwxyz"
    UPPERCASECHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    NONLETTERCHARACTERS = "_{}:^;"
    OPERATIONS = "*-+=[](){}^/|&!~><:;.,@%"
    LETTERS = LOWERCASECHARACTERS + UPPERCASECHARACTERS
    CHARACTERS = LOWERCASECHARACTERS + UPPERCASECHARACTERS + NONLETTERCHARACTERS
    OBJECTCHARACTERS = (
        LOWERCASECHARACTERS + UPPERCASECHARACTERS + NONLETTERCHARACTERS + DIGITS
    )
