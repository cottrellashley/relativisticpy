If you just want to see the grammar, see 'grammar.txt'

# Grammar File Syntax (How to read a grammar file)

```

Sequence: e1 e2
Ordered Choice: e1 / e2
Zero-or-more: e*
One-or-more: e+
Optional: e?
And-predicate: &e
Not-predicate: !e
Group: (e)

Peek: > e 
2Peek: >> e
3Peek: >>> e
nPeek: >...> e
```

# Grammar of Atoms

The structure of the grammar above the "atom" level, involving constructs like expressions, terms, factors, and so on, represents a hierarchical organization designed to parse and interpret the syntax of a programming language. This hierarchy is crucial for understanding how simple elements (atoms) are combined to form more complex expressions and statements.


```
statements      :   NEWLINE* statement (NEWLINE* statement)

statement       :   KEYWORD:print? expr

expr            :   (ID) EQUAL expr
                :   bool-expr ((KEYWORD:and|KEYWORD:or) bool-expr)*

bool-expr       :   NOT bool-expr
                :   arith-expr ((EQEQUAL|LESS|GREATER|LESSEQUAL|GREATEREQUAL) arith-expr)*

arith-expr      :   term ((PLUS|MINUS) term)*

term            :   factor ((MUL|DIV) factor)*
                :   SYMBOL (SYMBOL)*

factor          :   (PLUS|MINUS) factor
                :   power

power           :   atom ((CIRCUMFLEX|DOUBLESTAR) atom)*

```

# Atom Rules

The term "atom" is often used to refer to the simplest or most basic elements that can appear in expressions or statements. These are the "indivisible" parts of the language syntax, analogous to how atoms are the fundamental building blocks in chemistry and physics, which cannot be broken down further by chemical means. There must be a clear way to distinguish each 'atom' object which is composed of more than one token, in this case we use the 'peek(k)' operation so we can gain extra contex from k tokens ahead without actually advancing our current token position. This is therefore a 'LL(k) Parser'.

```
atom            :   INT|FLOAT|STRING|BOOL
                |   LPAR        -> LPAR expr RPAR
                |   RSQB        -> array

                |   SUM         -> sum
                |   DOSUM       -> do_sum
                |   PROD        -> product
                |   DOPROD      -> do_product
                |   INTEGRATE   -> integrate
                |   FRAC        -> fraction
                |   PROD        -> product
                |   LIMIT       -> limit
                |   PARTIAL     -> partial
                |   BEGIN       -> BeginMap( array, equation, matrix, pmatrix, etc... )

                |   ID      -> IF(TensorPeek = True)        -> tensor
                             / IF(FunctionDefPeek = True)   -> func-def
                             / IF(FunctionPeek = True)      -> function
                             / ID

                |   SYMBOL  -> IF(TensorPeek = True)        -> tensor
                             / IF(FunctionDefPeek = True)   -> func-def
                             / IF(FunctionPeek = True)      -> function
                             / SYMBOL

tensor          :   ID (UNDER|CIRCUMFLEX) LBRACE ( (ID|INT) ((EQUAL|COLON) (INT|atom))* RBRACE )*
                |   tensor ((EQUAL) expr)?
                |   tensor ((EQUAL) array)?
                |   tensor ((COLONEQUAL) array)?
                |   tensor tensor*

array           :   LSQB NEWLINE* expr (COMMA NEWLINE* expr)* NEWLINE* RSQB

matrix          :   BEGIN LBRACE ID:matrix RBRACE NEWLINE* ( expr ( AMPER NEWLINE* expr )* NEWLINE* DOUBLEBACKSLASH)+ NEWLINE* END LBRACE ID:matrix RBRACE

sum             :   SUM UNDER LBRACE (ID|SYMBOL) EQUAL INT RBRACE CIRCUMFLEX LBRACE expr RBRACE expr
do_sum          :   DOSUM UNDER LBRACE expr RBRACE CIRCUMFLEX LBRACE expr RBRACE expr
product         :   PROD UNDER LBRACE expr RBRACE CIRCUMFLEX LBRACE expr RBRACE expr
do_product      :   DOPROD UNDER LBRACE expr RBRACE CIRCUMFLEX LBRACE expr RBRACE expr
integrate       :   INTEGRATE UNDER LBRACE expr RBRACE CIRCUMFLEX LBRACE expr RBRACE expr

frac            :   SUM UNDER LBRACE expr RBRACE CIRCUMFLEX LBRACE expr RBRACE expr

func-def        :   FUNCTIONID? LPAR (ID (COMMA ID)*)? RPAR (EQUAL expr NEWLINE) ((EQUAL) expr)?

INT             :   [0-9]+              i.e. 974
FLOAT           :   [0-9]+ . [0-9]+     i.e. 13242.238
ID              :   [a-zA-Z_0-9]+       i.e. T_1, n2, n_0, n, x, y, fun_test , etc...
LATEXSYMBOL     :  \[a-zA-Z]+           i.e. \Theat , \theta , \pi , etc...
LATEXOPERATION  :  \[a-zA-Z]+           i.e. \frac , \sum , \int , etc...

```



# Token Types

```python
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
    INTEGRATE = "int"

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
            cls.LATEXNEWLINE.value: cls.NEWLINE
        }
```

# Semantic Analyzer

# Interpreter (actually just a traverser)

