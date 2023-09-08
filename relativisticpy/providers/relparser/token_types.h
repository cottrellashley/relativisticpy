// An enumeration of token types used by the lexer to generate tokens.

#define NONE                 "NONE"

// Single Character Tokens
#define PLUS                 "PLUS"
#define MINUS                "MINUS"
#define STAR                 "STAR"
#define SLASH                "SLASH"
#define EQUAL                "EQUAL"
#define LPAR                 "LPAR"
#define RPAR                 "RPAR"
#define COMMA                "COMMA"
#define LSQB                 "LSQB"
#define RSQB                 "RSQB"
#define DOT                  "DOT"
#define OPEN_CURLY_BRACE     "OPEN_CURLY_BRACE"
#define CLOSED_CURLY_BRACE   "CLOSED_CURLY_BRACE"
#define CIRCUMFLEX           "CIRCUMFLEX"
#define EXCLAMATION          "EXCLAMATION"
#define PERCENT              "PERCENT"
#define AMPER                "AMPER"
#define COLON                "COLON"
#define SEMI                 "SEMI"
#define LESS                 "LESS"
#define GREATER              "GREATER"
#define AT                   "AT"
#define LBRACE               "LBRACE"
#define RBRACE               "RBRACE"
#define VBAR                 "VBAR"
#define TILDE                "TILDE"

// Double Character Tokens
#define NOTEQUAL             "NOTEQUAL"
#define PERCENTEQUAL         "PERCENTEQUAL"
#define AMPEREQUAL           "AMPEREQUAL"
#define DOUBLESTAR           "DOUBLESTAR"
#define STAREQUAL            "STAREQUAL"
#define PLUSEQUAL            "PLUSEQUAL"
#define MINEQUAL             "MINEQUAL"
#define RARROW               "RARROW"
#define DOUBLESLASH          "DOUBLESLASH"
#define SLASHEQUAL           "SLASHEQUAL"
#define COLONEQUAL           "COLONEQUAL"
#define LEFTSHIFT            "LEFTSHIFT"
#define LESSEQUAL            "LESSEQUAL"
#define EQEQUAL              "EQEQUAL"
#define GREATEREQUAL         "GREATEREQUAL"
#define RIGHTSHIFT           "RIGHTSHIFT"
#define ATEQUAL              "ATEQUAL"
#define CIRCUMFLEXEQUAL      "CIRCUMFLEXEQUAL"
#define VBAREQUAL            "VBAREQUAL"
#define VBARVBAR             "VBARVBAR"

// Triple Character Tokens
#define DOUBLESTAREQUAL      "DOUBLESTAREQUAL"
#define ELLIPSIS             "ELLIPSIS"
#define DOUBLESLASHEQUAL     "DOUBLESLASHEQUAL"
#define LEFTSHIFTEQUAL       "LEFTSHIFTEQUAL"
#define RIGHTSHIFTEQUAL      "RIGHTSHIFTEQUAL"

// Arbitrarily Sized Tokens
#define OBJECT               "OBJECT"
#define FLOAT                "FLOAT"
#define INTEGER              "INTEGER"
#define FUNCTION             "FUNCTION"
#define ARRAY                "ARRAY"


// An enumeration of node types used by a parser.

#define SMALLER             '<'
#define SMALLERTHANOREQUAL  "<="
#define BIGGER              '>'
#define BIGGERTHANOREQUAL   ">="
#define ASSIGNMENT          ":="
#define EQUALEQUAL          "=="
#define NOTEQUAL            "!="
#define ADD                 '+'        // Addition operator
#define SUBTRACT            '-'        // Subtraction operator
#define MULTIPLY            '*'        // Multiplication operator
#define DIVIDE              '/'        // Division operator
#define EQUALS              '='        // The `=` symbol for assignment or comparison
#define EXPONENTIATION1     '^'        // The `^` symbol for exponentiation
#define EXPONENTIATION2     "**"
#define POSITIVE            "positive" // Positive operator '+'
#define AND                 '&'        // Logical AND operator
#define OR                  '|'        // Logical OR operator
