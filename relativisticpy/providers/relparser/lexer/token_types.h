// An enumeration of token types used by the lexer to generate tokens.

#define NONE                 0

// Single Character Tokens
#define PLUS                 1
#define MINUS                2
#define STAR                 3
#define SLASH                4
#define EQUAL                5
#define LPAR                 6
#define RPAR                 7
#define COMMA                8
#define LSQB                 9
#define RSQB                 10
#define DOT                  11
#define OPEN_CURLY_BRACE     12
#define CLOSED_CURLY_BRACE   13
#define CIRCUMFLEX           14
#define EXCLAMATION          15
#define PERCENT              16
#define AMPER                17
#define COLON                18
#define SEMI                 19
#define LESS                 20
#define GREATER              21
#define AT                   22
#define LBRACE               23
#define RBRACE               24
#define VBAR                 25
#define TILDE                26

// Double Character Tokens
#define NOTEQUAL             27
#define PERCENTEQUAL         28
#define AMPEREQUAL           29
#define DOUBLESTAR           30
#define STAREQUAL            31
#define PLUSEQUAL            32
#define MINEQUAL             33
#define RARROW               34
#define DOUBLESLASH          35
#define SLASHEQUAL           36
#define COLONEQUAL           37
#define LEFTSHIFT            38
#define LESSEQUAL            39
#define EQEQUAL              40
#define GREATEREQUAL         41
#define RIGHTSHIFT           42
#define ATEQUAL              43
#define CIRCUMFLEXEQUAL      44
#define VBAREQUAL            45
#define VBARVBAR             46

// Triple Character Tokens
#define DOUBLESTAREQUAL      47
#define ELLIPSIS             48
#define DOUBLESLASHEQUAL     49
#define LEFTSHIFTEQUAL       50
#define RIGHTSHIFTEQUAL      51

// Arbitrarily Sized Tokens
#define OBJECT               52
#define FLOAT                53
#define INTEGER              54
#define FUNCTION             55
#define ARRAY                56

