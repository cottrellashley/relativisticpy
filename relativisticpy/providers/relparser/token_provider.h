
typedef struct {
    Token* tokens;  // Pointer to an array of Tokens
    size_t size;    // Number of Tokens in the array
} TokenList;

// Assuming the following enum to represent the TokenType. 
// This is a partial representation, and you should include all the required tokens.
typedef enum TokenType {
    STAR,
    MINUS,
    PLUS,
    EQUAL,
    LSQB,
    RSQB,
    LPAR,
    RPAR,
    LBRACE,
    RBRACE,
    CIRCUMFLEX,
    SLASH,
    VBAR,
    AMPER,
    EXCLAMATION,
    TILDE,
    GREATER,
    LESS,
    COLON,
    DOT,
    COMMA,
    SEMI,
    AT,
    PERCENT,
    NOTEQUAL,
    PERCENTEQUAL,
    AMPEREQUAL,
    PLUSEQUAL,
    COLONEQUAL,
    EQEQUAL,
    VBARVBAR,
    ATEQUAL,
    CIRCUMFLEXEQUAL,
    VBAREQUAL,
    DOUBLESTAR,
    GREATEREQUAL,
    RIGHTSHIFT,
    DOUBLESLASH,
    MINEQUAL,
    RARROW,
    LESSEQUAL,
    LEFTSHIFTEQUAL,
    LEFTSHIFT,
    DOUBLESTAREQUAL,
    ELLIPSIS,
    DOUBLESLASHEQUAL,
    RIGHTSHIFTEQUAL,
    OP
} TokenType;

typedef struct {
    TokenType type;
    char *value;
} Token;

void new_token(TokenType type, char *value);
bool single_match_exists(char c1);
bool double_match_exists(char c1, char c2);
bool tripple_match_exists(char c1, char c2, char c3);
void new_single_operation_token(char c1);
void new_double_operation_token(char c1, char c2);
void new_tripple_operation_token(char c1, char c2, char c3);
Token* get_tokens();
void free_tokens();