

typedef struct {
    int type;   // Pointer to the iterable object (in this case, a string)
    char* value;  // Length of the iterable
} Token;


typedef struct {
    Token* tokens;  // Pointer to an array of Tokens
    size_t size;    // Number of Tokens in the array
} TokenList;