


// Forward declaration for required structs
struct ITokenProvider;
struct IIterator;
struct Characters;
struct TokenType;

typedef struct Lexer {
    struct ITokenProvider* p_token_provider;
    struct IIterator* p_characters;
} Lexer;
