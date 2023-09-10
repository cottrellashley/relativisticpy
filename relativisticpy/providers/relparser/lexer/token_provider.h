
typedef struct 
{
    Token* tokens;  // Pointer to an array of Tokens
    size_t size;    // Number of Tokens in the array
} TokenList;

typedef struct 
{
    char type[];
    char value[];
} Token;

void new_token(TokenList *token_list, char type[], char value[]);
bool single_match_exists(char c1);
bool double_match_exists(char c1, char c2);
int tokens_len(TokenList *p_token_list);
bool tripple_match_exists(char c1, char c2, char c3);
void new_single_operation_token(char c1);
void new_double_operation_token(char c1, char c2);
void new_tripple_operation_token(char c1, char c2, char c3);
Token* get_tokens();
void free_tokens();