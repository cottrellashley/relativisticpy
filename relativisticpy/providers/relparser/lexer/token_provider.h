#ifndef TOKEN_PROVIDER_H
#define TOKEN_PROVIDER_H

#define TYPE_SIZE 100
#define VALUE_SIZE 100

typedef struct 
{
    char type[TYPE_SIZE];
    char value[VALUE_SIZE];
} Token;

typedef struct 
{
    Token* tokens;  // Pointer to an array of Tokens
    size_t size;    // Number of Tokens in the array
} TokenList;

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

#endif
