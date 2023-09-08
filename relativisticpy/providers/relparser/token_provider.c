#include "token_provider.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

TokenList tokenList = {NULL, 0};

void new_token(TokenType type, char *value) {
    Token newToken = {type, value};
    tokenList.tokens = realloc(tokenList.tokens, sizeof(Token) * (tokenList.size + 1));
    tokenList.tokens[tokenList.size++] = newToken;
}

bool single_match_exists(char c1) {
    return _PyToken_OneChar(c1) != OP;
}

bool double_match_exists(char c1, char c2) {
    return _PyToken_TwoChars(c1, c2) != OP;
}

bool tripple_match_exists(char c1, char c2, char c3) {
    return _PyToken_ThreeChars(c1, c2, c3) != OP;
}

void new_single_operation_token(char c1) {
    if (single_match_exists(c1)) {
        new_token(_PyToken_OneChar(c1), &c1);
    }
}

void new_double_operation_token(char c1, char c2) {
    if (double_match_exists(c1, c2)) {
        char value[3] = {c1, c2, '\0'};
        new_token(_PyToken_TwoChars(c1, c2), value);
    }
}

void new_tripple_operation_token(char c1, char c2, char c3) {
    if (tripple_match_exists(c1, c2, c3)) {
        char value[4] = {c1, c2, c3, '\0'};
        new_token(_PyToken_ThreeChars(c1, c2, c3), value);
    }
}

Token* get_tokens() {
    return tokenList.tokens;
}

void free_tokens() {
    free(tokenList.tokens);
    tokenList.size = 0;
}
