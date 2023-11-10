#ifndef LEXER_H
#define LEXER_H

#include "iterator.h"

typedef struct Lexer {
    struct TokenProvider* p_token_provider;
    struct Iterator* p_characters;
} Lexer;

#endif