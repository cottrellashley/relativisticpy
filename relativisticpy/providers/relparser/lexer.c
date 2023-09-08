#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "p_lexer.h"
#include "p_characters.h"

// Constructor
Lexer* Lexer_new(struct ITokenProvider* p_token_provider) {
    Lexer* p_lexer = malloc(sizeof(Lexer));
    p_lexer->p_token_provider = p_token_provider;
    return p_lexer;
}

// Destructor
void Lexer_delete(Lexer* p_lexer) {
    free(p_lexer);
}

// Methods translated to functions
void Lexer_tokenize(Lexer* p_lexer, struct IIterator* p_characters) {
    p_lexer->p_characters = p_characters;
    Iterator_Advance(p_lexer->p_characters);
    while (Iterator_Current(p_lexer->p_characters) != NULL) {
        if (strchr(WHITESPACE, *Iterator_Current(p_lexer->p_characters))) { 
            Iterator_Advance(p_lexer->p_characters); // If we have any whitespace, just continue without doing anything.
        } else if (strchr(LETTERS, *Iterator_Current(p_lexer->p_characters))) {
            Lexer_operation(p_lexer);
        } else if (strchr(DIGITS, *Iterator_Current(p_lexer->p_characters))) {
            Lexer_number(p_lexer);
        } else if (strchr(OPERATIONS, *Iterator_Current(p_lexer->p_characters))) {
            Lexer_operation(p_lexer);
        }
    }
}

void Lexer_number(Lexer* p_lexer) {
    char* number = malloc(50);  // assuming maximum length of 50 for simplicity
    int count = 0;
    while (Iterator_Current(p_lexer->p_characters) != NULL 
           && (Characters_isDigit(Iterator_Current(p_lexer->p_characters)) 
               || *Iterator_Current(p_lexer->p_characters) == '.')) {
        strcat(number, Iterator_Current(p_lexer->p_characters));
        Iterator_Advance(p_lexer->p_characters);
    }
    for (int i = 0; i < strlen(number); i++) {
        if (number[i] == '.') {
            count++;
        }
    }
    if (count == 0) {
        ITokenProvider_new_token(p_lexer->p_token_provider, TokenType_INTEGER, number);
    } else if (count == 1) {
        ITokenProvider_new_token(p_lexer->p_token_provider, TokenType_FLOAT, number);
    } else {
        printf("Illegal Character '%s'", number);
        exit(EXIT_FAILURE);
    }
    free(number);
}

// Assuming max ops length of 100 for simplicity
void Lexer_operation(Lexer* p_lexer) {
    char ops[100];
    int ops_len = 0;

    while (Iterator_Current(p_lexer->p_characters) != NULL && Characters_isOperation(Iterator_Current(p_lexer->p_characters))) {
        ops[ops_len++] = *Iterator_Current(p_lexer->p_characters);
        Iterator_Advance(p_lexer->p_characters);
    }
    ops[ops_len] = '\0';  // Null terminate the ops string

    int i = 0;
    while (i < ops_len) {
        if (i + 2 < ops_len && ITokenProvider_isTriple(p_lexer->p_token_provider, ops[i]) && ITokenProvider_tripleMatchExists(p_lexer->p_token_provider, ops[i], ops[i+1], ops[i+2])) {
            ITokenProvider_newTripleOperationToken(p_lexer->p_token_provider, ops[i], ops[i+1], ops[i+2]);
            i += 3;
        } else if (i + 1 < ops_len && ITokenProvider_isDouble(p_lexer->p_token_provider, ops[i]) && ITokenProvider_doubleMatchExists(p_lexer->p_token_provider, ops[i], ops[i+1])) {
            ITokenProvider_newDoubleOperationToken(p_lexer->p_token_provider, ops[i], ops[i+1]);
            i += 2;
        } else {
            ITokenProvider_newSingleOperationToken(p_lexer->p_token_provider, ops[i]);
            i += 1;
        }
    }
}

void Lexer_object(Lexer* p_lexer) {
    char obj[100];  // Assuming max object length of 100 for simplicity
    int obj_len = 0;

    while (Iterator_Current(p_lexer->p_characters) != NULL && Characters_isObjectCharacter(Iterator_Current(p_lexer->p_characters))) {
        if (Characters_isCharacter(Iterator_Current(p_lexer->p_characters)) && Iterator_Peek(p_lexer->p_characters, 1) == '(') {
            obj[obj_len++] = *Iterator_Current(p_lexer->p_characters);
            Iterator_Advance(p_lexer->p_characters);
            obj[obj_len] = '\0';  // Null terminate the obj string
            ITokenProvider_newToken(p_lexer->p_token_provider, TokenType_FUNCTION, obj);
        } else if (!Characters_isObjectCharacter(Iterator_Peek(p_lexer->p_characters, 1)) && Iterator_Peek(p_lexer->p_characters, 1) != '(') {
            obj[obj_len++] = *Iterator_Current(p_lexer->p_characters);
            Iterator_Advance(p_lexer->p_characters);
            obj[obj_len] = '\0';  // Null terminate the obj string
            ITokenProvider_newToken(p_lexer->p_token_provider, TokenType_OBJECT, obj);
        } else {
            obj[obj_len++] = *Iterator_Current(p_lexer->p_characters);
            Iterator_Advance(p_lexer->p_characters);
        }
    }
}
