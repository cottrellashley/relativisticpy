// iterator.h
#ifndef TOKEN_ITERATOR_H
#define TOKEN_ITERATOR_H

#include <stddef.h>
#include <stdbool.h>

typedef struct {
    char* p_object;   // Pointer to the iterable object (in this case, a string)
    int length;  // Length of the iterable
    char* p_current_token;  // Pointer to the current item
    int location;   // Current location in the iterable
} TokenIterator;

void init_token_iterator(TokenIterator* p_iter, const Token* str);
void advance_token(TokenIterator* p_iter);
char peek_token(TokenIterator* p_iter, int n);
int token_len(TokenIterator* p_iter);
char current_token(TokenIterator* p_iter);
void cleanup_token(TokenIterator* p_iter);

#endif  // ITERATOR_H
