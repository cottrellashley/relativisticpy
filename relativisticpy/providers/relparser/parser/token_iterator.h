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

void Iterator_Init(TokenIterator* p_iter, const Token* str);
void Iterator_Advance(TokenIterator* p_iter);
char Iterator_Peek(TokenIterator* p_iter, int n);
int Iterator_Length(TokenIterator* p_iter);
char Iterator_Current(TokenIterator* p_iter);
void Iterator_Cleanup(TokenIterator* p_iter);

#endif  // ITERATOR_H
