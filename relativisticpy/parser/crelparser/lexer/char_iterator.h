// iterator.h
#ifndef CHAR_ITERATOR_H
#define CHAR_ITERATOR_H

#include <stddef.h>
#include <stdbool.h>

typedef struct {
    char* p_object;   // Pointer to the iterable object (in this case, a string)
    int length;  // Length of the iterable
    char* p_current_item;  // Pointer to the current item
    int location;   // Current location in the iterable
} CharIterator;

void init_char_iter(CharIterator* p_iter, const char* str);
void advance_char(CharIterator* p_iter);
char peek_char(CharIterator* p_iter, int n);
int string_len(CharIterator* p_iter);
char current_char(CharIterator* p_iter);
char* string_to_iterable(const char *input);
void free_char_iter(CharIterator* p_iter);

#endif  // ITERATOR_H
