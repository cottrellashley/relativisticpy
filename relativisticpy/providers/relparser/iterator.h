// iterator.h
#ifndef ITERATOR_H
#define ITERATOR_H

#include <stddef.h>
#include <stdbool.h>

typedef struct {
    char* object;   // Pointer to the iterable object (in this case, a string)
    size_t length;  // Length of the iterable
    char* current_item;  // Pointer to the current item
    int location;   // Current location in the iterable
} Iterator;

void Iterator_Init(Iterator* iter, const char* str);
void Iterator_Advance(Iterator* iter);
char Iterator_Peek(Iterator* iter, size_t n);
size_t Iterator_Length(Iterator* iter);
char Iterator_Current(Iterator* iter);
char* breakdownString(const char *input);
void Iterator_Cleanup(Iterator* iter);

#endif  // ITERATOR_H
