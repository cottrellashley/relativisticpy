#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "token_iterator.h"
#include "token_provider.h"

// Initialize the iterator struct
void init_token_iterator(TokenIterator* p_iter, TokenList* p_token_list) 
{
    p_iter->p_object = p_token_list;  // Get a copy of the string
    p_iter->length = token_len(p_token_list);
    p_iter->p_current_item = NULL;   // NULL indicates the iterator hasn't advanced yet (or it has gone passed the last char)
    p_iter->location = -1;
}

// Advance the iterator struct
void advance_token(TokenIterator* p_iter) 
{
    if (p_iter->location + 1 < p_iter->length) 
    {
        p_iter->location++;
        p_iter->p_current_item = &p_iter->p_object[p_iter->location]; // The pointer to the new current item has changed to the address of the next character.
        printf("Current item: %c\n", *p_iter->p_current_item);  // Use %c for chars
    } 
    else 
    {
        p_iter->p_current_item = NULL;   // End of the iterable
    }
}

// Peek ahead by n items
char peek_token(TokenIterator* p_iter, int n) 
{
    if (p_iter->location + n < p_iter->length) 
    {
        return p_iter->p_object[p_iter->location + n]; // Do I have to dereference p_obj here ?????
    } 
    else 
    {
        return '\0';   // Return null char if out of bounds
    }
}

// Get the current item
char current_token(TokenIterator* p_iter) 
{
    if (p_iter->p_current_item != NULL) 
    {
        return *p_iter->p_current_item;
    }
    return '\0';   // Return null char if iterator hasn't been advanced yet or has reached the end
}

// Get the length of the iterable
int token_len(TokenIterator* p_iter) 
{
    return p_iter->length;
}

void cleanup_token(TokenIterator* p_iter) 
{
    free(p_iter->p_object);
}