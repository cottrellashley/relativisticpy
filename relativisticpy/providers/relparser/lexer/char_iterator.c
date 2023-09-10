#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "char_iterator.h"

// ... [Same function implementations as before]

// Assuming that the iterable object is a list of integers for simplicity.
// Adjust it according to your needs.

// Initialize the iterator struct
void init_char_iter(CharIterator* p_iter, const char* str) 
{
    p_iter->p_object = string_to_iterable(str);  // Get a copy of the string
    p_iter->length = strlen(str);
    p_iter->p_current_item = NULL;   // NULL indicates the iterator hasn't advanced yet (or it has gone passed the last char)
    p_iter->location = -1;
}

// Advance the iterator struct
void advance_char(CharIterator* p_iter) 
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
char peek_char(CharIterator* p_iter, int n) 
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
char current_char(CharIterator* p_iter) 
{
    if (p_iter->p_current_item != NULL) 
    {
        return *p_iter->p_current_item;
    }
    return '\0';   // Return null char if iterator hasn't been advanced yet or has reached the end
}

// Get the length of the iterable
int string_len(CharIterator* p_iter) 
{
    return p_iter->length;
}

void free_char_iter(CharIterator* p_iter) 
{
    free(p_iter->p_object);
}

char* string_to_iterable(const char *p_input) 
{
    int length = strlen(p_input);
    
    // Allocate memory for the character array
    char *charArray = (char *)malloc((length + 1) * sizeof(char)); // +1 for the null terminator

    if (charArray == NULL) 
    {
        printf("Memory allocation failed\n");
        exit(1);
    }

    for(int i = 0; i < length; i++) 
    {
        charArray[i] = p_input[i];
    }
    
    charArray[length] = '\0'; // Null-terminate the character array

    return charArray;
}
