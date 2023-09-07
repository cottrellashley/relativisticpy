#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "iterator.h"

// ... [Same function implementations as before]

// Assuming that the iterable object is a list of integers for simplicity.
// Adjust it according to your needs.

// Initialize the iterator struct
void Iterator_Init(Iterator* iter, const char* str) {
    iter->object = breakdownString(str);  // Get a copy of the string
    iter->length = strlen(str);
    iter->current_item = NULL;   // NULL indicates the iterator hasn't advanced yet
    iter->location = -1;
}

// Advance the iterator struct
void Iterator_Advance(Iterator* iter) {
    if (iter->location + 1 < iter->length) {
        iter->location++;
        iter->current_item = &iter->object[iter->location];
        printf("Current item: %c\n", *iter->current_item);  // Use %c for chars
    } else {
        iter->current_item = NULL;   // End of the iterable
    }
}

// Peek ahead by n items
char Iterator_Peek(Iterator* iter, size_t n) {
    if (iter->location + n < iter->length) {
        return iter->object[iter->location + n];
    } else {
        return '\0';   // Return null char if out of bounds
    }
}

// Get the current item
char Iterator_Current(Iterator* iter) {
    if (iter->current_item != NULL) {
        return *iter->current_item;
    }
    return '\0';   // Return null char if iterator hasn't been advanced yet or has reached the end
}

// Get the length of the iterable
size_t Iterator_Length(Iterator* iter) {
    return iter->length;
}

void Iterator_Cleanup(Iterator* iter) {
    free(iter->object);
}

char* breakdownString(const char *input) {
    int length = strlen(input);
    
    // Allocate memory for the character array
    char *charArray = (char *)malloc((length + 1) * sizeof(char)); // +1 for the null terminator

    if (charArray == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    for(int i = 0; i < length; i++) {
        charArray[i] = input[i];
    }
    
    charArray[length] = '\0'; // Null-terminate the character array

    return charArray;
}