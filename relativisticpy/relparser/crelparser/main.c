// main.c
#include <stdio.h>
#include <string.h>
#include "lexer/char_iterator.h"
#include "lexer/token_provider.h"
#include "parser/token_iterator.h"

// Assuming you have definitions for the Iterator functions somewhere...

int main(int argc, char *argv[]) {
    // Check if an argument has been provided
    if (argc < 2) {
        printf("Please provide a string argument.\n");
        return 1; // Exit with an error code
    }

    const char *testStr = argv[1];

    CharIterator char_iterator; // Represents the String + Methods to allow us to peek etc...
    TokenList tokenList = {NULL, 0}; // As StringIterator is used, token_provider will generate tokens, appending them to this list -> pass a pointer.
    TokenIterator tokiter;

    init_char_iter(&char_iterator, testStr);

    advance_char(&char_iterator);

    while (current_char(&char_iterator) != '\0') {
        advance_char(&char_iterator);
    }

    free_char_iter(&char_iterator);

    return 0;
}
