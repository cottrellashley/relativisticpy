#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "lexer.h"
#include "characters.h"
#include "iterator.h"

// Constructor
Lexer* new_lexer(struct TokenProvider* p_token_provider) 
{
    Lexer* p_lexer = malloc(sizeof(Lexer));
    p_lexer->p_token_provider = p_token_provider;

    return p_lexer;
}

/*
 * Lexer_delete()
 * Will free up the memory space of the lexer struct.
 */
void free_lexer(Lexer* p_lexer) 
{
    free(p_lexer);
}

// Methods translated to functions
void tokenize_string(Lexer* p_lexer, struct Iterator* p_characters) 
{
    p_lexer->p_characters = p_characters;
    advance_char(p_lexer->p_characters);

    while (current_char(p_lexer->p_characters) != NULL) 
    {
        if (strchr(WHITESPACE, current_char(p_lexer->p_characters))) 
        {
            advance_char(p_lexer->p_characters); // If we have any whitespace, just continue without doing anything.
        } 
        else if (strchr(LETTERS, current_char(p_lexer->p_characters))) 
        {
            generate_object_token(p_lexer);
        } 
        else if (strchr(DIGITS, current_char(p_lexer->p_characters))) 
        {
            generate_number_token(p_lexer);
        } 
        else if (strchr(OPERATIONS, current_char(p_lexer->p_characters))) 
        {
            Lexer_operation(p_lexer);
        }

    }

}

void generate_number_token(Lexer* p_lexer) 
{
    char* number = malloc(50);  // assuming maximum length of 50 for simplicity
    int count = 0;

    while (current_char(p_lexer->p_characters) != NULL && (is_digit(current_char(p_lexer->p_characters)) || *current_char(p_lexer->p_characters) == '.')) 
    {
        strcat(number, current_char(p_lexer->p_characters));
        advance_char(p_lexer->p_characters);
    }

    for (int i = 0; i < strlen(number); i++) 
    {
        if (number[i] == '.') 
        {
            count++;
        }
    }
    if (count == 0) 
    {
        new_token(p_lexer->p_token_provider, TokenType_INTEGER, number);
    } 
    else if (count == 1) 
    {
        new_token(p_lexer->p_token_provider, TokenType_FLOAT, number);
    }
    else
    {
        printf("Illegal Character '%s'", number);

        exit(EXIT_FAILURE);
    }

    free(number);
}

// Assuming max ops length of 100 for simplicity
void generate_operaion_token(Lexer* p_lexer) 
{
    char ops[100];
    int ops_len = 0;

    while (current_char(p_lexer->p_characters) != NULL && is_operation(current_char(p_lexer->p_characters))) 
    {
        ops[ops_len++] = *current_char(p_lexer->p_characters);
        advance_char(p_lexer->p_characters);
    }

    ops[ops_len] = '\0';  // Null terminate the ops string
    int i = 0;

    while (i < ops_len) 
    {
        if (i + 2 < ops_len && is_triple_char_op(p_lexer->p_token_provider, ops[i]) && triple_match_exists(p_lexer->p_token_provider, ops[i], ops[i+1], ops[i+2])) 
        {
            new_tripple_token(p_lexer->p_token_provider, ops[i], ops[i+1], ops[i+2]);
            i += 3;
        } 
        else if (i + 1 < ops_len && is_double_char_op(p_lexer->p_token_provider, ops[i]) && double_match_exists(p_lexer->p_token_provider, ops[i], ops[i+1])) 
        {
            new_double_token(p_lexer->p_token_provider, ops[i], ops[i+1]);
            i += 2;
        } 
        else 
        {
            new_single_token(p_lexer->p_token_provider, ops[i]);
            i += 1;
        }
    
    }

}

void generate_object_token(Lexer* p_lexer) 
{
    char obj[100];  // Assuming max object length of 100 characters to prevent crazy people.
    int obj_len = 0; // The length of character iterate the object.

    while (current_char(p_lexer->p_characters) != NULL && is_object_char(current_char(p_lexer->p_characters))) 
    {
        if (is_character_char(current_char(p_lexer->p_characters)) && peek_char(p_lexer->p_characters, 1) == '(') 
        {
            obj[obj_len++] = *current_char(p_lexer->p_characters);
            advance_char(p_lexer->p_characters);
            obj[obj_len] = '\0';  // Null terminate the obj string
            new_token(p_lexer->p_token_provider, TokenType_FUNCTION, obj);
        } 
        else if (!is_object_char(peek_char(p_lexer->p_characters, 1)) && peek_char(p_lexer->p_characters, 1) != '(') 
        {
            obj[obj_len++] = *current_char(p_lexer->p_characters);
            advance_char(p_lexer->p_characters);
            obj[obj_len] = '\0';  // Null terminate the obj string
            new_token(p_lexer->p_token_provider, OBJECT, obj);

        } 
        else 
        {
            obj[obj_len++] = *current_char(p_lexer->p_characters);
            advance_char(p_lexer->p_characters);
        }

    }

}
