#include "token_types.h"
#include "token_provider.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

TokenList tokenList = {NULL, 0};

void new_token(char* type, char *value) 
{
    Token newToken = {type, value};
    tokenList.tokens = realloc(tokenList.tokens, sizeof(Token) * (tokenList.size + 1));
    tokenList.tokens[tokenList.size++] = newToken;
}

int tokens_len(TokenList *p_token_list)
{
    return 10; // Please IMPLEMENT!!
}

bool single_match_exists(char c1) 
{
    return one_char_token(c1) != NONE;
}

bool double_match_exists(char c1, char c2) 
{
    return two_char_token(c1, c2) != NONE;
}

bool tripple_match_exists(char c1, char c2, char c3) 
{
    return three_char_token(c1, c2, c3) != NONE;
}

void new_single_operation_token(char c1) 
{
    if (single_match_exists(c1)) 
    {
        new_token(one_char_token(c1), &c1);
    }
}

void new_double_operation_token(char c1, char c2) 
{
    if (double_match_exists(c1, c2)) 
    {
        char value[3] = {c1, c2, '\0'};
        new_token(two_char_token(c1, c2), value);
    }
}

void new_tripple_operation_token(char c1, char c2, char c3) 
{
    if (tripple_match_exists(c1, c2, c3)) 
    {
        char value[4] = {c1, c2, c3, '\0'};
        new_token(three_char_token(c1, c2, c3), value);
    }
}

Token* get_tokens() 
{
    return tokenList.tokens;
}

void free_tokens() 
{
    free(tokenList.tokens);
    tokenList.size = 0;
}


int one_char_token(int c1)
{
    switch (c1) 
    {
        case '!': return EXCLAMATION;
        case '%': return PERCENT;
        case '&': return AMPER;
        case '(': return LPAR;
        case ')': return RPAR;
        case '*': return STAR;
        case '+': return PLUS;
        case ',': return COMMA;
        case '-': return MINUS;
        case '.': return DOT;
        case '/': return SLASH;
        case ':': return COLON;
        case ';': return SEMI;
        case '<': return LESS;
        case '=': return EQUAL;
        case '>': return GREATER;
        case '@': return AT;
        case '[': return LSQB;
        case ']': return RSQB;
        case '^': return CIRCUMFLEX;
        case '{': return LBRACE;
        case '|': return VBAR;
        case '}': return RBRACE;
        case '~': return TILDE;
    }

    return NONE;
}

int two_char_token(int c1, int c2)
{
    switch (c1) 
    {
        case '!': 
            switch (c2) 
            {
                case '=': return NOTEQUAL;
            }
            break;
        case '%':
            switch (c2) 
            {
                case '=': return PERCENTEQUAL;
            }
            break;
        case '&':
            switch (c2) 
            {
                case '=': return AMPEREQUAL;
            }
            break;
        case '*':
            switch (c2) 
            {
                case '*': return DOUBLESTAR;
                case '=': return STAREQUAL;
            }
            break;
        case '+':
            switch (c2) 
            {
                case '=': return PLUSEQUAL;
            }
            break;
        case '-':
            switch (c2) 
            {
                case '=': return MINEQUAL;
                case '>': return RARROW;
            }
            break;
        case '/':
            switch (c2) 
            {
                case '/': return DOUBLESLASH;
                case '=': return SLASHEQUAL;
            }
            break;
        case ':':
            switch (c2) 
            {
                case '=': return COLONEQUAL;
            }
            break;
        case '<':
            switch (c2) 
            {
                case '<': return LEFTSHIFT;
                case '=': return LESSEQUAL;
                case '>': return NOTEQUAL;
            }
            break;
        case '=':
            switch (c2) 
            {
                case '=': return EQEQUAL;
            }
            break;
        case '>':
            switch (c2) 
            {
                case '=': return GREATEREQUAL;
                case '>': return RIGHTSHIFT;
            }
            break;
        case '@':
            switch (c2) 
            {
                case '=': return ATEQUAL;
            }
            break;
        case '^':
            switch (c2) 
            {
                case '=': return CIRCUMFLEXEQUAL;
            }
            break;
        case '|':
            switch (c2) 
            {
                case '=': return VBAREQUAL;
            }
            break;
    }
    return NONE;
}

int three_char_token(int c1, int c2, int c3)
{
    switch (c1) 
    {
    case '*':
        switch (c2) 
        {
        case '*':
            switch (c3) 
            {
                case '=': return DOUBLESTAREQUAL;
            }
            break;
        }
        break;
    case '.':
        switch (c2) 
        {
        case '.':
            switch (c3) 
            {
                case '.': return ELLIPSIS;
            }
            break;
        }
        break;
    case '/':
        switch (c2) 
        {
        case '/':
            switch (c3) 
            {
                case '=': return DOUBLESLASHEQUAL;
            }
            break;
        }
        break;
    case '<':
        switch (c2) 
        {
        case '<':
            switch (c3) 
            {
                case '=': return LEFTSHIFTEQUAL;
            }
            break;
        }
        break;
    case '>':
        switch (c2) 
        {
        case '>':
            switch (c3) 
            {
                case '=': return RIGHTSHIFTEQUAL;
            }
            break;
        }
        break;
    }
    return NONE;
}