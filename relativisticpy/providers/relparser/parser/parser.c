#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "node_types.h"
#include "node_provider.h"
#include "../lexer/token_provider.h"
#include "../lexer/token_types.h"

void raise_error(const char* message) 
{
    // Assuming you've defined some error handling mechanism
    fprintf(stderr, "%s\n", message);
    exit(1);
}

Node* parse(TokenIterator* iterating_tokens) 
{
    Node* result;

    // Setting the class field to the passed-in iterator
    this->iterating_tokens = iterating_tokens;
    iterating_tokens->advance();

    if (iterating_tokens->current() == NULL_PTR) 
    {
        return NULL_PTR;
    }

    int len = iiiterator_length(iterating_tokens); // Assuming there's some way to get the length of the iterator

    if (len >= 3) 
    {
        result = equation();
    } 
    else if (len < 3) 
    {
        result = statement();
    } 
    else if (iterating_tokens->current() != NULL_PTR) 
    {
        raise_error("Syntax Error");
    }
    
    return result;
}

Token* token_new(TokenType type, const char* value) 
{
    Token* token = malloc(sizeof(Token));
    token->type = type;
    // Assuming value is a null-terminated string and using strdup
    token->value = strdup(value);
    return token;
}

void token_free(Token* token) 
{
    if (token != NULL) 
    {
        free(token->value);
        free(token);
    }
}

Node* equation(void) 
{
    Token* current_token = iterating_tokens.current();
    Token* next_token = iterating_tokens.peek(1);
    // If the next_token is NULL, set it to a NONE token.
    if (next_token == NULL) 
    {
        next_token = token_new(NONE, "NONE");
    }
    
    if (current_token->type == OBJECT && next_token->type == EQUAL) 
    {
        Node* subject_variable = object();
        iterating_tokens.advance();
        Node* result = node_provider.new_node(EQUALS_NODE, subject_variable, statement());
        // If you allocated the NONE token earlier, free it now
        if (next_token->type == NONE) 
        {
            token_free(next_token);
        }
        return result;
    } 
    else if (current_token->type == OBJECT && next_token->type == COLONEQUAL) 
    {
        Node* subject_variable = object();
        iterating_tokens.advance();
        Node* result = node_provider.new_node(ASSIGNMENT_NODE, subject_variable, statement());
        // If you allocated the NONE token earlier, free it now
        if (next_token->type == NONE) 
        {
            token_free(next_token);
        }
        return result;
    } 
    else 
    {
        // If you allocated the NONE token earlier, free it now
        if (next_token->type == NONE) 
        {
            token_free(next_token);
        }
        return statement();
    }
}

Node* array(void) 
{
    NodeList* elements = create_node_list(10);  // Assuming an initial capacity of 10
    
    Token* current_token;
    if ((current_token = iterating_tokens.current())->type != RSQB) 
    {
        append_to_node_list(elements, statement());

        while ((current_token = iterating_tokens.current()) != NULL && current_token->type == COMMA) 
        {
            iterating_tokens.advance();
            append_to_node_list(elements, statement());
        }
    }

    if (current_token->type != RSQB) 
    {
        // Handle error: Raise error here, assuming you have a method for that.
        fprintf(stderr, "Syntax Error, expecting a CLOSED_SQUARE_BRACE token.");
        exit(1); // Or handle error as you see fit
    }

    iterating_tokens.advance();
    Node* result = node_provider.new_node(ARRAY_NODE, elements);

    // Free the elements list but not the nodes inside it
    free(elements->nodes);
    free(elements);

    return result;
}

Node* statement(void) 
{
    Node* result = bool_expr();

    Token* current_token;
    while ((current_token = iterating_tokens.current()) != NULL &&
           (current_token->type == AMPER || current_token->type == VBAR)) 
    {

        Node* array[2];
        array[0] = result;

        if (current_token->type == AMPER) 
        {
            iterating_tokens.advance();
            array[1] = bool_expr();
            result = node_provider.new_node(AND_NODE, array);
        } 
        else if (current_token->type == VBAR) 
        {
            iterating_tokens.advance();
            array[1] = bool_expr();
            result = node_provider.new_node(OR_NODE, array);
        }
    }

    return result;
}

Node* bool_expr(void) 
{
    Node* result = expr();

    Token* current_token;
    while ((current_token = iterating_tokens.current()) != NULL &&
           (current_token->type == LESS || current_token->type == LESSEQUAL ||
            current_token->type == GREATER || current_token->type == GREATEREQUAL ||
            current_token->type == EQEQUAL || current_token->type == NOTEQUAL)) 
    {

        Node* array[2];
        array[0] = result;

        if (current_token->type == LESS) 
        {
            iterating_tokens.advance();
            array[1] = expr();
            result = node_provider.new_node(LESS_NODE, array);
        } 
        else if (current_token->type == LESSEQUAL) 
        {
            iterating_tokens.advance();
            array[1] = expr();
            result = node_provider.new_node(LESSEQUAL_NODE, array);
        } 
        else if (current_token->type == GREATER) 
        {
            iterating_tokens.advance();
            array[1] = expr();
            result = node_provider.new_node(GREATER_NODE, array);
        } 
        else if (current_token->type == GREATEREQUAL) 
        {
            iterating_tokens.advance();
            array[1] = expr();
            result = node_provider.new_node(GREATEREQUAL_NODE, array);
        } 
        else if (current_token->type == EQEQUAL) 
        {
            iterating_tokens.advance();
            array[1] = expr();
            result = node_provider.new_node(EQEQUAL_NODE, array);
        } 
        else if (current_token->type == NOTEQUAL) 
        {
            iterating_tokens.advance();
            array[1] = expr();
            result = node_provider.new_node(NOTEQUAL_NODE, array);
        }
    }

    return result;
}

Node* expr(void) 
{
    Node* result = term();

    // Loop while current token is not NULL and token type is PLUS or MINUS
    Token* current_token;
    while ((current_token = iterating_tokens.current()) != NULL &&
           (current_token->type == PLUS || current_token->type == MINUS)) 
    {
        if (current_token->type == PLUS) 
        {
            iterating_tokens.advance();

            Node* array[2];
            array[0] = result;
            array[1] = term();

            result = node_provider.new_node(PLUS_NODE, array);
        } 
        else if (current_token->type == MINUS) 
        {
            iterating_tokens.advance();

            Node* array[2];
            array[0] = result;
            array[1] = term();

            result = node_provider.new_node(MINUS_NODE, array);
        }
    }

    return result;
}

Node* term(void) 
{
    Node* result = power();

    // Loop while current token is not NULL and token type is STAR or SLASH
    Token* current_token;
    while ((current_token = iterating_tokens.current()) != NULL &&
           (current_token->type == STAR || current_token->type == SLASH)) 
    {
        if (current_token->type == STAR) 
        {
            iterating_tokens.advance();

            Node* array[2];
            array[0] = result;
            array[1] = power();

            result = node_provider.new_node(MULTIPLY_NODE, array);
        } 
        else if (current_token->type == SLASH) 
        {
            iterating_tokens.advance();

            Node* array[2];
            array[0] = result;
            array[1] = power();

            result = node_provider.new_node(DIVIDE_NODE, array);
        }
    }

    return result;
}

Node* power(void) 
{
    Node* result = object();
    
    // Loop while current token is not NULL and token type is DOUBLESTAR or CIRCUMFLEX
    Token* current_token;
    while ((current_token = iterating_tokens.current()) != NULL && (current_token->type == DOUBLESTAR || current_token->type == CIRCUMFLEX)) 
    {
        if (current_token->type == CIRCUMFLEX) 
        {
            iterating_tokens.advance();
            
            Node* array[2];
            array[0] = result;
            array[1] = object();
            
            result = node_provider.new_node(EXPONENTIATION1_NODE, array);
        } 
        else if (current_token->type == DOUBLESTAR) 
        {
            iterating_tokens.advance();
            
            Node* array[2];
            array[0] = result;
            array[1] = object();
            
            result = node_provider.new_node(EXPONENTIATION2_NODE, array);
        }
    }
    
    return result;
}


Node* object(void) 
{
    Token* token = this->iterating_tokens->current();

    switch (token->type) 
    {
        case FLOAT:
            this->iterating_tokens->advance();
            return this->node_provider->new_node(FLOAT_NODE, token);
            
        case INTEGER:
            this->iterating_tokens->advance();
            return this->node_provider->new_node(INTEGER_NODE, token);

        case OBJECT:
            this->iterating_tokens->advance();
            return this->node_provider->new_node(OBJECT_NODE, token);

        case PLUS:
            this->iterating_tokens->advance();
            return this->node_provider->new_node(POSITIVE_NODE, object());

        case MINUS:
            this->iterating_tokens->advance();
            return this->node_provider->new_node(MINUS_NODE, object());

        case LSQB:
            {
                this->iterating_tokens->advance();
                Node** elements = NULL; // Assuming a dynamic array implementation
                int element_count = 0;

                while (this->iterating_tokens->current() != NULL_PTR && this->iterating_tokens->current()->type != RSQB) 
                {
                    add_to_elements(&elements, &element_count, statement());  // Assuming a function to add to the dynamic array
                    if (this->iterating_tokens->current()->type == COMMA) 
                    {
                        this->iterating_tokens->advance();
                    }
                }

                if (this->iterating_tokens->current()->type != RSQB) 
                {
                    raise_error("Syntax Error, expecting a CLOSED_SQUARE_BRACE token.");
                }

                this->iterating_tokens->advance();
                return this->node_provider->new_node(ARRAY_NODE, elements);
            }

        case FUNCTION:
            {
                this->iterating_tokens->advance();
                char* func_name = token->value;

                if (this->iterating_tokens->current()->type != LPAR) 
                {
                    raise_error("Syntax Error, expecting a OPEN_BRACE token.");
                }

                this->iterating_tokens->advance();
                Node** wrt_variables = find_variables(); // Assuming you have this function defined

                if (this->iterating_tokens->current()->type != RPAR) 
                {
                    raise_error("Syntax Error, expecting a CLOSE_BRACE token.");
                }

                this->iterating_tokens->advance();
                return this->node_provider->new_node(FUNCTION_NODE, func_name, wrt_variables);
            }

        case LPAR:
            {
                this->iterating_tokens->advance();
                Node* result = expr();

                if (this->iterating_tokens->current()->type != RPAR) 
                {
                    raise_error("Syntax Error, expecting a OPEN_BRACE token.");
                }

                this->iterating_tokens->advance();
                return result;
            }

        default:
            return NULL_PTR; // Or handle as appropriate for unexpected token types
    }
    
}