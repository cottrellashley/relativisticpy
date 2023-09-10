#ifndef NODE_TYPES_H
#define NODE_TYPES_H

// An enumeration of node types used by a parser. These strings will be surfaced to AST passed to Python object.
#define SMALLER             '<'
#define SMALLERTHANOREQUAL  "<="
#define BIGGER              '>'
#define BIGGERTHANOREQUAL   ">="
#define ASSIGNMENT          ":="
#define EQUALEQUAL          "=="
#define NOTEQUAL            "!="
#define ADD                 '+'        // Addition operator
#define SUBTRACT            '-'        // Subtraction operator
#define MULTIPLY            '*'        // Multiplication operator
#define DIVIDE              '/'        // Division operator
#define EQUALS              '='        // The `=` symbol for assignment or comparison
#define EXPONENTIATION1     '^'        // The `^` symbol for exponentiation
#define EXPONENTIATION2     "**"
#define POSITIVE            "positive" // Positive operator '+'
#define AND                 '&'        // Logical AND operator
#define OR                  '|'        // Logical OR operator

#endif