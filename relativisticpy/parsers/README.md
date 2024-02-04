# Lexer

The lexer (for simplicity reasons) is very slightly different than normal lexer when it comes to the IDENTIFIER token. Namely, we distinguish at the tokenizer level:

- IDENTIFIER = ID
- TENSORID = ID followed by ("_" or "^") followed by "{"
- FUNCTIONID = ID followed by "("

# Parser

# Semantic Analyzer

# Interpreter (actually just a traverser)

