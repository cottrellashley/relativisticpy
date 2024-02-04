from ..types.position import Position
from ..lexers.gr_lexer import GRLexer
from ..lexers.base import LexerResult, Token, TokenType
from typing import List
import pytest

def test_simple_token_generation():
    _int : LexerResult = GRLexer("1").tokenize()
    assert _int.code == "1"
    assert _int.tokens[0].type == TokenType.INT
    assert _int.tokens[0].value == '1'

    _float : LexerResult = GRLexer("31.12").tokenize()
    assert _float.code == "31.12"
    assert _float.tokens[0].type == TokenType.FLOAT
    assert _float.tokens[0].value == '31.12'

    _id : LexerResult = GRLexer("hello").tokenize()
    assert _id.code == "hello"
    assert _id.tokens[0].type == TokenType.ID
    assert _id.tokens[0].value == 'hello'

    _id_ : LexerResult = GRLexer("t_s").tokenize()
    assert _id_.code == "t_s"
    assert _id_.tokens[0].type == TokenType.ID
    assert _id_.tokens[0].value == 't_s'

    _tensor_id : LexerResult = GRLexer("T_{a}").tokenize()
    assert _tensor_id.code == "T_{a}"
    assert _tensor_id.tokens[0].type == TokenType.TENSORID
    assert _tensor_id.tokens[0].value == 'T'

    _func_id : LexerResult = GRLexer("f(a)").tokenize()
    assert _func_id.code == "f(a)"
    assert _func_id.tokens[0].type == TokenType.FUNCTIONID
    assert _func_id.tokens[0].value == 'f'

    _newline : LexerResult = GRLexer("\n").tokenize()
    assert _newline.code == "\n"
    assert _newline.tokens[0].type == TokenType.NEWLINE
    assert _newline.tokens[0].value == '\n'

    # Operations Tokens

    _star = GRLexer("*").tokenize()
    assert _star.code == "*"
    assert _star.tokens[0].type == TokenType.STAR
    assert _star.tokens[0].value == '*'

    _minus = GRLexer("-").tokenize()
    assert _minus.code == "-"
    assert _minus.tokens[0].type == TokenType.MINUS
    assert _minus.tokens[0].value == '-'

    _plus = GRLexer("+").tokenize()
    assert _plus.code == "+"
    assert _plus.tokens[0].type == TokenType.PLUS
    assert _plus.tokens[0].value == '+'

    _equal = GRLexer("=").tokenize()
    assert _equal.code == "="
    assert _equal.tokens[0].type == TokenType.EQUAL
    assert _equal.tokens[0].value == '='

    _lsqb = GRLexer("[").tokenize()
    assert _lsqb.code == "["
    assert _lsqb.tokens[0].type == TokenType.LSQB
    assert _lsqb.tokens[0].value == '['

    _rsqb = GRLexer("]").tokenize()
    assert _rsqb.code == "]"
    assert _rsqb.tokens[0].type == TokenType.RSQB
    assert _rsqb.tokens[0].value == ']'

    _lpar = GRLexer("(").tokenize()
    assert _lpar.code == "("
    assert _lpar.tokens[0].type == TokenType.LPAR
    assert _lpar.tokens[0].value == '('

    _rpar = GRLexer(")").tokenize()
    assert _rpar.code == ")"
    assert _rpar.tokens[0].type == TokenType.RPAR
    assert _rpar.tokens[0].value == ')'

    _lbrace = GRLexer("{").tokenize()
    assert _lbrace.code == "{"
    assert _lbrace.tokens[0].type == TokenType.LBRACE
    assert _lbrace.tokens[0].value == '{'

    _rbrace = GRLexer("}").tokenize()
    assert _rbrace.code == "}"
    assert _rbrace.tokens[0].type == TokenType.RBRACE
    assert _rbrace.tokens[0].value == '}'

    _circumflex = GRLexer("^").tokenize()
    assert _circumflex.code == "^"
    assert _circumflex.tokens[0].type == TokenType.CIRCUMFLEX
    assert _circumflex.tokens[0].value == '^'

    _slash = GRLexer("/").tokenize()
    assert _slash.code == "/"
    assert _slash.tokens[0].type == TokenType.SLASH
    assert _slash.tokens[0].value == '/'

    _vbar = GRLexer("|").tokenize()
    assert _vbar.code == "|"
    assert _vbar.tokens[0].type == TokenType.VBAR
    assert _vbar.tokens[0].value == '|'

    _amper = GRLexer("&").tokenize()
    assert _amper.code == "&"
    assert _amper.tokens[0].type == TokenType.AMPER
    assert _amper.tokens[0].value == '&'

    _exclamation = GRLexer("!").tokenize()
    assert _exclamation.code == "!"
    assert _exclamation.tokens[0].type == TokenType.EXCLAMATION
    assert _exclamation.tokens[0].value == '!'

    _tilde = GRLexer("~").tokenize()
    assert _tilde.code == "~"
    assert _tilde.tokens[0].type == TokenType.TILDE
    assert _tilde.tokens[0].value == '~'

    _greater = GRLexer(">").tokenize()
    assert _greater.code == ">"
    assert _greater.tokens[0].type == TokenType.GREATER
    assert _greater.tokens[0].value == '>'

    _less = GRLexer("<").tokenize()
    assert _less.code == "<"
    assert _less.tokens[0].type == TokenType.LESS
    assert _less.tokens[0].value == '<'

    _colon = GRLexer(":").tokenize()
    assert _colon.code == ":"
    assert _colon.tokens[0].type == TokenType.COLON
    assert _colon.tokens[0].value == ':'

    _dot = GRLexer(".").tokenize()
    assert _dot.code == "."
    assert _dot.tokens[0].type == TokenType.DOT
    assert _dot.tokens[0].value == '.'

    _comma = GRLexer(",").tokenize()
    assert _comma.code == ","
    assert _comma.tokens[0].type == TokenType.COMMA
    assert _comma.tokens[0].value == ','

    _semi : LexerResult = GRLexer(";").tokenize()
    assert _semi.code == ";"
    assert _semi.tokens[0].type == TokenType.SEMI
    assert _semi.tokens[0].value == ';'

    _at : LexerResult = GRLexer("@").tokenize()
    assert _at.code == "@"
    assert _at.tokens[0].type == TokenType.AT
    assert _at.tokens[0].value == '@'

    _percent : LexerResult = GRLexer("%").tokenize()
    assert _percent.code == "%"
    assert _percent.tokens[0].type == TokenType.PERCENT
    assert _percent.tokens[0].value == '%'

    _colon : LexerResult = GRLexer(":").tokenize()
    assert _colon.code == ":"
    assert _colon.tokens[0].type == TokenType.COLON
    assert _colon.tokens[0].value == ':'

    _under : LexerResult = GRLexer("_").tokenize()
    assert _under.code == "_"
    assert _under.tokens[0].type == TokenType.UNDER
    assert _under.tokens[0].value == '_'

    _lbrace : LexerResult = GRLexer("{").tokenize()
    assert _lbrace.code == "{"
    assert _lbrace.tokens[0].type == TokenType.LBRACE
    assert _lbrace.tokens[0].value == '{'

    _rbrace : LexerResult = GRLexer("}").tokenize()
    assert _rbrace.code == "}"
    assert _rbrace.tokens[0].type == TokenType.RBRACE
    assert _rbrace.tokens[0].value == '}'

    # Double Operators

    _not_equal : LexerResult = GRLexer("!=").tokenize()
    assert _not_equal.code == "!="
    assert _not_equal.tokens[0].type == TokenType.NOTEQUAL
    assert _not_equal.tokens[0].value == '!='

    _percent_equal : LexerResult = GRLexer("%=").tokenize()
    assert _percent_equal.code == "%="
    assert _percent_equal.tokens[0].type == TokenType.PERCENTEQUAL
    assert _percent_equal.tokens[0].value == '%='

    _amper_equal : LexerResult = GRLexer("&=").tokenize()
    assert _amper_equal.code == "&="
    assert _amper_equal.tokens[0].type == TokenType.AMPEREQUAL
    assert _amper_equal.tokens[0].value == '&='

    _plus_equal : LexerResult = GRLexer("+=").tokenize()
    assert _plus_equal.code == "+="
    assert _plus_equal.tokens[0].type == TokenType.PLUSEQUAL
    assert _plus_equal.tokens[0].value == '+='

    _colon_equal : LexerResult = GRLexer(":=").tokenize()
    assert _colon_equal.code == ":="
    assert _colon_equal.tokens[0].type == TokenType.COLONEQUAL
    assert _colon_equal.tokens[0].value == ':='

    _eq_equal : LexerResult = GRLexer("==").tokenize()
    assert _eq_equal.code == "=="
    assert _eq_equal.tokens[0].type == TokenType.EQEQUAL
    assert _eq_equal.tokens[0].value == '=='

    # _vbar_vbar : LexerResult = GRLexer("||").tokenize()
    # assert _vbar_vbar.code == "||"
    # assert _vbar_vbar.tokens[0].type == TokenType.VBARVBAR
    # assert _vbar_vbar.tokens[0].value == '||'

    _at_equal : LexerResult = GRLexer("@=").tokenize()
    assert _at_equal.code == "@="
    assert _at_equal.tokens[0].type == TokenType.ATEQUAL
    assert _at_equal.tokens[0].value == '@='

    _circumflex_equal : LexerResult = GRLexer("^=").tokenize()
    assert _circumflex_equal.code == "^="
    assert _circumflex_equal.tokens[0].type == TokenType.CIRCUMFLEXEQUAL
    assert _circumflex_equal.tokens[0].value == '^='

    _vbar_equal : LexerResult = GRLexer("|=").tokenize()
    assert _vbar_equal.code == "|="
    assert _vbar_equal.tokens[0].type == TokenType.VBAREQUAL
    assert _vbar_equal.tokens[0].value == '|='

    _double_star : LexerResult = GRLexer("**").tokenize()
    assert _double_star.code == "**"
    assert _double_star.tokens[0].type == TokenType.DOUBLESTAR
    assert _double_star.tokens[0].value == '**'

    _star_equal : LexerResult = GRLexer("*=").tokenize()
    assert _star_equal.code == "*="
    assert _star_equal.tokens[0].type == TokenType.STAREQUAL
    assert _star_equal.tokens[0].value == '*='

    _greater_equal : LexerResult = GRLexer(">=").tokenize()
    assert _greater_equal.code == ">="
    assert _greater_equal.tokens[0].type == TokenType.GREATEREQUAL
    assert _greater_equal.tokens[0].value == '>='

    _right_shift : LexerResult = GRLexer(">>").tokenize()
    assert _right_shift.code == ">>"
    assert _right_shift.tokens[0].type == TokenType.RIGHTSHIFT
    assert _right_shift.tokens[0].value == '>>'

    _double_slash : LexerResult = GRLexer("//").tokenize()
    assert _double_slash.code == "//"
    assert _double_slash.tokens[0].type == TokenType.DOUBLESLASH
    assert _double_slash.tokens[0].value == '//'

    _slash_equal : LexerResult = GRLexer("/=").tokenize()
    assert _slash_equal.code == "/="
    assert _slash_equal.tokens[0].type == TokenType.SLASHEQUAL
    assert _slash_equal.tokens[0].value == '/='

    _min_equal : LexerResult = GRLexer("-=").tokenize()
    assert _min_equal.code == "-="
    assert _min_equal.tokens[0].type == TokenType.MINEQUAL
    assert _min_equal.tokens[0].value == '-='

    _less_equal : LexerResult = GRLexer("<=").tokenize()
    assert _less_equal.code == "<="
    assert _less_equal.tokens[0].type == TokenType.LESSEQUAL
    assert _less_equal.tokens[0].value == '<='

    _left_shift_equal : LexerResult = GRLexer("<<=").tokenize()
    assert _left_shift_equal.code == "<<="
    assert _left_shift_equal.tokens[0].type == TokenType.LEFTSHIFTEQUAL
    assert _left_shift_equal.tokens[0].value == '<<='

    _double_star_equal : LexerResult = GRLexer("**=").tokenize()
    assert _double_star_equal.code == "**="
    assert _double_star_equal.tokens[0].type == TokenType.DOUBLESTAREQUAL
    assert _double_star_equal.tokens[0].value == '**='

    _ellipsis : LexerResult = GRLexer("...").tokenize()
    assert _ellipsis.code == "..."
    assert _ellipsis.tokens[0].type == TokenType.ELLIPSIS
    assert _ellipsis.tokens[0].value == '...'

    _double_slash_equal : LexerResult = GRLexer("//=").tokenize()
    assert _double_slash_equal.code == "//="
    assert _double_slash_equal.tokens[0].type == TokenType.DOUBLESLASHEQUAL
    assert _double_slash_equal.tokens[0].value == '//='

    _left_shift_equal : LexerResult = GRLexer("<<=").tokenize()
    assert _left_shift_equal.code == "<<="
    assert _left_shift_equal.tokens[0].type == TokenType.LEFTSHIFTEQUAL
    assert _left_shift_equal.tokens[0].value == '<<='

    _right_shift_equal : LexerResult = GRLexer(">>=").tokenize()
    assert _right_shift_equal.code == ">>="
    assert _right_shift_equal.tokens[0].type == TokenType.RIGHTSHIFTEQUAL
    assert _right_shift_equal.tokens[0].value == '>>='
