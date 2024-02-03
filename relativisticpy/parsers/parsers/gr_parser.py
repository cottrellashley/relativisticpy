################################################################################
##################### RELATIVITY PARSER GRAMMAR ################################
################################################################################
#
# statements      :   NEWLINE* statement (NEWLINE* statement)
#
# statement       :   KEYWORD:print? expr
#
# expr            :   (ID|TENSORID) EQUAL expr
#                 :   bool-expr ((KEYWORD:and|KEYWORD:or) bool-expr)*
#
# bool-expr       :   NOT bool-expr
#                 :   arith-expr ((EQEQUAL|LESS|GREATER|LESSEQUAL|GREATEREQUAL) arith-expr)*
#
# arith-expr      :   term ((PLUS|MINUS) term)*
#
# term            :   factor ((MUL|DIV) factor)*
#
# factor          :   (PLUS|MINUS) factor
#                 :   power
#
# power           :   atom ((CIRCUMFLEX|DOUBLESTAR) atom)*
#
# atom            :   INT|FLOAT|STRING|BOOL|ID|TENSORID|FUNCTIONID|LATEXID
#                 :   LPAR expr RPAR
#                 :   array
#                 :   func-def
#                 :   tensor
#
# tensor          :   TENSORID ((UNDER|CIRCUMFLEX) LBRACE ID ((EQUAL|COLON) (INT|atom))? RBRACE )*
#
# array           :   LPAR (expr ((COMMA) expr*)? RPAR)
#
# func-def        :   FUNCTIONID? LPAR (ID (COMMA ID)*)? RPAR (EQUAL expr NEWLINE)


from typing import List
from relativisticpy.parsers.parsers.base import BaseParser, ParserResult
from relativisticpy.parsers.lexers.base import Token, TokenType
from relativisticpy.parsers.types.base import AstNode, NodeType, UnaryNode, BinaryNode, ArrayNode
from relativisticpy.parsers.types.gr_nodes import TensorNode, Function, Definition
from relativisticpy.parsers.shared.error_messages import braces_unmatched_errors
from relativisticpy.parsers.types.position import Position


class GRParser(BaseParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse(self):
        if self.current_token == None:
            return None

        result = self.statements()
        return ParserResult(self.raw_code, result)

    def statements(self):
        statements = []
        start_position = self.current_token.start_position.copy()

        while self.current_token.type == TokenType.NEWLINE:
            self.advance_token()

        statement = self.statement()
        statements.append(statement)

        more_statements = True

        while True:
            newline_count = 0

            if self.current_token == None:
                break

            while (
                self.current_token != None
                and self.current_token.type == TokenType.NEWLINE
            ):
                self.advance_token()
                newline_count += 1
            if newline_count == 0:
                more_statements = False

            if not more_statements:
                break
            if not self.current_token:
                break
            statement = self.statement()
            if not statement:
                more_statements = False
                continue
            statements.append(statement)

        return statements

    ####################################### STATEMENT ##############################################
    # statement       :   KEYWORD:print? expr
    ################################################################################################
    def statement(self):
        start_position = self.current_token.start_position.copy()

        if self.current_token.type == TokenType.PRINT:
            self.advance_token()
            return self.new_print_node(start_position, [self.expr()])

        return self.expr()

    ###################################### EXPRESSIONS #############################################
    # expr            :   (ID|TENSORID) EQUAL expr
    #                 :   bool-expr ((KEYWORD:and|KEYWORD:or) bool-expr)*
    ################################################################################################
    def expr(self):
        start_position = self.current_token.start_position.copy()

        if self.current_token.type == TokenType.ID and self.peek(
            1, Token(TokenType.NONE, "")
        ).type in (TokenType.EQUAL, TokenType.COLONEQUAL):
            token = self.current_token
            # Skip over as we already know Token will be EQUAL
            self.advance_token()
            if self.current_token.type == TokenType.EQUAL:
                self.advance_token()
                return self.new_assignment_node(
                    start_position,
                    [
                        token.value,
                        self.expr(),
                    ],
                )
            elif self.current_token.type == TokenType.COLONEQUAL:
                self.advance_token()
                return self.new_definition_node(
                    start_position,
                    [
                        token.value,
                        self.expr(),
                    ],
                )

        else:
            result = self.bool_expr()

            while self.current_token != None and self.current_token.type in (
                TokenType.AND,
                TokenType.OR,
            ):
                if self.current_token.type == TokenType.AND:
                    self.advance_token()
                    result = self.new_and_node(
                        start_position, [result, self.bool_expr()]
                    )

                elif self.current_token.type == TokenType.OR:
                    self.advance_token()
                    result = self.new_and_node(
                        start_position, [result, self.bool_expr()]
                    )

            return result

    ############################## BOOLEAN EXPRESSION #############################################
    # bool-expr       :   NOT bool-expr
    #                 :   arith-expr ((EQEQUAL|LESS|GREATER|LESSEQUAL|GREATEREQUAL) arith-expr)*
    ################################################################################################
    def bool_expr(self):
        start_position = self.current_token.start_position.copy()

        if self.current_token.type == TokenType.NOT:
            self.advance_token()
            return self.new_not_node(start_position, [self.bool_expr()])

        result = self.arith_expr()
        while self.current_token != None and self.current_token.type in (
            TokenType.EQEQUAL,
            TokenType.LESS,
            TokenType.GREATER,
            TokenType.LESSEQUAL,
            TokenType.GREATEREQUAL,
        ):
            if self.current_token.type == TokenType.EQEQUAL:
                self.advance_token()
                result = self.new_eqequal_node(
                    start_position, [result, self.arith_expr()]
                )
            elif self.current_token.type == TokenType.LESS:
                self.advance_token()
                result = self.new_less_node(start_position, [result, self.arith_expr()])
            elif self.current_token.type == TokenType.GREATER:
                self.advance_token()
                result = self.new_greater_node(
                    start_position, [result, self.arith_expr()]
                )
            elif self.current_token.type == TokenType.LESSEQUAL:
                self.advance_token()
                result = self.new_lessequal_node(
                    start_position, [result, self.arith_expr()]
                )
            elif self.current_token.type == TokenType.GREATEREQUAL:
                self.advance_token()
                result = self.new_greaterequal_node(
                    start_position, [result, self.arith_expr()]
                )
        return result

    # arith-expr : term ((PLUS|MINUS) term)*
    def arith_expr(self):
        start_position = self.current_token.start_position.copy()
        result = self.term()

        while self.current_token != None and self.current_token.type in (
            TokenType.PLUS,
            TokenType.MINUS,
        ):
            if self.current_token.type == TokenType.PLUS:
                self.advance_token()
                result = self.new_add_node(start_position, [result, self.term()])
            elif self.current_token.type == TokenType.MINUS:
                self.advance_token()
                result = self.new_sub_node(start_position, [result, self.term()])
        return result

    def term(self):
        # Look for a factor and store it in result
        start_position = self.current_token.start_position.copy()
        result = self.factor()

        while self.current_token != None and self.current_token.type in (
            TokenType.STAR,
            TokenType.SLASH,
        ):
            if self.current_token.type == TokenType.STAR:
                self.advance_token()
                result = self.new_mul_node(start_position, [result, self.factor()])
            elif self.current_token.type == TokenType.SLASH:
                self.advance_token()
                result = self.new_div_node(start_position, [result, self.factor()])
        return result

    def factor(self):
        start_position = self.current_token.start_position.copy()

        if self.current_token.type == TokenType.PLUS:
            self.advance_token()
            return self.new_pos_node(start_position, [self.factor()])
        elif self.current_token.type == TokenType.MINUS:
            self.advance_token()
            return self.new_neg_node(start_position, [self.factor()])
        return self.power()

    def power(self):
        start_position = self.current_token.start_position.copy()
        result = self.atom()
        # Look for additional powers and construct power node
        while self.current_token != None and self.current_token.type in (
            TokenType.DOUBLESTAR,
            TokenType.CIRCUMFLEX,
        ):
            if self.current_token.type in [TokenType.CIRCUMFLEX, TokenType.DOUBLESTAR]:
                self.advance_token()
                result = self.new_pow_node(start_position, [result, self.atom()])
        return result

    def atom(self):
        token: Token = self.current_token
        start_position = self.current_token.start_position.copy()

        if token.type == TokenType.FLOAT:
            self.advance_token()
            return self.new_float_node(start_position, [token.value])

        elif token.type == TokenType.INT:
            self.advance_token()
            return self.new_int_node(start_position, [token.value])

        elif token.type == TokenType.ID:
            self.advance_token()
            return self.new_symbol_node(start_position, [token.value])

        elif token.type == TokenType.TENSORID:
            self.advance_token()
            return self.tensor(token, start_position)

        elif token.type == TokenType.LSQB:
            self.advance_token()
            return self.array(start_position)

        elif token.type == TokenType.FUNCTIONID:
            self.advance_token()
            return self.function(token, start_position)

        elif token.type == TokenType.LPAR:
            self.advance_token()
            result = self.expr()
            if self.current_token.type != TokenType.RPAR:
                return self.invalid_syntax_error(
                    "Syntax Error, expecting a '(' token.",
                    start_position,
                    self.current_token.end_position.copy(),
                    self.raw_code,
                )
            self.advance_token()
            return result

        else:
            return self.illegal_character_error(
                f"""Syntax Error, the type {token.value} at Line: {self.current_token.end_position.copy().line}, 
                Position: {self.current_token.end_position.copy().character} is not recognized or supported by RelativisticPy's Parser.""",
                start_position,
                self.current_token.end_position.copy(),
                self.raw_code,
            )

    def array(self, start_position):
        elements = []
        if self.current_token.type != TokenType.RSQB:
            while (
                self.current_token != None
                and self.current_token.type == TokenType.NEWLINE
            ):
                self.advance_token()

            elements.append(self.statement())
            while (
                self.current_token != None
                and self.current_token.type == TokenType.COMMA
            ):
                self.advance_token()
                while (
                    self.current_token != None
                    and self.current_token.type == TokenType.NEWLINE
                ):
                    self.advance_token()

                elements.append(self.statement())

                while (
                    self.current_token != None
                    and self.current_token.type == TokenType.NEWLINE
                ):
                    self.advance_token()

            while (
                self.current_token != None
                and self.current_token.type == TokenType.NEWLINE
            ):
                self.advance_token()
        if (
            self.current_token == None
        ):  ############################ <<<<<<<<<<<<<<<---------- PLEASE ADD THESE EVERY OTHER ERROR RAISING PLACE.
            return self.invalid_syntax_error(
                braces_unmatched_errors,
                start_position,
                self.peek_prev_token(ignore_NEWLINE=True).end_position.copy(),
                self.raw_code,
            )
        if self.current_token.type != TokenType.RSQB:
            return self.invalid_syntax_error(
                braces_unmatched_errors,
                start_position,
                self.current_token.end_position.copy(),
                self.raw_code,
            )
        self.advance_token()


        return self.new_array_node(start_position, elements)

    # tensor : TENSORID ((UNDER|CIRCUMFLEX) LBRACE ID ((EQUAL|COLON) (INT|atom))? RBRACE )*
    def tensor(self, token: Token, start_position):
        tensor_node = TensorNode()
        tensor_node.identifier = token.value
        tensor_node.position = self.current_token.start_position.copy()
        tensor_covariant = True

        while self.current_token != None and self.current_token.type in (
            TokenType.UNDER,
            TokenType.CIRCUMFLEX,
        ):
            if self.current_token.type not in [TokenType.UNDER, TokenType.CIRCUMFLEX]:
                return self.invalid_syntax_error(
                    "Syntax Error, expecting a '_' or '^' token. Please check documentation for tensor syntax.",
                    start_position,
                    self.current_token.end_position.copy(),
                    self.raw_code,
                )
            tensor_covariant = self.current_token.type == TokenType.UNDER

            self.advance_token()
            if self.current_token.type != TokenType.LBRACE:
                return self.invalid_syntax_error(
                    "Syntax Error, expecting a token: '{'",
                    start_position,
                    self.current_token.end_position.copy(),
                    self.raw_code,
                )

            self.advance_token()
            if self.current_token.type != TokenType.ID:
                return self.invalid_syntax_error(
                    "Syntax Error, expecting a IDENTIFIER: i.e. some variable.",
                    start_position,
                    self.current_token.start_position.copy(),
                    self.raw_code,
                )

            while self.current_token.type == TokenType.ID or self.peek(
                1, Token(TokenType.NONE, "")
            ).type in [
                TokenType.UNDER,
                TokenType.CIRCUMFLEX,
            ]:  # T_{a b} => Index 'a' and Index 'b' are covariant.
                if self.current_token.type == TokenType.ID:
                    index = self.current_token.value
                    value = None

                    self.advance_token()
                    if self.current_token.type in (TokenType.EQUAL, TokenType.COLON):
                        self.advance_token()
                        value = self.atom()

                    tensor_node.new_index(
                        identifier=index, covariant=tensor_covariant, values=value
                    )
                elif self.current_token.type == TokenType.RBRACE:
                    self.advance_token()
                    tensor_covariant = (
                        self.current_token.type == TokenType.UNDER
                    )  # reset the covariance
                    self.advance_token()
                    if self.current_token.type != TokenType.LBRACE:
                        return self.invalid_syntax_error(
                            "Syntax Error, expecting a token: '{'",
                            start_position,
                            self.current_token.end_position.copy(),
                            self.raw_code,
                        )
                    self.advance_token()

        if self.current_token.type not in [
            TokenType.RBRACE
        ]:  # Should enforce the end of the _{a b} indices obejct.
            return self.invalid_syntax_error(
                "Syntax Error, expecting a token: '}'",
                start_position,
                self.current_token.end_position.copy(),
                self.raw_code,
            )

        self.advance_token()

        if self.current_token == None:
            return tensor_node

        if self.current_token.type == TokenType.COLONEQUAL:
            self.advance_token()
            if self.current_token.type != TokenType.LSQB:
                return self.invalid_syntax_error(
                    "Syntax Error, expecting a token: '[' ",
                    start_position,
                    self.current_token.end_position.copy(),
                    self.raw_code,
                )
            
            tensor_node.tensor_components_ast = self.atom() # We do not need to perform the definiton check until the last minute -> just do if tensor.id = metric -> set metric in workbook state
            tensor_node.callback = 'tensor_assignment'
            return tensor_node

        if self.current_token.type == TokenType.EQUAL:
            self.advance_token()
            if self.current_token.type == TokenType.LSQB:
                tensor_node.tensor_components_ast = self.atom() # We do not need to perform the definiton check until the last minute -> just do if tensor.id = metric -> set metric in workbook state
                tensor_node.callback = 'tensor_assignment' # TODO: This currenctly 
                return tensor_node

            tensor_node.tensor_expr_ast = self.expr()
            tensor_node.callback = 'tensor_assignment'
            return tensor_node

        return tensor_node

    def function(self, token: Token, start_position):
        func_node = Function()

        func_node.identifier = token.value

        if self.current_token.type != TokenType.LPAR:
            self.invalid_syntax_error(
                "Syntax Error, expecting a OPEN_BRACE token: '(' .",
                start_position,
                self.current_token.end_position.copy(),
                self.raw_code,
            )

        self.advance_token()

        func_node.new_argument(
            self.arith_expr()
        )  # We only allow mathematical expressions as args
        while self.current_token != None and self.current_token.type == TokenType.COMMA:
            self.advance_token()
            func_node.new_argument(self.arith_expr())

        if self.current_token.type != TokenType.RPAR:
            return self.invalid_syntax_error(
                "Syntax Error, expecting a CLOSE_BRACE token: ')' .",
                start_position,
                self.current_token.end_position.copy(),
                self.raw_code,
            )

        self.advance_token()
        func_node.set_position(start_position)

        if self.current_token == None:
            return func_node

        # Is user defining a callable object | function ?
        if self.current_token.type == TokenType.EQUAL:
            self.advance_token()
            func_node.executable = self.expr()
            return func_node

        return func_node
