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
from relativisticpy.interpreter.parsers.base import BaseParser, ParserResult
from relativisticpy.interpreter.lexers.base import Token, TokenType
from relativisticpy.interpreter.nodes.base import (
    AstNode,
    NodeType,
    UnaryNode,
    BinaryNode,
    ArrayNode,
    Infinitesimal,
    Definition, 
    Call, 
    Def,
    TensorNode
)
from relativisticpy.interpreter.shared.error_messages import braces_unmatched_errors
from relativisticpy.interpreter.nodes.position import Position


class GRParser(BaseParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse(self):
        if self.current_token.type == TokenType.NONE:
            return None

        result = self.statements()
        return ParserResult(self.raw_code, result)

    def statements(self):
        statements = []
        position = self.current_token.position.copy()

        while self.current_token.type == TokenType.NEWLINE:
            self.advance_token()

        statement = self.statement()
        statements.append(statement)

        more_statements = True

        while True:
            newline_count = 0

            if self.current_token.type == TokenType.NONE:
                break

            while (
                (self.current_token.type != TokenType.NONE or self.current_token != None)
                and self.current_token.type == TokenType.NEWLINE
            ):
                self.advance_token()
                newline_count += 1
            if newline_count == 0:
                more_statements = False

            if not more_statements:
                break
            if self.current_token.type == TokenType.NONE:
                break
            if self.current_token.type == TokenType.RBRACE:
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

        if self.current_token.type == TokenType.PRINT:
            position = self.current_token.position.copy()
            self.advance_token()
            return self.new_print_node(position, [self.expr()])

        return self.expr()

    ###################################### EXPRESSIONS #############################################
    # expr            :   (ID|TENSORID) EQUAL expr
    #                 :   bool-expr ((KEYWORD:and|KEYWORD:or) bool-expr)*
    ################################################################################################
    def expr(self):
        position = self.current_token.position.copy()

        if self.current_token.type == TokenType.ID and self.peek_type(1) == TokenType.COLONEQUAL:
            token = self.current_token
            # Skip over as we already know Token will be EQUAL
            self.advance_token()
            if self.current_token.type == TokenType.COLONEQUAL:
                self.advance_token()
                return self.new_definition_node(
                    position,
                    [
                        token.value,
                        self.expr(),
                    ],
                )

        else:
            result = self.bool_expr()

            while self.current_token.type in (
                TokenType.AND,
                TokenType.OR,
            ):
                if self.current_token.type == TokenType.AND:
                    self.advance_token()
                    result = self.new_and_node(
                        position, [result, self.bool_expr()]
                    )

                elif self.current_token.type == TokenType.OR:
                    self.advance_token()
                    result = self.new_and_node(
                        position, [result, self.bool_expr()]
                    )

            return result

    ############################## BOOLEAN EXPRESSION #############################################
    # bool-expr       :   NOT bool-expr
    #                 :   arith-expr ((EQEQUAL|LESS|GREATER|LESSEQUAL|GREATEREQUAL) arith-expr)*
    ################################################################################################
    def bool_expr(self):
        position = self.current_token.position.copy()

        if self.current_token.type == TokenType.NOT:
            self.advance_token()
            return self.new_not_node(position, [self.bool_expr()])

        result = self.arith_equation()
        while self.current_token.type in (
            TokenType.EQEQUAL,
            TokenType.LESS,
            TokenType.GREATER,
            TokenType.LESSEQUAL,
            TokenType.GREATEREQUAL,
        ):
            if self.current_token.type == TokenType.EQEQUAL:
                self.advance_token()
                result = self.new_eqequal_node(
                    position, [result, self.arith_equation()]
                )
            elif self.current_token.type == TokenType.LESS:
                self.advance_token()
                result = self.new_less_node(position, [result, self.arith_expr()])
            elif self.current_token.type == TokenType.GREATER:
                self.advance_token()
                result = self.new_greater_node(
                    position, [result, self.arith_expr()]
                )
            elif self.current_token.type == TokenType.LESSEQUAL:
                self.advance_token()
                result = self.new_lessequal_node(
                    position, [result, self.arith_expr()]
                )
            elif self.current_token.type == TokenType.GREATEREQUAL:
                self.advance_token()
                result = self.new_greaterequal_node(
                    position, [result, self.arith_expr()]
                )
    
        if self.current_token.type == TokenType.VBAR:

            self.advance_token()
            self.confirm_syntax(self.current_token.type, TokenType.UNDER)
            self.advance_token()
            self.confirm_syntax(self.current_token.type, TokenType.LBRACE)
            self.advance_token()
            lhs = self.atom()
            self.confirm_syntax(self.current_token.type, TokenType.EQUAL)
            self.advance_token()
            rhs = self.arith_expr()

            return Call(
                    identifier = 'subs',
                    position = position,
                    args = [result, lhs, rhs]
                )
    
        return result
    
    # arith-expr : term ((PLUS|MINUS) term)*
    def arith_equation(self):
        position = self.current_token.position.copy()
        result = self.arith_expr()
        if self.current_token != None:
            if self.current_token.type == TokenType.EQUAL:
                self.advance_token()
                result = self.new_equation_node(position, [result, self.arith_expr()])
        return result

    # arith-expr : term ((PLUS|MINUS) term)*
    def arith_expr(self):
        position = self.current_token.position.copy()
        result = self.term()

        while self.current_token.type in (
            TokenType.PLUS,
            TokenType.MINUS,
        ):
            if self.current_token.type == TokenType.PLUS:
                self.advance_token()
                result = self.new_add_node(position, [result, self.term()])
            elif self.current_token.type == TokenType.MINUS:
                self.advance_token()
                result = self.new_sub_node(position, [result, self.term()])
        return result

    def term(self):
        # Look for a factor and store it in result
        position = self.current_token.position.copy()
        result = self.factor()

        while self.current_token.type in (
            TokenType.STAR,
            TokenType.SLASH,
        ):
            if self.current_token.type == TokenType.STAR:
                self.advance_token()
                result = self.new_mul_node(position, [result, self.factor()])
            elif self.current_token.type == TokenType.SLASH:
                self.advance_token()
                result = self.new_div_node(position, [result, self.factor()])
        return result

    def factor(self):
        position = self.current_token.position.copy()

        if self.current_token.type == TokenType.PLUS:
            self.advance_token()
            return self.new_pos_node(position, [self.factor()])
        elif self.current_token.type == TokenType.MINUS:
            self.advance_token()
            return self.new_neg_node(position, [self.factor()])
        return self.power()

    def power(self):
        position = self.current_token.position.copy()
        result = self.factorial()
        # Look for additional powers and construct power node
        while self.current_token.type in (
            TokenType.DOUBLESTAR,
            TokenType.CIRCUMFLEX,
        ):
            if self.current_token.type in [TokenType.CIRCUMFLEX, TokenType.DOUBLESTAR]:
                self.advance_token()
                result = self.new_pow_node(position, [result, self.factorial()])
        return result

    def factorial(self):
        position = self.current_token.position.copy()
        result = self.atom()
        if self.current_token.type == TokenType.EXCLAMATION:
            self.advance_token()
            return self.new_factorial_node(position, [result])
        return result

    ##############################  ATOM RULES  ######################################################
    #
    # atom            :   INT|FLOAT|STRING|BOOL
    #                 |   LPAR        -> LPAR expr RPAR
    #                 |   RSQB        -> array
    #
    #                 |   SUM         -> sum
    #                 |   DV          -> dv
    #                 |   PDV         -> pdv
    #                 |   DOSUM       -> do_sum
    #                 |   PROD        -> product
    #                 |   DOPROD      -> do_product
    #                 |   INTEGRATE   -> integrate
    #                 |   FRAC        -> fraction
    #                 |   PROD        -> product
    #                 |   LIMIT       -> limit
    #                 |   PARTIAL     -> partial
    #                 |   BEGIN       -> BeginMap( array, equation, matrix, pmatrix, etc... )
    #
    #                 |   ID          -> IF(TensorPeek = True)        -> tensor
    #                                 /  IF(FunctionDefPeek = True)   -> func-def
    #                                 /  IF(FunctionPeek = True)      -> function
    #                                 /  ID
    #
    #                 |   SYMBOL      -> IF(TensorPeek = True)        -> tensor
    #                                 /  IF(FunctionDefPeek = True)   -> func-def
    #                                 /  IF(FunctionPeek = True)      -> function
    #                                 /  SYMBOL
    #
    #                 |   D           -> IF(TensoPeek = True)         -> tensor
    #                                 /  D                            -> derivative
    #
    #                 |   PARTIAL     -> IF(TensoPeek = True)         -> tensor
    #                                 /  PARTIAL                      -> partial
    ################################################################################################
    def atom(self):
        token: Token = self.current_token
        position = self.current_token.position.copy()

        if token.type == TokenType.FLOAT:
            self.advance_token()
            return self.new_float_node(position, [token.value])

        elif token.type == TokenType.CONSTANT:
            self.advance_token()
            return self.new_constant_node(position, [token.value])

        elif token.type == TokenType.INT:
            self.advance_token()
            return self.new_int_node(position, [token.value])

        elif token.type in [TokenType.INFINITESIMAL, TokenType.PARTIAL]:
            if (
                self.peek_type(1) in [TokenType.UNDER, TokenType.CIRCUMFLEX]
                and self.peek_type(2) == TokenType.LBRACE
            ):
                self.advance_token()
                return self.tensor(token)
            self.advance_token()
            return self.infinitesimal(token)

        elif token.type in [TokenType.ID, TokenType.SYMBOL]:

            if (
                self.peek_type(1) in [TokenType.UNDER, TokenType.CIRCUMFLEX]
                and self.peek_type(2) == TokenType.LBRACE
            ):
                self.advance_token()
                return self.tensor(token)
            elif self.peek_type(1) == TokenType.APOSTROPHE:
                self.advance_token()
                return self.single_arg_function_derivative(token, position)
            elif self.peek_type(1) == TokenType.LPAR:
                self.advance_token()
                return self.function(token, position)
            elif self.peek_type(1) == TokenType.COLON:
                self.advance_token()
                return self.expr_function(token, position)
            else:
                self.advance_token()
                return self.new_symbol_node(position, [token.value])

        elif token.type in [
            TokenType.SUM,
            TokenType.DOSUM,
            TokenType.PROD,
            TokenType.DOPROD,
        ]:
            self.advance_token()
            return self.sum(token, position)

        elif token.type == TokenType.FRAC:
            self.advance_token()
            return self.frac(position)

        elif token.type == TokenType.LIMIT:
            self.advance_token()
            return self.limit(position)

        elif token.type == TokenType.SQRT:
            self.advance_token()
            return self.sqrt(position)

        elif token.type == TokenType.BEGIN:  # We are beggining a new object
            self.advance_token()
            self.confirm_syntax(self.current_token.type, TokenType.LBRACE)
            self.advance_token()
            self.confirm_syntax(self.current_token.type, TokenType.ID)
            if self.current_token.value == "matrix":
                return self.matrix(position)

        elif token.type == TokenType.LSQB:
            self.advance_token()
            return self.array(position)
        
        elif token.type == TokenType.VBAR:
            self.advance_token()
            result = self.arith_expr()
            self.confirm_syntax(self.current_token.type, TokenType.VBAR)
            self.advance_token()
            return self.new_absolute_node(position, [result])

        elif token.type == TokenType.LPAR:
            self.advance_token()
            result = self.bool_expr()
            self.confirm_syntax(self.current_token.type, TokenType.RPAR)
            self.advance_token()
            return result

        else:
            return self.illegal_character_error(
                f"""Syntax Error, the type {token.value} at Line: {self.current_token.position.end_pos.line}, 
                Position: {self.current_token.position.end_pos.character} is not recognized or supported by RelativisticPy's Parser.""",
            )

    ############################## ARRAY ATOM   ####################################################
    #  array           :   LSQB NEWLINE* (expr (COMMA NEWLINE* expr)* NEWLINE* RSQB)
    ################################################################################################
    def array(self, position):
        elements = []
        if self.current_token.type != TokenType.RSQB:

            self.ignore_newlines()
            elements.append(self.statement())

            while self.current_token.type == TokenType.COMMA:
                self.advance_token()

                self.ignore_newlines()
                elements.append(self.statement())
                self.ignore_newlines()

            self.ignore_newlines()

        if (
            self.current_token.type == TokenType.NONE
        ):  ############################ <<<<<<<<<<<<<<<---------- PLEASE ADD THESE EVERY OTHER ERROR RAISING PLACE.
            return self.invalid_syntax_error(
                braces_unmatched_errors,
                self.peek_prev_token(ignore_NEWLINE=True).position.copy(),
                self.raw_code,
            )
        if self.current_token.type != TokenType.RSQB:
            return self.invalid_syntax_error(
                braces_unmatched_errors,
                self.peek_prev_token(ignore_NEWLINE=True).position.copy(),
                self.raw_code,
            )
        self.advance_token()

        return self.new_array_node(position, elements)

    ############################## MATRIX ATOM ##########################################################################################################################################################
    #  matrix          :   BEGIN LBRACE ID:matrix RBRACE NEWLINE* expr ( '&' NEWLINE* expr )* NEWLINE* ('\\' NEWLINE* expr ( '&' NEWLINE* expr )* )* NEWLINE* END LBRACE ID:matrix RBRACE
    ######################################################################################################################################################################################################
    def matrix(self, position):
        i_elements = []
        j_elements = []

        self.advance_token()
        self.confirm_syntax(self.current_token.type, TokenType.RBRACE)
        self.advance_token()

        self.ignore_newlines()
        i_elements.append(self.statement())
        self.ignore_newlines()

        while self.current_token.type == TokenType.AMPER:
            self.advance_token()

            self.ignore_newlines()
            i_elements.append(self.statement())
            self.ignore_newlines()

        j_elements.append(self.new_array_node(position, i_elements))

        while self.current_token.type == TokenType.DOUBLEBACKSLASH:
            i_elements = []

            self.advance_token()
            self.ignore_newlines()
            if self.current_token.type == TokenType.END:
                continue

            i_elements.append(self.statement())

            while self.current_token.type == TokenType.AMPER:
                self.advance_token()

                self.ignore_newlines()
                i_elements.append(self.statement())
                self.ignore_newlines()

            self.ignore_newlines()
            j_elements.append(self.new_array_node(position, i_elements))
            self.ignore_newlines()

        self.ignore_newlines()
        self.confirm_syntax(self.current_token.type, TokenType.END)

        self.advance_token()
        self.confirm_syntax(self.current_token.type, TokenType.LBRACE)

        self.advance_token()
        self.confirm_tok_value(self.current_token.value, "matrix")

        self.advance_token()
        self.confirm_syntax(self.current_token.type, TokenType.RBRACE)

        self.advance_token()
        return self.new_array_node(position, j_elements)

    ############################## TENSOT ATOM ####################################################
    # tensor          :   ID ((UNDER|CIRCUMFLEX) LBRACE (ID ((EQUAL|COLON) (INT|atom))?)* RBRACE )*
    #                 :   tensor ((EQUAL) expr)?
    #                 :   tensor ((EQUAL) array)?
    #                 :   tensor ((COLONEQUAL) array)?
    #                 :   tensor tensor*  <- NOT IMPLEMENTED YET
    ###############################################################################################
    def tensor(self, token: Token):
        tensor_node = TensorNode(self.current_token.position.copy())
        tensor_node.identifier = token.value
        tensor_covariant = True

        while self.current_token.type in (
            TokenType.UNDER,
            TokenType.CIRCUMFLEX,
        ):
            self.confirm_syntax(
                self.current_token.type, [TokenType.UNDER, TokenType.CIRCUMFLEX]
            )
            tensor_covariant = self.current_token.type == TokenType.UNDER

            self.advance_token()
            self.confirm_syntax(self.current_token.type, TokenType.LBRACE)

            self.advance_token()
            self.confirm_syntax(
                self.current_token.type, [TokenType.SYMBOL, TokenType.ID]
            )

            while self.current_token.type in [
                TokenType.ID,
                TokenType.SYMBOL,
            ] or self.peek_type(1) in [
                TokenType.UNDER,
                TokenType.CIRCUMFLEX,
            ]:  # T_{a b} => Index 'a' and Index 'b' are covariant.
                if self.current_token.type in [TokenType.ID, TokenType.SYMBOL]:
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
                    self.confirm_syntax(self.current_token.type, TokenType.LBRACE)
                    self.advance_token()

        if self.current_token.value in TokenType.Keywords():
            return self.illegal_character_error(
                f"Syntax Error with object. Cannot use keyword as an index: {self.current_token.value}"
            )
        self.confirm_syntax(self.current_token.type, TokenType.RBRACE)
        self.advance_token()

        if self.current_token.type == TokenType.NONE:
            return tensor_node

        if self.current_token.type == TokenType.COLONEQUAL:
            self.advance_token()
            if self.current_token.type == TokenType.LSQB:
                tensor_node.component_ast = (
                    self.atom()
                )  # We do not need to perform the definiton check until the last minute -> just do if tensor.id = metric -> set metric in workbook state
                return tensor_node

            tensor_node.component_ast = self.expr()
            return tensor_node

        if self.current_token.type == TokenType.EQUAL:
            self.advance_token()
            if self.current_token.type == TokenType.LSQB:
                tensor_node.component_ast = (
                    self.atom()
                )  # Build Eq in sympy. We do not need to perform the definiton check until the last minute -> just do if tensor.id = metric -> set metric in workbook state
                return tensor_node

            tensor_node.component_ast = self.expr()
            return tensor_node

        return tensor_node

    ############################## ARRAY ATOM   ####################################################
    #  array           :   LSQB NEWLINE* (expr (COMMA NEWLINE* expr)* NEWLINE* RSQB)
    ################################################################################################
    def function(self, token: Token, position):

        identifier = token.value
        arguments = []

        self.confirm_syntax(self.current_token.type, TokenType.LPAR)

        self.advance_token()

        if not self.current_token.type == TokenType.RPAR:

            arguments.append(
                self.bool_expr()
            )  # We only allow mathematical expressions as args
            while self.current_token != None and self.current_token.type == TokenType.COMMA:
                self.advance_token()
                arguments.append(self.bool_expr())

        self.confirm_syntax(self.current_token.type, TokenType.RPAR)

        self.advance_token()

        if self.current_token.type == TokenType.NONE:
            pos = self.peek_prev_token(ignore_NEWLINE=True).position.copy()
            return Call(
                            identifier = identifier,
                            position = pos, 
                            args = arguments
                        )

        # Is user defining a function ?
        if self.current_token.type == TokenType.COLONEQUAL:
            self.advance_token()
            return Def(
                        identifier = identifier,
                        body = self.expr(),
                        position = self.current_token.position.copy(),
                        args = arguments
                    )

        return Call(
                        identifier = identifier,
                        position = self.current_token.position.copy(), 
                        args = arguments
                    )

    ############################## ARRAY ATOM   ####################################################
    #  array           :   ID COLON LPAR ((ID|SYMBOL) COMMA)* RARROW LBRACE statements RBRACE
    ################################################################################################
    def expr_function(self, token: Token, position):

        identifier = token.value
        arguments = []

        self.confirm_syntax(self.current_token.type, TokenType.COLON)
        self.advance_token()

        self.confirm_syntax(self.current_token.type, TokenType.LPAR)
        self.advance_token()

        self.confirm_syntax(self.current_token.type, [TokenType.RPAR, TokenType.SYMBOL, TokenType.ID])
        if self.current_token.type in [TokenType.SYMBOL, TokenType.ID]:

            arguments.append(
                self.new_symbol_node(self.current_token.position.copy(), [self.current_token.value])
            )  # We only allow variables\ids as args for expr_functions
            self.advance_token()
            while self.current_token.type == TokenType.COMMA:
                self.advance_token()
                self.confirm_syntax(self.current_token.type,[TokenType.SYMBOL, TokenType.ID])
                arguments.append(self.new_symbol_node(self.current_token.position.copy(), [self.current_token.value]))
                self.advance_token()

        self.confirm_syntax(self.current_token.type, TokenType.RPAR)

        self.advance_token()

        self.confirm_syntax(self.current_token.type, TokenType.RARROW)

        self.advance_token()

        self.confirm_syntax(self.current_token.type, TokenType.LBRACE)

        self.advance_token()

        statements = self.statements()

        self.ignore_newlines()

        self.confirm_syntax(self.current_token.type, TokenType.RBRACE)

        self.advance_token()

        return Def(
                    identifier = identifier,
                    body = statements,
                    position = position,
                    args = arguments
                )


    ############################## SUM ATOM   ####################################################
    #  sum             :   SUM UNDER LBRACE expr RBRACE CIRCUMFLEX LBRACE expr RBRACE expr
    ################################################################################################
    def sum(self, token: Token, position):
        identifier = token.value

        self.confirm_syntax(self.current_token.type, TokenType.UNDER)
        self.advance_token()

        self.confirm_syntax(self.current_token.type, TokenType.LBRACE)
        self.advance_token()

        self.confirm_syntax(self.current_token.type, [TokenType.ID, TokenType.SYMBOL])
        var = self.new_symbol_node(
            self.current_token.position, [self.current_token.value]
        )

        self.advance_token()
        self.confirm_syntax(self.current_token.type, TokenType.EQUAL)

        self.advance_token()
        start = self.arith_expr()
        self.confirm_syntax(self.current_token.type, TokenType.RBRACE)

        self.advance_token()
        self.confirm_syntax(self.current_token.type, TokenType.CIRCUMFLEX)

        self.advance_token()
        self.confirm_syntax(self.current_token.type, TokenType.LBRACE)

        self.advance_token()
        end = self.arith_expr()
        self.confirm_syntax(self.current_token.type, TokenType.RBRACE)

        self.advance_token()
        expression = self.arith_expr()

        return Call(
                        identifier = identifier,
                        position = position,
                        args = [expression, var, start, end]
                    )

    ############################## LIMIT ATOM   ####################################################
    #  limit             :   LIMIT UNDER LBRACE (SYMBOL|ID) RARROW arith_expr RBRACE arith_expr
    ################################################################################################
    def limit(self, position):

        self.confirm_syntax(self.current_token.type, TokenType.UNDER)
        self.advance_token()

        self.confirm_syntax(self.current_token.type, TokenType.LBRACE)
        self.advance_token()

        self.confirm_syntax(self.current_token.type, [TokenType.ID, TokenType.SYMBOL])
        var = self.new_symbol_node(
            self.current_token.position, [self.current_token.value]
        )
        self.advance_token()

        self.confirm_syntax(self.current_token.type, TokenType.RARROW)
        self.advance_token()

        to_expr = self.arith_expr()
        self.confirm_syntax(self.current_token.type, TokenType.RBRACE)
        self.advance_token()

        expression = (
            self.arith_expr()
        )  ### <<<<<<<<<<<<<<<<<<<<<<<<<------------------- BAD CODE = INPROPER DOWNWARD DEPENDENCY. This lower level code should not need to know the order in which to place the arguments.
        return Call(
                        identifier = TokenType.LIMIT.value,
                        position = position,
                        args = [expression, var, to_expr]
                    )

    ############################## FRAC ATOM ######################################################
    #  frac             :   FRAC LBRACE arith_expr RBRACE LBRACE arith_expr RBRACE
    ###############################################################################################
    def frac(self, position):
        pos = self.current_token.position
        self.confirm_syntax(self.current_token.type, TokenType.LBRACE)
        self.advance_token()
        numerator = self.arith_expr()
        self.confirm_syntax(self.current_token.type, TokenType.RBRACE)
        self.advance_token()
        self.confirm_syntax(self.current_token.type, TokenType.LBRACE)
        self.advance_token()
        denominator = self.arith_expr()
        self.confirm_syntax(self.current_token.type, TokenType.RBRACE)
        self.advance_token()
        return self.new_div_node(pos, [numerator, denominator])

    ############################## SQRT ATOM ######################################################
    #  sqrt             :   SQRT LBRACE arith_expr RBRACE
    ###############################################################################################
    def sqrt(self, position):
        pos = self.current_token.position
        self.confirm_syntax(self.current_token.type, TokenType.LBRACE)
        self.advance_token()
        argument = self.arith_expr()
        self.confirm_syntax(self.current_token.type, TokenType.RBRACE)
        self.advance_token()
        return Call(
                        identifier = TokenType.SQRT.value,
                        position = pos,
                        args = [argument]
                    )

    ############################## ATOM ###########################################################
    #  1-arg-func-derivative    :   ID APOSTROPHE+ LPAR SYMBOL RPAR
    ###############################################################################################
    def single_arg_function_derivative(self, token: Token, position):
        identifier = token.value

        d_order = 1

        self.confirm_syntax(self.current_token.type, TokenType.APOSTROPHE)
        self.advance_token()

        while self.current_token.type == TokenType.APOSTROPHE:
            d_order += 1
            self.advance_token()
        
        self.confirm_syntax(self.current_token.type, TokenType.LPAR)
        self.advance_token()

        arguments = self.arith_expr()

        self.confirm_syntax(self.current_token.type, TokenType.RPAR)
        self.advance_token()

        symbol_function = Call(identifier=identifier, position=position, args=[arguments])

        return Call(
                        identifier = 'func_derivative',
                        position = position,
                        args = [symbol_function, arguments, self.new_int_node(position, [str(d_order)])]
                    )

    ############################## INFINITESSIMAL ATOM   ####################################################
    #  diff_operator        :   D ( > RBRACE | > DASH )
    #                       |   D arith_expr
    #                       |   D LPAR arith_expr RPAR
    #                       |   D LBRACE arith_expr RBRACE
    #                       |   D CIRCUMFLEX INT LBRACE arith_expr RBRACE !(CIRCUMFLEX INT)
    #                       |   D !(CIRCUMFLEX INT) LBRACE arith_expr RBRACE CIRCUMFLEX INT
    ###############################################################################################
    def infinitesimal(self, token):
        diff_order = self.new_int_node(
            self.current_token.position.copy(), ["1"]
        )  # defaults to first order derivaitve
        pos = self.current_token.position.copy()
        circumflex_tag = False
        diff_order_as_int = 1

        # d^n{expr} OR d^n(expr)
        if self.current_token.type == TokenType.CIRCUMFLEX:
            circumflex_tag = True
            self.advance_token()
            self.confirm_syntax(self.current_token.type, [TokenType.INT, TokenType.SYMBOL, TokenType.ID])
            if self.current_token.type == TokenType.INT:
                diff_order = self.new_int_node(
                    self.current_token.position.copy(), [self.current_token.value]
                )
                diff_order_as_int = int(self.current_token.value)
            elif self.current_token.type in [TokenType.SYMBOL, TokenType.ID]:
                diff_order = self.new_symbol_node(
                    self.current_token.position.copy(), [self.current_token.value]
                )
            self.advance_token()

        # d{x} OR d{x^n} OR d{x}^n
        if self.current_token.type == TokenType.LBRACE:
            self.advance_token()
            expr = self.arith_expr()
            self.confirm_syntax(self.current_token.type, TokenType.RBRACE)

            # d^2{x}^2 Is not valid
            if self.peek_type(1) == TokenType.CIRCUMFLEX and circumflex_tag:
                return self.illegal_character_error(
                    "Syntax Error with diff object. Cannot have two '^' within one infinitesimal object i.e. d^n{ <expr> }^n is not allowed."
                )
            self.advance_token()

            if self.current_token.type == TokenType.CIRCUMFLEX:
                self.advance_token()
                self.confirm_syntax(self.current_token.type, [TokenType.INT, TokenType.SYMBOL, TokenType.ID])
                if self.current_token.type == TokenType.INT:
                    diff_order = self.new_int_node(
                        self.current_token.position.copy(), [self.current_token.value]
                    )
                    diff_order_as_int = int(self.current_token.value)
                elif self.current_token.type in [TokenType.SYMBOL, TokenType.ID]:
                    diff_order = self.new_symbol_node(
                        self.current_token.position.copy(), [self.current_token.value]
                    )
                self.advance_token()

            inft = Infinitesimal(pos, [expr])
            inft.expression = expr
            inft.diff_order = diff_order
            inft.diff_order_as_int = diff_order_as_int
            return inft

        expr = self.arith_expr()
        if self.current_token != None:
            inft = Infinitesimal(pos, [expr])
        else:
            inft = Infinitesimal(pos, [expr])
        inft.expression = expr
        self.confirm_syntax(self.current_token.type, [TokenType.INT, TokenType.SYMBOL, TokenType.ID])
        if self.current_token.type == TokenType.INT:
            diff_order = self.new_int_node(
                self.current_token.position.copy(), [self.current_token.value]
            )
            diff_order_as_int = int(self.current_token.value)
        elif self.current_token.type in [TokenType.SYMBOL, TokenType.ID]:
            diff_order = self.new_symbol_node(
                self.current_token.position.copy(), [self.current_token.value]
            )
        inft.is_partial = False
        return inft
