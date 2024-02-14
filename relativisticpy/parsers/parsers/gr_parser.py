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
from relativisticpy.parsers.types.base import (
    AstNode,
    NodeType,
    UnaryNode,
    BinaryNode,
    ArrayNode,
    Infinitesimal,
)
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

        if self.current_token.type == TokenType.ID and self.peek_type(1) in (
            TokenType.EQUAL,
            TokenType.COLONEQUAL,
        ):
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
        if self.current_token != None:
            if self.current_token.type == TokenType.VBAR:
                func_node = Function()
                func_node.identifier = "subs"
                self.advance_token()
                self.confirm_syntax(self.current_token.type, TokenType.UNDER)
                self.advance_token()
                self.confirm_syntax(self.current_token.type, TokenType.LBRACE)
                self.advance_token()
                lhs = self.atom()
                self.confirm_syntax(self.current_token.type, TokenType.EQUAL)
                self.advance_token()
                rhs = self.arith_expr()
                func_node.new_argument(result)
                func_node.new_argument(lhs)
                func_node.new_argument(rhs)
                return func_node
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
        start_position = self.current_token.start_position.copy()

        if token.type == TokenType.FLOAT:
            self.advance_token()
            return self.new_float_node(start_position, [token.value])

        elif token.type == TokenType.CONSTANT:
            self.advance_token()
            return self.new_constant_node(start_position, [token.value])

        elif token.type == TokenType.INT:
            self.advance_token()
            return self.new_int_node(start_position, [token.value])

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
                return self.single_arg_function_derivative(token, start_position)
            elif self.peek_type(1) == TokenType.LPAR:
                self.advance_token()
                return self.function(token, start_position)
            elif self.peek_type(1) == TokenType.COLON:
                self.advance_token()
                return self.function(token, start_position)
            else:
                self.advance_token()
                return self.new_symbol_node(start_position, [token.value])

        elif token.type in [
            TokenType.SUM,
            TokenType.DOSUM,
            TokenType.PROD,
            TokenType.DOPROD,
        ]:
            self.advance_token()
            return self.sum(token, start_position)

        elif token.type == TokenType.FRAC:
            self.advance_token()
            return self.frac(start_position)

        elif token.type == TokenType.LIMIT:
            self.advance_token()
            return self.limit(start_position)

        elif token.type == TokenType.SQRT:
            self.advance_token()
            return self.sqrt(start_position)

        elif token.type == TokenType.BEGIN:  # We are beggining a new object
            self.advance_token()
            self.confirm_syntax(self.current_token.type, TokenType.LBRACE)
            self.advance_token()
            self.confirm_syntax(self.current_token.type, TokenType.ID)
            if self.current_token.value == "matrix":
                return self.matrix(start_position)

        elif token.type == TokenType.LSQB:
            self.advance_token()
            return self.array(start_position)

        elif token.type == TokenType.LPAR:
            self.advance_token()
            result = self.bool_expr()
            self.confirm_syntax(self.current_token.type, TokenType.RPAR)
            self.advance_token()
            return result

        else:
            return self.illegal_character_error(
                f"""Syntax Error, the type {token.value} at Line: {self.current_token.end_position.copy().line}, 
                Position: {self.current_token.end_position.copy().character} is not recognized or supported by RelativisticPy's Parser.""",
            )

    ############################## ARRAY ATOM   ####################################################
    #  array           :   LSQB NEWLINE* (expr (COMMA NEWLINE* expr)* NEWLINE* RSQB)
    ################################################################################################
    def array(self, start_position):
        elements = []
        if self.current_token.type != TokenType.RSQB:

            self.ignore_newlines()
            elements.append(self.statement())

            while (
                self.current_token != None
                and self.current_token.type == TokenType.COMMA
            ):
                self.advance_token()

                self.ignore_newlines()
                elements.append(self.statement())
                self.ignore_newlines()

            self.ignore_newlines()

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

    ############################## MATRIX ATOM ##########################################################################################################################################################
    #  matrix          :   BEGIN LBRACE ID:matrix RBRACE NEWLINE* expr ( '&' NEWLINE* expr )* NEWLINE* ('\\' NEWLINE* expr ( '&' NEWLINE* expr )* )* NEWLINE* END LBRACE ID:matrix RBRACE
    ######################################################################################################################################################################################################
    def matrix(self, start_position):
        i_elements = []
        j_elements = []

        self.advance_token()
        self.confirm_syntax(self.current_token.type, TokenType.RBRACE)
        self.advance_token()

        self.ignore_newlines()
        i_elements.append(self.statement())
        self.ignore_newlines()

        while self.current_token != None and self.current_token.type == TokenType.AMPER:
            self.advance_token()

            self.ignore_newlines()
            i_elements.append(self.statement())
            self.ignore_newlines()

        j_elements.append(self.new_array_node(start_position, i_elements))

        while (
            self.current_token != None
            and self.current_token.type == TokenType.DOUBLEBACKSLASH
        ):
            i_elements = []

            self.advance_token()
            self.ignore_newlines()
            if self.current_token.type == TokenType.END:
                continue

            i_elements.append(self.statement())

            while (
                self.current_token != None
                and self.current_token.type == TokenType.AMPER
            ):
                self.advance_token()

                self.ignore_newlines()
                i_elements.append(self.statement())
                self.ignore_newlines()

            self.ignore_newlines()
            j_elements.append(self.new_array_node(start_position, i_elements))
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
        return self.new_array_node(start_position, j_elements)

    ############################## TENSOT ATOM ####################################################
    # tensor          :   ID ((UNDER|CIRCUMFLEX) LBRACE (ID ((EQUAL|COLON) (INT|atom))?)* RBRACE )*
    #                 :   tensor ((EQUAL) expr)?
    #                 :   tensor ((EQUAL) array)?
    #                 :   tensor ((COLONEQUAL) array)?
    #                 :   tensor tensor*  <- NOT IMPLEMENTED YET
    ###############################################################################################
    def tensor(self, token: Token):
        tensor_node = TensorNode(self.current_token.start_position.copy())
        tensor_node.identifier = token.value
        tensor_covariant = True

        while self.current_token != None and self.current_token.type in (
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

        if self.current_token == None:
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
    def function(self, token: Token, start_position):
        func_node = Function()

        func_node.identifier = token.value

        self.confirm_syntax(self.current_token.type, TokenType.LPAR)

        self.advance_token()

        func_node.new_argument(
            self.arith_expr()
        )  # We only allow mathematical expressions as args
        while self.current_token != None and self.current_token.type == TokenType.COMMA:
            self.advance_token()
            func_node.new_argument(self.arith_expr())

        self.confirm_syntax(self.current_token.type, TokenType.RPAR)

        self.advance_token()
        func_node.set_position(start_position)

        if self.current_token == None:
            return func_node

        # Is user defining a callable object | function ?
        if self.current_token.type == TokenType.COLONEQUAL:
            self.advance_token()
            func_node.executable = self.expr()
            return func_node

        return func_node

    ############################## SUM ATOM   ####################################################
    #  sum             :   SUM UNDER LBRACE expr RBRACE CIRCUMFLEX LBRACE expr RBRACE expr
    ################################################################################################
    def sum(self, token: Token, start_position):
        func_node = Function()
        func_node.identifier = token.value

        self.confirm_syntax(self.current_token.type, TokenType.UNDER)
        self.advance_token()

        self.confirm_syntax(self.current_token.type, TokenType.LBRACE)
        self.advance_token()

        self.confirm_syntax(self.current_token.type, [TokenType.ID, TokenType.SYMBOL])
        var = self.new_symbol_node(
            self.current_token.start_position, [self.current_token.value]
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

        func_node.new_argument(expression)
        func_node.new_argument(var)
        func_node.new_argument(start)
        func_node.new_argument(end)

        return func_node

    ############################## LIMIT ATOM   ####################################################
    #  limit             :   LIMIT UNDER LBRACE (SYMBOL|ID) RARROW arith_expr RBRACE arith_expr
    ################################################################################################
    def limit(self, start_position):
        func_node = Function()

        func_node.identifier = TokenType.LIMIT.value

        self.confirm_syntax(self.current_token.type, TokenType.UNDER)
        self.advance_token()

        self.confirm_syntax(self.current_token.type, TokenType.LBRACE)
        self.advance_token()

        self.confirm_syntax(self.current_token.type, [TokenType.ID, TokenType.SYMBOL])
        var = self.new_symbol_node(
            self.current_token.start_position, [self.current_token.value]
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
        func_node.new_argument(expression)
        func_node.new_argument(var)
        func_node.new_argument(to_expr)
        return func_node

    ############################## FRAC ATOM ######################################################
    #  frac             :   FRAC LBRACE arith_expr RBRACE LBRACE arith_expr RBRACE
    ###############################################################################################
    def frac(self, start_position):
        pos = self.current_token.start_position
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
    def sqrt(self, start_position):
        func_node = Function()
        func_node.identifier = TokenType.SQRT.value
        pos = self.current_token.start_position
        self.confirm_syntax(self.current_token.type, TokenType.LBRACE)
        self.advance_token()
        func_node.new_argument(self.arith_expr())
        self.confirm_syntax(self.current_token.type, TokenType.RBRACE)
        self.advance_token()
        return func_node

    ############################## ATOM ###########################################################
    #  1-arg-func-derivative    :   ID APOSTROPHE+ LPAR SYMBOL RPAR
    ###############################################################################################
    def single_arg_function_derivative(self, token: Token, start_position):
        func_node = Function()
        func_arg = Function()
        func_arg.identifier = token.value
        func_node.identifier = 'func_derivative'
        d_order = 1

        self.confirm_syntax(self.current_token.type, TokenType.APOSTROPHE)
        self.advance_token()

        while self.current_token.type == TokenType.APOSTROPHE:
            d_order += 1
            self.advance_token()
        
        self.confirm_syntax(self.current_token.type, TokenType.LPAR)
        self.advance_token()

        args = self.arith_expr()

        self.confirm_syntax(self.current_token.type, TokenType.RPAR)
        self.advance_token()

        func_arg.new_argument(args)
        func_node.new_argument(func_arg)
        func_node.new_argument(args)
        func_node.new_argument(self.new_int_node(start_position, [str(d_order)]))
        return func_node

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
            self.current_token.end_position.copy(), ["1"]
        )  # defaults to first order derivaitve
        circumflex_tag = False
        diff_order_as_int = 1

        # d^n{expr} OR d^n(expr)
        if self.current_token.type == TokenType.CIRCUMFLEX:
            circumflex_tag = True
            self.advance_token()
            self.confirm_syntax(self.current_token.type, TokenType.INT)
            diff_order = self.new_int_node(
                self.current_token.end_position.copy(), [self.current_token.value]
            )
            diff_order_as_int = int(self.current_token.value)
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
                self.confirm_syntax(self.current_token.type, TokenType.INT)
                diff_order = self.new_int_node(
                    self.current_token.end_position.copy(), [self.current_token.value]
                )
                diff_order_as_int = int(self.current_token.value)
                self.advance_token()

            inft = Infinitesimal(self.current_token.end_position.copy(), [expr])
            inft.expression = expr
            inft.diff_order = diff_order
            inft.diff_order_as_int = diff_order_as_int
            return inft

        expr = self.arith_expr()
        if self.current_token != None:
            inft = Infinitesimal(self.current_token.end_position.copy(), [expr])
        else:
            inft = Infinitesimal(token.end_position.copy(), [expr])
        inft.expression = expr
        self.confirm_syntax(self.current_token.type, TokenType.INT)
        inft.diff_order = self.new_int_node(
            self.current_token.end_position.copy(), [self.current_token.value]
        )
        inft.is_partial = False
        return inft
