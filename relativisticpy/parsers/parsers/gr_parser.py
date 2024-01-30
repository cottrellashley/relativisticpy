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
from relativisticpy.parsers.types.gr_nodes import Tensor, Function, Definition
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
        tensor_node = Tensor()
        tensor_node.identifier = token.value
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
            return self.new_tensor_node(start_position, [tensor_node])

        if self.current_token.type == TokenType.COLONEQUAL:
            self.advance_token()
            if self.current_token.type != TokenType.LSQB:
                return self.invalid_syntax_error(
                    "Syntax Error, expecting a token: '[' ",
                    start_position,
                    self.current_token.end_position.copy(),
                    self.raw_code,
                )

            return self.new_tensor_comp_definition_node(
                start_position,
                [tensor_node, self.atom()],
            )

        if self.current_token.type == TokenType.EQUAL:
            self.advance_token()
            if self.current_token.type == TokenType.LSQB:
                return self.new_tensor_comp_assignment_node(
                    start_position,
                    [self.new_tensor_node(start_position, [tensor_node]), self.atom()],
                )
            return self.new_tensor_expr_assignment_node(
                start_position,
                [self.new_tensor_node(start_position, [tensor_node]), self.expr()],
            )

        return self.new_tensor_node(start_position, [tensor_node])

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


    def new_add_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.ADD,
            position=position,
            callback='add',
            args=args
        )
    
    def new_sub_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.SUB,
            position=position,
            callback='sub',
            args=args
        )
    
    def new_mul_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.MUL,
            position=position,
            callback='mul',
            args=args
        )
    
    def new_div_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.DIV,
            position=position,
            callback='div',
            args=args
        )
    
    def new_pow_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.POW,
            position=position,
            callback='pow',
            args=args
        )
    
    ###################################
    ### SIMPLE DATA TYPE NODES 
    ###################################

    def new_int_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return UnaryNode(
            type=NodeType.INT,
            position=position,
            callback='int',
            inferenced_type='int',
            args=args
        )
    
    def new_float_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return UnaryNode(
            type=NodeType.FLOAT,
            position=position,
            callback='float',
            inferenced_type='float',
            args=args
        )
    
    def new_symbol_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return UnaryNode(
            type=NodeType.SYMBOL,
            position=position,
            callback='symbol',
            inferenced_type='symbol',
            args=args
        )
    
    def new_neg_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return UnaryNode(
            type=NodeType.NEG,
            position=position,
            callback='neg',
            inferenced_type=None,
            args=args
        )

    def new_pos_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return UnaryNode(
            type=NodeType.POS,
            position=position,
            callback='pos',
            inferenced_type=None,
            args=args
        )
    
    def new_array_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return ArrayNode(
            type=NodeType.ARRAY,
            position=position,
            callback='array',
            inferenced_type='array',
            args=args
        )
    
    def new_tensor_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return AstNode(
            type=NodeType.TENSOR,
            position=position,
            callback='tensor',
            inferenced_type='tensor',
            args=args
        )
    
    def new_not_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return UnaryNode(
            type=NodeType.NOT,
            position=position,
            callback='not_',
            inferenced_type=None,
            args=args
        )
    
    def new_and_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.AND,
            position=position,
            callback='and_',
            args=args
        )
    
    def new_or_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.OR,
            position=position,
            callback='or',
            args=args
        )
    
    def new_eqequal_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.EQEQUAL,
            position=position,
            callback='eqequal',
            args=args
        )
    
    def new_less_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.LESS,
            position=position,
            callback='less',
            args=args
        )
    
    def new_greater_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.GREATER,
            position=position,
            callback='greater',
            args=args
        )
    
    def new_lessequal_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.LESSEQUAL,
            position=position,
            callback='lessequal',
            args=args
        )
    
    def new_greaterequal_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.GREATEREQUAL,
            position=position,
            callback='greaterequal',
            args=args
        )
    
    def new_assignment_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.ASSIGNMENT,
            position=position,
            callback='assignment',
            args=args
        )
    
    def new_definition_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return Definition(
            position=position,
            args=args
        )
    
    def new_tensor_comp_assignment_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.TENSOR_COMPONENT_ASSIGNMENT,
            position=position,
            callback='tensor_component_assignment',
            args=args
        )
    
    def new_tensor_expr_assignment_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.TENSOR_EXPR_ASSIGNMENT,
            position=position,
            callback='tensor_expr_assignment',
            args=args
        )
    
    def new_function_def_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return BinaryNode(
            type=NodeType.FUNCTION_DEF,
            position=position,
            callback='function_def',
            args=args
        )
    
    def new_print_node(self, position: Position, args: List[AstNode]) -> AstNode:
        return UnaryNode(
            type=NodeType.PRINT,
            position=position,
            callback='print_',
            inferenced_type=None,
            args=args
        )
    
    def new_tensor_comp_definition_node(self, position: Position, args: List[AstNode]) -> AstNode: ## Must become a Tensor node
        return_ = Tensor() # Should be returning this. So that when 'tensor_definition' is called, left child is this return = Tensor() object and the right child is the array expression to be set.
        return BinaryNode(
            type=NodeType.TENSOR_COMPONENT_DEFINITION,
            position=position,
            callback='tensor_definition',
            args=args
        )