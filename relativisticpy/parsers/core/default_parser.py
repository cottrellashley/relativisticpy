from relativisticpy.utils.regex import str_is_tensors, is_symbol_object_key
from relativisticpy.parsers.shared.interfaces.iterator import IIterator
from relativisticpy.parsers.shared.interfaces.node_provider import INodeProvider
from relativisticpy.parsers.shared.interfaces.parser_ import IParser
from relativisticpy.parsers.shared.constants import NodeType, TokenType
from relativisticpy.parsers.shared.models.token import Token


class Parser(IParser):
    def __init__(self, node_provider: INodeProvider):
        """
        Constructor for Parser class.

        Args:
            tokens: a list of tokens representing the input expression to parse
        """
        self.node_provider = node_provider

    def raise_error(self, error_message):
        """
        Helper method to raise exceptions with a custom error message.

        Args:
            error_message: a string representing the error message to raise
        """
        raise Exception(error_message)

    def parse(self, iterating_tokens: IIterator):
        """
        Parse the input expression.

        Returns:
            The resulting expression tree if the input was valid, None otherwise.
        """
        self.iterating_tokens = iterating_tokens
        self.iterating_tokens.advance()

        if self.iterating_tokens.current() == None:
            return None

        elif len(self.iterating_tokens) >= 3:
            result = self.equation()

        elif len(self.iterating_tokens) < 3:
            result = self.statement()

        elif self.iterating_tokens.current() != None:
            self.raise_error("Syntax Error")
        return result

    def equation(self):
        current_token = self.iterating_tokens.current()
        next_token = self.iterating_tokens.peek(1, Token(TokenType.NONE, "NONE"))
        if (
            current_token.type == TokenType.OBJECT
            and next_token.type == TokenType.EQUAL
        ):
            subject_variable = self.object("object_key")
            self.iterating_tokens.advance()
            return self.node_provider.new_node(
                NodeType.EQUALS, [subject_variable, self.statement()]
            )
        elif (
            current_token.type == TokenType.OBJECT
            and next_token.type == TokenType.COLONEQUAL
            and str_is_tensors(current_token.value)
        ):
            subject_variable = self.object("tensor_key")
            self.iterating_tokens.advance()
            return self.node_provider.new_node(
                NodeType.TENSOR_INIT, [subject_variable, self.statement()]
            )
        elif (
            current_token.type == TokenType.OBJECT
            and next_token.type == TokenType.COLONEQUAL
            and is_symbol_object_key(current_token.value)
        ):
            subject_variable = self.object("symbol_key")
            self.iterating_tokens.advance()
            return self.node_provider.new_node(
                NodeType.SYMBOL_DEFINITION, [subject_variable, self.statement()]
            )
        elif (
            current_token.type == TokenType.OBJECT
            and next_token.type == TokenType.COLONEQUAL
        ):
            subject_variable = self.object("object_key")
            self.iterating_tokens.advance()
            return self.node_provider.new_node(
                NodeType.ASSIGNMENT, [subject_variable, self.statement()]
            )
        else:
            return self.statement()

    def array(self):
        elements = []
        if self.iterating_tokens.current().type != TokenType.RSQB:
            elements.append(self.statement())
            while (
                self.iterating_tokens.current() != None
                and self.iterating_tokens.current().type == TokenType.COMMA
            ):
                self.iterating_tokens.advance()
                elements.append(self.statement())
        if self.iterating_tokens.current().type != TokenType.RSQB:
            self.raise_error("Syntax Error, expecting a CLOSED_SQUARE_BRACE token.")
        self.iterating_tokens.advance()
        return self.node_provider.new_node(NodeType.ARRAY, elements)

    def statement(self):
        result = self.bool()
        while (
            self.iterating_tokens.current() != None
            and self.iterating_tokens.current().type
            in (TokenType.AMPER, TokenType.VBAR)
        ):
            if self.iterating_tokens.current().type == TokenType.AMPER:
                self.iterating_tokens.advance()
                result = self.node_provider.new_node(
                    NodeType.AND, [result, self.bool()]
                )
            elif self.iterating_tokens.current().type == TokenType.VBAR:
                self.iterating_tokens.advance()
                result = self.node_provider.new_node(NodeType.OR, [result, self.bool()])
        return result

    def bool(self):
        """
        bool : expr (LESS|LESSQUAL|GREATER|GREATEEQUAL|EQUALEQUAL|NOTEQUAL) expr
        """
        result = self.expr()
        while (
            self.iterating_tokens.current() != None
            and self.iterating_tokens.current().type
            in (
                TokenType.LESS,
                TokenType.LESSEQUAL,
                TokenType.GREATER,
                TokenType.GREATEREQUAL,
                TokenType.EQEQUAL,
                TokenType.NOTEQUAL,
            )
        ):
            if self.iterating_tokens.current().type == TokenType.LESS:
                self.iterating_tokens.advance()
                result = self.node_provider.new_node(
                    NodeType.LESS, [result, self.expr()]
                )
            elif self.iterating_tokens.current().type == TokenType.LESSEQUAL:
                self.iterating_tokens.advance()
                result = self.node_provider.new_node(
                    NodeType.LESSEQUAL, [result, self.expr()]
                )
            elif self.iterating_tokens.current().type == TokenType.GREATER:
                self.iterating_tokens.advance()
                result = self.node_provider.new_node(
                    NodeType.GREATER, [result, self.expr()]
                )
            elif self.iterating_tokens.current().type == TokenType.GREATEREQUAL:
                self.iterating_tokens.advance()
                result = self.node_provider.new_node(
                    NodeType.GREATEREQUAL, [result, self.expr()]
                )
            elif self.iterating_tokens.current().type == TokenType.EQEQUAL:
                self.iterating_tokens.advance()
                result = self.node_provider.new_node(
                    NodeType.EQEQUAL, [result, self.expr()]
                )
            elif self.iterating_tokens.current().type == TokenType.NOTEQUAL:
                self.iterating_tokens.advance()
                result = self.node_provider.new_node(
                    NodeType.NOTEQUAL, [result, self.expr()]
                )
        return result

    def expr(self):
        """
        Parse an expression as defined by the grammar:
            expr = term (PLUS | MINUS) term.

        Returns:
            An expression tree that represents the input expression.
        """
        # Look for a term and store it in X
        result = self.term()
        # Look for additional terms after (PLUS|MINUS) tokens and then construct expression tree
        while (
            self.iterating_tokens.current() != None
            and self.iterating_tokens.current().type
            in (TokenType.PLUS, TokenType.MINUS)
        ):
            if self.iterating_tokens.current().type == TokenType.PLUS:
                self.iterating_tokens.advance()
                result = self.node_provider.new_node(
                    NodeType.PLUS, [result, self.term()]
                )
            elif self.iterating_tokens.current().type == TokenType.MINUS:
                self.iterating_tokens.advance()
                result = self.node_provider.new_node(
                    NodeType.MINUS, [result, self.term()]
                )
        return result

    def term(self):
        """
        Parse a term.

        Returns:
            A term node that represents the input term.
        """
        # Look for a factor and store it in result
        result = self.power()
        # Look for additional factors after (MUL|DIV) tokens and then construct term node
        while (
            self.iterating_tokens.current() != None
            and self.iterating_tokens.current().type
            in (TokenType.STAR, TokenType.SLASH)
        ):
            if self.iterating_tokens.current().type == TokenType.STAR:
                self.iterating_tokens.advance()
                result = self.node_provider.new_node(
                    NodeType.MULTIPLY, [result, self.power()]
                )
            elif self.iterating_tokens.current().type == TokenType.SLASH:
                self.iterating_tokens.advance()
                result = self.node_provider.new_node(
                    NodeType.DIVIDE, [result, self.power()]
                )
        return result

    def power(self):
        """
        Parse a power.

        Returns:
            A power node that represents the input power.
        """
        result = self.object()
        # Look for additional powers and construct power node
        while (
            self.iterating_tokens.current() != None
            and self.iterating_tokens.current().type
            in (TokenType.DOUBLESTAR, TokenType.CIRCUMFLEX)
        ):
            if self.iterating_tokens.current().type == TokenType.CIRCUMFLEX:
                self.iterating_tokens.advance()
                result = self.node_provider.new_node(
                    NodeType.EXPONENTIATION1, [result, self.object()]
                )
            elif self.iterating_tokens.current().type == TokenType.DOUBLESTAR:
                self.iterating_tokens.advance()
                result = self.node_provider.new_node(
                    NodeType.EXPONENTIATION2, [result, self.object()]
                )
        return result

    def object(self, context: str = None):
        """
        Parse an object.

        Returns:
            An object node that represents the input object.
        """
        token = self.iterating_tokens.current()

        if token.type == TokenType.FLOAT and context == None:
            self.iterating_tokens.advance()
            return self.node_provider.new_node(NodeType.FLOAT, token)

        elif token.type == TokenType.OBJECT and context == "object_key":
            self.iterating_tokens.advance()
            return self.node_provider.new_node(NodeType.VARIABLEKEY, token)

        elif token.type == TokenType.OBJECT and context == "tensor_key":
            self.iterating_tokens.advance()
            return self.node_provider.new_node(NodeType.TENSOR_KEY, token)

        elif token.type == TokenType.OBJECT and context == "symbol_key":
            self.iterating_tokens.advance()
            return self.node_provider.new_node(NodeType.SYMBOL_KEY, token)

        elif token.type == TokenType.INTEGER and context == None:
            self.iterating_tokens.advance()
            return self.node_provider.new_node(NodeType.INTEGER, token)

        elif token.type == TokenType.OBJECT and context == None:
            self.iterating_tokens.advance()
            return self.node_provider.new_node(NodeType.OBJECT, token)

        elif token.type == TokenType.PLUS and context == None:
            self.iterating_tokens.advance()
            return self.node_provider.new_node(NodeType.POSITIVE, [self.object()])

        elif token.type == TokenType.MINUS and context == None:
            self.iterating_tokens.advance()
            return self.node_provider.new_node(NodeType.NEGATIVE, [self.object()])

        elif token.type == TokenType.LSQB and context == None:
            self.iterating_tokens.advance()
            elements = []
            if self.iterating_tokens.current().type != TokenType.RSQB:
                elements.append(self.statement())
                while (
                    self.iterating_tokens.current() != None
                    and self.iterating_tokens.current().type == TokenType.COMMA
                ):
                    self.iterating_tokens.advance()
                    elements.append(self.statement())
            if self.iterating_tokens.current().type != TokenType.RSQB:
                self.raise_error("Syntax Error, expecting a CLOSED_SQUARE_BRACE token.")
            self.iterating_tokens.advance()
            return self.node_provider.new_node(NodeType.ARRAY, elements)

        elif token.type == TokenType.FUNCTION and context == None:
            self.iterating_tokens.advance()
            func_name = token.value
            if self.iterating_tokens.current().type != TokenType.LPAR:
                self.raise_error("Syntax Error, expecting a OPEN_BRACE token.")
            self.iterating_tokens.advance()
            wrt_variables = self.find_variables()
            if self.iterating_tokens.current().type != TokenType.RPAR:
                self.raise_error("Syntax Error, expecting a CLOSE_BRACE token.")
            self.iterating_tokens.advance()
            return self.node_provider.new_node(
                NodeType.FUNCTION, [func_name, wrt_variables]
            )

        elif token.type == TokenType.LPAR and context == None:
            self.iterating_tokens.advance()
            result = self.expr()
            if self.iterating_tokens.current().type != TokenType.RPAR:
                self.raise_error("Syntax Error, expecting a OPEN_BRACE token.")
            self.iterating_tokens.advance()
            return result

        self.raise_error(
            f"Syntax Error, the type {token.value} is not recognized or supported by RelativisticPy's Parser."
        )

    # ===== Helper Methods ======

    def find_variables(self):
        """
        Helper method to find and extract variables from the input.

        Returns:
            A list of VariableNodes representing the extracted variables.
        """
        tokens = []
        tokens.append(self.statement())
        while (
            self.iterating_tokens.current() != None
            and self.iterating_tokens.current().type == TokenType.COMMA
        ):
            self.iterating_tokens.advance()
            # if self.iterating_tokens.current().type != NodeType.OBJECT:
            #     self.raise_error("Syntax Error, expecting a VARIABLE token.")
            tokens.append(self.statement())
        return tokens
