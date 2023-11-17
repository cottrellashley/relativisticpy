from relativisticpy.parsers.shared.interfaces.iterator import IIterator
from relativisticpy.parsers.shared.interfaces.lexer import ILexer
from relativisticpy.parsers.shared.interfaces.tokens import ITokenProvider
from relativisticpy.parsers.shared.constants import TokenType
from relativisticpy.parsers.shared.constants import Characters


class Lexer(ILexer):
    def __init__(self, token_provider: ITokenProvider):
        self.token_provider = token_provider

    def tokenize(self, characters: IIterator):
        self.characters = characters
        self.characters.advance()
        # If the token is longer than a single character, we need to defined how to generate it.
        while self.characters.current() != None:
            if self.characters.current() in Characters.WHITESPACE.value:
                self.characters.advance()

            elif self.characters.current() in Characters.LETTERS.value:
                self._object()

            elif self.characters.current() in Characters.DIGITS.value:
                self._number()

            elif self.characters.current() in Characters.OPERATIONS.value:
                self._operation()

            else:
                raise Exception(f"Illegal Character '{self.characters.current()}'")
        a = self.token_provider.get_tokens()
        return a

    def _number(self):
        number = ""
        while self.characters.current() != None and (
            self.characters.current() in Characters.DIGITS.value
            or self.characters.current() == "."
        ):
            number += self.characters.current()
            self.characters.advance()
        count = number.count(".")
        if count == 0:
            self.token_provider.new_token(TokenType.INTEGER, number)
        elif count == 1:
            self.token_provider.new_token(TokenType.FLOAT, number)
        else:
            raise Exception(f"Illegal Character '{number}'")

    def _operation(self):
        ops = []
        while (
            self.characters.current() != None
            and self.characters.current() in Characters.OPERATIONS.value
        ):
            ops.append(self.characters.current())
            self.characters.advance()

        i = 0
        while i < len(ops):
            if (
                i + 2 < len(ops)
                and ops[i] in self.token_provider.tripples()
                and self.token_provider.tripple_match_exists(
                    ops[i], ops[i + 1], ops[i + 2]
                )
            ):
                self.token_provider.new_tripple_operation_token(
                    ops[i], ops[i + 1], ops[i + 2]
                )
                i += 3
            elif (
                i + 1 < len(ops)
                and ops[i] in self.token_provider.doubles()
                and self.token_provider.double_match_exists(ops[i], ops[i + 1])
            ):
                self.token_provider.new_double_operation_token(ops[i], ops[i + 1])
                i += 2
            else:
                self.token_provider.new_single_operation_token(ops[i])
                i += 1

    def _object(self):
        obj = ""
        while (
            self.characters.current() != None
            and self.characters.current() in Characters.OBJECTCHARACTERS.value
        ):
            if (
                self.characters.current() in Characters.CHARACTERS.value
                and self.characters.peek(1, "") == "("
            ):
                obj += self.characters.current()
                self.characters.advance()
                self.token_provider.new_token(TokenType.FUNCTION, obj)
            elif (
                self.characters.peek(1, "")
                not in Characters.OBJECTCHARACTERS.value + "("
            ):
                obj += self.characters.current()
                self.characters.advance()
                self.token_provider.new_token(TokenType.OBJECT, obj)
            elif self.characters.peek(1, None) == None:
                obj += self.characters.current()
                self.token_provider.new_token(TokenType.OBJECT, obj)
                self.characters.advance()
            else:
                obj += self.characters.current()
                self.characters.advance()
