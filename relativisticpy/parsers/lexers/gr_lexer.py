from relativisticpy.parsers.lexers.base import BaseLexer, Characters, LexerResult, TokenType

class GRLexer(BaseLexer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tokenize(self):
        # If the token is longer than a single character, we need to defined how to generate it.
        while self.current_char() != None:
            if self.current_char() in Characters.WHITESPACE.value:
                self.advance_char()

            elif self.current_char() in Characters.COMMENT.value:
                self._skip_comment()

            elif self.current_char() in Characters.DELINIMATORS.value:
                start_pos = self.current_pos()
                self.advance_char()
                self.token_provider.new_token(
                    TokenType.NEWLINE,
                    TokenType.NEWLINE.value,
                    start_pos=start_pos,
                    end_pos=self.current_pos(),
                )

            elif self.current_char() == TokenType.BACKSLASH.value:
                self._build_latex()

            elif self.current_char() in Characters.LETTERS.value:
                self._identifiers()

            elif self.current_char() in Characters.DIGITS.value:
                self._build_number()

            elif self.current_char() in TokenType.SINGLES():
                self._operation()

            else:
                raise Exception(f"Illegal Character '{self.current_char()}'")

        # Left the while loop => end of strings => wrap up the tokens.
        tokens = self.token_provider.get_tokens()
        return LexerResult(self.raw_code, tokens)

    def _build_number(self):
        number = ""
        start_pos = self.current_pos()
        while self.current_char() != None and (
            self.current_char() in Characters.NUMBER.value
        ):
            number += self.current_char()
            self.advance_char()
        count = number.count(".")
        if count == 0:
            self.token_provider.new_token(
                TokenType.INT, number, start_pos=start_pos, end_pos=self.current_pos()
            )
        elif count == 1:
            self.token_provider.new_token(
                TokenType.FLOAT, number, start_pos=start_pos, end_pos=self.current_pos()
            )
        else:
            raise Exception(f"Illegal Character '{number}'")

    def _operation(self):
        ops = []
        start_pos = self.current_pos()
        doubles_dic = TokenType.DOUBLES()
        triples_dic = TokenType.TRIPPLES()
        while (
            self.current_char() != None
            and self.current_char() in TokenType.SINGLES()
        ):
            ops.append(self.current_char())
            self.advance_char()

        i = 0
        while i < len(ops):
            if (
                i + 2 < len(ops)
                and ops[i] in triples_dic
                and self.token_provider.tripple_match_exists(
                    ops[i], ops[i + 1], ops[i + 2]
                )
            ):
                self.token_provider.new_tripple_operation_token(
                    ops[i],
                    ops[i + 1],
                    ops[i + 2],
                    start_pos=start_pos,
                    end_pos=self.current_pos(),
                )
                i += 3
            elif (
                i + 1 < len(ops)
                and ops[i] in doubles_dic
                and self.token_provider.double_match_exists(ops[i], ops[i + 1])
            ):
                self.token_provider.new_double_operation_token(
                    ops[i], ops[i + 1], start_pos=start_pos, end_pos=self.current_pos()
                )
                i += 2
            else:
                self.token_provider.new_single_operation_token(
                    ops[i], start_pos=start_pos, end_pos=self.current_pos()
                )
                i += 1

    # Here we are building: function - object - tensor
    def _identifiers(self):
        start_pos = self.current_pos()
        obj = ""
        while self.current_char() != None and self.current_char() in Characters.IDS.value:
            obj += self.current_char()

            if self.peek_char(1, None) == "_" and self.peek_char(2, None) in Characters.IDS.value:
                self.advance_char()
                obj += self.current_char()

            self.advance_char()
        
        if obj in TokenType.Keywords():
            self.token_provider.new_token(TokenType.Keywords()[obj], obj, start_pos=start_pos, end_pos=self.current_pos() )
        else:
            self.token_provider.new_token(TokenType.ID, obj, start_pos=start_pos, end_pos=self.current_pos() )

    def _build_latex(self):
        start_pos = self.current_pos()
        obj = ""
        self.advance_char()

        while self.current_char() != None and self.current_char() in Characters.LETTERS.value:
            obj += self.current_char()
            self.advance_char()

        if self.current_char() == TokenType.BACKSLASH.value:
            self.token_provider.new_token(
                TokenType.DOUBLEBACKSLASH,
                obj,
                start_pos=start_pos,
                end_pos=self.current_pos(),
            )
            self.advance_char()
        elif obj not in TokenType.LaTeX():
            raise ValueError("TOKENIZER LEVEL ERROR: The latex keyword entered is not supported.")
        
        if obj in TokenType.LaTeX():
            self.token_provider.new_token(
                TokenType.LaTeX()[obj],
                obj,
                start_pos=start_pos,
                end_pos=self.current_pos(),
            )



    def _skip_comment(self):
        self.advance_char()

        while self.current_char() != "\n":
            self.advance_char()

        self.advance_char()
