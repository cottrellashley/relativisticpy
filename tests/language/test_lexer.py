"""The lexer: plain-text maths, LaTeX and Unicode all become one token stream."""

from textwrap import dedent

import pytest

from relativisticpy.errors import ParseError
from relativisticpy.language import Source, TokenKind, tokenize

VALUED = {TokenKind.IDENT, TokenKind.NUMBER, TokenKind.STRING, TokenKind.COMMAND}


def lex(text: str) -> str:
    """Render the tokens of ``text`` compactly: ``x + 1`` gives ``x + 1``.

    Identifiers, numbers and strings print as their value, commands as
    ``\\name``, environments as ``begin:name``/``end:name`` and every other
    token as its symbol; line breaks print as ``⏎``.
    """
    rendered = []
    for token in tokenize(text)[:-1]:
        kind = token.kind
        if kind is TokenKind.STRING:
            rendered.append(repr(token.value))
        elif kind is TokenKind.COMMAND:
            rendered.append("\\" + token.value)
        elif kind in VALUED:
            rendered.append(token.value)
        elif kind is TokenKind.BEGIN:
            rendered.append(f"begin:{token.value}")
        elif kind is TokenKind.END:
            rendered.append(f"end:{token.value}")
        elif kind is TokenKind.NEWLINE:
            rendered.append("⏎")
        else:
            rendered.append(kind.value)
    return " ".join(rendered)


def kinds(text: str) -> list[TokenKind]:
    return [token.kind for token in tokenize(text)]


class TestPlainText:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("42", "42"),
            ("3.14", "3.14"),
            (".5", ".5"),
            ("6.674e-11", "6.674e-11"),
            ("3E5", "3E5"),
            ("2e", "2 e"),  # Euler's number, not an exponent
            ("2x", "2 x"),
            ("1.5.2", "1.5 .2"),
        ],
    )
    def test_numbers(self, text, expected):
        assert lex(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"10\,000", "10000"),
            (r"299\,792\,458", "299792458"),
            ("10~000 10\u2009000 10\u202f000", "10000 10000 10000"),
            (r"3.141\,592", "3.141592"),
            (r"6.022\,140\,76", "6.02214076"),  # the last group of decimals may be shorter
            (r"1\,000.5", "1000.5"),
            (r"10\,00", "10 00"),  # not a group of three
            (r"2\,x", "2 x"),
        ],
    )
    def test_digits_grouped_in_threes_are_one_number(self, text, expected):
        assert lex(text) == expected

    def test_a_grouped_number_spans_its_groups(self):
        assert tokenize(r"10\,000")[0].span.text == r"10\,000"

    def test_a_point_before_a_unicode_digit_is_a_full_stop(self):
        # ² and ₀ are digits to str.isdigit, but they are scripts, not part of a number
        assert lex("5.²") == "5 . ^ { 2 }"
        assert lex("x.₀") == "x . _ { 0 }"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("x", "x"),
            ("theta", "theta"),
            ("TempOne", "TempOne"),
            ("eq0", "eq0"),
            ("abs_func", "abs_func"),
            ("x_0", "x_0"),
            ("T_a", "T_a"),
            # an underscore followed by anything else is a subscript
            ("g_{mu}", "g _ { mu }"),
            ("x_", "x _"),
            (r"R_\mu", "R _ mu"),
        ],
    )
    def test_identifiers(self, text, expected):
        assert lex(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # a script without braces is one letter, as TeX reads it
            ("R^a_b", "R ^ a _ b"),
            ("G^t_t", "G ^ t _ t"),
            (r"\delta^a_b", "delta ^ a _ b"),
            ("A^i_j", "A ^ i _ j"),
            ("x^a_b", "x ^ a _ b"),
            (r"\Gamma^r_tt", "Gamma ^ r _ t t"),
            ("x^ab", "x ^ a b"),
            (r"\pounds_VT", r"\pounds _ V T"),
            (r"\omega_\mu{}^a{}_cX^c", "omega _ mu { } ^ a { } _ c X ^ c"),
            # or a Greek letter spelled out, as plain text writes \mu
            ("x^mu + y^theta", "x ^ mu + y ^ theta"),
            ("x^munu", "x ^ mu nu"),
            # a number is whole, and so is a name in a style command's braces
            ("x^10", "x ^ 10"),
            (r"x_\mathrm{max} + y_{\rm eff}", "x _ max + y _ { eff }"),
            # names written with an underscore are names, as plain text writes them
            ("r_s^2 + x_0 + T_x_0", "r_s ^ 2 + x_0 + T_x_0"),
        ],
    )
    def test_a_script_without_braces_is_one_letter(self, text, expected):
        assert lex(text) == expected

    @pytest.mark.parametrize("text", ["oo", "x^oo", r"\int_0^oo", "∫_0^oo", "a^{oo}"])
    def test_oo_is_one_symbol_infinity_a_script_whole(self, text):
        # x^oo was x^o o, and oo in LaTeX or Unicode the product of two o's
        infinity = [t for t in tokenize(text)[:-1] if t.span.text == "oo"]
        assert len(infinity) == 1
        assert infinity[0].kind is TokenKind.IDENT
        assert infinity[0].value == "oo"
        assert infinity[0].verbatim

    def test_a_name_with_oo_in_it_is_a_name(self):
        assert lex("foo + oon + oo_1 + x^ooa") == "foo + oon + oo_1 + x ^ o oa"
        assert not any(t.verbatim for t in tokenize("foo oon")[:-1])

    def test_escaped_underscores_join_a_name(self):
        assert lex(r"geodesic\_equations(x) x\_0a") == "geodesic_equations ( x ) x_0a"

    def test_an_escaped_underscore_needs_a_name_after_it(self):
        with pytest.raises(ParseError, match=r"unsupported LaTeX escape '\\_'"):
            lex(r"x\_θ")

    def test_single_character_operators(self):
        assert lex("+ - * / ^ _ ! = < > ( ) [ ] { } | , : ; &") == (
            "+ - * / ^ _ ! = < > ( ) [ ] { } | , : ; &"
        )

    def test_two_character_operators(self):
        assert lex("** := == != <= >= ->") == "** := == != <= >= ->"

    def test_longest_match_wins(self):
        assert lex("a**b") == "a ** b"
        assert lex("x:=1") == "x := 1"
        assert lex("f : (x) -> x") == "f : ( x ) -> x"
        assert lex("n!=1") == "n != 1"

    def test_strings(self):
        assert lex("symbols('x y')") == "symbols ( 'x y' )"
        assert lex('"text"') == "'text'"

    def test_primes_follow_names_without_space(self):
        assert lex("f'(x)") == "f ' ( x )"
        assert lex("f''(x)") == "f ' ' ( x )"
        assert lex("y'") == "y '"

    def test_quote_after_space_starts_a_string(self):
        assert lex("f 'x'") == "f 'x'"

    def test_comments_run_to_the_end_of_the_line(self):
        assert lex("x := 1 # a comment\ny") == "x := 1 ⏎ y"
        assert lex("x % a LaTeX comment\ny") == "x ⏎ y"

    def test_each_line_break_is_a_token(self):
        assert lex("a\n\nb") == "a ⏎ ⏎ b"

    def test_whitespace_is_recorded_on_the_following_token(self):
        tokens = tokenize("a b(c) d")
        assert [t.space_before for t in tokens[:-1]] == [False, True, False, False, False, True]

    def test_eof_is_always_last(self):
        assert kinds("") == [TokenKind.EOF]
        assert kinds("x")[-1] is TokenKind.EOF


class TestWords:
    """LaTeX's functions and operators, written in letters as plain text and Unicode write
    them, are LaTeX's commands where they apply to what follows them."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # sin²x was s i n² x, with i the imaginary unit, and log_2 8 a name times 8
            ("sin x", r"\sin x"),
            ("sin(x)", r"\sin ( x )"),
            ("sin^2 x", r"\sin ^ 2 x"),
            ("sin²x", r"\sin ^ { 2 } x"),
            ("cos θ", r"\cos theta"),
            (r"sin \theta + sin\,x", r"\sin theta + \sin x"),
            ("ln|x|", r"\ln | x |"),
            ("log_2 8", r"\log _ 2 8"),
            ("int_0^1 x dx", r"\int _ 0 ^ 1 x dx"),
            ("int x dx", r"\int x dx"),
            ("lim_{x->0} f", r"\lim _ { x -> 0 } f"),
            ("lim_(x->0) f", r"\lim _ ( x -> 0 ) f"),
            ("sum_{n=1}^oo a_n", r"\sum _ { n = 1 } ^ oo a_n"),
            ("min_x x^2", r"\min _ x x ^ 2"),
            ("sup_x f", r"\sup _ x f"),
            ("argmax_x f", "argmax _ x f"),
        ],
    )
    def test_a_word_that_applies_is_the_command(self, text, expected):
        assert lex(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("sin + 1", "sin + 1"),
            ("max := 10", "max := 10"),
            ("max_iter := 100", "max_iter := 100"),
            ("max_output is 1000", "max_output is 1000"),
            ("sum_n + 1", "sum_n + 1"),
            ("sum(a, b)", "sum ( a , b )"),
            ("sinx + sin_theta x", "sinx + sin_theta x"),
            ("sin'(x)", "sin ' ( x )"),
            ("x = 5 min", "x = 5 min"),
            ("max if x > 0", "max if x > 0"),
            # the Reynolds and Prandtl numbers, and i n t in an exponent
            ("Re^(1/2) Pr^(1/3)", "Re ^ ( 1 / 2 ) Pr ^ ( 1 / 3 )"),
            ("e^{int}", "e ^ { int }"),
            (r"\mathrm{sin} x", "sin x"),
        ],
    )
    def test_a_word_that_applies_to_nothing_is_a_name(self, text, expected):
        assert lex(text) == expected


class TestLatexSymbols:
    def test_greek_letters_are_names(self):
        assert lex(r"\alpha \mu\nu \Gamma \Omega") == "alpha mu nu Gamma Omega"

    def test_command_and_plain_spellings_agree(self):
        assert tokenize(r"\theta")[0].value == tokenize("theta")[0].value == "theta"

    def test_variant_letterforms_are_the_same_symbol(self):
        assert lex(r"\varphi \vartheta \varepsilon \varrho") == "phi theta epsilon rho"
        assert lex(r"\varOmega \varGamma \varPhi") == "Omega Gamma Phi"

    def test_varpi_is_its_own_symbol(self):
        assert lex(r"\varpi \pi") == "varpi pi"

    def test_letter_like_commands(self):
        assert lex(r"\infty \hbar \ell") == "infty hbar ell"

    @pytest.mark.parametrize(
        ("command", "kind"),
        [
            (r"\cdot", TokenKind.STAR),
            (r"\times", TokenKind.STAR),
            (r"\div", TokenKind.SLASH),
            (r"\le", TokenKind.LESS_EQ),
            (r"\leq", TokenKind.LESS_EQ),
            (r"\geqslant", TokenKind.GREATER_EQ),
            (r"\neq", TokenKind.NOT_EQ),
            (r"\to", TokenKind.ARROW),
            (r"\rightarrow", TokenKind.ARROW),
            (r"\equiv", TokenKind.DEFINE),
            (r"\coloneqq", TokenKind.DEFINE),
            (r"\vert", TokenKind.BAR),
        ],
    )
    def test_operator_commands(self, command, kind):
        assert kinds(command)[0] is kind

    def test_structural_commands_are_left_for_the_parser(self):
        assert lex(r"\frac{1}{2} \sqrt{x} \partial \nabla \sin") == (
            r"\frac { 1 } { 2 } \sqrt { x } \partial \nabla \sin"
        )

    def test_unknown_commands_are_left_for_the_parser(self):
        assert lex(r"\bowtie") == r"\bowtie"


class TestLatexTypesetting:
    def test_inline_math_delimiters_are_ignored(self):
        assert lex(r"$x$ \( y \)") == "x y"

    def test_a_display_is_one_formula(self):
        # Line breaks inside \[ ... \] or $$ ... $$ do not end the statement.
        assert lex("$$y$$") == "begin:displaymath y end:displaymath"
        assert lex(r"\[ x \]") == "begin:displaymath x end:displaymath"
        assert lex(r"\] x $$") == "x"  # unmatched delimiters are skipped

    def test_spacing_commands_are_whitespace(self):
        assert lex(r"a\,b\;c\:d\!e\ f~g\quad h\qquad i") == "a b c d e f g h i"
        assert all(t.space_before for t in tokenize(r"a\,b")[1:-1])

    def test_unicode_spaces_are_spacing_and_a_byte_order_mark_is_nothing(self):
        assert lex("\ufeffx =\u00a01\u2009+\u202f2") == "x = 1 + 2"
        tokens = tokenize("\ufeffa\u00a0b\u2009c")
        assert [(t.value, t.space_before, t.explicit_space) for t in tokens[:-1]] == [
            ("a", False, False),
            ("b", True, True),
            ("c", True, True),
        ]

    def test_explicit_spacing_is_distinguished_from_whitespace(self):
        tokens = tokenize(r"a b\,c\quad d~e")
        assert [(t.value, t.space_before, t.explicit_space) for t in tokens[:-1]] == [
            ("a", False, False),
            ("b", True, False),
            ("c", True, True),
            ("d", True, True),
            ("e", True, True),
        ]

    def test_layout_commands_are_dropped_with_their_arguments(self):
        assert lex(r"x \label{eq:schwarzschild} \tag*{1} \vspace{5mm} \hspace*{1em} y") == "x y"
        assert lex(r"\displaystyle \sum\limits") == r"\sum"

    def test_line_break_commands_end_the_line(self):
        assert lex(r"a \newline b \linebreak[4] c") == "a ⏎ b ⏎ c"

    def test_plain_tex_rows_end_with_cr(self):
        assert lex(r"\matrix{a & b \cr c & d}") == r"begin:matrix a & b \\ c & d end:matrix"

    def test_text_commands_become_names(self):
        assert lex(r"\text{otherwise} \operatorname{tr}(A) \text{if }") == "otherwise tr ( A ) if"

    def test_text_with_spaces_becomes_a_string(self):
        assert lex(r"\text{for all } x") == "'for all' x"

    def test_empty_text_is_dropped(self):
        assert lex(r"a \text{} b") == "a b"

    def test_style_commands_are_transparent(self):
        assert lex(r"\mathrm{d}x") == "d x"
        assert lex(r"\mathbf{F}_{\mu} \boldsymbol{\nabla}") == r"F _ { mu } \nabla"
        assert lex(r"\mathcal{L} \mathrm{e}^{\mathrm{i}\pi}") == "L e ^ { i pi }"

    def test_names_in_text_and_style_commands_are_verbatim(self):
        tokens = tokenize(r"\mathrm{mass} \text{eff} m")
        assert [(t.value, t.verbatim) for t in tokens[:-1]] == [
            ("mass", True),
            ("eff", True),
            ("m", False),
        ]

    def test_style_command_without_braces_styles_the_next_token(self):
        assert lex(r"\mathbf x") == "x"

    def test_row_separator(self):
        assert lex(r"a \\ b") == r"a \\ b"

    def test_row_separator_spacing_argument_is_dropped(self):
        assert lex(r"a \\[2pt] b \\ [x]") == r"a \\ b \\ [ x ]"

    def test_a_starred_row_separator_leaves_no_star(self):
        # \\* only forbids a page break; its star is not a Hodge dual on the next row
        assert lex(r"a \\* b \\*[2pt] c") == r"a \\ b \\ c"

    def test_backslash_newline_continues_the_line(self):
        assert lex("a + \\\nb") == "a + b"

    def test_literal_braces_group(self):
        assert lex(r"\{ x \}") == "( x )"


class TestDelimiters:
    def test_left_right_parentheses(self):
        assert lex(r"\left( x \right)") == "( x )"

    def test_display_brackets_and_braces_group(self):
        assert lex(r"\left[ x \right]") == "( x )"
        assert lex(r"\left\{ x \right\}") == "( x )"

    def test_absolute_value(self):
        assert lex(r"\left| x \right|") == "| x |"
        assert lex(r"\left\lvert x \right\rvert") == "| x |"

    def test_evaluation_bar(self):
        assert lex(r"\left. f \right|_{x=0}") == "( f ) | _ { x = 0 }"

    def test_invisible_right_delimiter(self):
        assert lex(r"\left( x \right.") == "( x )"

    def test_floor_and_ceiling(self):
        assert lex(r"\left\lfloor x \right\rfloor") == r"\lfloor x \rfloor"
        assert lex(r"\left\lceil x \right\rceil") == r"\lceil x \rceil"

    def test_nested_pairs(self):
        assert lex(r"\left( \left| x \right| \right)") == "( | x | )"

    def test_sized_delimiters(self):
        assert lex(r"\bigl[ x \bigr] \Big( y \Big) \bigg| z \bigg|") == "( x ) ( y ) | z |"

    def test_size_command_without_delimiter_is_ignored(self):
        assert lex(r"\big x") == "x"

    def test_a_sized_slash_divides(self):
        assert lex(r"a \big/ b \Bigg/ c") == "a / b / c"

    def test_unmatched_left(self):
        with pytest.raises(ParseError, match=r"'\\left' has no matching '\\right'"):
            tokenize(r"\left( x")

    def test_unmatched_right(self):
        with pytest.raises(ParseError, match=r"'\\right' has no matching '\\left'"):
            tokenize(r"x \right)")

    def test_left_needs_a_delimiter(self):
        with pytest.raises(ParseError, match=r"\\left must be followed by a delimiter"):
            tokenize(r"\left x")


class TestEnvironments:
    def test_begin_and_end(self):
        assert lex(r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}") == (
            r"begin:pmatrix 1 & 2 \\ 3 & 4 end:pmatrix"
        )

    def test_starred_environment(self):
        assert lex(r"\begin{align*} x \end{align*}") == "begin:align* x end:align*"

    def test_alignment_markers_are_ignored_in_alignment_environments(self):
        assert lex(r"\begin{align} a &= b \\ c &= d \end{align}") == (
            r"begin:align a = b \\ c = d end:align"
        )

    def test_the_second_marker_of_a_column_pair_separates_equations(self):
        tokens = tokenize(r"\begin{align} a &= 1 & b &= 2 \\ c &= 3 \end{align}")
        assert [t.value for t in tokens if t.wide_space] == ["b"]  # not c, after the row break

    def test_a_bracket_broken_over_rows_is_one_bracket(self):
        source = r"\begin{align} y &= \left( a \right. \\ & \left. + b \right) \end{align}"
        assert lex(source) == "begin:align y = ( a + b ) end:align"

    def test_the_rows_of_multline_are_lines_of_one_equation(self):
        assert lex(r"\begin{multline} a \\ + b \end{multline}") == (
            "begin:multline a ⏎ + b end:multline"
        )

    def test_ampersands_separate_columns_in_a_matrix_inside_an_alignment(self):
        source = r"\begin{align} g &= \begin{pmatrix} a & b \end{pmatrix} \end{align}"
        assert lex(source) == "begin:align g = begin:pmatrix a & b end:pmatrix end:align"

    def test_array_column_specification_is_dropped(self):
        assert lex(r"\begin{array}{|c@{}c|} 1 & 2 \end{array}") == "begin:array 1 & 2 end:array"

    def test_mismatched_end(self):
        with pytest.raises(ParseError, match=r"\\end\{bmatrix\} does not match \\begin\{pmatrix\}"):
            tokenize(r"\begin{pmatrix} 1 \end{bmatrix}")

    def test_unclosed_environment(self):
        with pytest.raises(ParseError, match=r"\\begin\{cases\} has no matching \\end\{cases\}"):
            tokenize(r"\begin{cases} x")

    def test_end_without_begin(self):
        with pytest.raises(ParseError, match=r"without a matching \\begin"):
            tokenize(r"\end{matrix}")


class TestUnicode:
    def test_greek_letters(self):
        assert lex("α β θ ϑ φ ϕ μ µ Γ Ω") == "alpha beta theta theta phi phi mu mu Gamma Omega"

    def test_symbols(self):
        assert lex("∞ ℏ ℓ π ⊙") == "infty hbar ell pi odot"

    def test_symbols_are_one_symbol_as_their_commands_are(self):
        # as \hbar is: hbar is not h b a r
        assert all(t.verbatim for t in tokenize("∞ ℏ ℓ ϖ ⊙ θ")[:-1])

    def test_operators(self):
        assert lex("a·b×c÷d−e ≤ ≥ ≠ → ≡ ≔") == "a * b * c / d - e <= >= != -> := :="

    def test_operator_symbols_become_commands(self):
        assert lex("∂ ∇ √ ∑ ∏ ∫ ∬ ∭ ∮") == (
            r"\partial \nabla \sqrt \sum \prod \int \iint \iiint \oint"
        )

    def test_superscripts_are_spelled_out(self):
        assert lex("r²") == "r ^ { 2 }"
        assert lex("x⁻¹²") == "x ^ { - 12 }"
        assert lex("xⁿ") == "x ^ { n }"

    def test_subscripts_are_spelled_out(self):
        assert lex("x₀") == "x _ { 0 }"

    def test_prime_symbol(self):
        assert lex("f′(x)") == "f ' ( x )"

    def test_vulgar_fractions_and_the_fraction_slash_divide(self):
        # ½ and 3⁄4 were refused as characters the language had no use for
        assert lex("x + ½") == "x + ( 1 / 2 )"
        assert lex("3⁄4 + ¾") == "3 / 4 + ( 3 / 4 )"

    @pytest.mark.parametrize("text", ["2½", "2 ½"])
    def test_a_number_before_a_vulgar_fraction_is_refused(self, text):
        # 2½ is two and a half, and the product of 2 and ½ is 1
        error = lexing_error(text)
        assert error.message == f"{text} is a whole number and a fraction side by side"
        assert error.hint == "for the mixed number, write 2 + 1/2; for the product, 2 \\cdot 1/2"

    def test_greek_capitals_that_look_latin_are_latin(self):
        assert lex("Α Μ") == "A M"


class TestSpans:
    def test_token_spans_cover_their_source_text(self):
        tokens = tokenize(r"g_{\mu\nu} := \frac{1}{2}")
        assert [t.span.text for t in tokens[:-1]] == [
            "g", "_", "{", r"\mu", r"\nu", "}", ":=", r"\frac", "{", "1", "}", "{", "2", "}",
        ]  # fmt: skip

    def test_eof_span_is_empty_and_at_the_end(self):
        eof = tokenize("abc")[-1]
        assert (eof.span.start, eof.span.end) == (3, 3)

    def test_line_and_column(self):
        token = tokenize("x := 1\n  y")[4]
        assert (token.value, token.span.line, token.span.column) == ("y", 2, 3)

    def test_source_name_appears_in_errors(self):
        with pytest.raises(ParseError) as error:
            tokenize(Source("x @ y", name="metric.relpy"))
        assert "metric.relpy:1:3" in str(error.value)


class TestErrors:
    def test_unexpected_character_is_underlined(self):
        with pytest.raises(ParseError) as error:
            tokenize("x := 1 @ 2")
        assert str(error.value) == dedent("""\
            unexpected character '@'
             --> <input>:1:8
              |
            1 | x := 1 @ 2
              |        ^""")

    def test_unterminated_string(self):
        with pytest.raises(ParseError, match="unterminated string"):
            tokenize("symbols('x y)")

    def test_string_cannot_span_lines(self):
        with pytest.raises(ParseError, match="unterminated string"):
            tokenize("'x\ny'")

    def test_lone_backslash(self):
        with pytest.raises(ParseError, match="a backslash must start a LaTeX command"):
            tokenize("x \\")

    def test_unsupported_escape(self):
        with pytest.raises(ParseError, match=r"unsupported LaTeX escape '\\#'"):
            tokenize(r"\#")

    def test_unclosed_command_argument(self):
        with pytest.raises(ParseError, match=r"unclosed argument of \\text"):
            tokenize(r"\text{abc")

    def test_a_source_that_says_it_cannot_be_read(self):
        source = Source(r"x = \loop", unreadable=(r"the macro \loop expands without end", 4, 9))
        with pytest.raises(ParseError, match="expands without end") as error:
            tokenize(source)
        assert error.value.span is not None
        assert error.value.span.text == r"\loop"

    def test_layout_command_needs_its_argument(self):
        with pytest.raises(ParseError, match=r"\\label must be followed by a braced argument"):
            tokenize(r"\label x")


class TestLexerDetails:
    def test_braced_arguments_may_nest_and_escape(self):
        assert lex(r"x \label{eq:{a}\%b} y") == "x y"

    def test_operatorname_star(self):
        assert lex(r"\operatorname*{argmax}") == "argmax"

    @pytest.mark.parametrize("text", [r"\operatorname*{arg\,max}", r"\operatorname{arg max}"])
    def test_an_operator_name_is_one_name_however_it_is_spaced(self, text):
        assert lex(text) == "argmax"
        assert [t.kind for t in tokenize(text)] == [TokenKind.IDENT, TokenKind.EOF]

    def test_unclosed_style_argument(self):
        with pytest.raises(ParseError, match=r"unclosed argument of \\mathrm") as error:
            tokenize(r"\mathrm{abc")
        assert error.value.hint == r"add '}' where the argument ends, as in \mathrm{abc}"

    def test_left_with_an_unsupported_delimiter(self):
        with pytest.raises(ParseError, match=r"\\left must be followed by a delimiter"):
            tokenize(r"\left\alpha x \right)")

    def test_sized_floor(self):
        assert lex(r"\big\lfloor x \big\rfloor") == r"\lfloor x \rfloor"

    def test_norm_bars(self):
        assert lex(r"\|x\|") == "| x |"

    def test_unclosed_optional_argument_is_left_alone(self):
        assert lex(r"a \newline[ b") == "a ⏎ [ b"


def lexing_error(text: str) -> ParseError:
    with pytest.raises(ParseError) as caught:
        tokenize(text)
    return caught.value


class TestStringsThatAreNotRaw:
    """In a Python string that is not raw, "\\frac" arrives as a form feed and "rac"."""

    @pytest.mark.parametrize(
        ("text", "message", "hint"),
        [
            ("\frac{1}{2}", r"Python turned \f in \frac into a form feed", r'r"\frac{1}{2}"'),
            ("\beta + 1", r"Python turned \b in \beta into a backspace", r'r"\beta + 1"'),
            ("\alpha", r"Python turned \a in \alpha into an alert", r'r"\alpha"'),
            ("\vec{v}", r"Python turned \v in \vec into a vertical tab", r'r"\vec{v}"'),
            ("x + \theta", r"Python turned \t in \theta into a tab", r'r"x + \theta"'),
            ("\tau", r"Python turned \t in \tau into a tab", r'r"\tau"'),
            ("\rho", r"Python turned \r in \rho into a carriage return", r'r"\rho"'),
            ("x := 1\n\nabla_\\mu A^\\mu", r"Python turned \n in \nabla into a line break", None),
        ],
    )
    def test_escapes_that_begin_commands_are_errors(self, text, message, hint):
        error = lexing_error(text)
        assert error.message == message
        assert "control characters" in error.notes[0]
        assert error.hint is not None
        assert error.hint.startswith("write the string as a raw string")
        if hint is not None:
            assert error.hint.endswith(hint)

    def test_the_caret_is_on_the_control_character(self):
        _line, carets = str(lexing_error("x + \theta")).splitlines()[3:5]
        assert carets.index("^") == len("1 | x + ")

    @pytest.mark.parametrize(
        "text",
        [
            "x := 1\n\ty := 2",  # a tab before what no command ends with
            "\tanswer := 42",  # a tab-indented name: \tanswer is no command
            "x := 1\nu := 2",  # u completes \nu, but a line may well start with u
            "x := 1\r\ny := 2",  # a line ending, as on Windows
            "x := 1\n\neq := 2",  # eq completes \neq, but is a name too
        ],
    )
    def test_line_breaks_and_tabs_that_are_meant_are_read(self, text):
        assert tokenize(text)[-1].kind is TokenKind.EOF


class TestCharacters:
    @pytest.mark.parametrize(
        ("text", "message", "hint"),
        [
            ("𝒜 + 1", "unexpected character '𝒜' (U+1D49C MATHEMATICAL SCRIPT CAPITAL A)",
             r"\mathcal{A}"),
            ("Ｘ + 1", "unexpected character 'Ｘ' (U+FF38 FULLWIDTH LATIN CAPITAL LETTER X)", "X"),
            ("ψ\u0304 + 1", "unexpected character U+0304 COMBINING MACRON, an accent on ψ",
             r"\bar{\psi}"),
            ("x\u0302", "unexpected character U+0302 COMBINING CIRCUMFLEX ACCENT, an accent on x",
             r"\hat{x}"),
            ("x 😀 y", "unexpected character '😀' (U+1F600 GRINNING FACE)", None),
        ],
    )  # fmt: skip
    def test_an_unexpected_character_is_named_with_what_to_write(self, text, message, hint):
        error = lexing_error(text)
        assert error.message == message
        assert error.hint == (None if hint is None else f"write {hint}")


class TestCommandArguments:
    @pytest.mark.parametrize(
        ("text", "message"),
        [
            (r"\operatorname{tr", r"unclosed argument of \operatorname"),
            (r"\textbf", r"\textbf must be followed by a braced argument, as in \textbf{word}"),
            (r"\mathrm{x", r"unclosed argument of \mathrm"),
            (r"\mathcal", r"\mathcal needs an argument, as in \mathcal{A}"),
            (r"x + \mathbf", r"\mathbf needs an argument, as in \mathbf{v}"),
            (r"\begin", r"\begin must be followed by a braced argument, as in \begin{equation}"),
        ],
    )
    def test_messages_name_the_command(self, text, message):
        assert lexing_error(text).message == message

    def test_an_unclosed_argument_is_marked_at_its_brace(self):
        lines = str(lexing_error(r"\operatorname{tr")).splitlines()
        assert lines[4] == "  |              ^ this '{' is never closed"
        assert lines[5] == r"hint: add '}' where the argument ends, as in \operatorname{tr}"


class TestWhereTokensAre:
    def test_a_style_command_is_part_of_the_token_it_styles(self):
        # so that a statement starting with \mathrm{solve} is quoted from its backslash
        tokens = tokenize(r"\mathrm{solve}(x^2 = 4, x) + \mathrm{d} s + \mathbf{u v}")
        assert [t.span.text for t in tokens if t.kind is TokenKind.IDENT][:2] == [
            r"\mathrm{solve}",
            "x",
        ]
        idents = [t.span.text for t in tokens if t.kind is TokenKind.IDENT]
        assert idents[-4:] == [r"\mathrm{d}", "s", r"\mathbf{u", "v}"]

    def test_the_end_of_input_is_just_after_the_last_token(self):
        eof = tokenize("x := 3\ny +\n\n")[-1]
        assert (eof.span.line, eof.span.column) == (2, 4)


class TestRemainders:
    @pytest.mark.parametrize("text", ["x % 2", "y := n % k == 0", "(a + b) % 7 == 0", "x%2"])
    def test_a_python_remainder_is_an_error(self, text):
        error = lexing_error(text)
        assert error.message.startswith("'%' starts a comment, so ")
        assert error.hint is not None
        assert "for a comment, write #" in error.hint

    @pytest.mark.parametrize(
        "text",
        [
            "x := 1 % the radius of the star",
            "M := 1.989e30 % kg",  # a unit, as a comment
            "% a line of its own",
            r"\alpha % 2",
            "x %",
        ],
    )
    def test_a_comment_is_a_comment(self, text):
        assert tokenize(text)[0].kind in (TokenKind.IDENT, TokenKind.EOF)


class TestTokenDescriptions:
    @pytest.mark.parametrize(
        ("text", "description"),
        [
            ("x", "name 'x'"),
            ("42", "number '42'"),
            ("'hi'", "string 'hi'"),
            (r"\frac", r"\frac"),
            (r"\begin{pmatrix}", r"\begin{pmatrix}"),
            ("+", "'+'"),
            ("\n", "line break"),
        ],
    )
    def test_describe(self, text, description):
        token = tokenize(text + (r"\end{pmatrix}" if "begin" in text else ""))[0]
        assert token.describe() == description

    def test_repr(self):
        assert repr(tokenize("x")[0]) == "Token(IDENT, 'x', 0:1)"

    def test_spans_from_different_sources_cannot_merge(self):
        a, b = tokenize("x")[0].span, tokenize("x")[0].span
        with pytest.raises(ValueError, match="different sources"):
            a.to(b)
