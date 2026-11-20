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


class TestLatexSymbols:
    def test_greek_letters_are_names(self):
        assert lex(r"\alpha \mu\nu \Gamma \Omega") == "alpha mu nu Gamma Omega"

    def test_command_and_plain_spellings_agree(self):
        assert tokenize(r"\theta")[0].value == tokenize("theta")[0].value == "theta"

    def test_variant_letterforms_are_the_same_symbol(self):
        assert lex(r"\varphi \vartheta \varepsilon \varrho") == "phi theta epsilon rho"

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
        assert lex(r"\otimes") == r"\otimes"


class TestLatexTypesetting:
    def test_math_mode_delimiters_are_ignored(self):
        assert lex("$x$ $$y$$") == "x y"
        assert lex(r"\[ x \] \( y \)") == "x y"

    def test_spacing_commands_are_whitespace(self):
        assert lex(r"a\,b\;c\:d\!e\ f~g\quad h\qquad i") == "a b c d e f g h i"
        assert all(t.space_before for t in tokenize(r"a\,b")[1:-1])

    def test_layout_commands_are_dropped_with_their_arguments(self):
        assert lex(r"x \label{eq:schwarzschild} \tag*{1} \vspace{5mm} \hspace*{1em} y") == "x y"
        assert lex(r"\displaystyle \sum\limits") == r"\sum"

    def test_line_break_commands_end_the_line(self):
        assert lex(r"a \newline b \linebreak[4] c \cr d") == "a ⏎ b ⏎ c ⏎ d"

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

    def test_style_command_without_braces_styles_the_next_token(self):
        assert lex(r"\mathbf x") == "x"

    def test_row_separator(self):
        assert lex(r"a \\ b") == r"a \\ b"

    def test_row_separator_spacing_argument_is_dropped(self):
        assert lex(r"a \\[2pt] b \\ [x]") == r"a \\ b \\ [ x ]"

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
        assert lex("∞ ℏ ℓ π") == "infty hbar ell pi"

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

    def test_layout_command_needs_its_argument(self):
        with pytest.raises(ParseError, match=r"\\label must be followed by a braced argument"):
            tokenize(r"\label x")
