r"""Notation from papers and textbooks, as their LaTeX writes it.

Equations in papers end sentences, use plain TeX (``{a \over b}``, ``{\rm d}``),
chain relations through derivations, and carry their own conventions for
indices. Each case here is written in the style of a real text.
"""

import pytest

from relativisticpy.errors import ParseError
from relativisticpy.language import parse, parse_expression, to_sexpr, tokenize


def sexpr(text: str) -> str:
    return to_sexpr(parse_expression(text))


def program(text: str) -> list[str]:
    return [to_sexpr(statement) for statement in parse(text).statements]


class TestPlainTex:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"{1\over 2} x", "(* (/ 1 2) x)"),
            (r"{a+b \over c}", "(/ (+ a b) c)"),
            (r"{{dx^\mu}\over{d\lambda}}", "(d/dlambda (x ^[mu]))"),  # still a derivative
            ("{1\\over\n 2}", "(/ 1 2)"),  # whatever the line breaks
            (r"\sqrt{1 \over x}", "(sqrt (/ 1 x))"),
            (r"{n \choose k}", "(call binomial n k)"),
        ],
    )
    def test_over(self, text, expected):
        assert sexpr(text) == expected

    def test_font_switches_last_to_the_end_of_their_group(self):
        tokens = tokenize(r"{\rm Tr}\,M + {\cal L}")
        assert [(t.value, t.verbatim) for t in tokens if t.kind.name == "IDENT"] == [
            ("Tr", True),
            ("M", False),
            ("L", True),
        ]

    def test_plain_tex_matrices(self):
        assert sexpr(r"\pmatrix{a & b \cr c & d}") == "(pmatrix (a b) (c d))"
        assert sexpr(r"\cases{a = 1 \cr b = 2 \cr}") == "(list (= a 1) (= b 2))"

    def test_cases_written_as_an_array_in_a_brace(self):
        source = r"""
            \left\{ \begin{array}{ll}
                x & \mbox{if } x > 0 \\
                0 & \mbox{otherwise}
            \end{array} \right.
        """
        assert sexpr(source) == "(piecewise (x if (> x 0)) (0 if otherwise))"
        # in brackets that close, an array is still a matrix
        assert sexpr(r"\left( \begin{array}{cc} 1 & 2 \end{array} \right)") == "(array (1 2))"

    def test_symbol_commands_are_single_symbols(self):
        assert [t.verbatim for t in tokenize(r"\hbar \ell")[:2]] == [True, True]


class TestSentences:
    @pytest.mark.parametrize("text", [r"E = m c^2\ .", "E = m c^2,", r"E = m c^2 .\label{e}"])
    def test_punctuation_ends_a_sentence_not_an_equation(self, text):
        assert program(text) == ["(= E (* m (^ c 2)))"]

    def test_commas_and_wide_spaces_separate_equations(self):
        assert program(r"a = 1,\quad b = 2 \qquad c = 3") == ["(= a 1)", "(= b 2)", "(= c 3)"]

    def test_a_wide_space_inside_brackets_does_not(self):
        assert program(r"f(x \quad y)") == ["(call f (* x y))"]

    def test_connectives_between_equations(self):
        assert program(r"a = b \Rightarrow c = d") == ["(= a b)", "(= c d)"]

    def test_a_display_is_one_formula_whatever_its_lines(self):
        assert program("\\[ a =\n b +\n c \\]") == ["(= a (+ b c))"]


class TestDerivations:
    def test_relations_chain(self):
        assert sexpr("a = b = c") == "(chain a = b = c)"

    def test_rows_continue_with_a_relation_or_an_aligned_operator(self):
        source = r"""
            \begin{eqnarray}
              R &=& a + b \nonumber \\
                & & {} - c \\
                &=& d\ ,
            \end{eqnarray}
        """
        assert program(source) == ["(chain R = (- (+ a b) c) = d)"]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"\alpha \sim l^2", "(~ alpha (^ l 2))"),
            (r"\rho \propto a^{-3}", "(∝ rho (^ a (- 3)))"),
            (r"x \approx 1.5", "(≈ x 1.5)"),
            (r"p \in M", "(∈ p M)"),
            (r"\phi : M \rightarrow N", "(chain phi : M → N)"),
            (r"\mu \leftrightarrow \nu", "(↔ mu nu)"),
        ],
    )
    def test_relations_that_are_read_but_not_computed(self, text, expected):
        assert sexpr(text) == expected

    def test_equiv_states_an_identity_when_it_cannot_define(self):
        assert program(r"\phi(x, 1) \equiv y = 2") == ["(chain (call phi x 1) ≡ y = 2)"]
        assert program(r"\Lambda \equiv 2") == ["(:= Lambda 2)"]


class TestIndices:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"T^{\mu_1 \cdots \mu_k}", "(T ^[mu_1 ⋯ mu_k])"),
            (r"x^{\mu'}", "(x ^[mu'])"),
            (r"\Lambda^{\mu'}{}_{\nu}", "(Lambda ^[mu'] _[nu])"),
            (r"A_{\mu_{p+1}}", "(A _[mu_{p+1}])"),
            (r"(\phi_{t*})_\mu", "((phi _[t*]) _[mu])"),
            (r"R_{\rho\sigma{\mu\nu}}", "(R _[rho sigma mu nu])"),  # as from a macro
            (r"\phi,{}_\mu", "(phi _[, mu])"),
            (r"x^\mu'", "(x ^[mu'])"),
            (r"{\Gamma^r}_{tt}", "(Gamma ^[r] _[t t])"),  # MTW's staggered indices
            (r"{T_{\alpha}}^{\beta}", "(T _[alpha] ^[beta])"),
        ],
    )
    def test_index_labels(self, text, expected):
        assert sexpr(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("r_+^2", "(r _+ ^2)"),
            (r"r_\pm", "(r _±)"),
            ("i^- = 1", "(= (^ i -) 1)"),
            ("r^* + 1", "(+ (^ r *) 1)"),
            (r"{}^{(3)}R_{ij}", "(^{(3)}R _[i j])"),
        ],
    )
    def test_labels_on_symbols(self, text, expected):
        assert sexpr(text) == expected


class TestOperators:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"\Box \phi", "(* □ phi)"),
            (r"\pounds_V T", "(* (£ _[V]) T)"),
            (r"\pounds_VT", "(* (£ _[V]) T)"),  # as LaTeX reads it, one letter
            (r"A \wedge B", "(∧ A B)"),
            (r"a \pm b", "(± a b)"),
            (r"\pm b", "(± b)"),
            (r"(*F)_{\mu\nu}", "((* F) _[mu nu])"),
            (r"f \circ g", "(∘ f g)"),
            (r"a + \cdots + b", "(+ (+ a ⋯) b)"),
            (r"\sum_b x^b", "(sum b (x ^[b]))"),
        ],
    )
    def test_operators(self, text, expected):
        assert sexpr(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"\int d^4x \sqrt{-g} R", "(int (* (sqrt (- g)) R) (d^4 x))"),
            (r"\int d^nx\, f", "(int f (d^n x))"),
            (r"\int_M {\rm d}\omega", "(int (* d omega) over M)"),
        ],
    )
    def test_integrals_of_papers(self, text, expected):
        assert sexpr(text) == expected

    def test_a_function_argument_ends_at_brackets(self):
        assert sexpr(r"\cos\theta\left(\frac{d\phi}{d\lambda}\right)^2") == (
            "(* (call cos theta) (^ (d/dlambda phi) 2))"
        )


class TestAnalysis:
    """The notation of analysis and statistics: spaces, memberships, elementwise
    products."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"\mathbb{R}^{500}", "(^ ℝ 500)"),  # not R, the Ricci scalar
            (r"\mathbb{E}_t[X_{ta}]", "(* (𝔼 _[t]) (list (X _[t a])))"),
            (r"\mathbb C", "ℂ"),
            ("ℝ^3", "(^ ℝ 3)"),  # typed directly
            (r"\mathbb{1}_a", "(𝟙 _[a])"),
            (r"\mathbb{Ricci}", "Ricci"),  # more than a letter: only a typeface
        ],
    )
    def test_blackboard_bold_is_a_name_of_its_own(self, text, expected):
        assert sexpr(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"a, b \in \mathcal{A}", ["(∈ (tuple a b) A)"]),
            (r"t \in \mathcal{T}, \quad a, b \in \mathcal{A}", ["(∈ t T)", "(∈ (tuple a b) A)"]),
            (r"x, y", ["x", "y"]),  # without a membership, still two statements
            (r"p \in M, \quad q", ["(∈ p M)", "q"]),
            (r"p_{1}, p_{2} \in \mathbb{R}^3", ["(∈ (tuple (p _[1]) (p _[2])) (^ ℝ 3))"]),
            # prose, and names set apart by wide spacing, are not members
            (r"\text{where} \quad x, y \in \mathbb{R}", ["(∈ (tuple x y) ℝ)"]),
            (r"\text{for}, a \in \mathcal{A}", ["for", "(∈ a A)"]),
            (r"t \quad a, b \in \mathcal{A}", ["t", "(∈ (tuple a b) A)"]),
        ],
    )
    def test_members_listed_together_are_one_declaration(self, text, expected):
        assert program(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"A \odot B", "(⊙ A B)"),
            ("x ⊙ y", "(⊙ x y)"),
            (r"A_{ab} \odot B_{ab} + C", "(+ (⊙ (A _[a b]) (B _[a b])) C)"),
            (r"M_\odot", "(M _[odot])"),  # a subscript: the Sun
            (r"2 M_\odot r", "(* (* 2 (M _[odot])) r)"),
            ("M_⊙", "(M _[odot])"),
            # the right operand is all the factors after it, maybe signed
            (r"w_a \odot \mathbb{E}_t[X_{ta}]", "(⊙ w_a (* (𝔼 _[t]) (list (X _[t a]))))"),
            (r"A \odot -B", "(⊙ A (- B))"),
            ("A ⊙ B C + D", "(+ (⊙ A (* B C)) D)"),
            (r"A \odot B \odot C", "(⊙ (⊙ A B) C)"),
        ],
    )
    def test_odot_between_operands_is_the_elementwise_product(self, text, expected):
        assert sexpr(text) == expected


def test_unknown_notation_still_fails_clearly():
    with pytest.raises(ParseError, match="unknown LaTeX command"):
        parse_expression(r"\bowtie x")
