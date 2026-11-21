r"""Reading LaTeX documents: their macros and display equations."""

import pytest

from relativisticpy import Workbook
from relativisticpy.documents import (
    Document,
    Macro,
    expand,
    find_macros,
    math_blocks,
    strip_comments,
)


def verdicts(text: str) -> list[str]:
    """What checking a document finds, statement by statement."""
    return [finding.verdict.value for finding in Workbook().check(Document(text)).findings]


class TestMacros:
    def test_newcommand_without_arguments(self):
        macros = find_macros(r"\newcommand{\be}{\begin{equation}}")
        assert macros["be"] == Macro("be", 0, r"\begin{equation}")

    def test_newcommand_with_arguments_and_a_default(self):
        macros = find_macros(r"\newcommand{\pd}[2][x]{\frac{\partial #2}{\partial #1}}")
        assert macros["pd"] == Macro("pd", 2, r"\frac{\partial #2}{\partial #1}", default="x")

    def test_def_with_parameters(self):
        assert find_macros(r"\def\p#1{{\partial}_{#1}}")["p"] == Macro("p", 1, r"{\partial}_{#1}")

    def test_let_and_declaremathoperator(self):
        macros = find_macros(r"\let\lie\pounds \DeclareMathOperator{\Tr}{Tr}")
        assert macros["lie"].body == r"\pounds"
        assert macros["Tr"].body == r"\operatorname{Tr}"

    def test_renewcommand_replaces_and_providecommand_does_not(self):
        macros = find_macros(
            r"\newcommand{\a}{1} \renewcommand{\a}{2} \newcommand{\b}{3} \providecommand{\b}{4}"
        )
        assert (macros["a"].body, macros["b"].body) == ("2", "3")


class TestExpansion:
    macros = find_macros(
        r"\def\mn{{\mu\nu}} \def\p#1{{\partial}_{#1}} \newcommand{\pd}[2][x]{#1#2}"
    )

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"g_\mn", r"g_{\mu\nu}"),
            (r"\p{\mu} \phi", r"{\partial}_{\mu} \phi"),
            (r"\p\mu", r"{\partial}_{\mu}"),  # an argument may be one command
            (r"\p 0", r"{\partial}_{0}"),  # or one character
            (r"\pd{y}", "xy"),
            (r"\pd[t]{y}", "ty"),
            (r"\p{\mn}", r"{\partial}_{{\mu\nu}}"),  # macros inside arguments expand too
            (r"\mnx", r"\mnx"),  # \mnx is another command
        ],
    )
    def test_expand(self, text, expected):
        assert expand(text, self.macros) == expected

    def test_a_macro_that_never_stops_expanding(self):
        with pytest.raises(ValueError, match="expands without end"):
            expand(r"\loop", find_macros(r"\def\loop{\loop}"))


class TestEquations:
    def test_every_kind_of_display(self):
        document = Document(
            r"""
            Inline $x$ is not a display.
            \begin{equation} a = 1 \end{equation}
            \begin{eqnarray} b &=& 2 \end{eqnarray}
            \begin{align*} c &= 3 \end{align*}
            $$ d = 4 $$
            \[ e = 5 \]
            """
        )
        environments = [equation.environment for equation in document.equations()]
        assert environments == ["equation", "eqnarray", "align*", "$$", "displaymath"]

    def test_shorthands_for_environments(self):
        document = Document(
            "\\newcommand{\\be}{\\begin{equation}}\n\\newcommand{\\ee}{\\end{equation}}\n"
            "Text.\n\\be x = 1 \\ee\n"
        )
        [equation] = document.equations()
        assert equation.source.text == r"\begin{equation} x = 1 \end{equation}"
        assert equation.line == 4

    def test_positions_are_the_file_s(self):
        document = Document("One.\nTwo, then $$ x =\n y $$")
        [equation] = document.equations()
        assert equation.source.origin == (2, 11)

    def test_comments_are_removed_but_escaped_percent_signs_stay(self):
        assert strip_comments("a % note\nb \\% c") == "a \nb \\% c"
        assert Document("% $$ x $$\n$$ y $$").equations()[0].source.text == "$$ y $$"

    def test_sections_are_counted(self):
        document = Document(
            r"""
            $$ a $$
            \section{One}
            $$ b $$ $$ c $$
            \chapter*{Two}
            $$ d $$
            """
        )
        assert [equation.section for equation in document.equations()] == [0, 1, 1, 2]

    def test_read_a_file(self, tmp_path):
        path = tmp_path / "paper.tex"
        path.write_text(r"\begin{equation} E = m c^2 \end{equation}")
        document = Document.read(path)
        assert document.name == str(path)
        assert len(document.equations()) == 1


class TestReading:
    def test_a_display_is_latex_even_without_a_command(self):
        # in a display, ab is a times b, so the identity holds
        assert verdicts(r"\begin{equation} (a+b)^2 = a^2 + 2ab + b^2 \end{equation}") == ["holds"]

    def test_a_markdown_display_is_latex(self):
        [block] = math_blocks("Prose.\n\n$$\n(a+b)^2 - a^2 - 2ab - b^2\n$$\n")
        assert block.latex
        assert Workbook().exe(block) == [0]

    def test_an_index_range_is_not_a_definition(self):
        document = r"""
            \begin{equation} \mu, \nu = 0, 1, 2, 3 \end{equation}
            \begin{equation} \nu + 1 = 1 \end{equation}
        """
        assert verdicts(document) == ["not computed", "differs"]  # \nu is not 0

    def test_words_between_equations_separate_them(self):
        document = r"""
            \begin{equation} a = 1 \quad \text{and} \quad b = 2 \end{equation}
            \begin{equation} a + b = 3 \end{equation}
        """
        assert verdicts(document) == ["defines", "defines", "holds"]
