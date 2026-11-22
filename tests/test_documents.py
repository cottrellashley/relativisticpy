r"""Reading LaTeX documents: their macros and display equations."""

import time
from pathlib import Path

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
from relativisticpy.errors import ParseError
from relativisticpy.language import parse


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

    def test_let_copies_what_its_target_means_then(self):
        macros = find_macros(r"\newcommand{\a}{A} \let\b\a \renewcommand{\a}{B}")
        assert expand(r"\a \b", macros) == "B A"

    @pytest.mark.parametrize(
        ("definitions", "text"),
        [
            (r"\let\old\vec \renewcommand{\vec}[1]{\old{\mathbf{#1}}}", r"\vec{x}"),
            (r"\let\oldleft\left \renewcommand{\left}{\oldleft}", r"\left( x \right)"),
        ],
    )
    def test_let_of_a_command_is_that_command_even_once_redefined(self, definitions, text):
        # \old is the command \vec, which the new \vec does not expand again, endlessly
        expected = {r"\vec{x}": r"\vec{\mathbf{x}}"}.get(text, text)
        assert expand(text, find_macros(definitions)) == expected

    @pytest.mark.parametrize(
        ("definition", "text", "expected"),
        [
            (r"\newcommand{\abs}[1]{\lvert#1\rvert}", r"\abs{x}", r"\lvert x\rvert"),
            (r"\newcommand{\pair}[2]{#1#2}", r"\pair{\alpha}{b}", r"\alpha b"),
            (r"\newcommand{\R}{R}", r"\alpha\R", r"\alpha R"),
        ],
    )
    def test_an_expansion_does_not_join_a_command_beside_it(self, definition, text, expected):
        assert expand(text, find_macros(definition)) == expected


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

    def test_the_environments_of_amsmath_breqn_and_ieeetrantools(self):
        # alignat, IEEEeqnarray, dmath and math were skipped, and nothing said so
        text = r"""
            \begin{alignat}{2} x &= 1 &\quad y &= 2 \end{alignat}
            \begin{IEEEeqnarray}{rCl} q & = & 5 \end{IEEEeqnarray}
            \begin{dmath} p = 6 \end{dmath}
            \begin{dgroup}
                \begin{dmath} u = 1 \end{dmath} \begin{dmath} v = 2 \end{dmath}
            \end{dgroup}
            \begin{math} m = 7 \end{math}
        """
        document = Document(text)
        environments = [equation.environment for equation in document.equations()]
        assert environments == ["alignat", "IEEEeqnarray", "dmath", "dgroup", "math"]
        assert verdicts(text) == ["defines"] * 7

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

    def test_let_copies_a_macro_of_a_package(self, tmp_path):
        (tmp_path / "macros.sty").write_text(r"\newcommand{\R}{\mathcal{R}}")
        path = tmp_path / "paper.tex"
        path.write_text(r"\usepackage{macros} \let\Ric\R \renewcommand{\R}{X} $$ \Ric $$")
        [equation] = Document.read(path).equations()
        assert equation.source.text == r"$$ \mathcal{R} $$"

    def test_an_equation_whose_macros_do_not_expand_is_not_parsed(self):
        document = Document(
            "\\def\\loop{\\loop}\n"
            "\\newcommand{\\foo}[2][a]{#1 #2}\n"
            "$$ x = \\loop $$\n"
            "$$ \\foo[b $$\n"
            "$$ y = 2 $$\n"
        )
        findings = Workbook().check(document).findings  # the check goes on after them
        assert [(f.verdict.value, f.line, f.detail) for f in findings] == [
            ("not parsed", 3, "the macro \\loop expands without end"),
            ("not parsed", 4, "cannot expand \\foo: unclosed '['"),
            ("defines", 5, ""),
        ]

    def test_a_document_may_end_in_a_backslash(self):
        assert len(Document("$$ x $$ \\").equations()) == 1

    def test_positions_are_the_file_s_through_macros(self):
        document = Document(
            "\\newcommand{\\half}{\\frac{1}{2}}\n"
            "\\newcommand{\\two}{a\n+ b}\n"
            "Text \\[ \\half + \\two + @ \\]\n"
        )
        [equation] = document.equations()
        with pytest.raises(ParseError) as error:
            parse(equation.source)
        span = error.value.span
        assert (span.line, span.column) == (4, 24)  # where @ is in the file
        assert span.render().splitlines()[2:] == [
            r"4 |      \[ \half + \two + @ \]",
            "  |                        ^",
        ]

    def test_an_error_in_a_macro_points_at_the_macro(self):
        [equation] = Document("\\newcommand{\\bad}{\\oops}\nThen \\[ x = \\bad \\]").equations()
        with pytest.raises(ParseError, match=r"unknown LaTeX command \\oops") as error:
            parse(equation.source)
        line, carets = error.value.span.render().splitlines()[2:]
        assert line == r"2 |      \[ x = \bad \]"
        assert carets.index("^") == line.index(r"\bad")
        assert carets.count("^") == len(r"\bad")

    def test_equations_are_found_in_linear_time(self):
        lines = (rf"\section{{S{i}}} $$ x_{{{i}}} = {i} $$" for i in range(8000))
        document = Document("\n".join(lines))
        started = time.perf_counter()
        equations = document.equations()
        assert len(equations) == 8000
        assert time.perf_counter() - started < 3  # eleven seconds when quadratic

    @pytest.mark.parametrize(
        "text",
        [
            r"\iffalse $$ a = 1 $$ \fi $$ b = 2 $$",
            r"\iffalse \ifx\a\b $$ a = 1 $$ \fi $$ a = 1 $$ \fi $$ b = 2 $$",
            r"\newif\ifdraft \iffalse \ifdraft $$ a $$ \fi $$ a $$ \fi $$ b = 2 $$",
            r"\iffalse \ifthenelse{x}{y}{z} $$ a = 1 $$ \fi $$ b = 2 $$",  # a macro
            r"\iffalse $$ a = 1 $$ \else $$ b = 2 $$ \fi",
            r"\begin{comment} $$ a = 1 $$ \end{comment} $$ b = 2 $$",
            r"\newenvironment{eq}{\begin{equation}}{\end{equation}} $$ b = 2 $$",
            r"$$ b = 2 $$ \iff",  # the relation, not a conditional
        ],
    )
    def test_text_tex_skips_has_no_equations(self, text):
        assert [e.source.text for e in Document(text).equations()] == ["$$ b = 2 $$"]

    def test_spaces_before_the_braces_of_an_environment(self):
        document = Document(
            r"\begin {equation} a = 1 \end {equation} \begin{equation} b \end{equation}"
        )
        assert [e.source.text for e in document.equations()] == [
            r"\begin {equation} a = 1 \end {equation}",
            r"\begin{equation} b \end{equation}",
        ]

    def test_an_unclosed_display_hides_nothing_after_it(self):
        document = Document(r"$$ a = 1 \[ b = 2 \] \begin{equation} c \end{equation}")
        assert [e.environment for e in document.equations()] == ["displaymath", "equation"]


class TestIncludedFiles:
    """A paper's equations are also in the files it reads with \\input and \\include."""

    @pytest.fixture
    def paper(self, tmp_path):
        (tmp_path / "section1.tex").write_text(
            "\\section{One}\n\\begin{equation}\nx := 2\n\\end{equation}\n"
            "\\input{part}\n\\begin{equation}\nx^2 = 4\n\\end{equation}\n"
        )
        (tmp_path / "part.tex").write_text("Text.\n$$ y := x + 1 $$\n")
        (tmp_path / "section2.tex").write_text("\\begin{equation}\nx^3 = 9\n\\end{equation}\n")
        main = tmp_path / "main.tex"
        main.write_text(
            "\\documentclass{article}\n\\begin{document}\n$$ a := 1 $$\n"
            "\\input{section1}\n\\include{section2}\n\\input missing\n$$ x + 1 = 3 $$\n"
            "\\end{document}\n"
        )
        return tmp_path

    def test_their_equations_are_read_in_order_with_their_own_positions(self, paper):
        document = Document.read(paper / "main.tex")
        found = [
            (Path(e.source.name).name, e.line, e.source.text.split()[1])
            for e in document.equations()
        ]
        assert found == [
            ("main.tex", 3, "a"),
            ("section1.tex", 2, "x"),
            ("part.tex", 2, "y"),
            ("section1.tex", 6, "x^2"),
            ("section2.tex", 1, "x^3"),
            ("main.tex", 7, "x"),
        ]

    def test_a_check_reads_them(self, paper):
        findings = Workbook().check(paper / "main.tex").findings
        assert [f.verdict.value for f in findings] == [
            "defines", "defines", "defines", "holds", "differs", "holds",
        ]  # fmt: skip
        assert findings[4].span.render().splitlines()[0].endswith("section2.tex:2:1")

    def test_their_sections_are_counted_after_those_before_them(self, paper):
        sections = [equation.section for equation in Document.read(paper / "main.tex").equations()]
        assert sections == [0, 1, 1, 1, 1, 1]

    def test_a_file_that_is_not_found_is_named(self, paper):
        assert Document.read(paper / "main.tex").missing == [("missing", 6)]

    def test_a_file_that_includes_itself_is_read_once(self, tmp_path):
        (tmp_path / "loop.tex").write_text("$$ x := 1 $$\n\\input{loop}\n")
        assert len(Document.read(tmp_path / "loop.tex").equations()) == 1


class TestDefinitionsThatCannotBeUsed:
    """A macro whose definition or call is wrong says so, rather than that it is unknown."""

    @pytest.mark.parametrize(
        ("definition", "equation", "message"),
        [
            (r"\newcommand{\vv}[1]{\vec{#1}", r"\vv{x} = 1",
             r"the definition of \vv on line 1 cannot be read: unclosed '{'"),
            (r"\def\half#1/#2{\frac{#1}{#2}}", r"\half 1/2 = 0.5",
             r"\half is defined on line 1 with delimited parameters, #1/#2, which are not "
             "supported"),
            (r"\newcommand{\pd}[2]{\frac{\partial #1}{\partial #2}}", r"\pd{f} = 0",
             r"\pd takes 2 arguments, but 1 is given here"),
            (r"\newcommand{\pd}[2]{\frac{\partial #1}{\partial #2}}", r"x + \pd{f}",
             r"\pd takes 2 arguments, but 1 is given here"),
        ],
    )  # fmt: skip
    def test_the_message_says_what_is_wrong_with_the_macro(self, definition, equation, message):
        [found] = Document(
            f"{definition}\n\\begin{{equation}} {equation} \\end{{equation}}\n"
        ).equations()
        with pytest.raises(ParseError) as error:
            parse(found.source)
        assert error.value.message == message

    def test_a_missing_argument_marks_the_call(self):
        text = "\\newcommand{\\pd}[2]{\\frac{\\partial #1}{\\partial #2}}\n$$ \\pd{f} = 0 $$\n"
        [found] = Document(text).equations()
        with pytest.raises(ParseError) as error:
            parse(found.source)
        assert error.value.span is not None
        assert error.value.span.written == r"\pd{f}"

    def test_a_macro_that_expands_to_an_unknown_command_says_so(self):
        text = "\\newcommand{\\Gam}{\\Gama}\n\\[ \\Gam^t_{tr} = 0 \\]\n"
        [found] = Document(text).equations()
        with pytest.raises(ParseError, match=r"unknown LaTeX command \\Gama") as error:
            parse(found.source)
        assert error.value.notes == [r"\Gam, defined on line 1, expands to \Gama"]

    def test_a_document_says_what_it_holds(self):
        document = Document("\\newcommand{\\R}{R}\n$$ \\R = 0 $$\n\\[ x \\]\n", "notes.tex")
        assert repr(document) == "Document('notes.tex': 4 lines, 2 display equations, 1 macro)"


class TestMarkdown:
    @pytest.mark.parametrize(
        ("text", "blocks"),
        [
            ("A stray ` here.\n\n$$ x $$\n\nThen `code` and $$ y $$", [" x ", " y "]),
            ("Code `$$ a $$` and ``a ` b $$ c $$`` but $$ x $$", [" x "]),
            ("<!-- $$ hidden $$ -->\n$$ x $$", [" x "]),
            ("Prose.\n\n    $$ code $$\n\n$$ x $$", [" x "]),
            ("- an item\n\n    $$ x $$\n", [" x "]),  # in the item, not code
            ("```\n$$ code $$\n```\n$$ x $$", [" x "]),
        ],
    )
    def test_blocks_outside_code_and_comments(self, text, blocks):
        assert [block.text for block in math_blocks(text)] == blocks

    def test_blocks_know_their_lines(self):
        blocks = math_blocks("a\n\n$$ x $$\n\nb\n$$\ny\n$$")
        assert [block.origin for block in blocks] == [(3, 3), (6, 3)]

    def test_every_form_of_display_is_run(self):
        # only $$ blocks ran: the others were skipped without a word
        text = (
            "```relativisticpy\nx := 2\nx + 1\n```\n\n"
            "\\[ y := 3 \\]\n\n"
            "\\begin{equation}\nz := 4\n\\end{equation}\n\n"
            "```math\nx + y + z\n```\n\n"
            "```python\nprint('code, not run')\n```\n\n"
            "$$ x + 10 $$\n"
        )
        blocks = math_blocks(text, "notes.md")
        assert [(block.text.strip(), block.latex) for block in blocks] == [
            ("x := 2\nx + 1", False),
            ("y := 3", True),
            ("\\begin{equation}\nz := 4\n\\end{equation}", True),
            ("x + y + z", True),
            ("x + 10", True),
        ]
        assert [block.origin[0] for block in blocks] == [2, 6, 8, 13, 20]
        results = [result for block in blocks for result in Workbook().exe(block)]
        assert results[0] == 3
        with Workbook() as wb:
            assert [r for block in blocks for r in wb.exe(block)] == [3, 9, 12]


class TestReading:
    def test_a_display_is_latex_even_without_a_command(self):
        # in a display, ab is a times b, so the identity holds
        assert verdicts(r"\begin{equation} (a+b)^2 = a^2 + 2ab + b^2 \end{equation}") == ["holds"]

    def test_a_markdown_display_is_latex(self):
        [block] = math_blocks("Prose.\n\n$$\n(a+b)^2 - a^2 - 2ab - b^2\n$$\n")
        assert block.latex
        assert Workbook().exe(block) == [0]

    def test_unicode_symbols_are_one_symbol(self):
        # ℏ is \hbar, not h b a r, though a LaTeX statement splits runs of letters
        assert str(Workbook().expr("ℏ ω + ℓ")) == "ell + hbar*omega"

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

    def test_a_statement_is_quoted_from_its_first_command(self):
        # it was quoted from inside \mathrm{, as "solve}(x^2 = 4, x)" and "d} s^2 = ..."
        document = Document(
            "\\newcommand{\\dd}{\\mathrm{d}}\n"
            "$$ \\mathrm{solve}(x^2 = 4, x) $$\n$$ \\dd s^2 = -\\dd t^2 + \\dd x^2 $$\n"
        )
        findings = Workbook().check(document).findings
        assert [finding.span.text for finding in findings] == [
            r"\mathrm{solve}(x^2 = 4, x)",
            r"\mathrm{d} s^2 = -\mathrm{d} t^2 + \mathrm{d} x^2",
        ]
        assert findings[1].span.written == r"\dd s^2 = -\dd t^2 + \dd x^2"  # as the file has it
