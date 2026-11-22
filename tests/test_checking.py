r"""Checking documents: every statement gets a verdict, in the document's order."""

import io
import json
import re
import runpy
import sys
import tarfile
import time
from pathlib import Path

import pytest
import sympy

from relativisticpy import Workbook, checking, display
from relativisticpy.checking import STRICT, Verdict, _time_limit, line_range
from relativisticpy.cli import main
from relativisticpy.documents import Document
from relativisticpy.errors import FileError, NotComputedError, UsageError
from relativisticpy.runtime import executor

BENCHMARKS = Path(__file__).parent.parent / "benchmarks"
DOCS = Path(__file__).parent.parent / "docs"
PAPER = r"""
\documentclass{article}
\newcommand{\be}{\begin{equation}}
\newcommand{\ee}{\end{equation}}
\begin{document}
\section{Polar coordinates}
The plane in polar coordinates is
\be ds^2 = dr^2 + r^2 d\theta^2 \ . \ee
Its connection is
\begin{eqnarray}
  \Gamma^r_{\theta\theta} &=& -r \ , \cr
  \Gamma^\theta_{r\theta} &=& {1 \over r} \ , \cr
  \Gamma^\theta_{\theta\theta} &=& 1 \ ,
\end{eqnarray}
and the plane is flat, $R = 0$:
\be R^r{}_{\theta r\theta} = 0 \ . \ee
Approximately, \be \pi \approx 3.14 \ . \ee
\section{Elsewhere}
Here $r$ is something else: \be r = 2, \qquad r^2, \qquad \Gamma^r_{rr} = 0, \qquad \frac{a}{ \ee
\end{document}
"""


def verdicts(report):
    return [(finding.line, finding.verdict) for finding in report.findings]


def judged(equation):
    """The verdict on a document of one equation, and why."""
    report = Workbook().check(Document(rf"\begin{{equation}} {equation} \end{{equation}}"))
    [finding] = report.findings
    return finding.verdict, finding.detail


def test_every_statement_gets_a_verdict():
    report = Workbook().check(Document(PAPER, "paper.tex"))
    assert verdicts(report) == [
        (8, Verdict.DEFINES),
        (11, Verdict.HOLDS),
        (12, Verdict.HOLDS),
        (13, Verdict.DIFFERS),
        (16, Verdict.HOLDS),
        (17, Verdict.NOT_COMPUTED),
        (19, Verdict.NOT_PARSED),
    ]


def test_the_findings_say_why():
    findings = Workbook().check(Document(PAPER, "paper.tex")).findings
    minus_one = display.unicode(sympy.Integer(-1))  # in the notation results are shown in
    assert findings[3].detail == f"the left side minus the right is {minus_one}"
    assert findings[3].difference == -1
    assert findings[4].detail == "both sides are zero here"
    assert "'≈' is read but not computed" in findings[5].detail
    assert findings[6].span.line == 19


def test_each_section_starts_afresh():
    # Without the metric of the first section, the second has no connection.
    report = Workbook().check(Document(PAPER.replace(r", \qquad \frac{a}{", ""), "paper.tex"))
    last = report.findings[-3:]
    assert [finding.verdict for finding in last] == [
        Verdict.DEFINES,
        Verdict.COMPUTES,
        Verdict.NOT_UNDERSTOOD,
    ]
    assert "needs coordinates" in last[-1].detail


def test_part_of_a_document_is_read_after_what_comes_before_it():
    # DOC-06, CLI-18: the line element above the lines runs first, for its definitions
    report = Workbook().check(Document(PAPER), lines=(10, 14))
    assert report.equations == 1
    assert verdicts(report) == [
        (11, Verdict.HOLDS),
        (12, Verdict.HOLDS),
        (13, Verdict.DIFFERS),
    ]
    assert report.context == 1  # the line element, run but not reported
    assert "Also run, for their definitions, and not reported: 1 statement" in (report.describe())


def test_index_notation_is_checked_as_notation():
    report = Workbook().check(
        Document(r"""
        \begin{equation} g^{\mu\nu} g_{\nu\sigma} = \delta^\mu_\sigma \end{equation}
        \begin{equation} G_{\mu\nu} = 8\pi G T_{\mu\nu} \end{equation}
        """)
    )
    # DOC-11: a law in index notation relates tensors nothing defines, as one written
    # with components does
    notation = "index notation, not an identity: G_{μν} = 8π G T_{μν}"
    assert [(finding.verdict, finding.detail) for finding in report.findings] == [
        (Verdict.HOLDS, "an identity of index notation"),
        (Verdict.RELATES, f"nothing defines T: {notation}"),
    ]


class TestScalarEquations:
    """An equation holds when it is proven; numbers tell one that differs from one that
    is not decided."""

    @pytest.mark.parametrize(
        "identity",
        [
            r"\sum_{n=1}^{3} n = 6",
            r"\cosh^2 x - \sinh^2 x = 1",
            r"\sin^2(x/2) = (1 - \cos x)/2",
            r"\cos^2 1 + \sin^2 1 = 1",
            "0.1 + 0.2 = 0.3",
            r"1.1 \times 3 = 3.3",
            r"1.1 x \times 3 = 3.3 x",
            r"\partial_r (\alpha(r)^2) = 2 \alpha(r) \partial_r \alpha(r)",
        ],
    )
    def test_identities_hold(self, identity):
        assert judged(identity) == (Verdict.HOLDS, "")

    @pytest.mark.parametrize(
        ("equation", "difference"),
        [
            ("0.1 + 0.2 = 0.30001", -9.99999999995449e-6),
            ("x + 1 = x", 1),  # false whatever x is
        ],
    )
    def test_equations_that_numbers_refute_differ(self, equation, difference):
        verdict, detail = judged(equation)
        assert verdict is Verdict.DIFFERS
        shown = display.unicode(sympy.sympify(difference))
        assert detail == f"the left side minus the right is {shown}"

    @pytest.mark.parametrize(
        ("equation", "unknown"),
        [
            ("x^2 + y^2 = 1", ["x", "y"]),
            # an equation for alpha, which nothing defines; not an identity, nor a mistake
            (r"\partial_r (\alpha(r)^2) = \alpha(r) \partial_r \alpha(r)", ["r", "alpha(r)"]),
            (r"\Omega^2 = \frac{GM}{r^3}", ["G", "M", "Omega", "r"]),
        ],
    )
    def test_equations_of_quantities_nothing_defines_relate(self, equation, unknown):
        # DOC-11, AGT-13: a law or an equation to solve is not a mistake
        verdict, detail = judged(equation)
        assert verdict is Verdict.RELATES
        names = [display.unicode(sympy.sympify(name)) for name in unknown]
        listed = ", ".join(names[:-1]) + " and " + names[-1]
        assert detail.startswith(f"nothing defines {listed}: the left side minus the right is ")

    def test_a_map_into_pairs_relates(self):
        # Carroll's chart of the sphere (2.4): an unknown function's value compared with a
        # pair, and the pair with its value, relate what nothing defines; neither is a mistake
        report = Workbook().check(
            Document(r"""\begin{equation}
            \phi_1(x^1,x^2,x^3) \equiv (y^1,y^2) = \left( {{2x^1}\over{1-x^3}}\ ,\
            {{2x^2}\over{1-x^3}}\right)
            \end{equation}""")
        )
        assert [finding.verdict for finding in report] == ["relates", "relates"]
        first = report.findings[0].detail
        assert first == (
            "nothing defines x, y and φ₁(x, x², x³): φ₁(x, x², x³) and (y, y²) are related, "
            "not shown equal"
        )
        assert judged("(1, 2) = (1, 3)") == (Verdict.DIFFERS, "the two sides are not equal")

    def test_quantities_the_metric_is_written_with_are_known(self):
        # Carroll's Kretschmann scalar (7.30) and the sphere's curvature are mistakes
        report = Workbook().check(
            Document(r"""
            \begin{equation} ds^2 = -(1 - \frac{2GM}{r}) dt^2 + \frac{dr^2}{1 - \frac{2GM}{r}}
                + r^2 d\Omega^2 \end{equation}
            \begin{equation} R^{\mu\nu\rho\sigma}R_{\mu\nu\rho\sigma} = \frac{12 G^2 M^2}{r^6}
            \end{equation}
            \section{The sphere}
            \begin{equation} ds^2 = a^2 (d\theta^2 + \sin^2\theta d\phi^2) \end{equation}
            \begin{equation} R = \frac{1}{a^2} \end{equation}
            """)
        )
        assert [f.verdict for f in report] == ["defines", "differs", "defines", "differs"]
        assert (
            report.findings[1].difference
            == 36 * sympy.Symbol("G") ** 2 * sympy.Symbol("M") ** 2 / sympy.Symbol("r") ** 6
        )

    def test_names_declared_by_a_relation_are_known(self):
        # an index range declares nu: nu + 1 = 1 is a condition on it, which differs
        report = Workbook().check(
            Document(r"""
            \begin{equation} \mu, \nu = 0, 1, 2, 3 \end{equation}
            \begin{equation} \nu + 1 = 1 \end{equation}
            """)
        )
        assert [finding.verdict for finding in report] == ["not computed", "differs"]

    @pytest.mark.parametrize("equation", [r"\sqrt{x^2} = x", r"\log(a b) = \log a + \log b"])
    def test_equations_true_only_where_they_were_tried_are_not_decided(self, equation):
        # True for positive numbers, but not proven: not a mistake in the text either
        assert judged(equation) == (
            Verdict.NOT_COMPUTED,
            "the sides agree numerically, but this was not proven",
        )

    def test_equations_that_are_not_numbers_are_not_decided(self):
        assert judged(r"\lim_{x \to \infty} f(x) = 1") == (
            Verdict.NOT_COMPUTED,
            "not decided: the sides could not be computed as numbers",
        )

    def test_values_keep_their_types(self):
        x = sympy.Symbol("x")
        wb = Workbook()
        assert wb.expr("0.1 + 0.2 = 0.3") is sympy.true
        assert wb.expr("0.1 + 0.2 = 0.30001") is sympy.false
        assert wb.expr(r"\sqrt{x^2} = x") == sympy.Eq(sympy.sqrt(x**2), x)

    def test_a_false_relation_says_why(self):
        assert judged(r"2 \neq 2") == (Verdict.DIFFERS, "the two sides are equal")
        assert judged("3 < 2") == (Verdict.DIFFERS, "the inequality does not hold")
        assert judged("x + 1 = x") == (Verdict.DIFFERS, "the left side minus the right is 1")


def test_a_check_neither_uses_nor_changes_the_workbooks_definitions():
    wb = Workbook()
    wb.exe("x := 3")
    assert verdicts(wb.check(Document(r"\begin{equation} x = 3 \end{equation}"))) == [
        (1, Verdict.DEFINES)
    ]
    wb.check(Document(PAPER))  # whose sections each start afresh
    assert wb.expr("x") == 3


def test_no_statement_stops_a_check():
    report = Workbook().check(
        Document(r"""
        \begin{equation} [1, 2] < [3, 4] \end{equation}
        \begin{equation} 5.² = 25 \end{equation}
        \begin{equation} 2 + 2 = 4 \end{equation}
        """)
    )
    assert [finding.verdict for finding in report.findings] == [
        Verdict.NOT_UNDERSTOOD,
        Verdict.NOT_PARSED,
        Verdict.HOLDS,
    ]


def test_a_failure_of_relativisticpy_is_a_finding(monkeypatch):
    def broken(self, source, *, name="<input>"):
        raise RuntimeError("something broke")
        yield  # a generator, as Workbook.each is

    monkeypatch.setattr(Workbook, "each", broken)
    report = Workbook().check(Document(PAPER))
    assert verdicts(report) == [
        (8, Verdict.NOT_UNDERSTOOD),
        (11, Verdict.NOT_UNDERSTOOD),  # where the equation starts, within its environment
        (16, Verdict.NOT_UNDERSTOOD),
        (17, Verdict.NOT_UNDERSTOOD),
        (19, Verdict.NOT_PARSED),  # whose equation does not parse
    ]
    finding = report.findings[0]
    assert finding.detail == "internal error: RelativisticPy failed while checking this statement"
    assert finding.notes == ("Python says: RuntimeError: something broke",)
    assert "please report it" in finding.hint
    assert finding.reason == "internal errors of RelativisticPy"


@pytest.mark.parametrize(
    "text",
    [
        r"\let\old\vec \renewcommand{\vec}[1]{\old{#1}} \begin{equation} \vec{x} \end{equation}",
        r"\newcommand{\foo}[1][x]{#1} \begin{equation} \foo[ = 1 \end{equation}",
        "\\begin{equation} x = 1 \\end{equation} \\",
    ],
)
def test_a_document_that_cannot_be_read_gives_a_report(text):
    assert Workbook().check(Document(text)).findings


def test_a_document_whose_equations_cannot_be_found_is_a_finding(monkeypatch):
    def broken(self):
        raise ValueError(r"the macro \vec expands without end")

    monkeypatch.setattr(Document, "equations", broken)
    findings = Workbook().check(Document(PAPER, "paper.tex")).findings
    assert [(finding.verdict, finding.detail) for finding in findings] == [
        (
            Verdict.NOT_PARSED,
            r"cannot read the document: ValueError: the macro \vec expands without end",
        )
    ]


def test_a_file_is_read_with_its_packages_and_the_files_it_inputs(tmp_path):
    (tmp_path / "macros.sty").write_text(r"\newcommand{\half}{\frac{1}{2}}")
    (tmp_path / "numbers.tex").write_text(r"\newcommand{\two}{2}")
    paper = tmp_path / "paper.tex"
    paper.write_text(
        "\\usepackage{macros}\n\\input{numbers}\n"
        r"\begin{equation} \half \cdot \two = 1 \end{equation}"
    )
    assert verdicts(Workbook().check(paper)) == [(3, Verdict.HOLDS)]


def test_the_summary_counts_verdicts_and_says_what_they_mean():
    # CLI-17: what each verdict means, and how long the check took
    summary = Workbook().check(Document(PAPER, "paper.tex")).summary()
    head = summary.splitlines()[0]
    assert re.fullmatch(r"paper\.tex: 5 equations, 7 statements, checked in \d+\.\d s", head)
    assert "  ✓ holds               3   proven equal, 1 with both sides zero" in summary
    assert "  ✗ differs             1   of known quantities, whose sides are not equal" in summary


def test_one_equation_is_one_equation():
    # CLI-41: plurals agree
    report = Workbook().check(r"\begin{equation} 1 = 1 \end{equation}")
    assert report.summary().startswith("<document>: 1 equation, 1 statement, checked in")


def test_the_summary_says_why_statements_are_not_understood():
    # CLI-17: "not understood" broken down by reason, the commonest first
    report = Workbook().check(
        Document(r"""
        \begin{equation} T_{\mu\nu} + S_{\mu\nu} \end{equation}
        \begin{equation} T_{\mu\nu} - S_{\mu\nu} \end{equation}
        \begin{equation} \Gamma^t_{tt} \end{equation}
        """)
    )
    lines = report.summary().splitlines()
    assert lines[1].endswith("what RelativisticPy cannot do yet, most often:")
    assert lines[2].split() == ["2", "T", "is", "not", "a", "tensor"]
    assert lines[3].split()[0] == "1"
    assert report.to_dict()["reasons"] == {
        "T is not a tensor": 2,
        "the Christoffel connection needs coordinates": 1,
    }


def test_a_notebook_is_checked_by_its_blocks(tmp_path):
    notes = tmp_path / "notes.md"
    notes.write_text("Prose.\n\n$$\nx := 2\n$$\n\nThen $$ x^2 = 4 $$\n")
    report = Workbook().check(notes)
    assert verdicts(report) == [(4, Verdict.DEFINES), (7, Verdict.HOLDS)]


def test_a_computation_over_its_time_limit_is_not_computed(monkeypatch):
    # The limit reaches through the executor, which reports other failures as errors.
    def slowly(component):
        time.sleep(5)
        return component

    monkeypatch.setattr(executor, "simplify_component", slowly)
    curvature = Document(r"""
    \begin{equation} ds^2 = -dt^2 + dx^2 \end{equation}
    \begin{equation} R_{txtx} = 0 \end{equation}
    """)
    report = Workbook().check(curvature, time_limit=0.1)
    finding = report.findings[1]
    # CLI-39: the statement as written, and how to give it longer
    assert (finding.verdict, finding.detail) == (
        Verdict.NOT_COMPUTED,
        "over the time limit of 0.1 s",
    )
    assert finding.text == "R_{txtx} = 0"
    assert finding.reason == checking.TIME_LIMIT
    assert "--time-limit" in finding.hint
    # CLI-20, AGT-02: a statement over the time limit fails --strict, and is listed
    assert report.failed()
    assert not report.ok
    assert report.listing() == [finding]


def test_what_a_statement_over_the_time_limit_defines_is_missed_later(monkeypatch):
    # CLI-39: a statement that needs what one over the time limit defines says so
    def slowly(component):
        time.sleep(5)
        return component

    monkeypatch.setattr(executor, "simplify_component", slowly)
    document = Document(r"""
    \begin{equation} ds^2 = -dt^2 + dx^2 \end{equation}
    \begin{equation} R_{txtx} = 0 \end{equation}
    \begin{equation} T_{tt} \end{equation}
    """)
    last = Workbook().check(document, time_limit=0.1).findings[-1]
    assert last.verdict is Verdict.NOT_UNDERSTOOD
    assert "line 3 was over the time limit, so what it defines may be unknown here" in last.notes


def test_judging_is_within_the_time_limit(monkeypatch):
    def slowly(value, session, state):
        time.sleep(5)
        return Verdict.HOLDS, "", None

    monkeypatch.setattr(checking, "_judge", slowly)
    start = time.monotonic()
    square = Document(r"\begin{equation} x^2 = x x \end{equation}")
    finding = Workbook().check(square, time_limit=0.1).findings[0]
    assert (finding.verdict, finding.detail) == (
        Verdict.NOT_COMPUTED,
        "over the time limit of 0.1 s",
    )
    assert time.monotonic() - start < 2


@pytest.mark.parametrize("seconds", [0, -1, float("nan")])
def test_a_time_limit_is_a_positive_number_of_seconds(seconds):
    with pytest.raises(ValueError, match="a time limit is a positive number of seconds") as error:
        Workbook().check(Document(PAPER), time_limit=seconds)
    assert isinstance(error.value, UsageError)


def test_an_infinite_time_limit_is_none():
    report = Workbook().check(Document(PAPER), time_limit=float("inf"))
    assert verdicts(report)[:2] == [(8, Verdict.DEFINES), (11, Verdict.HOLDS)]


def test_the_corpus_is_unpacked_on_pythons_without_extraction_filters(tmp_path, monkeypatch):
    # Python 3.10.0 to 3.10.11 have no data filter, and refuse the argument for one.
    corpus = runpy.run_path(str(BENCHMARKS / "corpus.py"))
    paper = rb"\begin{equation} x \end{equation}"
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        member = tarfile.TarInfo("notes/paper.tex")
        member.size = len(paper)
        tar.addfile(member, io.BytesIO(paper))
        tar.addfile(tarfile.TarInfo("../outside.tex"))  # what the filter is there to refuse

    def extractall(self, path=".", members=None, **options):
        raise TypeError("extractall() got an unexpected keyword argument 'filter'")

    monkeypatch.delattr(tarfile, "data_filter")
    monkeypatch.setattr(tarfile.TarFile, "extractall", extractall)
    corpus["unpack"](archive.getvalue(), tmp_path / "src")
    assert (tmp_path / "src" / "notes" / "paper.tex").read_bytes() == paper
    assert not (tmp_path / "outside.tex").exists()


def test_the_table_of_the_corpus_agrees_with_its_prose():
    # docs/checking.md gives the numbers of benchmarks/corpus.py for Carroll's notes twice.
    text = (DOCS / "checking.md").read_text(encoding="utf-8")
    row = re.search(r"\| Carroll, .* \| (\d+) \| ([\d.]+)% \| (\d+) \| (\d+) \| (\d+) \|", text)
    assert row is not None
    equations, parsed, hold, differ, relate = row.groups()
    assert f"Of their {equations} display equations" in text
    assert f"({parsed}%)\nparse" in text
    assert f" {hold} statements hold" in text
    assert f"sides are not equal, {differ} involve only what the\n" in text
    assert f"The other {relate} relate" in text


def test_a_statement_can_be_given_a_time_limit():
    def spin() -> None:
        start = time.monotonic()
        while time.monotonic() - start < 5:
            pass

    with pytest.raises(TimeoutError), _time_limit(0.05):
        spin()
    with _time_limit(None):  # no limit
        pass


class TestVerdicts:
    def test_a_verdict_is_its_name(self):
        # API-08, AGT-12: count("differs") counts, and a verdict is JSON as it is
        report = Workbook().check(Document(PAPER, "paper.tex"))
        assert Verdict.DIFFERS == "differs"
        assert report.count("differs") == report.count(Verdict.DIFFERS) == 1
        assert {"differs": 1}[Verdict.DIFFERS] == 1
        assert f"{Verdict.NOT_PARSED}" == str(Verdict.NOT_PARSED) == "not parsed"

    def test_a_verdict_that_does_not_exist_is_refused(self):
        with pytest.raises(UsageError, match="'differ' is not a verdict"):
            Workbook().check(Document(PAPER)).count("differ")

    def test_check_judges_errors_by_their_type_never_their_words(self, monkeypatch):
        # ERR-33: a reworded message keeps its verdict
        from relativisticpy.compiler import Compiler

        def refuse(self, statement, scope):
            raise NotComputedError("worded in any way at all", span=statement.span)

        monkeypatch.setattr(Compiler, "statement", refuse)
        report = Workbook().check(r"\begin{equation} x = 1 \end{equation}")
        assert [finding.verdict for finding in report] == ["not computed"]


class TestIllFormed:
    """ERR-06: the paper's own mistakes of notation, apart from what RelativisticPy
    cannot do yet; only where the notation is surely at fault."""

    @pytest.mark.parametrize(
        ("equation", "detail"),
        [
            (r"R_{\mu\mu} = 0", r"the index \mu appears twice in one term, both times down"),
            (
                r"R_{\mu\nu} + g^{\mu\nu} = 0",
                r"the terms of a sum have different free indices: _{\mu\nu} and ^{\mu\nu}",
            ),
            (
                "V^a = W^b",
                "the two sides have different free indices: ^{a} on the left, ^{b} on the right",
            ),
            (r"A_\mu B^\mu C_\mu = 0", r"the index \mu appears 3 times in one term"),
            (
                r"\eta_{\rho\nu}\eta^{\nu\mu} = \delta^\rho_\mu",  # Carroll's (1.56), as written
                r"the two sides have different free indices: ^{\mu}_{\rho} on the left, "
                r"^{\rho}_{\mu} on the right",
            ),
        ],
    )
    def test_notation_that_breaks_the_rules(self, equation, detail):
        report = Workbook().check(rf"\begin{{equation}} {equation} \end{{equation}}")
        [finding] = report.findings
        assert (finding.verdict, finding.detail) == (Verdict.ILL_FORMED, detail)
        assert finding.hint
        assert report.failed()
        assert report.listing() == [finding]

    @pytest.mark.parametrize(
        "equation",
        [
            r"F^{ij} = \epsilon^{ijk}B_k",  # B_k, a name or B with k: not judged
            "a_i b_i = 1",  # the Euclidean convention
            r"\chi^\mu = \zeta^\mu + \Omega_H\eta^\mu",  # H is a label, not an index
            r"x^{\mu'} = x^\mu + a^\mu",  # an index in another frame
            r"\Gamma^r_{rr} = 0",  # r names a component, wherever coordinates are
            r"g_{\theta\theta} = r^2",
            r"\nabla_\mu V^\mu{}_\mu = 0",  # a derivative, which may be read otherwise
            r"A^{\rm odd} = \Pi_{\rm c}\, r\, F",  # words set in roman, not indices
            r"T^{\mu\nu}_{\rm dust} = \rho U^\mu U^\nu",
        ],
    )
    def test_notation_that_may_be_read_otherwise_is_not_blamed(self, equation):
        report = Workbook().check(rf"\begin{{equation}} {equation} \end{{equation}}")
        assert Verdict.ILL_FORMED not in [finding.verdict for finding in report]


class TestFindings:
    def test_a_finding_points_where_editors_do(self):
        # CLI-24: paper.tex:12:5, which terminals and editors link to
        report = Workbook().check(Document(PAPER, "paper.tex"))
        differs = report.findings[3]
        assert (differs.file, differs.line, differs.column) == ("paper.tex", 13, 3)
        assert differs.location == "paper.tex:13:3"
        assert differs.describe().startswith(
            r"paper.tex:13:3: ✗ differs  \Gamma^\theta_{\theta\theta} &=& 1"
        )

    def test_the_statement_is_quoted_as_written(self):
        # DOC-36, AGT-30: macros as the file has them, and no command cut in half
        report = Workbook().check(
            Document(
                "\\newcommand{\\dd}{\\mathrm{d}}\n"
                r"\begin{equation} \dd s^2 = a^2 (\dd\theta^2 + \sin^2\theta \dd\phi^2) "
                r"\end{equation}"
                "\n$$ \\mathrm{solve}(x^2 = 4, x) $$",
                "sphere.tex",
            )
        )
        assert [finding.text for finding in report] == [
            r"\dd s^2 = a^2 (\dd\theta^2 + \sin^2\theta \dd\phi^2)",
            r"\mathrm{solve}(x^2 = 4, x)",
        ]

    def test_a_statement_that_is_not_parsed_keeps_its_hint_and_caret(self):
        # CLI-21, ERR-17: the whole statement, the error's place in it, and its hint
        report = Workbook().check(Document(r"$$ y = \frca{1}{2} x + \sqrt{x} $$", "hints.tex"))
        [finding] = report.findings
        assert finding.verdict is Verdict.NOT_PARSED
        assert finding.text == r"y = \frca{1}{2} x + \sqrt{x}"
        assert (finding.line, finding.column) == (1, 8)
        assert finding.hint == r"did you mean \frac?"
        described = finding.describe()
        assert "hint: did you mean \\frac?" in described
        assert "^^^^^" in described

    def test_nothing_is_cut_short_without_a_width(self):
        # CLI-08, DOC-37: the difference is the finding; a terminal wraps it
        long = " + ".join(f"\\sin(x_{n})" for n in range(30))
        report = Workbook().check(Document(rf"$$ {long} = {long} + 1 + y $$", "long.tex"))
        [finding] = report.findings
        assert "…" not in finding.describe()
        assert long in finding.describe()
        assert all(len(line) <= 60 for line in finding.describe(width=60).splitlines())

    def test_a_tensor_equation_says_where_it_differs(self):
        # CLI-40: how many components differ, and the first few, by coordinate
        report = Workbook().check(
            Document(r"""
            \begin{equation} ds^2 = -(1 - \frac{2M}{r} + \frac{Q^2}{r^2}) dt^2
                + \frac{dr^2}{1 - \frac{2M}{r} + \frac{Q^2}{r^2}} + r^2 d\Omega^2 \end{equation}
            \begin{equation} R_{\mu\nu} = 0 \end{equation}
            """)
        )
        finding = report.findings[1]
        assert finding.verdict is Verdict.DIFFERS
        assert finding.detail.startswith("differs in 4 of 16 components: at tt, ")
        assert finding.detail.endswith("; and 1 more")

    def test_a_division_by_zero_is_not_computed(self):
        # ERR-29: not the jargon of complex infinity, and not a difference
        verdict, detail = judged(r"\frac{1}{0} = 1")
        assert (verdict, detail) == (
            Verdict.NOT_COMPUTED,
            "the left side is infinite here: a division by zero",
        )

    def test_a_huge_integer_never_stops_a_check(self):
        # ERR-01: Python will not print an integer of 35 660 digits
        report = Workbook().check(Document("$$ x := 10000! $$\n$$ x $$"))
        finding = report.findings[1]
        assert finding.verdict is Verdict.COMPUTES
        assert "an integer of 35,660 digits" in finding.detail

    def test_a_finding_is_data(self):
        # AGT-12, ERR-33: every finding and the whole report as JSON
        report = Workbook().check(Document(PAPER, "paper.tex"))
        data = report.to_dict()
        assert json.loads(json.dumps(data)) == data
        assert data["counts"] == {
            "holds": 3,
            "differs": 1,
            "defines": 1,
            "not computed": 1,
            "not parsed": 1,
        }
        differs = data["findings"][3]
        assert differs["verdict"] == "differs"
        assert (differs["file"], differs["line"], differs["column"]) == ("paper.tex", 13, 3)
        assert differs["statement"] == r"\Gamma^\theta_{\theta\theta} &=& 1"
        assert differs["difference"]["text"] == display.text(sympy.Integer(-1))
        assert data["ok"] is False

    def test_findings_and_reports_show_what_they_are(self):
        # API-07: not dataclass dumps with byte offsets
        report = Workbook().check(Document(PAPER, "paper.tex"))
        assert repr(report.findings[3]).startswith(
            r"Finding(paper.tex:13:3 ✗ differs: \Gamma^\theta_{\theta\theta} &=& 1; "
        )
        assert repr(report) == report.describe()
        assert repr(report).startswith("paper.tex: 5 equations, 7 statements")


class TestReports:
    def test_a_report_is_a_sequence_of_findings(self):
        report = Workbook().check(Document(PAPER, "paper.tex"))
        assert len(report) == 7
        assert list(report) == report.findings

    def test_what_fails_is_listed_and_the_rest_counted(self):
        # ERR-06, CLI-07, DOC-39: what fails is listed by default, and nothing is hidden
        report = Workbook().check(Document(PAPER, "paper.tex"))
        text = report.describe()
        assert "paper.tex:13:3: ✗ differs" in text
        assert "paper.tex:19:93: ! not parsed" in text
        assert "paper.tex:11:3: ✓ holds" not in text
        assert text.endswith(
            "Not listed: 3 holds, 1 defines, 1 not computed. --all lists every statement."
        )
        assert "paper.tex:11:3: ✓ holds" in report.describe(all=True)

    @pytest.mark.parametrize(
        ("policy", "failed"),
        [
            (STRICT, True),
            ({"differs"}, True),
            (["not-parsed"], True),
            ({"ill-formed", "time limit"}, False),
            ("holds", True),
        ],
    )
    def test_a_policy_says_what_fails(self, policy, failed):
        # AGT-02, CLI-20: --fail-on in Python
        report = Workbook().check(Document(PAPER, "paper.tex"))
        assert report.failed(policy) is failed

    def test_ok_is_what_strict_passes(self):
        assert Workbook().check(r"\begin{equation} 1 + 1 = 2 \end{equation}").ok
        assert not Workbook().check(r"\begin{equation} 1 + 1 = 3 \end{equation}").ok

    def test_a_document_without_equations_fails_strict_and_says_why(self, tmp_path):
        # CLI-20, AGT-01: checking nothing is not a pass
        program = tmp_path / "plain.relpy"
        program.write_text("x := 3\nx^2\n")
        report = Workbook().check(program)
        assert report.equations == 0
        assert report.failed()
        assert not report.failed({"differs"})
        assert "no display equations were found in" in report.warnings[0]
        assert "warning: no display equations" in report.describe()


class TestLines:
    """--lines: DOC-06, CLI-18, CLI-41."""

    SPHERE = r"""
\section{The sphere}
The two-sphere of radius $a$ has the line element
\begin{equation}
  ds^2 = a^2 (d\theta^2 + \sin^2\theta\, d\phi^2)
\end{equation}
Its curvature is
\begin{equation}
  R = \frac{1}{a^2}
\end{equation}
"""

    def test_a_passage_after_its_line_element_finds_the_mistake(self):
        report = Workbook().check(Document(self.SPHERE, "sphere.tex"), lines=(7, 11))
        assert verdicts(report) == [(9, Verdict.DIFFERS)]  # R = 2/a^2, not 1/a^2

    def test_an_environment_that_reaches_into_the_lines_counts(self):
        report = Workbook().check(Document(PAPER), lines=(12, 12))
        assert report.equations == 1
        assert verdicts(report) == [(12, Verdict.HOLDS)]

    def test_one_line_and_open_ranges(self):
        document = Document(PAPER)
        assert verdicts(Workbook().check(document, lines=13)) == [(13, Verdict.DIFFERS)]
        assert [line for line, _ in verdicts(Workbook().check(document, lines=(16, None)))] == [
            16,
            17,
            19,
        ]
        assert [line for line, _ in verdicts(Workbook().check(document, lines=(None, 9)))] == [8]

    @pytest.mark.parametrize(
        ("text", "lines"),
        [("120:340", (120, 340)), ("13", (13, 13)), ("10:", (10, None)), (":10", (None, 10))],
    )
    def test_the_lines_of_the_command_line(self, text, lines):
        assert line_range(text) == lines

    @pytest.mark.parametrize("text", ["13:9", "five", ":", "0:3"])
    def test_lines_that_are_not_a_range_are_refused(self, text):
        with pytest.raises(UsageError):
            line_range(text)

    def test_a_range_past_the_end_says_so(self):
        report = Workbook().check(Document(PAPER, "paper.tex"), lines=(1000, 2000))
        assert report.warnings == ["lines 1000–2000 of paper.tex have no equations in them"]


class TestFiles:
    def test_a_file_that_is_not_utf8_is_refused_as_a_run_refuses_it(self, tmp_path):
        # CLI-33: never decoded silently with replacement characters
        paper = tmp_path / "latin1.tex"
        paper.write_bytes(b"\\begin{equation}\n\\text{caf\xe9} = 1\n\\end{equation}\n")
        with pytest.raises(FileError, match="is not UTF-8 text: the byte 0xE9 at line 2"):
            Workbook().check(paper)

    def test_a_jupyter_notebook_is_refused_by_name(self, tmp_path):
        notebook = tmp_path / "notes.ipynb"
        notebook.write_text("{}")
        with pytest.raises(FileError, match="is a Jupyter notebook"):
            Workbook().check(notebook)


class TestCommandLine:
    def test_check_lists_what_fails_and_counts_the_rest(self, tmp_path, capsys):
        # CLI-07: what fails is listed by default, and the rest is counted, with --all
        paper = tmp_path / "paper.tex"
        paper.write_text(PAPER)
        assert main(["check", str(paper)]) == 0
        output = capsys.readouterr().out
        assert output.startswith(f"{paper}: 5 equations, 7 statements")
        assert f"{paper}:13:3: ✗ differs" in output
        assert f"{paper}:19:93: ! not parsed" in output
        assert f"{paper}:11:3:" not in output  # holds: counted, not listed
        assert "Not listed: 3 holds, 1 defines, 1 not computed. --all lists" in output

    def test_all_lists_everything(self, tmp_path, capsys):
        paper = tmp_path / "paper.tex"
        paper.write_text(PAPER)
        assert main(["check", str(paper), "--all", "--lines", "15:20"]) == 0
        output = capsys.readouterr().out
        assert f"{paper}:19:93: ! not parsed" in output

    def test_bad_arguments(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            main(["check", "paper.tex", "--lines", "five"])
        assert "--lines takes line numbers, as in --lines 120:340" in capsys.readouterr().err
        assert main(["check", str(tmp_path / "missing.tex")]) == 2  # the command line's fault

    @pytest.mark.parametrize("seconds", ["0", "-1", "nan", "soon"])
    def test_a_time_limit_is_a_positive_number(self, seconds, capsys):
        with pytest.raises(SystemExit):
            main(["check", "paper.tex", "--time-limit", seconds])
        assert "--time-limit: takes a positive number of seconds" in capsys.readouterr().err

    def test_inf_is_no_time_limit(self, tmp_path, capsys):
        paper = tmp_path / "paper.tex"
        paper.write_text(PAPER)
        assert main(["check", str(paper), "--time-limit", "inf"]) == 0
        assert f"{paper}:13:3: ✗ differs" in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("equation", "status"),
        [("2 + 2 = 4", 0), (r"\pi \approx 3.14", 0), ("2 + 2 = 5", 1), (r"\frac{a}{", 1)],
    )
    def test_strict_fails_when_an_equation_differs_or_is_not_parsed(
        self, equation, status, tmp_path
    ):
        paper = tmp_path / "paper.tex"
        paper.write_text(rf"\begin{{equation}} {equation} \end{{equation}}")
        assert main(["check", str(paper), "--strict"]) == status
        assert main(["check", str(paper)]) == 0

    def test_workers(self, tmp_path, capsys):
        paper = tmp_path / "paper.tex"
        paper.write_text(PAPER)
        assert main(["check", "-j", "2", str(paper)]) == 0
        assert f"{paper}:13:3: ✗ differs" in capsys.readouterr().out

    def test_output_that_cannot_show_the_marks(self, tmp_path, monkeypatch):
        # Output redirected on Windows encodes with the ANSI code page, without ✓ or ✗.
        paper = tmp_path / "paper.tex"
        paper.write_text(PAPER)
        output = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        monkeypatch.setattr(sys, "stdout", output)
        assert main(["check", str(paper)]) == 0
        output.flush()
        printed = output.buffer.getvalue().decode("cp1252")
        # CLI-26: marks of their own in ASCII, never the ? of "not understood"
        assert f"{paper}:13:3: x differs" in printed
        assert "  + holds " in printed


def test_a_file_a_paper_includes_that_is_missing_is_reported(tmp_path):
    paper = tmp_path / "paper.tex"
    paper.write_text("\\input{chapter}\n$$ x := 2 $$\n", encoding="utf-8")
    report = Workbook().check(str(paper))
    assert report.warnings == [
        f"{paper}:1: the file chapter it includes was not found, so its equations were not checked"
    ]
