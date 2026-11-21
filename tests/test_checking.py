r"""Checking documents: every statement gets a verdict, in the document's order."""

import time

import pytest

from relativisticpy import Workbook
from relativisticpy.checking import Verdict, _time_limit
from relativisticpy.cli import main
from relativisticpy.documents import Document
from relativisticpy.runtime import executor

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
    assert findings[3].detail == "the left side minus the right is -1"
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


def test_part_of_a_document():
    report = Workbook().check(Document(PAPER), lines=(10, 14))
    assert report.equations == 1
    assert [finding.verdict for finding in report.findings][:2] == [
        Verdict.NOT_UNDERSTOOD,  # without the line element above, no metric
        Verdict.NOT_UNDERSTOOD,
    ]


def test_index_notation_is_checked_as_notation():
    report = Workbook().check(
        Document(r"""
        \begin{equation} g^{\mu\nu} g_{\nu\sigma} = \delta^\mu_\sigma \end{equation}
        \begin{equation} G_{\mu\nu} = 8\pi G T_{\mu\nu} \end{equation}
        """)
    )
    assert [(finding.verdict, finding.detail) for finding in report.findings] == [
        (Verdict.HOLDS, "an identity of index notation"),
        (Verdict.COMPUTES, r"index notation, not an identity: G_{\mu\nu} = 8 \pi G T_{\mu\nu}"),
    ]


def test_the_summary_counts_verdicts():
    summary = Workbook().check(Document(PAPER, "paper.tex")).summary()
    assert summary.splitlines()[0] == "paper.tex: 5 equations, 7 statements"
    assert "✓ holds               3   (1 with both sides zero)" in summary


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

    wb = Workbook()
    wb.exe(r"ds^2 = -dt^2 + dx^2")
    monkeypatch.setattr(executor, "simplify_component", slowly)
    curvature = Document(r"\begin{equation} R_{txtx} = 0 \end{equation}")
    finding = wb.check(curvature, time_limit=0.1).findings[0]
    assert (finding.verdict, finding.detail) == (Verdict.NOT_COMPUTED, "took longer than 0.1 s")


def test_a_statement_can_be_given_a_time_limit():
    def spin() -> None:
        start = time.monotonic()
        while time.monotonic() - start < 5:
            pass

    with pytest.raises(TimeoutError), _time_limit(0.05):
        spin()
    with _time_limit(None):  # no limit
        pass


class TestCommandLine:
    def test_check_lists_what_holds_and_what_differs(self, tmp_path, capsys):
        paper = tmp_path / "paper.tex"
        paper.write_text(PAPER)
        assert main(["check", str(paper)]) == 0
        output = capsys.readouterr().out
        assert output.startswith(f"{paper}: 5 equations, 7 statements")
        assert "line 11     ✓ holds" in output
        assert "line 13     ✗ differs" in output
        assert "not parsed" in output.splitlines()[5]  # in the summary only
        assert "line 19" not in output

    def test_all_lists_everything(self, tmp_path, capsys):
        paper = tmp_path / "paper.tex"
        paper.write_text(PAPER)
        assert main(["check", str(paper), "--all", "--lines", "15:20"]) == 0
        output = capsys.readouterr().out
        assert "line 19     ! not parsed" in output

    def test_bad_arguments(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            main(["check", "paper.tex", "--lines", "five"])
        assert "--lines takes two line numbers" in capsys.readouterr().err
        assert main(["check", str(tmp_path / "missing.tex")]) == 1
