"""The language reference: generated from the language's tables, complete, current and
short enough to read at once, and there for ``relativisticpy help`` wherever it runs."""

import re
from pathlib import Path

import pytest

from relativisticpy import help as offline
from relativisticpy import reference
from relativisticpy.checking import Verdict
from relativisticpy.compiler.builtins import BUILTINS, CONSTANTS
from relativisticpy.compiler.compiler import _LINEAR_ALGEBRA, _NOTATION_OPERATIONS, _STATISTICS
from relativisticpy.compiler.session import ASSUMPTION_STATEMENTS, RENAMING_STATEMENTS
from relativisticpy.documents import DISPLAY_ENVIRONMENTS
from relativisticpy.errors import UsageError
from relativisticpy.language import vocabulary

DOCS = Path(__file__).parent.parent / "docs"
PAGE = reference.markdown()


def test_the_page_in_the_docs_is_current():
    written = (DOCS / "reference.md").read_text(encoding="utf-8")
    assert written == PAGE, (
        "docs/reference.md is out of date: write it again with "
        "uv run python -m relativisticpy.reference --write docs/reference.md"
    )


def code(name):
    """Whether ``name`` is on the page as code, alone or in a list of commands."""
    return re.search(rf"`(?:[^`\n]* )?{re.escape(name)}(?:[ (][^`\n]*)?`", PAGE) is not None


@pytest.mark.parametrize("name", sorted(BUILTINS))
def test_every_builtin_is_described(name):
    assert code(name)


@pytest.mark.parametrize("name", sorted(CONSTANTS))
def test_every_constant_is_described(name):
    assert code(name)


@pytest.mark.parametrize("name", sorted(vocabulary.COMMANDS | set(vocabulary.UNSUPPORTED_COMMANDS)))
def test_every_command_is_described(name):
    assert re.search(rf"\\{name}(?![A-Za-z])", PAGE)


@pytest.mark.parametrize(
    "name",
    sorted(
        set().union(*vocabulary.ENVIRONMENTS.values())
        | DISPLAY_ENVIRONMENTS
        | {*ASSUMPTION_STATEMENTS, *RENAMING_STATEMENTS, "Coordinates", "Kretschmann"}
        | {*_STATISTICS, *_LINEAR_ALGEBRA, *_NOTATION_OPERATIONS}
    ),
)
def test_every_environment_statement_and_operation_is_described(name):
    assert code(name)


@pytest.mark.parametrize("verdict", list(Verdict))
def test_every_verdict_is_described(verdict):
    assert f"| {verdict.mark} | {verdict.value} | {verdict.meaning} |" in PAGE


def test_the_meanings_of_definitions_are_given():
    for written in ("`x := 2`", "`x = 2`", r"\equiv", "`ds^2 = ...`", r"`i, j \in \mathcal{H}`"):
        assert written in PAGE


def test_an_agent_reads_it_at_once():
    # About 10,000 tokens: a token of this page is about three characters.
    assert len(PAGE) < 32_000


def test_its_tables_are_well_formed():
    for line in PAGE.splitlines():
        if line.startswith("|"):
            assert line.endswith((" |", "---|")), line


def test_help_has_it_without_its_page(monkeypatch, tmp_path):
    (tmp_path / "language.md").write_text("# The language\n", encoding="utf-8")
    monkeypatch.setattr(offline, "_folder", lambda: tmp_path)
    assert offline.topics() == [("language", "The language"), ("reference", reference.TITLE)]
    assert offline.page("reference") == PAGE


def test_help_has_it_without_any_pages(monkeypatch):
    monkeypatch.setattr(offline, "_folder", lambda: None)
    assert offline.topics() == [("reference", reference.TITLE)]
    assert offline.page("reference") == PAGE
    with pytest.raises(UsageError, match="no help on 'tensors'") as raised:
        offline.page("tensors")
    assert "the documentation is not installed" in raised.value.message
    assert "relativisticpy help reference" in raised.value.hint


def test_help_reads_its_page_where_there_is_one():
    assert offline.page("reference") == PAGE
    assert ("reference", reference.TITLE) in offline.topics()


def test_a_topic_is_never_a_path():
    with pytest.raises(UsageError, match=r"no help on '\.\./README'"):
        offline.page("../README")


def test_the_module_writes_and_prints_the_page(tmp_path, capsysbinary):
    assert reference.main(["--write", str(tmp_path / "reference.md")]) == 0
    assert (tmp_path / "reference.md").read_text(encoding="utf-8") == PAGE
    assert reference.main([]) == 0
    assert capsysbinary.readouterr().out.decode("utf-8") == PAGE
    assert reference.main(["--wrong"]) == 2
