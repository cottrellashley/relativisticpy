"""The language reference: generated from the language's tables, complete, current and
short enough to read at once, and there for ``relativisticpy help`` wherever it runs. It
indexes the capabilities by family, and ``relativisticpy help NAME`` describes each in
full, from what its integration declares."""

import re
from pathlib import Path

import pytest
import sympy

from relativisticpy import display, integrations, reference
from relativisticpy import help as offline
from relativisticpy.checking import Verdict
from relativisticpy.compiler.builtins import BUILTINS, CONSTANTS
from relativisticpy.compiler.compiler import _LINEAR_ALGEBRA, _NOTATION_OPERATIONS, _STATISTICS
from relativisticpy.compiler.session import ASSUMPTION_STATEMENTS, RENAMING_STATEMENTS
from relativisticpy.documents import DISPLAY_ENVIRONMENTS
from relativisticpy.errors import UsageError
from relativisticpy.integrations import Example, function
from relativisticpy.language import vocabulary

DOCS = Path(__file__).parent.parent / "docs"
PAGE = reference.markdown()
CAPABILITIES = integrations.capabilities()
TOPICS = {topic for topic, _ in offline.topics()} | set(offline.ALIASES)


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
def test_every_builtin_is_indexed_and_described_in_full(name):
    assert code(name)
    described = offline.page(f"{name}()")
    assert described == reference.capability(name)
    assert described.startswith(f"# {integrations.find(name).name}\n")
    if name.lower() not in TOPICS:  # a page comes first, as latex is
        assert offline.page(name) == described


@pytest.mark.parametrize("capability", CAPABILITIES, ids=lambda capability: capability.name)
def test_every_capability_is_described_in_full_by_its_declaration(capability):
    described = reference.capability(capability.name)
    assert code(capability.name)
    assert described.startswith(f"# {capability.name}\n\n")
    assert capability.summary[1:] in described
    for form in (*capability.aliases, *capability.latex, *capability.unicode):
        assert f"`{form}" in described
    for example in capability.examples:
        assert display.unicode(example.answer) in described
        assert example.source in described
        for line in example.program.strip().splitlines():
            assert " ".join(line.split()) in described
    assert f"`{capability.module}`" in described
    assert reference._KINDS[capability.kind] in described
    if capability.requires is not None:
        assert reference._LIBRARIES[capability.requires] in described


def test_the_help_of_a_capability_says_what_computes_it_and_how_to_use_it():
    assert reference.capability("besselj") == (
        "# besselj\n"
        "\n"
        "The Bessel function of the first kind, of order `\\nu`.\n"
        "\n"
        "- **Written:** `besselj(nu, x)`, `J_{\\nu}(x)`\n"
        "- **Arguments:** 2, `nu` and `x`\n"
        "- **Takes:** numbers and expressions\n"
        "- **Computed by:** SymPy, exactly; on data, SciPy "
        '(`pip install "relativisticpy[numeric]"`)\n'
        "- **Kind:** a function, declared by the integration "
        "`relativisticpy.integrations.bessel`: Bessel and Airy functions\n"
        "\n"
        "## Examples\n"
        "\n"
        "- `J_0(1)` gives `0.7651976865579666` (scipy.special.jv(0, 1.0))\n"
        "- `J_{1/2}(\\pi/2)` gives `2/π` (DLMF 10.16.1)\n"
        "\n"
        "## Notes\n"
        "\n"
        "- Given a tensor, it applies to each component, and given data, to each number.\n"
    )
    spacetime = reference.capability("Minkowski")
    assert "`cartesian`, t, x, y, z; `spherical`, t, r, θ, φ" in spacetime
    assert "`g_{\\mu\\nu} := Minkowski()`, then `g_{tt}`, gives `−1`" in spacetime
    assert "Its inverse is `InverseLaplaceTransform`" in reference.capability("LaplaceTransform")
    assert "`\\varphi(n)` is read as totient only where `n` is an integer" in (
        reference.capability("totient")
    )
    assert "`asin(x)`, `arcsin(x)`, `\\arcsin`" in reference.capability("arcsin")
    assert "`extrema`, `kkt`, `simplex`" in reference.capability("minimize")
    assert "`fft`, to the same numbers, faster" in reference.capability("dft_sum")


def test_the_page_on_the_command_line_shows_the_help_of_a_function_as_it_is():
    page = (DOCS / "cli.md").read_text(encoding="utf-8")
    shown = page.split("$ relativisticpy help besselj\n", 1)[1].split("```", 1)[0]
    assert shown == reference.capability("besselj")


def test_the_help_of_a_new_capability_is_written_from_its_declaration():
    x = sympy.Symbol("x")

    @function(
        "gudermannian",
        latex=r"\operatorname{gd}(x)",
        summary="the Gudermannian function; it relates the circular and hyperbolic functions",
        examples=[
            Example(r"\operatorname{gd}(0)", 0, "DLMF 4.23.39"),
            Example("gudermannian(x)", 2 * sympy.atan(sympy.tanh(x / 2)), "DLMF 4.23.39"),
        ],
    )
    class Gudermannian:
        def exact(self, x):
            return 2 * sympy.atan(sympy.tanh(x / 2))

    with integrations.installed(Gudermannian):
        described = offline.page("gudermannian")
        assert "`gudermannian` `\\operatorname{gd}(x)`" in reference.markdown()
    assert described.startswith("# gudermannian\n\nThe Gudermannian function; it relates")
    assert "- **Written:** `gudermannian(x)`, `\\operatorname{gd}(x)`" in described
    assert "- `\\operatorname{gd}(0)` gives `0` (DLMF 4.23.39)" in described
    assert f"declared by the integration `{Gudermannian.module}`" in described
    with pytest.raises(UsageError, match="no help on 'gudermannian'"):
        offline.page("gudermannian")


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
    # About 12,000 tokens, a token of this page being about three characters. The
    # capabilities are an index, a line for each family with every name and form, and
    # relativisticpy help <name> describes each in full: a new capability adds its names
    # to a line, and a new integration a line.
    assert len(PAGE) < 36_000


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


def test_help_describes_a_function_without_any_pages(monkeypatch):
    monkeypatch.setattr(offline, "_folder", lambda: None)
    assert offline.page("besselj") == reference.capability("besselj")


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
