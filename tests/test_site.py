"""The documentation site, as built: what its pages copy, and what it gives agents."""

import doctest
import html
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
DOCS = ROOT / "docs"

pytest.importorskip("mkdocs", reason="the site is built with the docs group of dependencies")
pytest.importorskip("material", reason="the site is built with the docs group of dependencies")
pytestmark = pytest.mark.skipif(
    not (ROOT / "mkdocs.yml").is_file(), reason="the site is built from the source tree"
)


def tool(name):
    """A module of tools/, the hooks the site is built with."""
    spec = importlib.util.spec_from_file_location(f"tools_{name}", ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    """The configuration the site is built with, after building it."""
    from mkdocs.commands.build import build
    from mkdocs.config import load_config

    folder = tmp_path_factory.mktemp("site")
    configuration = load_config(str(ROOT / "mkdocs.yml"), site_dir=str(folder), strict=True)
    build(configuration)
    return configuration


@pytest.fixture(scope="module")
def site(config):
    return Path(config["site_dir"])


def built(site, page):
    """The HTML a page of docs/ is built as."""
    path = site / ("index.html" if page.stem == "index" else f"{page.stem}/index.html")
    return path.read_text(encoding="utf-8")


SESSION = re.compile(r'<div class="language-pycon highlight"><pre(?P<attributes>[^>]*)>')
COPY = re.compile(r'data-copy="(?P<code>[^"]*)"')


@pytest.mark.parametrize("page", sorted(DOCS.glob("*.md")), ids=lambda page: page.name)
def test_copying_an_example_gives_its_code_alone(site, page):
    examples = doctest.DocTestParser().get_examples(page.read_text(encoding="utf-8"))
    copies = []
    for block in SESSION.finditer(built(site, page)):
        copied = COPY.search(block["attributes"])
        assert copied is not None, "a session that copies its prompts"
        code = html.unescape(copied["code"])
        assert not re.search(r"^(>>>|\.\.\.)( |$)", code, re.MULTILINE)
        compile(code, page.name, "exec")  # code that runs, not output
        copies.append(code + "\n")
    assert "".join(copies) == "".join(example.source for example in examples)


def test_a_session_copies_as_doctest_reads_it():
    code_copy = tool("code_copy")
    session = '>>> x = """\n... a\n... """\n>>> x\n\'a\\n\'\n...\n>>>\n'
    assert code_copy.session_code(session) == 'x = """\na\n"""\nx\n'
    assert code_copy.session_code("$ relativisticpy\n") is None


def test_every_page_is_in_the_navigation(config):
    listed = {page for _, _, page in tool("navigation").pages(config["nav"])}
    assert listed == {page.name for page in DOCS.glob("*.md")}


def test_llms_txt_indexes_every_page_in_a_few_lines(config, site):
    navigation = tool("navigation")
    entries = navigation.pages(config["nav"])
    text = (site / "llms.txt").read_text(encoding="utf-8")
    summary = config["site_description"]
    assert text == navigation.llms_txt(entries, DOCS, config["site_url"], summary)
    assert text.startswith(f"# RelativisticPy\n\n> {summary}\n")
    for _, title, page in entries:
        assert f"- [{title}]({config['site_url']}{navigation.url(page)}index.html.md): " in text
    for fact in ("relativisticpy help reference", ":=", 'r"\\frac{1}{2}"', "--format json"):
        assert fact in text
    assert len(text) < 6_000  # about 1,500 tokens


def test_llms_full_txt_holds_every_page_in_order(config, site):
    text = (site / "llms-full.txt").read_text(encoding="utf-8")
    positions = []
    for _, _, page in tool("navigation").pages(config["nav"]):
        source = (DOCS / page).read_text(encoding="utf-8").strip()
        assert source in text, page
        positions.append(text.index(source))
    assert positions == sorted(positions)


def test_every_page_has_a_copy_in_markdown_where_llms_txt_points(config, site):
    navigation = tool("navigation")
    for _, _, page in navigation.pages(config["nav"]):
        copy = site / navigation.url(page) / "index.html.md"
        assert copy.read_text(encoding="utf-8") == (DOCS / page).read_text(encoding="utf-8")


def test_a_page_named_before_it_is_written_is_left_out(tmp_path):
    navigation = tool("navigation")
    (tmp_path / "glossary.md").write_text("# Glossary\n", encoding="utf-8")
    reference = {"Reference": [{"Command line": "cli.md"}, {"Glossary": "glossary.md"}]}
    nav = [{"Home": "index.md"}, reference]
    assert navigation.pending(nav, tmp_path) == [
        {"Home": "index.md"},
        {"Reference": [{"Glossary": "glossary.md"}]},
    ]
    (tmp_path / "cli.md").write_text("# Command line\n", encoding="utf-8")
    assert navigation.pending(nav, tmp_path) == nav


def test_a_page_is_described_by_its_header_or_its_first_sentence():
    describe = tool("navigation").description
    header = (
        "# T\n\n> **For:** you. **You will\n> learn:** what\n> [this](x.md) is. **Before:** y\n"
    )
    assert describe(header) == "what this is."
    prose = "# T\n\n```pycon\n>>> 1\n\n```\n\n**A name** is a [thing](x.md). More.\n"
    assert describe(prose) == "A name is a thing."
