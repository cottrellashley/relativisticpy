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
def site(tmp_path_factory):
    from mkdocs.commands.build import build
    from mkdocs.config import load_config

    folder = tmp_path_factory.mktemp("site")
    build(load_config(str(ROOT / "mkdocs.yml"), site_dir=str(folder), strict=True))
    return folder


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
