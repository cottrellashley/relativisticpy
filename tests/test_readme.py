"""The README, which is also the page of the package on PyPI: its program runs and prints
what it says, its links lead somewhere, and its mathematics shows on GitHub and PyPI.

Its sessions, ``>>>``, are doctests, which pytest runs with the README itself."""

import contextlib
import io
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

#: A block of Python that is a script, not a session, and the block of text after it.
SCRIPT = re.compile(
    r"```python\n(?P<code>.*?)```\n\nprints\n\n```text\n(?P<output>.*?)```", re.DOTALL
)


def test_the_first_program_prints_what_the_readme_says():
    scripts = SCRIPT.findall(README)
    assert scripts, "the README shows a program and what it prints"
    for code, output in scripts:
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            exec(compile(code, "README.md", "exec"), {})
        assert printed.getvalue() == output


def test_every_block_of_python_is_a_session_or_a_script_shown_with_what_it_prints():
    blocks = re.findall(r"```python\n(.*?)```", README, re.DOTALL)
    scripts = [block for block in blocks if not block.lstrip().startswith(">>>")]
    assert all(">>>" not in script for script in scripts)
    assert len(scripts) == len(SCRIPT.findall(README))


def test_its_links_within_the_repository_lead_to_files():
    links = re.findall(r"\]\(([^)]+)\)", README) + re.findall(r'(?:href|src)="([^"]+)"', README)
    local = [link for link in links if not re.match(r"[a-z]+:", link) and not link.startswith("#")]
    assert local
    for link in local:
        assert (ROOT / link.split("#")[0]).exists(), link


def test_its_mathematics_shows_on_github_and_pypi():
    # Neither renders \( ... \), and PyPI renders no mathematics at all: inline maths is
    # written as code.
    assert r"\(" not in README
    assert not re.search(r"(?<![`$])\$[^$`\n]+\$(?!\$)", README)
