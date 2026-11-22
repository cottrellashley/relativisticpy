"""The wheel carries the documentation, and ``relativisticpy help`` finds it there."""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
PAGES = sorted(path.name for path in (ROOT / "docs").glob("*.md"))

pytestmark = pytest.mark.skipif(
    shutil.which("uv") is None or not (ROOT / "pyproject.toml").is_file(),
    reason="building the wheel needs uv and the source tree",
)


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    out = tmp_path_factory.mktemp("dist")
    built = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out), str(ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    (path,) = out.glob("*.whl")
    return path


def test_the_wheel_has_every_page_and_nothing_else_of_the_docs(wheel):
    names = zipfile.ZipFile(wheel).namelist()
    shipped = sorted(name.split("/")[-1] for name in names if "/_docs/" in name)
    assert shipped == PAGES
    assert "reference.md" in shipped
    assert all(name.startswith(("relativisticpy/", "relativisticpy-")) for name in names)


def test_help_reads_the_pages_from_the_installed_package(wheel, tmp_path):
    installed = tmp_path / "site-packages"
    zipfile.ZipFile(wheel).extractall(installed)
    script = (
        "import json, relativisticpy, relativisticpy.help as h; "
        "print(json.dumps([relativisticpy.__file__, str(h._folder()), h.topics(), "
        "h.page('reference')]))"
    )
    environment = {**os.environ, "PYTHONPATH": str(installed)}
    ran = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=environment, check=True
    )
    module, folder, topics, reference = json.loads(ran.stdout)
    assert Path(module).is_relative_to(installed)
    assert Path(folder) == installed / "relativisticpy" / "_docs"
    assert [f"{topic}.md" for topic, _ in topics] == PAGES
    assert reference == (ROOT / "docs" / "reference.md").read_text(encoding="utf-8")
