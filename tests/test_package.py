"""What the package exports: the result types a script needs, and nothing it leaks; each
imported when it is first asked for, so that importing a module of it costs nothing."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import relativisticpy

EXPORTED = (
    "Space", "IndexEquation", "IndexExpression", "Step", "Document", "Report", "Finding",
    "Verdict", "TensorEquation", "UndecidedError", "holds", "display",
)  # fmt: skip


def test_the_result_types_are_exported():  # API-30
    for name in EXPORTED:
        assert name in relativisticpy.__all__
        assert hasattr(relativisticpy, name)


def test_nothing_of_importlib_leaks():  # API-30, AGT-34
    assert not hasattr(relativisticpy, "version")
    assert not hasattr(relativisticpy, "PackageNotFoundError")
    assert isinstance(relativisticpy.__version__, str)


def test_latex_is_the_one_printer():
    assert relativisticpy.latex is relativisticpy.display.latex


# -- importing it ---------------------------------------------------------------------------


def _imported(script: str) -> str:
    ran = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=120
    )
    return ran.stdout.strip()


def test_importing_a_module_of_the_package_imports_nothing_else():
    # The command line starts again with a fixed hash seed before it imports anything
    # that takes time (relativisticpy.__main__), so the package itself costs nothing.
    heavy = ("sympy", "relativisticpy.workbook", "relativisticpy.cli")
    script = (
        "import sys, relativisticpy, relativisticpy.errors, relativisticpy.__main__; "
        f"print([name for name in {heavy!r} if name in sys.modules])"
    )
    assert _imported(script) == "[]"


def test_a_name_is_imported_when_it_is_first_asked_for():
    script = (
        "import sys, relativisticpy; before = 'relativisticpy.workbook' in sys.modules; "
        "relativisticpy.Workbook; print(before, 'relativisticpy.workbook' in sys.modules)"
    )
    assert _imported(script) == "False True"


def test_every_exported_name_is_there_and_typed():
    import relativisticpy.workbook

    tree = ast.parse(Path(relativisticpy.__file__).read_text(encoding="utf-8"))
    (checking,) = [
        node
        for node in tree.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING"
    ]
    typed = {
        alias.asname or alias.name
        for node in checking.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    typed |= {
        node.target.id
        for node in checking.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert typed == set(relativisticpy.__all__)  # what a type checker sees
    for name in relativisticpy.__all__:
        assert getattr(relativisticpy, name) is not None
    assert set(relativisticpy.__all__) <= set(dir(relativisticpy))
    from relativisticpy import Workbook

    assert Workbook is relativisticpy.workbook.Workbook


def test_a_name_the_package_does_not_have_is_an_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute 'nothing'"):
        relativisticpy.nothing  # noqa: B018
    assert relativisticpy.checking.Verdict is relativisticpy.Verdict  # a module, by its name


# -- the Pythons it supports ----------------------------------------------------------------


def test_no_comprehension_calls_super_without_its_class():
    # Before Python 3.12 a comprehension, like a lambda, is a function of its own, where
    # super() finds no class: it runs on the newest Python and fails on the oldest.
    nested = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Lambda)
    found = []
    for path in sorted(Path(relativisticpy.__file__).parent.rglob("*.py")):
        for outer in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(outer, nested):
                continue
            found += [
                f"{path.name}:{node.lineno}"
                for node in ast.walk(outer)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "super"
                and not node.args
            ]
    assert not found, f"name the parent's method before the comprehension: {sorted(set(found))}"
