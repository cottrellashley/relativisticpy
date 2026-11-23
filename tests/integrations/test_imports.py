"""What the modules of the integrations take from other modules: public names alone. A name
that begins with an underscore is its module's own, to change as the module does; another
module that reaches for it breaks then, unseen (the review of the engines, 29)."""

from __future__ import annotations

import ast
from pathlib import Path

from relativisticpy import integrations

#: The private names still reached, each by the module that reaches it: those of the time
#: budget and of the assumptions, whose modules another line of work has in hand, and two
#: of the framework's own, which the modules of that work take too. Each leaves this list
#: when its name is public.
STILL_PRIVATE = {
    "_distributions.py": {"core._budget", "core._OverBudget", "assumptions._general", "_declare"},
    "_optimizers.py": {"_budget", "_OverBudget", "assumptions._general", "assumptions._assumed"},
    "_transforms.py": {"_declare", "_style"},
}


def private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


def reached(path: Path) -> set[str]:
    """The private names of other modules of RelativisticPy that a module imports, or
    reaches as attributes of a module it imports. A module of the framework, as
    ``_transforms``, is no private name: it is imported whole."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules, found = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("relativisticpy"):
            for alias in node.names:
                modules.add(alias.asname or alias.name)
                if private(alias.name) and node.module != "relativisticpy.integrations":
                    found.add(alias.name)
    for node in ast.walk(tree):
        attribute = isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        if attribute and node.value.id in modules and private(node.attr):
            found.add(f"{node.value.id}.{node.attr}")
    return found


def test_an_integration_takes_no_private_name_of_another_module():
    package = Path(integrations.__file__).parent
    found = {
        f"{path.name}: {name}"
        for path in sorted(package.glob("*.py"))
        for name in reached(path) - STILL_PRIVATE.get(path.name, set())
    }
    assert not found, (
        f"take a name without an underscore, made where it is defined: {sorted(found)}"
    )


def test_what_is_private_is_told_apart_from_a_module_of_the_framework(tmp_path):
    module = tmp_path / "taking.py"
    module.write_text(
        "from relativisticpy.integrations import _transforms as transforms\n"
        "from relativisticpy.integrations._forms import _style, patterns\n"
        "from relativisticpy.runtime import core\n"
        "core._budget(1)\ncore.summation\ntransforms._close\n"
    )
    assert reached(module) == {"_style", "core._budget", "transforms._close"}
