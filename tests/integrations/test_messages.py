"""What the conformance suite says of capabilities that break the pattern, written in a
temporary module as an agent would write one: each failure names the capability and says
what to do, in words an agent can act on."""

from __future__ import annotations

import importlib
import sys
import textwrap

import pytest

from relativisticpy import integrations

#: A module of capabilities, each broken in ways the checks find.
BROKEN = '''
"""Capabilities broken in every way the conformance suite checks."""

import scipy.special  # imported with the module, not when a capability is used
import sympy

from relativisticpy.integrations import Example, function


@function(
    "brokenj",
    summary="the Fresnel integral |S|.",
    requires="scipyy",
    examples=[Example("brokenj(0)", 2, "")],
)
class BrokenJ:
    def exact(self, x):
        return sympy.fresnels(x)

    def numeric(self, xp, x):
        return xp.cos(x)


@function(
    "brokenk",
    summary="the identity, which fails inside on a list",
    examples=[Example("brokenk(1)", 1, "standard"), Example("brokenk(2)", 2, "standard")],
)
class BrokenK:
    def exact(self, x):
        if isinstance(x, list):
            raise KeyError(x)  # as a mistake in an integration would
        return x
'''


@pytest.fixture
def broken(tmp_path, monkeypatch):
    """The temporary module, imported as an outside package's would be."""
    (tmp_path / "broken_integration.py").write_text(textwrap.dedent(BROKEN), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("broken_integration")
    yield module
    sys.modules.pop("broken_integration", None)


def told(failures: list[str], *words: str) -> bool:
    """Whether one failure says all of ``words``."""
    return any(all(word in failure for word in words) for failure in failures)


def test_every_failure_names_the_capability_and_says_what_to_do(broken):
    found = integrations.check(broken.BrokenJ)
    for failures in found.values():
        for failure in failures:
            assert "brokenj" in failure
            assert any(
                verb in failure
                for verb in ("give", "declare", "fix", "import", "write", "add", "refuse")
            ), failure


def test_the_declaration_is_told_what_it_lacks(broken):
    failures = integrations.check(broken.BrokenJ, ["declaration"])["declaration"]
    assert told(failures, "give a summary, one phrase without a full stop")
    assert told(failures, "has 1 example(s): give at least two")
    assert told(failures, "says not where its answer comes from: give its source")
    assert told(failures, "requires scipyy, which no extra of RelativisticPy installs")
    assert told(failures, "imports scipy when it is imported; import it inside exact or numeric")
    assert len(failures) == 5


def test_an_example_with_the_wrong_answer_is_told(broken):
    (failure,) = integrations.check(broken.BrokenJ, ["examples"])["examples"]
    assert failure == (
        "brokenj's example brokenj(0) gives 0, not 2: fix exact, or the example's answer"
    )


def test_numeric_that_disagrees_with_exact_is_told_where(broken):
    failures = integrations.check(broken.BrokenJ, ["agreement"])["agreement"]
    assert told(failures, "numeric and exact differ at (x=0.3)", "fix numeric", "points=[...]")


def test_a_function_that_is_wrong_on_data_is_told(broken):
    failures = integrations.check(broken.BrokenJ, ["lifting"])["lifting"]
    assert told(failures, "brokenj applied to data gives", "not its value at each component")


def test_a_library_declared_but_not_used_is_told(broken):
    failures = integrations.check(broken.BrokenJ, ["errors"])["errors"]
    assert failures == [
        "brokenj computes without scipyy: does it need it? declare requires only then"
    ]


def test_the_wrong_kind_of_argument_that_fails_inside_is_told(broken):
    (failure,) = integrations.check(broken.BrokenK, ["errors"])["errors"]
    assert failure.startswith("brokenk([1, 2]) fails inside brokenk")
    assert "refuse what it cannot take with an EvaluationError in exact" in failure


def test_a_call_that_does_not_read_back_is_told(broken):
    failures = integrations.check(broken.BrokenJ, ["display"])["display"]
    assert told(failures, "prints in text as fresnels(x), which reads back as fresnels(x), in")
    assert told(failures, "fresnels is a function the language does not know")
    assert told(failures, "name the capability fresnels, as its calls print, or give it that alias")
    assert told(failures, "declare a latex form the language reads as brokenj")


def test_a_summary_with_a_bar_is_the_first_sentence_of_its_help(broken):
    # The index of the reference lists names and forms, and the help, a sentence, holds a
    # bar that a table could not.
    assert integrations.check(broken.BrokenJ, ["reference"])["reference"] == []


def test_what_passes_is_told_nothing(broken):
    found = integrations.check(broken.BrokenK)
    assert found.pop("errors")  # its exact fails inside on a list
    (imports,) = found.pop("declaration")  # its module imports SciPy, as brokenj's does
    assert imports.startswith("brokenk: its module imports scipy when it is imported")
    assert all(failures == [] for failures in found.values()), found


def test_a_class_without_exact_is_refused_where_it_is_declared():
    with pytest.raises(TypeError, match="declare how it is computed with SymPy"):

        @integrations.function("brokenl", summary="nothing", examples=[])
        class BrokenL:
            pass


def test_the_broken_capabilities_are_not_part_of_the_language(broken):
    names = {capability.name for capability in integrations.capabilities()}
    assert not names & {"brokenj", "brokenk"}


EXAMPLES = [
    integrations.Example("brokenroot(4)", 2, "math.sqrt(4)"),
    integrations.Example("brokenroot(9)", 3, "math.sqrt(9)"),
]


def test_a_function_of_data_past_its_points_is_held_to_exact():
    # W_0 of data gave real parts only, where exact is complex (the review of the engines, 6)
    import sympy

    class Root:
        def exact(self, x):
            return sympy.sqrt(x)

        def numeric(self, xp, x):
            return xp.sqrt(xp.abs(x))

    root = integrations.function(
        "brokenroot", summary="the square root", examples=EXAMPLES, points=[(0.3,), (0.7,)]
    )(Root)
    (failure,) = integrations.check(root, ["lifting"])["lifting"]
    assert failure.startswith(
        "brokenroot applied to data past its points gives 0.547722557505 at (x=-0.3), where "
        "exact gives 0+0.547722557505j: "
    )
    assert failure.endswith("give a complex value where exact has one, or compute there with exact")


def test_a_function_of_data_far_past_its_points_is_held_to_exact():
    # L_40 of data had lost its digits where L_2 kept them (the review of the engines, 5)
    import sympy

    class Near:
        def exact(self, x):
            return sympy.sin(x)

        def numeric(self, xp, x):
            return xp.where(xp.abs(x) < 10, xp.sin(x), 0.0)

    near = integrations.function(
        "brokennear", summary="the sine, near 0", examples=EXAMPLES, points=[(0.3,), (0.7,)]
    )(Near)
    (failure,) = integrations.check(near, ["lifting"])["lifting"]
    assert failure.startswith(
        "brokennear applied to data past its points gives 0 at (x=40.3), where exact gives "
        "0.514740116924: "
    )


def test_a_function_of_data_that_computes_where_exact_refuses_is_told():
    import sympy

    from relativisticpy.errors import EvaluationError

    class Count:
        def exact(self, n):
            if sympy.sympify(n).is_number and float(n) % 1:
                raise EvaluationError(f"brokencount takes whole numbers, not {n}")
            return n

        def numeric(self, xp, n):
            return xp.floor(n)

    count = integrations.function(
        "brokencount", summary="a count", examples=EXAMPLES, points=[(1,), (2,)]
    )(Count)
    (failure,) = integrations.check(count, ["lifting"])["lifting"]
    assert failure.startswith(
        "brokencount applied to data past its points is not refused at (n=1.5), where exact "
        "refuses: it gives [(1+0j)]"
    )
