"""The conformance suite: every capability the integrations declare, held to the pattern.

Each check of :mod:`relativisticpy.integrations` runs for every capability it applies to,
and a failure says, in words, what to fix. ``uv run pytest tests/integrations -k besselj``
checks one capability.
"""

from __future__ import annotations

import pytest

from relativisticpy import integrations

CAPABILITIES = integrations.capabilities()

#: Where a capability does not pass a check yet, and why: what the language would need.
GAPS: dict[tuple[str, str], str] = {
    ("display", "conjugate"): r"\overline{x} reads as the symbol xbar (design/notation.md, 8)",
}


def applicable(check: str) -> list[object]:
    """The capabilities a check applies to, each a case named after it, and expected to
    fail where it is a gap."""
    cases = []
    for capability in CAPABILITIES:
        if not integrations.applies(check, capability):
            continue
        gap = GAPS.get((check, capability.name))
        marks = [pytest.mark.xfail(reason=gap, strict=True)] if gap else []
        cases.append(pytest.param(capability, id=capability.name, marks=marks))
    return cases


def passes(capability: integrations.Capability, check: str) -> None:
    failures = integrations.check(capability, [check])[check]
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("capability", applicable("declaration"))
def test_declaration(capability):
    passes(capability, "declaration")


@pytest.mark.parametrize("capability", applicable("examples"))
def test_examples(capability):
    passes(capability, "examples")


@pytest.mark.parametrize("capability", applicable("agreement"))
def test_agreement(capability):
    passes(capability, "agreement")


@pytest.mark.parametrize("capability", applicable("lifting"))
def test_lifting(capability):
    passes(capability, "lifting")


@pytest.mark.parametrize("capability", applicable("errors"))
def test_errors(capability):
    passes(capability, "errors")


@pytest.mark.parametrize("capability", applicable("display"))
def test_display(capability):
    passes(capability, "display")


@pytest.mark.parametrize("capability", applicable("reference"))
def test_reference(capability):
    passes(capability, "reference")


@pytest.mark.parametrize("capability", applicable("transform"))
def test_transform(capability):
    passes(capability, "transform")


@pytest.mark.parametrize("capability", applicable("rewrite"))
def test_rewrite(capability):
    passes(capability, "rewrite")


@pytest.mark.parametrize("capability", applicable("feasibility"))
def test_feasibility(capability):
    passes(capability, "feasibility")


@pytest.mark.parametrize("capability", applicable("optimality"))
def test_optimality(capability):
    passes(capability, "optimality")


def test_every_check_is_run_for_some_capability():
    assert {check.name for check in integrations.CHECKS} == {
        "declaration", "examples", "agreement", "lifting", "errors", "display", "reference",
        "transform", "rewrite", "feasibility", "optimality",
    }  # fmt: skip
    for check in integrations.CHECKS:
        assert applicable(check.name), f"no capability is checked for {check.name}"
