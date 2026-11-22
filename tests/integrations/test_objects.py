r"""The object integrations, held to the pattern of their kind: what a Python object becomes
when a program reads it, as ``\texttt{prices}``. Each declares the types it takes, gives
examples whose objects and answers are known, and its objects meet the rest of the language
without failing inside RelativisticPy; the reference says what each becomes."""

from __future__ import annotations

import importlib.util

import pytest

from relativisticpy import Workbook, integrations
from relativisticpy.errors import DataError
from relativisticpy.integrations import Data, Example, PythonFunction, Scalar, python_object
from relativisticpy.integrations._objects import conversion_of

CONVERSIONS = integrations.conversions()


def applicable(check: str) -> list[object]:
    return [
        pytest.param(each, id=each.name)
        for each in CONVERSIONS
        if integrations.applies(check, each)
    ]


def passes(conversion: integrations.ObjectConversion, check: str) -> None:
    if check != "declaration" and conversion.requires and not _installed(conversion.requires):
        pytest.skip(f"{conversion.requires} is not installed, so its objects cannot be made")
    failures = integrations.check(conversion, [check])[check]
    assert not failures, "\n".join(failures)


def _installed(library: str) -> bool:
    return importlib.util.find_spec(library) is not None


@pytest.mark.parametrize("conversion", applicable("declaration"))
def test_declaration(conversion):
    passes(conversion, "declaration")


@pytest.mark.parametrize("conversion", applicable("examples"))
def test_examples(conversion):
    passes(conversion, "examples")


@pytest.mark.parametrize("conversion", applicable("errors"))
def test_errors(conversion):
    passes(conversion, "errors")


@pytest.mark.parametrize("conversion", applicable("reference"))
def test_reference(conversion):
    passes(conversion, "reference")


def test_the_kinds_of_python_object_are_each_taken():
    assert {conversion.name for conversion in CONVERSIONS} >= {
        "numbers", "arrays", "pandas", "sparse", "xarray", "sympy", "functions", "lists",
    }  # fmt: skip
    assert integrations.check(CONVERSIONS[0]).keys() == {
        "declaration", "examples", "errors", "reference",
    }  # fmt: skip


# -- which integration takes an object ------------------------------------------------------


def test_the_most_specific_class_is_taken_before_a_protocol():
    import numpy as np
    import pandas as pd

    taken = {
        "pandas": pd.DataFrame({"a": [1.0]}),  # it has __array__, but its class is declared
        "arrays": np.float64(0.5),  # a float too, but NumPy's class comes first
        "numbers": 3,
        "lists": [1, 2],
        "sympy": integrations.find("sin").exact(1),
        "functions": len,
    }
    for name, value in taken.items():
        found = conversion_of(value)
        assert getattr(found, "name", None) == name, (name, found)
    assert conversion_of({"a": 1}) is None
    assert conversion_of(object()) is None


# -- an integration that breaks the pattern is told what to fix ------------------------------


class Quantity:
    """A library's own value: a number with a unit."""

    def __init__(self, magnitude: float) -> None:
        self.magnitude = magnitude


def _quantities() -> dict[str, Quantity]:
    return {"q": Quantity(3.0)}


@python_object(
    "quantities",
    takes=f"{Quantity.__module__}.{Quantity.__qualname__}",
    summary="a quantity: its magnitude, a number",
    examples=[
        Example(r"\texttt{q} + 1", 4.0, "3 + 1", given=_quantities),
        Example(r"2 \texttt{q}", 6.0, "2 × 3", given=_quantities),
    ],
)
class Quantities:
    def becomes(self, value: Quantity) -> Scalar:
        return Scalar(value.magnitude)


def test_an_integration_is_checked_before_it_joins_the_language():
    with integrations.installed(Quantities):
        found = conversion_of(Quantity(1.0))
        assert found is Quantities
        assert Workbook(scope={"q": Quantity(2.0)}).expr(r"\texttt{q}^2") == 4.0
    assert conversion_of(Quantity(1.0)) is None
    failures = integrations.check(Quantities)
    assert failures["declaration"] == []
    assert failures["examples"] == []
    assert failures["errors"] == []
    assert failures["reference"] == []


@python_object(
    "Broken Things",
    takes=["thing", "builtins.complex"],
    summary="things.",
    examples=[Example(r"\texttt{z} + 1", 5, "")],
)
class Broken:
    def becomes(self, value: complex) -> Data:
        return Data([value])


def test_what_breaks_the_pattern_is_said_in_words():
    failures = integrations.check(Broken)
    declared = "\n".join(failures["declaration"])
    assert "name an object integration with one word in lower case" in declared
    assert "give a summary, one phrase without a full stop" in declared
    assert "'thing', which is neither a class by its module and name" in declared
    assert "has 1 example(s): give at least two" in declared
    assert "says not where its answer comes from" in declared
    assert "gives no objects: give them, as given=lambda" in declared


def test_an_example_that_fails_says_how():
    @python_object(
        "doubled",
        takes=f"{Quantity.__module__}.{Quantity.__qualname__}",
        summary="a quantity, doubled",
        examples=[
            Example(r"\texttt{q}", 3.0, "its magnitude", given=_quantities),
            Example("1 + 1", 2, "standard", given=_quantities),
        ],
    )
    class Doubled:
        def becomes(self, value: Quantity) -> Scalar:
            return Scalar(2 * value.magnitude)

    failures = integrations.check(Doubled, ["examples"])["examples"]
    assert any(r"\texttt{q} gives 6.0, not 3.0 (its magnitude): fix becomes" in f for f in failures)
    assert any("1 + 1 reads no object" in f for f in failures)


def test_a_declaration_without_becomes_is_refused_at_once():
    with pytest.raises(TypeError, match="has no becomes method; declare what an object becomes"):

        @python_object("nothing", takes="builtins.int", summary="nothing", examples=[])
        class Nothing:
            pass


def test_what_becomes_cannot_read_is_said_in_relativisticpys_words():
    @python_object(
        "failing",
        takes=f"{Quantity.__module__}.{Quantity.__qualname__}",
        summary="a quantity that cannot be read",
        examples=[],
    )
    class Failing:
        def becomes(self, value: Quantity) -> PythonFunction:
            raise ValueError("no magnitude")

    with integrations.installed(Failing), pytest.raises(DataError) as raised:
        Workbook(scope={"q": Quantity(1.0)}).expr(r"\texttt{q}")
    assert "cannot be read: ValueError: no magnitude" in raised.value.message
