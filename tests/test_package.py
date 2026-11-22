"""What the package exports: the result types a script needs, and nothing it leaks."""

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
