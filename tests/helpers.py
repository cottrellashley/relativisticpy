"""Assertions shared by the test suites."""

import sympy


def same(actual, expected) -> bool:
    """Whether two scalars, arrays or tensors are mathematically equal."""
    actual = getattr(actual, "components", actual)
    expected = getattr(expected, "components", expected)
    if isinstance(actual, sympy.NDimArray) or isinstance(expected, sympy.NDimArray):
        actual, expected = sympy.Array(actual), sympy.Array(expected)
        if actual.shape != expected.shape:
            return False
        flat = actual.reshape(len(actual)).tolist(), expected.reshape(len(expected)).tolist()
        return all(sympy.simplify(a - b) == 0 for a, b in zip(*flat, strict=True))
    return sympy.simplify(sympy.sympify(actual) - sympy.sympify(expected)) == 0


def zero(value) -> bool:
    """Whether every component of a value simplifies to zero."""
    components = getattr(value, "components", value)
    if isinstance(components, sympy.NDimArray):
        return all(sympy.simplify(c) == 0 for c in components.reshape(len(components)).tolist())
    return sympy.simplify(components) == 0
