"""The intermediate representation.

The compiler lowers a program to a graph of typed operations; the executor
runs the part of the graph the program's outputs need.
"""

from relativisticpy.ir import ops
from relativisticpy.ir.graph import Graph, Node, dependencies, render
from relativisticpy.ir.types import (
    SCALAR,
    VALUE,
    EquationType,
    NotationAlgebra,
    NotationType,
    ScalarType,
    TensorType,
    Type,
    ValueType,
)

__all__ = [
    "SCALAR",
    "VALUE",
    "EquationType",
    "Graph",
    "Node",
    "NotationAlgebra",
    "NotationType",
    "ScalarType",
    "TensorType",
    "Type",
    "ValueType",
    "dependencies",
    "ops",
    "render",
]
