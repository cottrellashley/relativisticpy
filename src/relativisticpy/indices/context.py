from dataclasses import dataclass
from src.relativisticpy.indices.base import BaseTensorIndices

@dataclass
class IndicesProductContext:

    result : BaseTensorIndices
    "Represents the resulting indices AFTER two parent indices have been combined via some operation (* , + / -) and context has been added."

    parentA : BaseTensorIndices
    "Represents one of the parent indices which has been combined via some operation, with context information injected into it."

    parentB : BaseTensorIndices
    "Represents one of the parent indices which has been combined via some operation, with context information injected into it."