from dataclasses import dataclass
from typing import List


@dataclass
class AstNode:
    node: str
    handler: str
    args: List["AstNode"]
