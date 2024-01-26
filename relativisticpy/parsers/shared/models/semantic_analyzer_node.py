from dataclasses import dataclass
from typing import List


@dataclass
class SANode:
    node: str
    callback: str # Should be re_named to "node_callback"
    callback_return_type: str
    args: List["SANode"]
