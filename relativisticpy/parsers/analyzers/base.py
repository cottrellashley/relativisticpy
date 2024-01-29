from dataclasses import dataclass
from typing import List
from relativisticpy.parsers.types.base import AstNode


@dataclass
class ActionTree:
    ast: AstNode
    return_type: str
    returns_object: bool

@dataclass
class GrScriptTree:
    action_trees: List[ActionTree]
    contains_error: bool
    display_error_str: List[str]
