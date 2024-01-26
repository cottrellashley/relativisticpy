from dataclasses import dataclass
from typing import Any
from relativisticpy.parsers.shared.models.position import Position


@dataclass
class Error:
    "Error object. This can be surfaced all the way up to the user."

    position: Position
    "Position of error."

    message: str 
    "The error message."

    context: Any # this could be what objects are involved, which file, etc ...
    "Context of the error."
