
from dataclasses import dataclass
from relativisticpy.parser.shared.constants import TokenType


@dataclass
class Token:
    type: TokenType = None
    value: any      = None