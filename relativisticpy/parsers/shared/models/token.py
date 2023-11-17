from dataclasses import dataclass
from relativisticpy.parsers.shared.constants import TokenType


@dataclass
class Token:
    type: TokenType = None
    value: any = None
