from dataclasses import dataclass

@dataclass
class Position: # TODO: Convert to object called TokenPosition which has start and end within it.

    line: int
    "Line number."

    character: int
    "Character position in this line."

    def copy(self) -> "Position": return self


@dataclass
class TokenPosition:

    start_pos: Position
    "Position of the first character of the token."

    end_pos: Position
    "Position of the last character of the token."

    @property
    def len(self) -> int: return self.end_pos.character - self.start_pos.character

    def __len__(self) -> int: return self.len
    def copy(self) -> "TokenPosition": return self
