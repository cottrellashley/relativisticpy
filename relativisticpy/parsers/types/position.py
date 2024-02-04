from dataclasses import dataclass

@dataclass
class Position: # TODO: Convert to object called TokenPosition which has start and end within it.

    "Line number."
    line: int

    "Character position in this line."
    character: int


    def length(self, pos: "Position") -> int:
        "Computes the length between two positions."
        pass

    def copy(self) -> "Position": return self
