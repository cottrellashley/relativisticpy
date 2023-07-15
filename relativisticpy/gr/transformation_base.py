# Standard
from dataclasses import dataclass

@dataclass
class Basis:
    as_string: str
    as_symbol: any

@dataclass
class Transformation:
    as_string: str
    as_dict: dict

@dataclass
class CoordinateTransformation:
    new_basis: Basis
    transformation: Transformation