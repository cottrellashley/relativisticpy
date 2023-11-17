# This deserializes a string into a Coordinate transformation object.

from relativisticpy.symengine import SymbolArray
from relativisticpy.deserializers import Mathify

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


class TransformationDeserializer:
    def __init__(self, transformation: str, basis: str):
        """
        Args:
            transformation  = [ x1 = expr1 , x2 = expr2 , x3 = expr3 , ... , xN = exprN ]
            basis           = [ y1, y2 , y3, ... , yN]
        """
        self.transformation = transformation
        self.basis = basis

    def deserialize(self):
        return CoordinateTransformation(
            new_basis=Basis(as_string=str(self.basis), as_symbol=self.get_basis()),
            transformation=Transformation(
                as_string=str(self.transformation), as_dict=self.get_dict()
            ),
        )

    def get_dict(self):
        if isinstance(self.transformation, str):
            return self.get_dict_from_string()
        elif isinstance(self.transformation, SymbolArray) and isinstance(
            self.basis, SymbolArray
        ):
            return {key: value for key, value in zip(self.basis, self.transformation)}
        else:
            raise ValueError(
                "Transformation object must either be a string or Sympy MutableDenseNDimArray."
            )

    def get_basis(self):
        if isinstance(self.basis, str):
            return self.get_symbols_from_string()
        elif isinstance(self.basis, SymbolArray):
            return self.basis
        else:
            raise ValueError(
                "Transformation basis object must either be a string or Sympy MutableDenseNDimArray."
            )

    def get_dict_from_string(self):
        # First we convert to '[ x1 = expr1 , x2 = expr2 , x3 = expr3 , ... , xN = exprN ]' to:
        # list of list [ ['x1' , 'expr1' ], ['x2' , 'expr2' ], ['x3' , 'expr3' ], ... , ['xN' , 'exprN' ]]
        to_list = lambda string: [
            i.split("=")
            for i in string.replace("]", "")
            .replace("[", "")
            .replace(" ", "")
            .split(",")
        ]

        # Second we do a dictionary conprehension to convert this to a dictionary with Mathifyed objects.
        deserializer = lambda list: {
            Mathify(key): Mathify(value) for key, value in list
        }

        # Return dictionary
        return deserializer(to_list(self.transformation))

    def get_symbols_from_string(self):
        return Mathify(self.basis)
