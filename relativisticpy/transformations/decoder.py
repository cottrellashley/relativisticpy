# This deserializes a string into a Coordinate transformation object.
from relativisticpy.transformations.base import CoordinateTransformation, Transformation, Basis
from relativisticpy.core.simpify import Simpify

class TransformationDeserializer:

    def __init__(self, string: str):
        """
        Args:
            string = <string representation of >
        """
        self.string = string

    def deserialize(self):

        return CoordinateTransformation(
            new_basis= Basis(
                as_string   = self.basis,
                as_symbol   = self.get_symbols()

            ),
            transformation= Transformation(
                as_string   = self.transformation,
                as_dict     = self.get_dict()
            )
        )

    def get_dict(self):
        # First we convert to '[ x1 = expr1 , x2 = expr2 , x3 = expr3 , ... , xN = exprN ]' to:
        # list of list [ ['x1' , 'expr1' ], ['x2' , 'expr2' ], ['x3' , 'expr3' ], ... , ['xN' , 'exprN' ]]
        to_list = lambda string : [i.split('=') for i in string.replace(']','').replace('[', '').replace(' ','').split(',')]

        # Second we do a dictionary conprehension to convert this to a dictionary with simpifyed objects.
        deserializer = lambda list : {Simpify().parse(key) : Simpify().parse(value) for key, value in list}

        # Return dictionary
        return deserializer(to_list(self.transformation))

    def get_symbols(self):
        return Simpify().parse(self.basis)

