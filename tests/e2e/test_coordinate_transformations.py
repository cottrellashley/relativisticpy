


class TransformationDeserializer:
    def __init__(self, transformation: str, basis: str):
        """
        Args:
            transformation  = [ x1 = expr1 , x2 = expr2 , x3 = expr3 , ... , xN = exprN ]
            basis           = [ y1, y2 , y3, ... , yN]
        """
        self.transformation = transformation
        self.basis = basis