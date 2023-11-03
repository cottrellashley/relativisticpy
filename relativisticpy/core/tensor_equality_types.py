from enum import Enum

class TensorEqualityType:
    """ Enumerator containing all different ways in which tensors can be equivalent to each other (within the context of python objects)."""
    pass

class RankEquality(TensorEqualityType):
    """ Requires indices rank to match. Set of symbol match not required. """
    NAME = 'RankEquality'
    pass

class IndicesSymbolEquality(TensorEqualityType):
    NAME = 'IndicesSymbolEquality'
    pass

class IndicesOrderEquality(TensorEqualityType):
    NAME = 'IndicesOrderEquality'
    pass

class RankSymbolOrderEquality(TensorEqualityType):
    NAME = 'RankSymbolOrderEquality'
    pass

class InstanceEquality(TensorEqualityType):
    NAME = 'InstanceEquality'
    pass

class ExactComponentsEquality(TensorEqualityType):
    NAME = 'ExactComponentsEquality'
    pass

class NonZeroComponentsEquality(TensorEqualityType):
    NAME = 'NonZeroComponentsEquality'
    pass
