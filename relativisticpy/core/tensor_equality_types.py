from enum import Enum


class TensorEqualityType(Enum):
    """A TensorEqualityType enumeration.
    
    Attributes:
        RankEquality: Equal if tensors have the same rank.
        IndicesSymbolEquality: True if set of symbols match, in no particular order or rank.
                               Cardinality/size must be equal.
        IndicesOrderEquality: ...
        ...
    """

    RankEquality = 'RankEquality'
    IndicesSymbolEquality = 'IndicesSymbolEquality'
    IndicesOrderEquality = 'IndicesOrderEquality'
    RankSymbolOrderEquality = 'RankSymbolOrderEquality'
    InstanceEquality = 'InstanceEquality'
    ExactComponentsEquality = 'ExactComponentsEquality'
    NonZeroComponentsEquality = 'NonZeroComponentsEquality'