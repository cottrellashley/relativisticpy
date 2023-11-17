from enum import Enum


class TensorEqualityType(Enum):
    """All the ways in which two tensors can differ."""

    RankEquality = "RankEquality"
    IndicesSymbolEquality = "IndicesSymbolEquality"
    IndicesOrderEquality = "IndicesOrderEquality"
    RankSymbolOrderEquality = "RankSymbolOrderEquality"
    InstanceEquality = "InstanceEquality"
    ExactComponentsEquality = "ExactComponentsEquality"
    NonZeroComponentsEquality = "NonZeroComponentsEquality"
