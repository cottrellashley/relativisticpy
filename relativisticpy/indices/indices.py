from relativisticpy.index.data_structure import IndexDataStructure
from relativisticpy.indices.data_structure import TensorIndicesObject
from sympy.tensor.array import MutableDenseNDimArray

class Indices(TensorIndicesObject):

    def __init__(self, index_objects : list[IndexDataStructure], basis : MutableDenseNDimArray):
        TensorIndicesObject.__init__(self, 
                            indices         = index_objects, 
                            basis           = basis,
                            dimention       = int(len(basis)),
                            rank            = (int([x.comp_type for x in index_objects].count('contravariant')), int([x.comp_type for x in index_objects].count('covariant'))),
                            scalar          = len(index_objects) == 0,
                            shape           = tuple([i.dimention for i in index_objects]),
                            valid           = True
                            ),
        self.is_valid_index_structure = self.__validate_index_structure()

    def __validate_index_structure(self):
        """Validate the index from the scope of this class. Not the whole index."""
        return True

    def _is_valid_indices(self, indices):
        return indices
