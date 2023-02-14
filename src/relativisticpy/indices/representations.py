import re
from src.relativisticpy.indices.indices import Indices
from src.relativisticpy.index.representations import IndexRepresentationA
from src.relativisticpy.helpers.string_to_sympy.sympy_parser import SympyParser


class IndicesRepresentationA(Indices):
    def __init__(self, indices, basis):
        self.indices_string_representation = self._is_valid_indices(indices)
        self.basis = basis
        Indices.__init__(self, self.return_index_repr_objects(), self.basis)
        
    def return_index_repr_objects(self):
        lis = []
        individual_indices = [item for item in re.split('(?=[_^])', self.indices_string_representation) if item]
        for i in range(len(individual_indices)):
            lis.append(IndexRepresentationA(individual_indices[i], i, self.basis))
        return tuple(lis)
            
    def __repr__(self):
        return f"IndicesRepr({self.indices}, {self.basis})"

    def return_slices(self):
        return tuple([x.slc() for x in self.return_index_instances()])

    def _is_valid_indices(self, indices):
        return indices    

    