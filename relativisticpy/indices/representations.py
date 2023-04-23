import re
from relativisticpy.indices.indices import Indices
from relativisticpy.index.representations import IndexRepresentationA

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

    def is_representation_A(self, string):
        """
        Example of string to match this category:
            ^{a}^{b}_{theta = 0}_{phi=1}
        
        Conditions for this category to be recognized: 
            - Cannot contain anything but: = or { or } or _ or ^ or [a-zAZ0-9]
            - Has correct pattern: _{}^{}...
            - Between every curly brackets {}, there must be at least one [A-Za-z]+ character/word.
        """
        if isinstance(string, str):
            return bool(re.search("^((\^|\_)(\{)(\}))+$", re.sub('[^\^^\_^\{^\}]',"", string).replace(" ",'')))
        else:
            return False

    