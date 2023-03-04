import re
from sympy import MutableDenseNDimArray
from src.relativisticpy.indices.representations import IndicesRepresentationA
from src.relativisticpy.indices.data_structure import TensorIndicesObject
from src.relativisticpy.shared.helpers.string_to_sympy.sympy_parser import SympyParser


class TensorRepresentationIdentifier:
    """
    IMPORTANT: Always try and break this class by checking whether a string passes the match of more than one Category!!
    Optional Use class.
    """
    def __init__(self, components, indices, basis):
        self.indices = indices
        self.components = components
        self.basis = basis

    def get_basis(self):
        if isinstance(self.basis, str):
            return MutableDenseNDimArray(SympyParser(self.basis).convertToSympyObject())
        elif isinstance(self.basis, MutableDenseNDimArray):
            return self.basis
        else:
            raise ValueError("The object type you have entered for the basis is not suppoerted.")
        
    def get_components(self):
        if isinstance(self.components, str):
            return MutableDenseNDimArray(SympyParser(self.components).convertToSympyObject())
        #elif isinstance(self.components, MutableDenseNDimArray):
        else:
            return self.components
        # else:
        #     raise ValueError("The object type you have entered for the components is not suppoerted.")

    def get_indices(self) -> TensorIndicesObject:
        if isinstance(self.indices, str) and self.is_representation_A(self.indices):
            return IndicesRepresentationA(self.indices, self.get_basis())
        else:
            return self.indices

    def get_dimention(self):
        indices = self.get_indices()
        return indices.dimention

    def get_shape(self):
        indices = self.get_indices()
        return indices.shape

    def get_rank(self):
        indices = self.get_indices()
        return indices.rank

    def get_scalar_bool(self):
        indices = self.get_indices()
        return indices.scalar

    def rank_fits_tensor(self):
        return True
        
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

    def is_representation_B(self):
        """
        Example of string to match this category:
            [^a, ^b, _theta = 0, _phi = 1]
        """
        return NotImplementedError

    def is_representation_C(self):
        """
        Example of string to match this category:
            ^{a}^{b}_{theta:0}_{phi:1}
        
        Conditions for this category to be recognized: 
            - Cannot contain anything but: = or { or } or _ or ^ or [a-zAZ0-9]
            - Has correct pattern: _{}^{}...
            - Between every curly brackets {}, there must be at least one [A-Za-z]+ character/word.
        """
        return NotImplementedError

    def is_latex_representation(self):
        """
        Example of string to match this category:
            _{mu = 0}^{nu sigma = 0 phi}
        """
        return NotImplementedError