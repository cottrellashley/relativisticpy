# Standard Library
from typing import List

# External Modules
from relativisticpy.utils import tensor_trace_product
from relativisticpy.deserializers import tensor_from_string
from relativisticpy.symengine import SymbolArray

# This Module
from relativisticpy.core.indices import Indices, Idx
from relativisticpy.core.einsum_convention import einstein_convention

@einstein_convention
class MultiIndexObject:
    __slots__ = "components", "indices", "basis", "_subcomponents"

    @classmethod
    def from_string(cls, indices_str, comp_str, basis_str):
        return tensor_from_string(Idx, Indices, MultiIndexObject, indices_str, comp_str, basis_str)

    def __init__(self, indices: Indices, components: SymbolArray = None, basis: SymbolArray = None):
        self.components = components
        self.basis = basis
        self._subcomponents = None
        self.indices = indices

        if self.indices.basis == None:
            self.indices.basis = basis

        if indices.anyrunnig:
            if basis != None:
                indices.basis = basis
                self._subcomponents = self.get_subcomponents(indices)
                self.indices = indices.get_non_running()
            else:
                raise ValueError(f'Basis parameter must be provided to initialize {self} with non-running indices.')
    
    @property
    def rank(self): return self.indices.rank
    @property
    def scalar(self): return self.rank == (0,0)
    @property
    def shape(self): return self.indices.shape
    @property
    def dimention(self): return len(self.basis)

    @property
    def subcomponents(self): return self._subcomponents

    @subcomponents.setter
    def subcomponents(self, value: SymbolArray): self._subcomponents = value

    # Dunders
    def __post_init__(self) -> None: self.__set_self_summed() # After __init__ -> check and perform self-sum i.e. G_{a}^{a}_{b}_{c}
    def __neg__(self): return MultiIndexObject(self.indices, -self.components, self.basis)
    def get_subcomponents(self, indices: Indices):
        self._subcomponents = self.components[indices.__index__()]
        return self._subcomponents

    def __add__(self, other: 'MultiIndexObject') -> 'MultiIndexObject':
        operation = lambda a, b : a + b
        result = self.additive_operation(other, operation) # Implementation inserted by decorator
        return MultiIndexObject(components = result.components, indices = result.indices, basis = self.basis)

    def __sub__(self, other: 'MultiIndexObject') -> 'MultiIndexObject':
        operation = lambda a, b : a - b
        result = self.additive_operation(other, operation)
        return MultiIndexObject(components = result.components, indices = result.indices, basis = self.basis)

    def __mul__(self, other: 'MultiIndexObject') -> 'MultiIndexObject':
        if isinstance(other, (float, int)): # If we're number then just multiply every component by it (assuming the SymbolArray implements the * method ... )
            return MultiIndexObject(components = other*self.components, indices = self.indices, basis = self.basis)
        operation = lambda a, b : a * b
        result = self.einsum_operation(other, operation)
        return MultiIndexObject(components = result.components, indices = result.indices, basis = self.basis)

    def __rmul__(self, other: 'MultiIndexObject') -> 'MultiIndexObject':
        if isinstance(other, (float, int)): # If we're number then just multiply every component by it.
            return MultiIndexObject(components = other*self.components, indices = self.indices, basis = self.basis)
        return self * other

    def __truediv__(self, other: 'MultiIndexObject') -> 'MultiIndexObject':
        if isinstance(other, (float, int)): # If we're number then just divide every component by it.
            return MultiIndexObject(components = self.components/other, indices = self.indices, basis = self.basis)
        else:
            raise ValueError("Cannot divide with anything other than int or float.")

    def comps_contraction(self, other: 'MultiIndexObject', idcs: List[List[int]]): return tensor_trace_product(self.components, other.components, idcs)

    # Privates
    def __set_self_summed(self) -> None:
        if self.indices.self_summed:
            result = self.selfsum_operation()
            self.components = result.components
            self.indices = result.indices
        else:
            pass
