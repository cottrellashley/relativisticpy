from itertools import product
from operator import itemgetter
from typing import Union

# Dependencies
from sympy import MutableDenseNDimArray

# Internal
from relativisticpy.core.indices import Indices
from relativisticpy.core.decorators import einstein_convention
from relativisticpy.core.helpers import transpose_list

@einstein_convention
class MultiIndexObject:
    __slots__ = "components", "indices", "basis"

    def __init__(self, components: MutableDenseNDimArray, indices: Indices, basis: MutableDenseNDimArray):
        self.components = components
        self.indices = indices
        self.basis = basis
        if self.indices.basis == None:
            self.indices.basis = basis
    
    @property
    def rank(self): return self.indices.rank
    @property
    def scalar(self): return self.rank == (0,0)
    @property
    def shape(self): return self.indices.shape
    @property
    def dimention(self): return len(self.basis)

    # Dunders
    def __getitem__(self, indices): return self.components[indices.__index__()]
    def __neg__(self): self.components = -self.components; return self
    def __post_init__(self) -> None: self.__set_self_summed() # After __init__ -> check and perform self-sum i.e. G_{a}^{a}_{b}_{c}
    
    def __setitem__(self, indices: Indices, other_expr: Union['MultiIndexObject', MutableDenseNDimArray]):
        """
        Mapps resulting components into the shape of the setted indices.
        Indices must match in symbols and covariance, but in any order.
        If other_expr is another MultiIndexObject, we not only map the components, but match the indices.
            T[Indices] = T1[Indices]*T2[Indices] + T3[Indices]
        Else, if other_expr just an array, we replace the components of the tesnor with components specifies in the argument of the setter.
            T[Indices] = Array of components -> instace T(components, indices, basis)
        """
        if isinstance(other_expr, MultiIndexObject) and other_expr.indices == indices:
            summed_index_locations = transpose_list(indices._get_all_repeated_locations(other_expr.indices))
            all = [(IndexA, IndexB) for (IndexA, IndexB) in list(product(indices, other_expr.indices)) if itemgetter(*summed_index_locations[0])(IndexA) == itemgetter(*summed_index_locations[1])(IndexB)]
            for idxa, idxb  in all:
                self.components[idxa] = other_expr.components[idxb]
            return MultiIndexObject(self.components, indices, self.basis)
        elif isinstance(other_expr, MutableDenseNDimArray) and other_expr.shape == self.components.shape:
            for idxa in self.indices:
                self.components[idxa] = other_expr[idxa]
            return MultiIndexObject(self.components, indices, self.basis)
        else:
            raise ValueError(f'The object you are trying to set and/or map to the {self} has the a shape which does not match {self.shape}.')

    def __add__(self, other: 'MultiIndexObject') -> 'MultiIndexObject':
        operation = lambda a, b : a + b
        result = self.additive_operation(other, operation) # Implementation inserted by decorator
        return MultiIndexObject(components = result.components, indices = result.indices, basis = self.basis)

    def __sub__(self, other: 'MultiIndexObject') -> 'MultiIndexObject':
        operation = lambda a, b : a - b
        result = self.additive_operation(other, operation)
        return MultiIndexObject(components = result.components, indices = result.indices, basis = self.basis)

    def __mul__(self, other: 'MultiIndexObject') -> 'MultiIndexObject':
        if isinstance(other, (float, int)): # If we're number then just multiply every component by it.
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

    def coordinate_transformation(self, transformation):
        pass

    # Privates
    def __set_self_summed(self) -> None:
        if self.indices.self_summed:
            result = self.selfsum_operation()
            self.components = result.components
            self.indices = result.indices
        else:
            pass

# __add__ , __sub__ BUG: When we add two MultiIndexObjects, the order seems to matter when combined with multiplication.
# This violates the commutativity of MultiIndexObjects: A*(B + C) == A*B + A*C
# Source of BUG:
# Say we are subtracting: ((1,0),(0,1)) 
# Then we are essentially calculating 
# Result[(1,0)] = T1[(1,0)] - T2[(0,1)] 
# but if we had inverted the inputs and inputted the reverse instead
# Result[(1,0)] = T2[(1,0)] - T1[(0,1)] 
# (Since for addition we take the resulting indices to be the first MultiIndexObject.)
# this same component in the above woule have been T2[(0,1)] - T1[(1,0)] 
# which in the new order, in the resulting components is now at Result[(0,1)] 
# A fix would involve storing commutation rules within the object 
# i.e. We can also find the symmetries and then impose '[]' = [Idx(a)] as an Idx property => this index is commutated with Idx(a) 
# also '{}' = [Idx(a)] as an Idx property => this index is anti-commutated with Idx(a) 
# furthermore we can even find out from a non-resulting Indices object, whether the parent tensor can be further decomposed into 
# a sum of indices symmetries 