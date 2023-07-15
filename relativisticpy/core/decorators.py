# Standard
from dataclasses import dataclass
from itertools import product
from typing import Union
from operator import itemgetter

# External
from sympy import MutableDenseNDimArray

# Internal
from relativisticpy.core.indices import Indices
from relativisticpy.core.helpers import transpose_list

@dataclass
class TensorProduct:
    components: MutableDenseNDimArray
    indices: Indices

def einstein_convention(cls):
    """Class decorator. Injects the einstein summation convention implementation into class."""

    def additive_operation(self, tensor, operation):
        A = self.components
        B = tensor.components
        resulting_indices = self.indices.additive_product(tensor.indices)
        zeros = resulting_indices.zeros_array()
        for i in resulting_indices:
            zeros[i] = sum([operation(A[idx_A], B[idx_B]) for idx_A, idx_B in resulting_indices.generator(i)])
        return TensorProduct(components=zeros, indices=resulting_indices)

    def einsum_operation(self, tensor, operation):
        A = self.components
        B = tensor.components
        resulting_indices = self.indices.einsum_product(tensor.indices)
        zeros = resulting_indices.zeros_array()
        for i in resulting_indices:
            zeros[i] = sum([operation(A[idx_A], B[idx_B]) for idx_A, idx_B in resulting_indices.generator(i)])
        return TensorProduct(components=zeros, indices=resulting_indices)

    def selfsum_operation(self):
        resulting_indices = self.indices.self_product()
        zeros = resulting_indices.zeros_array()
        for i in resulting_indices:
            zeros[i] = sum([self.components[Indices] for Indices in resulting_indices.generator(i)])
        return TensorProduct(components=zeros, indices=resulting_indices)

    # Dunders
    def setitem(self, indices: Indices, other_expr: Union[cls, MutableDenseNDimArray]):
        print('hello')
        if isinstance(other_expr, cls) or issubclass(cls, type(other_expr)) or issubclass(type(other_expr), cls):
            summed_index_locations = transpose_list(indices._get_all_repeated_locations(other_expr.indices))
            all = [(IndexA, IndexB) for (IndexA, IndexB) in list(product(indices, other_expr.indices)) if itemgetter(*summed_index_locations[0])(IndexA) == itemgetter(*summed_index_locations[1])(IndexB)]
            for idxa, idxb  in all:
                MutableDenseNDimArray(self.components)[idxa] = other_expr.components[idxb]
            return cls(self.components, indices, other_expr.basis)
        elif isinstance(other_expr, cls) and other_expr.shape == self.components.shape:
            for idxa in self.indices:
                self.components[idxa] = other_expr[idxa]
            return cls(self.components, indices, self.basis)
        else:
            raise ValueError(f'The object you are trying to set and/or map to the {self} has the a shape which does not match {self.shape}.')

    def getitem(self, indices: Indices): return self.components[indices.__index__()]
    def neg(self): self.components = -self.components; return self

    cls.additive_operation = additive_operation
    cls.einsum_operation = einsum_operation
    cls.selfsum_operation = selfsum_operation
    cls.__setitem__ = setitem
    cls.__neg__ = neg
    cls.__getitem__ = getitem
    return cls