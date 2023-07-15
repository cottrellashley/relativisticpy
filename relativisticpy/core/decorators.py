# Standard Library
from dataclasses import dataclass
from itertools import product
from operator import itemgetter
from typing import Union

# External Modules
from relativisticpy.providers import SymbolArray, IMultiIndexArray, transpose_list

# This Module
from relativisticpy.core.indices import Indices

@dataclass
class TensorProduct:
    components: SymbolArray
    indices: Indices

def einstein_convention(cls: IMultiIndexArray):
    """Class decorator. Injects the einstein summation convention implementation into class."""

    def additive_operation(self: IMultiIndexArray, tensor: IMultiIndexArray, operation) -> TensorProduct:
        A = self.components
        B = tensor.components
        resulting_indices = self.indices.additive_product(tensor.indices)
        zeros = resulting_indices.zeros_array()
        for i in resulting_indices:
            zeros[i] = sum([operation(A[idx_A], B[idx_B]) for idx_A, idx_B in resulting_indices.generator(i)])
        return TensorProduct(components=zeros, indices=resulting_indices)

    def einsum_operation(self: IMultiIndexArray, tensor: IMultiIndexArray, operation) -> TensorProduct:
        A = self.components
        B = tensor.components
        resulting_indices = self.indices.einsum_product(tensor.indices)
        zeros = resulting_indices.zeros_array()
        for i in resulting_indices:
            zeros[i] = sum([operation(A[idx_A], B[idx_B]) for idx_A, idx_B in resulting_indices.generator(i)])
        return TensorProduct(components=zeros, indices=resulting_indices)

    def selfsum_operation(self: IMultiIndexArray) -> TensorProduct:
        resulting_indices = self.indices.self_product()
        zeros = resulting_indices.zeros_array()
        for i in resulting_indices:
            zeros[i] = sum([self.components[Indices] for Indices in resulting_indices.generator(i)])
        return TensorProduct(components=zeros, indices=resulting_indices)

    def setitem(self: IMultiIndexArray, indices: Indices, other_expr: Union[IMultiIndexArray, SymbolArray]):
        if issubclass(type(self), type(other_expr)):
            print('hello there')
            summed_index_locations = transpose_list(indices._get_all_repeated_locations(other_expr.indices))
            all = [(IndexA, IndexB) for (IndexA, IndexB) in list(product(indices, other_expr.indices)) if itemgetter(*summed_index_locations[0])(IndexA) == itemgetter(*summed_index_locations[1])(IndexB)]
            comp = SymbolArray(self.components) if not isinstance(self.components, SymbolArray) else self.components
            for idxa, idxb  in all:
                comp[idxa] = other_expr.components[idxb]
            self.components = comp
            self.basis = other_expr.basis
            self.indices = indices
        elif isinstance(other_expr, SymbolArray) and other_expr.shape == self.components.shape:
            comp = SymbolArray(self.components) if not isinstance(self.components, SymbolArray) else self.components
            for idxa in self.indices:
                comp[idxa] = other_expr[idxa]
            self.components = comp
            self.basis = self.basis
            self.indices = indices
        else:
            raise ValueError(f'The object you are trying to set and/or map to the {self} has the a shape which does not match {self.shape}.')

    def getitem(self, indices: Indices): return self.components[indices.__index__()]
    def neg(self): self.components = -self.components

    cls.additive_operation = additive_operation
    cls.einsum_operation = einsum_operation
    cls.selfsum_operation = selfsum_operation
    cls.__setitem__ = setitem
    cls.__neg__ = neg
    cls.__getitem__ = getitem
    return cls