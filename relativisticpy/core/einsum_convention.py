# Standard Library
from dataclasses import dataclass
from itertools import product
from operator import itemgetter
from typing import Union

# External Modules
from relativisticpy.utils import IMultiIndexArray, transpose_list
from relativisticpy.symengine import SymbolArray

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

    def getitem(self: IMultiIndexArray, idcs: Indices):
        # This should be implemented as follows:
        # 1. If the indices cov and contravarient indices structure matches the self.indices, then just return the current components
        # 2. If the indices does not match the self.indices, we must then perform a summation with the metric tensor in order to return the components
        #    which represent the indices structure given by the input parameter
        res = self.components
        deltas = self.indices.covariance_delta(idcs) # { 'raise': [0,1], 'lower': [3] }
        
        if len(deltas) > 0:
            for delta in deltas:
                res = getattr(self.metric, delta[0])(res, delta[1])
            return res

        if isinstance(idcs, Indices):
            return self.components[idcs.__index__()]

        return self.components


    def neg(self):
        comps = -self.components
        self.components = comps
        return self

    # Monkey patch the methods onto classes which need Einstein Summation Abilitiy.
    cls.additive_operation = additive_operation
    cls.einsum_operation = einsum_operation
    cls.selfsum_operation = selfsum_operation
    cls.__setitem__ = setitem
    cls.__neg__ = neg
    cls.__getitem__ = getitem

    return cls