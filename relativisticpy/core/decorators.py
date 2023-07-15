# Standard
from dataclasses import dataclass

# External
from sympy import MutableDenseNDimArray

# Internal
from relativisticpy.core.indices import Indices

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

    cls.additive_operation = additive_operation
    cls.einsum_operation = einsum_operation
    cls.selfsum_operation = selfsum_operation
    return cls