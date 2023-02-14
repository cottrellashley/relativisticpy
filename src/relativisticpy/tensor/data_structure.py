from dataclasses import dataclass
from src.relativisticpy.indices.products import TensorIndicesArithmetic, TensorIndicesProduct
from src.relativisticpy.tensor.base import BaseTensor
from src.relativisticpy.tensor.context import TensorContext

@dataclass
class TensorObject(BaseTensor):
    
    context : TensorContext = None

    # Sometimes Tensor Operations will need to look inside what the tensor context is, and calculate depending on that.

    # Sometimes Tensor Operations will need to look inside what the tensor context is, and calculate depending on that.

    def __add__(self, tensor : BaseTensor):
        # Get the sympy array components of each tensor
        A = self.components
        B = tensor.components

        # Get the resulting index structure of that tensor.
        indices_product = self.indices + tensor.indices
        resulting_indices = indices_product.result

        # From the resulting index structure, get a zero components array from that indices.
        zeros = resulting_indices.get_parent_tensor_zeros()

        # Based on the indices being multiplied, we take that information and multiply components in that order.
        indices_to_multiply = TensorIndicesArithmetic(self.indices, tensor.indices)
        new = lambda i : sum([A[Indices[0]]+B[Indices[1]] for Indices in indices_to_multiply.component(i)])

        # We replace the new zero components shaped like the answer with the new calculated components.
        for i in resulting_indices:
            zeros[i] = new(i)

        # We return a new tensor object with those new properties.
        return TensorObject(
                    zeros, 
                    resulting_indices, 
                    self.basis,
                    resulting_indices.rank, 
                    self.dimention, 
                    resulting_indices.shape, 
                    context = TensorContext(prev_a = self, prev_b = tensor)
                    )

    def __sub__(self, tensor : BaseTensor):
        # Get the sympy array components of each tensor
        A = self.components
        B = tensor.components

        # Get the resulting index structure of that tensor.
        indices_product = self.indices + tensor.indices
        resulting_indices = indices_product.result

        # From the resulting index structure, get a zero components array from that indices.
        zeros = resulting_indices.get_parent_tensor_zeros()

        # Based on the indices being multiplied, we take that information and multiply components in that order.
        indices_to_multiply = TensorIndicesArithmetic(self.indices, tensor.indices)
        new = lambda i : sum([A[Indices[0]]-B[Indices[1]] for Indices in indices_to_multiply.component(i)])

        # We replace the new zero components shaped like the answer with the new calculated components.
        for i in resulting_indices:
            zeros[i] = new(i)

        # We return a new tensor object with those new properties.
        return TensorObject(
                    zeros, 
                    resulting_indices, 
                    self.basis,
                    resulting_indices.rank, 
                    self.dimention, 
                    resulting_indices.shape, 
                    context = TensorContext(prev_a = self, prev_b = tensor)
                    )

    def __mul__(self,  tensor : BaseTensor):
        # If we just have one number input, just multiply every component by that number (Sympy Arrays already handle that.)
        if isinstance(tensor, (float, int)):
            return TensorObject(
                        tensor*self.components, 
                        self.indices, 
                        self.basis,
                        self.rank, 
                        self.dimention, 
                        self.shape
                        )
        # Get the sympy array components of each tensor
        A = self.components
        B = tensor.components

        # Get the resulting index structure of that tensor.
        indices_product = self.indices * tensor.indices
        resulting_indices = indices_product.result

        # From the resulting index structure, get a zero components array from that indices.
        zeros = resulting_indices.get_parent_tensor_zeros()

        # Based on the indices being multiplied, we take that information and multiply components in that order.
        indices_to_multiply = TensorIndicesProduct(self.indices, tensor.indices)
        new = lambda i : sum([A[Indices[0]]*B[Indices[1]] for Indices in indices_to_multiply.component(i)])

        # We replace the new zero components shaped like the answer with the new calculated components.
        for i in resulting_indices:
            zeros[i] = new(i)

        # We return a new tensor object with those new properties.
        return TensorObject(
                    zeros, 
                    resulting_indices, 
                    self.basis,
                    resulting_indices.rank, 
                    self.dimention, 
                    resulting_indices.shape, 
                    context = TensorContext(prev_a = self, prev_b = tensor)
                    )

    def __rmul__(self, tensor):
        # If we just have one number input, just multiply every component by that number (Sympy Arrays already handle that.)
        if isinstance(tensor, (float, int)):
            return TensorObject(
                        tensor*self.components, 
                        self.indices, 
                        self.basis,
                        self.rank, 
                        self.dimention, 
                        self.shape
                        )
        return self * tensor

    def __truediv__(self, tensor):
        if isinstance(tensor, (float, int)):
            return TensorObject(
                        self.components/tensor, 
                        self.indices, 
                        self.basis,
                        self.rank, 
                        self.dimention, 
                        self.shape
                        )
        else:
            raise ValueError("Cannot divide with anything other than int or float.")
