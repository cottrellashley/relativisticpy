from asyncio import constants
from dataclasses import dataclass
from src.relativisticpy.indices.products import *
from src.relativisticpy.base_tensor.base import BaseTensor
from src.relativisticpy.base_tensor.context import TensorContext

@dataclass
class TensorObject(BaseTensor):
    
    context : TensorContext = None

    def __neg__(self):
        return TensorObject(
            components  =-self.components,
            indices     =self.indices,
            basis       =self.basis,
            rank        = self.rank,
            dimention   =self.dimention,
            shape       =self.shape,
            scalar=self.scalar,
            context=self.context
        )

    def __post_init__(self):
        self.set_metric_context()
        self.set_self_summed()
    
    def set_self_summed(self):
        if self.indices.self_summed:
            contex = self.indices.get_self_summed_contex()

            # Get the resulting index structure of that tensor.
            resulting_indices = contex.result

            # From the resulting index structure, get a zero components array from that indices.
            zeros = resulting_indices.parent_tensor_as_zeros()

            pre_summed_components = self.components 

            # Based on the indices being multiplied, we take that information and multiply components in that order.
            indices_to_multiply = TensorSelfSummed(self.indices)
            new = lambda i : sum([pre_summed_components[Indices] for Indices in indices_to_multiply.component(i)])

            # We replace the new zero components shaped like the answer with the new calculated components.
            for i in resulting_indices:
                zeros[i] = new(i)

            # We return a new tensor object with those new properties.
            self.components = zeros
            self.indices = resulting_indices

        else:
            pass


    def set_metric_context(self):
        if self.context:
            if self.context.is_metric_tensor:
                for i in self.indices.indices:
                    i.metric_parent = True

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
        zeros = resulting_indices.parent_tensor_as_zeros()

        # Based on the indices being multiplied, we take that information and multiply components in that order.
        indices_to_multiply = TensorIndicesArithmetic(self.indices, tensor.indices)
        # This is the source of the error: 
        # Say we are subtracting: ((1,0),(0,1)) 
        # Then we are essentially doing 
        # Result[(1,0)] = T1[(1,0)] - T2[(0,1)] 
        # but if we had inverted the inputs and inputted the reverse instead
        # Result[(1,0)] = T2[(1,0)] - T1[(0,1)] 
        # this same component in the above woule have been T2[(0,1)] - T1[(1,0)] 
        # which in the new order, in the resulting components is now at Result[(0,1)] 
        new = lambda i : sum([A[idx_A]+B[idx_B] for idx_A, idx_B in indices_to_multiply.component(i)])

        # We replace the new zero components shaped like the answer with the new calculated components.
        for i in resulting_indices:
            zeros[i] = new(i)

        # We return a new tensor object with those new properties.
        return TensorObject(
                                components  = zeros, 
                                indices     = resulting_indices, 
                                basis       = self.basis,
                                rank        = resulting_indices.rank, 
                                dimention   = self.dimention, 
                                shape       = resulting_indices.shape,
                                scalar      = self.scalar,
                                context     = TensorContext(prev_a = self, prev_b = tensor)
                            )

    def __sub__(self, tensor : BaseTensor):
        # Get the sympy array components of each tensor
        A = self.components
        B = tensor.components

        # Get the resulting index structure of that tensor.
        indices_product = tensor.indices + self.indices 
        resulting_indices = indices_product.result

        # From the resulting index structure, get a zero components array from that indices.
        zeros = resulting_indices.parent_tensor_as_zeros()

        # Based on the indices being multiplied, we take that information and multiply components in that order.
        indices_to_multiply = TensorIndicesArithmetic(tensor.indices, self.indices)
        new = lambda i : sum([B[idx_A]-A[idx_B] for idx_A, idx_B in indices_to_multiply.component(i)])

        # We replace the new zero components shaped like the answer with the new calculated components.
        for i in resulting_indices:
            zeros[i] = new(i)

        # We return a new tensor object with those new properties.
        return TensorObject(
                                components  = zeros, 
                                indices     = resulting_indices, 
                                basis       = self.basis,
                                rank        = resulting_indices.rank, 
                                dimention   = self.dimention, 
                                shape       = resulting_indices.shape,
                                scalar      = self.scalar,
                                context     = TensorContext(prev_a = self, prev_b = tensor)
                            )

    def __mul__(self,  tensor : BaseTensor):
        # If we just have one number input, just multiply every component by that number (Sympy Arrays already handle that.)
        if isinstance(tensor, (float, int)):
            return TensorObject(
                                    components  = tensor*self.components, 
                                    indices     = self.indices, 
                                    basis       = self.basis,
                                    rank        = self.rank, 
                                    dimention   = self.dimention, 
                                    shape       = self.shape,
                                    scalar      = self.scalar
                                )
        # Get the sympy array components of each tensor
        A = self.components
        B = tensor.components

        # Get the resulting index structure of that tensor.
        indices_product = self.indices * tensor.indices
        resulting_indices = indices_product.result

        # From the resulting index structure, get a zero components array from that indices.
        zeros = resulting_indices.parent_tensor_as_zeros()

        # Based on the indices being multiplied, we take that information and multiply components in that order.
        indices_to_multiply = TensorIndicesProduct(self.indices, tensor.indices)
        new = lambda i : sum([A[Indices[0]]*B[Indices[1]] for Indices in indices_to_multiply.component(i)])

        # We replace the new zero components shaped like the answer with the new calculated components.
        for i in resulting_indices:
            zeros[i] = new(i)

        # We return a new tensor object with those new properties.
        return TensorObject(
                                components  = zeros, 
                                indices     = resulting_indices, 
                                basis       = self.basis,
                                scalar      = self.scalar,
                                rank        = resulting_indices.rank, 
                                dimention   = self.dimention, 
                                shape       = resulting_indices.shape, 
                                context     = TensorContext(prev_a = self, prev_b = tensor)
                            )

    def __rmul__(self, tensor):
        # If we just have one number input, just multiply every component by that number (Sympy Arrays already handle that.)
        if isinstance(tensor, (float, int)):
            return TensorObject(
                                    components  = tensor*self.components, 
                                    indices     = self.indices, 
                                    basis       = self.basis,
                                    rank        = self.rank, 
                                    dimention   = self.dimention, 
                                    shape       = self.shape,
                                    scalar      = self.scalar
                                )
        return self * tensor

    def __truediv__(self, tensor):
        if isinstance(tensor, (float, int)):
            return TensorObject(
                                    components = self.components/tensor, 
                                    indices = self.indices, 
                                    basis = self.basis,
                                    rank = self.rank, 
                                    dimention = self.dimention, 
                                    shape = self.shape,
                                    scalar = self.scalar
                                )
        else:
            raise ValueError("Cannot divide with anything other than int or float.")
