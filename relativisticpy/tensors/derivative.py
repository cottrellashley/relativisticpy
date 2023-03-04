import sympy as smp

from relativisticpy.base_tensor.gr_tensor import GrTensor



class Derivative(GrTensor):

    # def __init__(self, tensor : GrTensor):
    #     self.tensor = tensor

    def Derivative(self):
        A = self.parent_tensor_as_zeros()
        for j in range(self.tensor.dimention):
            for i in self.tensor.indices:
                A[list(i) + list([j])] = smp.diff(self.tensor.components[i],self.tensor.basis[j])
        return smp.simplify(A)

    def parent_tensor_as_zeros(self):
        new = list(self.tensor.shape) + [int(self.tensor.dimention)]
        return smp.MutableDenseNDimArray.zeros(*new)

    def __mul__(self,tensor : GrTensor):
        self.tensor = tensor
        indices_context = self.indices * self.tensor.indices 
        result_indices = indices_context.result

        A = self.parent_tensor_as_zeros()
        for j in range(self.tensor.dimention):
            for i in self.tensor.indices:
                A[list([j]) + list(i)] = smp.diff(self.tensor.components[i],self.tensor.basis[j])
        return GrTensor(smp.simplify(A), result_indices, self.basis)