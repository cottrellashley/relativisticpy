# External Modules
from relativisticpy.algebras import Indices, EinsumArray
from relativisticpy.diffgeom.geotensor import GrTensor
from relativisticpy.symengine import diff, simplify

class Derivative(EinsumArray):

    def __init__(self, indices: Indices, wrt):
        super().__init__(indices = indices, basis = wrt)

    def __mul__(self, other: EinsumArray) -> EinsumArray:
        self.components = other.basis
        operation = lambda a, b : diff(b, a)
        result = self.einsum_operation(other, operation)
        return type(other)(components = simplify(result.components), indices = result.indices, basis = other.basis)
