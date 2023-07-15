# External
from sympy import diff, simplify

# Internal
from relativisticpy.core.multi_index_obj import MultiIndexObject
from relativisticpy.core.decorators import einstein_convention

@einstein_convention
class Derivative(MultiIndexObject):

    def __init__(self, indices):
        self.indices = indices

    def __mul__(self, other : MultiIndexObject):
        self.components = other.basis
        self.basis = other.basis
        operation = lambda a, b : simplify(diff(b, a))
        result = self.einsum_operation(other, operation)
        return MultiIndexObject(components = result.components, indices = result.indices, basis = other.basis)