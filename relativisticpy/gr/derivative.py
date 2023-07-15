# External
import sympy as smp

# Internal
from relativisticpy.core.multi_index_obj import MultiIndexObject
from relativisticpy.core.decorators import einstein_convention

@einstein_convention
class Derivative(MultiIndexObject):

    def __init__(self, components, indices):
        self.indices = indices
        self.components = components
        self.basis = components

    def __mul__(self, other : MultiIndexObject):
        operation = lambda a, b : smp.simplify(smp.diff(b, a))
        result = self.einsum_operation(other, operation)
        return MultiIndexObject(components = result.components, indices = result.indices, basis = other.basis)