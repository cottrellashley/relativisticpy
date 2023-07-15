# External Modules
from relativisticpy.core import MultiIndexObject, einstein_convention, Indices
from relativisticpy.providers import IMultiIndexArray, diff, simplify


@einstein_convention
class Derivative(MultiIndexObject):

    def __init__(self, indices: Indices):
        self.indices = indices

    def __mul__(self, other: IMultiIndexArray) -> IMultiIndexArray:
        self.components = other.basis
        self.basis = other.basis
        operation = lambda a, b : diff(b, a)
        result = self.einsum_operation(other, operation)
        return MultiIndexObject(components = simplify(result.components), indices = result.indices, basis = other.basis)