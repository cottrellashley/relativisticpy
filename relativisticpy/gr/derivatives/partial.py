# External Modules
from relativisticpy.core import EinsteinArray, einstein_convention, Indices
from relativisticpy.symengine import diff, simplify


@einstein_convention
class Derivative(EinsteinArray):

    SYMBOL = "DerivativeSymbol"
    NAME = "Derivative"

    def __init__(self, indices: Indices):
        self.indices = indices

    def __mul__(self, other: EinsteinArray) -> EinsteinArray:
        self.components = other.basis
        self.basis = other.basis
        operation = lambda a, b : diff(b, a)
        result = self.einsum_operation(other, operation)
        return EinsteinArray(components = simplify(result.components), indices = result.indices, basis = other.basis)
