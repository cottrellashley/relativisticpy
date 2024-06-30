# External Modules
from relativisticpy.algebras import Indices, EinsumArray
from relativisticpy.symengine import diff, simplify
from relativisticpy.symengine.sympy import SymbolArray


class Derivative(EinsumArray):

    def __init__(self, indices: Indices, wrt: SymbolArray):
        super().__init__(indices=indices, components=wrt)

    def __mul__(self, other: EinsumArray) -> EinsumArray:
        def operation(a, b):
            return diff(b, a)

        return self._product_copy(
            other,
            operation,
            idx_op=Indices.EINSUM_GENERATOR,
            new_type_cls=EinsumArray
        )
