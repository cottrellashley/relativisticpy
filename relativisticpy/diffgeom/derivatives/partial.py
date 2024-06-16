# External Modules
from relativisticpy.algebras import Indices, EinsumArray
from relativisticpy.symengine import diff, simplify

class Derivative(EinsumArray):

    def __init__(self, indices: Indices, wrt):
        super().__init__(indices = indices, components = wrt)

    def __mul__(self, other: EinsumArray) -> EinsumArray:
        operation = lambda a, b : diff(b, a)
        result = self._product_copy(
                                        other, 
                                        operation,
                                        idx_op = Indices.EINSUM_GENERATOR,
                                        new_type_cls = type(other)
                                    )
        result.components = simplify(result.components)
        
        return type(result)(*result.args)
