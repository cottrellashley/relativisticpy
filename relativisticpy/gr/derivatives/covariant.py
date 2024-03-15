# Standard Library
from typing import Union
from itertools import product

# External Modules
from relativisticpy.core import Indices, Idx, EinsteinArray, Metric

# This Module
from relativisticpy.gr.connection import Connection
from relativisticpy.symengine import diff, simplify, trigsimp
from relativisticpy.utils.helpers import tensor_trace_product


class CovDerivative(EinsteinArray):
    """
    Class for the covariant derivative of a tensor. It is a subclass of Einstein.
    """

    @classmethod
    def from_metric(metric: Metric) -> "CovDerivative":
        pass

    @classmethod
    def from_connection(connection: Connection) -> "CovDerivative":
        pass

    def __init__(self, indices: Indices, metric: Metric):
        super().__init__(indices=indices, basis=metric.basis)
        self.connection_components = Connection.init_from_metric(metric).components

    def __mul__(self, other: EinsteinArray) -> EinsteinArray:
        self.components = other.basis
        operation = lambda a, b: diff(b, a)
        result = self.einsum_operation(other, operation)
        pdiff = EinsteinArray(
            components=simplify(result.components),
            indices=result.indices,
            basis=other.basis,
        )
        
        # Method 2 where we use the einsum operation to compute the covariant derivative
        total2 = pdiff
        for index in other.indices.indices:
            cd_idx = self.indices.indices[0].symbol
            if index.covariant:
                connection_indices = Indices(-Idx('dummy_index'), Idx(cd_idx), Idx(index.symbol))
                connection_indices.basis = self.basis
                operand_indices = other.indices.replace(old=index, new=Idx('dummy_index'))
                total2 = total2 - EinsteinArray(
                                        components=self.connection_components,
                                        indices=connection_indices,
                                        basis=self.basis,
                                    ) * EinsteinArray(
                                        components=other.components,
                                        indices=operand_indices,
                                        basis=self.basis,
                                    )
            else:
                connection_indices = Indices(-Idx(index.symbol), Idx(cd_idx), Idx('dummy_index'))
                connection_indices.basis = self.basis
                operand_indices = other.indices.replace(old=index, new=-Idx('dummy_index'))
                total2 += EinsteinArray(
                                       components=self.connection_components,
                                        indices=connection_indices,
                                        basis=self.basis,
                                    ) * EinsteinArray(
                                        components=other.components,
                                        indices=operand_indices,
                                        basis=self.basis,
                                    )
                
        zeros = total2.indices.zeros_array()
        for i in product(*[range(i) for i in total2.components.shape]):
            obj = total2.components[i]
            if obj == 0:
                continue

            simplified = trigsimp(obj)
            zeros[i] = simplify(simplified)

        total2.components = zeros

        return total2
