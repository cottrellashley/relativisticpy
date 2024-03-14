# Standard Library
from typing import Union

# External Modules
from relativisticpy.core import Indices, EinsteinArray, Metric

# This Module
from relativisticpy.gr.connection import Connection
from relativisticpy.symengine import diff, simplify
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

        total = pdiff
        for index in other.indices.indices:
            if index.covariant:
                total = total - EinsteinArray(
                                        components=tensor_trace_product(
                                            self.connection_components,
                                            other.components,
                                            [[0, index.order]],
                                        ),
                                        indices=result.indices,
                                        basis=self.basis,
                                    )
            else:
                total += EinsteinArray(
                                        components=tensor_trace_product(
                                            self.connection_components,
                                            other.components,
                                            [[2, index.order]],
                                        ),
                                        indices=result.indices,
                                        basis=self.basis,
                                    )
        
        # Method 2 where we use the einsum operation to compute the covariant derivative
        # total2 = pdiff
        # for index in other.indices.indices:
        #     if index.covariant:
        #         total2 -= EinsteinArray(
        #                                 components=self.connection_components,
        #                                 indices=result.indices,
        #                                 basis=self.basis,
        #                             ) * EinsteinArray(
        #                                 components=other.components,
        #                                 indices=result.indices,
        #                                 basis=self.basis,
        #                             )
        #     else:
        #         total += EinsteinArray(
        #                                components=self.connection_components,
        #                                 indices=result.indices,
        #                                 basis=self.basis,
        #                             ) * EinsteinArray(
        #                                 components=other.components,
        #                                 indices=result.indices,
        #                                 basis=self.basis,
        #                             )

        return total
