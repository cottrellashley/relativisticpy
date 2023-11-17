# Standard Library
from typing import Union

# External Modules
from relativisticpy.core import Indices, EinsteinArray, Metric

# This Module
from relativisticpy.gr.connection import Connection

class CovDerivative(EinsteinArray):

    @classmethod
    def from_metric(metric: Metric) -> 'CovDerivative':
        pass

    @classmethod
    def from_connection(connection: Connection) -> 'CovDerivative':
        pass
    
    def __init__(self, arg : Union[Metric, Connection], indices : Indices):
        super().__init__(components  =   CovDerivative.from_metric(arg) if isinstance(arg, Metric) else CovDerivative.from_connection(arg) if isinstance(arg, Connection) else ValueError(f'arg must be of types: {type(Metric)} or {type(Connection)}'), 
                         indices     =   indices,
                         basis       =   arg.basis
                        )

    def __mul__(self, other : EinsteinArray) -> EinsteinArray:
        self.__cov_derivative_object(other) # The number of connections in expression depends on indices stricture of other component
        pass
        
    def __cov_derivative_object(self, other: EinsteinArray) -> EinsteinArray:
        pass
