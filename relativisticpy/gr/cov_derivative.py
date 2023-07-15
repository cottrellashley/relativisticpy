from typing import Union

from relativisticpy.core.indices import Indices
from relativisticpy.core.multi_index_obj import MultiIndexObject
from relativisticpy.gr.metric import Metric
from relativisticpy.gr.connection import Connection

class CovDerivative(MultiIndexObject):

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


    def __mul__(self, other : MultiIndexObject) -> MultiIndexObject:
        self.__cov_derivative_object(other) # The number of connections in expression depends on indices stricture of other component
        pass
        
    def __cov_derivative_object(self, other: MultiIndexObject) -> MultiIndexObject:
        pass
