from typing import Union

from relativisticpy.core.indices import Indices
from relativisticpy.core.multi_index_obj import MultiIndexObject
from relativisticpy.gr.metric import Metric
from relativisticpy.gr.connection import Connection

class Riemann(MultiIndexObject):

    @classmethod
    def from_metric(metric: Metric) -> 'Riemann':
        pass

    @classmethod
    def from_connection(connection: Connection) -> 'Riemann':
        pass
    
    def __init__(self, arg : Union[Metric, Connection], indices : Indices):
        super().__init__(components  =   Riemann.from_metric(arg) if isinstance(arg, Metric) else Riemann.from_connection(arg) if isinstance(arg, Connection) else ValueError(f'arg must be of types: {type(Metric)} or {type(Connection)}'), 
                         indices     =   indices,
                         basis       =   arg.basis
                        )
