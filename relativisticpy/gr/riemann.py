# Standard Library
from typing import Union

# External Modules
from relativisticpy.core import Indices, MultiIndexObject

# This Module
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
        super().__init__(
                         components  =   Riemann.from_metric(arg) if isinstance(arg, Metric) else Riemann.from_connection(arg) if isinstance(arg, Connection) else ValueError(f'arg must be of types: {type(Metric)} or {type(Connection)}'), 
                         indices     =   indices,
                         basis       =   arg.basis
                        )
