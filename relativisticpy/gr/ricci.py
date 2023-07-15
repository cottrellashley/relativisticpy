from typing import Union

from relativisticpy.core.indices import Indices
from relativisticpy.core.multi_index_obj import MultiIndexObject
from relativisticpy.gr.metric import Metric
from relativisticpy.gr.connection import Connection
from relativisticpy.gr.riemann import Riemann

class Ricci(MultiIndexObject):

    @classmethod
    def from_metric(metric: Metric) -> 'Ricci':
        pass

    @classmethod
    def from_connection(connection: Connection) -> 'Ricci':
        pass

    @classmethod
    def from_riemann(riemann: Riemann) -> 'Ricci':
        pass
    
    def __init__(self, arg : Union[Metric, Connection], indices : Indices):
        __comps = Ricci.from_metric(arg) if isinstance(arg, Metric) else Ricci.from_connection(arg) if isinstance(arg, Connection) else Ricci.from_riemann(arg) if isinstance(arg, Riemann) else ValueError(f'arg must be of types: {type(Metric)} or {type(Connection)}')
        super().__init__(components = __comps, indices = indices, basis = arg.basis)
