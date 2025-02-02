


from abc import abstractproperty
from relativisticpy.typing.core import MultiIndexArrayType


class MetricType(MultiIndexArrayType):

    @abstractproperty
    def _(self) -> 'MetricType': pass

    @abstractproperty
    def inv(self) -> 'MetricType': pass