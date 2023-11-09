"""
Interfaces Provider. 
This only exists to inprove the type hinting experience and for simplifications of what information each class it capable of returning.
"""

from abc import ABC, abstractproperty, abstractmethod
from typing import Optional, Union, Tuple
from sympy import Symbol
from relativisticpy.utils import SymbolArray

class IIdx:

    @abstractproperty
    def symbol(self) -> Symbol: pass

    @abstractproperty
    def covariant(self) -> bool: pass

    @abstractproperty
    def values(self) -> Optional[Union[list, int]]: pass

    @abstractproperty
    def order(self) -> int: pass

    @abstractproperty
    def running(self) -> bool: pass

    @abstractproperty
    def dimention(self) -> int: pass

    @abstractproperty
    def basis(self) -> SymbolArray: pass

    @abstractmethod
    def zeros_array(self) -> SymbolArray: pass

    @abstractmethod
    def find(self, key: 'IIdx') -> int: pass


class IIndices:

    @abstractproperty
    def generator(self) -> SymbolArray: pass

    @abstractproperty
    def basis(self) -> SymbolArray: pass

    @abstractproperty
    def indices(self) -> SymbolArray: pass

    @abstractproperty
    def dimention(self) -> int: pass

    @abstractproperty
    def scalar(self) -> bool: pass

    @abstractproperty
    def shape(self) -> Tuple[int]: pass

    @abstractproperty
    def rank(self) -> Tuple[int]: pass

    @abstractproperty
    def self_summed(self) -> bool: pass

    @abstractmethod
    def einsum_product(self, other: 'IIndices') -> 'IIndices': pass

    @abstractmethod
    def self_product(self, other: 'IIndices') -> 'IIndices': pass

    @abstractmethod
    def additive_product(self, other: 'IIndices') -> 'IIndices': pass

class IMultiIndexArray(ABC):

    @abstractproperty
    def components(self) -> SymbolArray: pass

    @abstractproperty
    def indices(self) -> IIndices: pass

    @abstractproperty
    def basis(self) -> SymbolArray: pass

    @abstractproperty
    def dimention(self) -> SymbolArray: pass

class IMetric(IMultiIndexArray):

    @abstractproperty
    def _(self) -> 'IMetric': pass

    @abstractproperty
    def inv(self) -> 'IMetric': pass