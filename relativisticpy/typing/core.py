"""
Type Provider. 
This only exists to inprove the type hinting experience and for simplifications of what information each class it capable of returning.
"""

from abc import ABC, abstractproperty, abstractmethod
from typing import Optional, Union, Tuple
from sympy import Symbol

class IdexType:
    """ Type for the individual Index object within Indices. """

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
    def basis(self): pass

    @abstractmethod
    def zeros_array(self): pass

    @abstractmethod
    def find(self, key: 'IdexType') -> int: pass


class IndicesType:

    @abstractproperty
    def generator(self): pass

    @abstractproperty
    def basis(self): pass

    @abstractproperty
    def indices(self): pass

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
    def einsum_product(self, other: 'IndicesType') -> 'IndicesType': pass

    @abstractmethod
    def self_product(self, other: 'IndicesType') -> 'IndicesType': pass

    @abstractmethod
    def additive_product(self, other: 'IndicesType') -> 'IndicesType': pass

class MultiIndexArrayType(ABC):

    @abstractproperty
    def components(self): pass

    @abstractproperty
    def indices(self) -> IndicesType: pass

    @abstractproperty
    def basis(self): pass

    @abstractproperty
    def dimention(self): pass
