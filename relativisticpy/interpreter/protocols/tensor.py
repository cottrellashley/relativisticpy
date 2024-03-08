from typing import Protocol, List
from relativisticpy.interpreter.protocols.symbolic import SymbolArray

class Idx(Protocol):

    # Properties
    @property
    def symbol(self) -> bool:
        ...

    @property
    def dimention(self) -> int:
        ...

    @property
    def basis(self) -> SymbolArray:
        ...

class Indices(Protocol):

    # Properties
    @property
    def anyrunnig(self) -> bool:
        ...

    @property
    def indices(self) -> List[Idx]:
        ...

    @property
    def basis(self) -> List[Idx]:
        ...

    # Types of equality
    def rank_eq(self, other: 'Indices') -> bool:
        ...
    def symbol_eq(self, other: 'Indices') -> bool:
        ...
    def symbol_and_symbol_rank_eq(self, other: 'Indices') -> bool:
        ...
    def symbol_order_eq(self, other: 'Indices') -> bool:
        ...
    def symbol_order_rank_eq(self, other: 'Indices') -> bool:
        ...
    
    # Other methods
    def get_reshape(self, other: 'Indices') -> 'Indices':
        ...
    def get_non_running(self) -> 'Indices':
        ...

class TensorNode(Protocol):
    pass

class Tensor(Protocol):

    @property
    def indices(self) -> Indices:
        ...

    @property
    def basis(self) -> SymbolArray:
        ...

    @property
    def components(self) -> SymbolArray:
        ...

    @property
    def subcomponents(self) -> SymbolArray:
        "Returns the sub-components of the tensor (if user defined tensor by calling sub-component from indices.)"
        ...
    
    @classmethod
    def from_node(cls, node) -> 'Tensor':
        ...

    def reshape_tensor_components(self, indices: Indices) -> 'Tensor':
        " Re-shapes the tensor components with respect to new indices. "
        ...
