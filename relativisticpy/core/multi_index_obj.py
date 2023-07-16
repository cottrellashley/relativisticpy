# External Modules
from relativisticpy.providers import SymbolArray, IMultiIndexArray, Sympify

# This Module
from relativisticpy.core.indices import Indices
from relativisticpy.core.decorators import einstein_convention

@einstein_convention
class MultiIndexObject(IMultiIndexArray):
    __slots__ = "components", "indices", "basis"
    default_comp = 'd'
    _cls_idcs = Indices

    def __init__(self, indices: Indices, components: SymbolArray = None, basis: SymbolArray = None):
        self.components = components
        self.indices = indices
        self.basis = basis
        if self.indices.basis == None:
            self.indices.basis = basis
    
    @property
    def rank(self): return self.indices.rank
    @property
    def scalar(self): return self.rank == (0,0)
    @property
    def shape(self): return self.indices.shape
    @property
    def dimention(self): return len(self.basis)

    # Dunders
    def __post_init__(self) -> None: self.__set_self_summed() # After __init__ -> check and perform self-sum i.e. G_{a}^{a}_{b}_{c}
    def __getitem__(self, indices: Indices): return self.components[indices.__index__()]
    def __neg__(self): return MultiIndexObject(self.indices, -self.components, self.basis)

    def __add__(self, other: IMultiIndexArray) -> IMultiIndexArray:
        operation = lambda a, b : a + b
        result = self.additive_operation(other, operation) # Implementation inserted by decorator
        return MultiIndexObject(components = result.components, indices = result.indices, basis = self.basis)

    def __sub__(self, other: IMultiIndexArray) -> IMultiIndexArray:
        operation = lambda a, b : a - b
        result = self.additive_operation(other, operation)
        return MultiIndexObject(components = result.components, indices = result.indices, basis = self.basis)

    def __mul__(self, other: IMultiIndexArray) -> IMultiIndexArray:
        if isinstance(other, (float, int)): # If we're number then just multiply every component by it.
            return MultiIndexObject(components = other*self.components, indices = self.indices, basis = self.basis)
        operation = lambda a, b : a * b
        result = self.einsum_operation(other, operation)
        return MultiIndexObject(components = result.components, indices = result.indices, basis = self.basis)

    def __rmul__(self, other: IMultiIndexArray) -> IMultiIndexArray:
        if isinstance(other, (float, int)): # If we're number then just multiply every component by it.
            return MultiIndexObject(components = other*self.components, indices = self.indices, basis = self.basis)
        return self * other

    def __truediv__(self, other: IMultiIndexArray) -> IMultiIndexArray:
        if isinstance(other, (float, int)): # If we're number then just divide every component by it.
            return MultiIndexObject(components = self.components/other, indices = self.indices, basis = self.basis)
        else:
            raise ValueError("Cannot divide with anything other than int or float.")

    def coordinate_transformation(self, transformation):
        pass

    @classmethod
    def from_string(cls, indices, components, basis):
        return cls(indices = cls._cls_idcs.from_string(indices), components = Sympify(components), basis = Sympify(basis))


    # Privates
    def __set_self_summed(self) -> None:
        if self.indices.self_summed:
            result = self.selfsum_operation()
            self.components = result.components
            self.indices = result.indices
        else:
            pass
