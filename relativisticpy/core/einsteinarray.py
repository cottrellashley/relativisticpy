# Standard Library
from typing import List, Callable
from itertools import product

# External Modules
from relativisticpy.utils import tensor_trace_product
from relativisticpy.deserializers import tensor_from_string
from relativisticpy.symengine import SymbolArray

# This Module
from relativisticpy.core.indices import Indices, Idx
from relativisticpy.core.einsum_convention import einstein_convention


@einstein_convention
class EinsteinArray:
    """
    A class representing arrays that operate under the Einstein summation convention.
    It supports operations like addition, subtraction, multiplication, and division,
    following the rules of tensor algebra.

    Attributes:
        indices (Indices): The indices of the tensor.
        components (SymbolArray): The tensor's components.
        basis (SymbolArray): The basis vectors for the tensor space.
        subcomponents (SymbolArray): The subcomponents derived from the tensor components.

    Methods:
        from_string: Class method to create an instance from string representations.
        scalar: Property indicating if the tensor is a scalar.
        shape: Property to get the shape of the tensor.
        dimention: Property to get the dimension of the tensor space.
        add, subtract, multiply, etc.: Methods implementing tensor operations.

    Example:
        >>> tensor = EinsteinArray.from_string("_{mu}_{nu}", "[[1, 0],[0, 1]]", "[x, y]")
        >>> print(tensor.rank)
    """

    @classmethod
    def from_string(cls, indices_str, comp_str, basis_str):
        """
        Creates an EinsteinArray object from string representations of its components.

        Args:
            indices_str (str): A string representing the indices of the tensor.
            comp_str (str): A string representing the tensor's components.
            basis_str (str): A string representing the basis vectors.

        Returns:
            EinsteinArray: An instance of EinsteinArray.

        Example:
            >>> EinsteinArray.from_string("ij", "[[1,2],[3,4]]", "basis")
        """
        return tensor_from_string(
            Idx, Indices, EinsteinArray, indices_str, comp_str, basis_str
        )

    def __init__(
        self,
        indices: Indices,
        components: SymbolArray = None,
        basis: SymbolArray = None,
    ):
        self.components = components
        self.abasis = basis
        self._subcomponents = None
        self.indices = indices

        if self.indices.basis == None: # Need a better solution (EinArray should not know indices implementation.)
            self.indices.basis = basis

        # I think this is trying to first remove an array 
        if indices.anyrunnig: # Need a better solution (EinArray should not know indices implementation.)
            if basis != None:
                indices.basis = basis
                self._subcomponents = self.get_subcomponents(indices)
                self.indices = indices.get_non_running()
            else:
                raise ValueError(
                    f"Basis parameter must be provided to initialize {self} with non-running indices."
                )
        self.__post_init__(basis)

    @property
    def rank(self):
        return self.indices.rank

    @property
    def scalar(self) -> bool:
        return self.rank == (0, 0)
    
    @property
    def scalar_comp_value(self):
        if self.scalar:
            return list(self.components)[0]
        else:
            return self.components

    @property
    def shape(self):
        return self.indices.shape

    @property
    def basis(self): return self.abasis

    @property
    def dimention(self):
        return len(self.basis)

    @property
    def subcomponents(self):
        return self._subcomponents

    @subcomponents.setter
    def subcomponents(self, value: SymbolArray):
        self._subcomponents = value

    @basis.setter
    def basis(self, value: SymbolArray) -> None:
        self.basis = value
        self.indices.basis = value

    # Dunders
    def __post_init__(self, basis = None) -> None:
        self.__set_self_summed(basis)  # After __init__ -> check and perform self-sum i.e. G_{a}^{a}_{b}_{c}

    def __neg__(self):
        self.components = -self.components
        return self

    def get_subcomponents(self, indices: Indices):
        self._subcomponents = self.components[indices.__index__()]
        return self._subcomponents

    def reshape_tensor_components(self, indices: Indices):
        reshape_tuple_order = self.indices.get_reshape(indices)
        indices.basis = self.basis
        new_components = self.rearrange_components(reshape_tuple_order)
        return type(self)(indices, new_components, indices.basis)
    
    def rearrange_components(self, new_order):
        # Determine the shape of the new array
        new_shape = [self.components.shape[i] for i in new_order]

        # Create a new array with the same data but new shape
        new_array = SymbolArray.zeros(*new_shape)

        # Iterate over each possible index in the new array
        for new_index in product(*[range(s) for s in new_shape]):
            # Map the new index to the corresponding index in the original array
            original_index = tuple(new_index[new_order.index(i)] for i in range(len(new_order)))
            # Assign the value from the original array to the new index in the new array
            new_array[new_index] = self.components[original_index]

        return new_array


    def __add__(self, other: "EinsteinArray") -> "EinsteinArray":
        operation = lambda a, b: a + b
        result = self.additive_operation(
            other, operation
        )  # Implementation inserted by decorator
        return EinsteinArray(
            components=result.components, indices=result.indices, basis=self.basis
        )

    def __sub__(self, other: "EinsteinArray") -> "EinsteinArray":
        operation = lambda a, b: a - b
        result = self.additive_operation(other, operation)
        return EinsteinArray(
            components=result.components, indices=result.indices, basis=self.basis
        )

    def __mul__(self, other: "EinsteinArray") -> "EinsteinArray":
        if isinstance(
            other, (float, int)
        ):  # If we're number then just multiply every component by it (assuming the SymbolArray implements the * method ... )
            return EinsteinArray(
                components=other * self.components,
                indices=self.indices,
                basis=self.basis,
            )
        if not other.scalar and self.scalar:
            return EinsteinArray(
                components=other.components * self.components,
                indices=other.indices,
                basis=other.basis,
            )
        if other.scalar:
            return EinsteinArray(
                components=other.components * self.components,
                indices=self.indices,
                basis=self.basis,
            )

        operation = lambda a, b: a * b
        result = self.einsum_operation(other, operation)
        ein_array = EinsteinArray(
            components=result.components, indices=result.indices, basis=self.basis
        )
        return ein_array.scalar_comp_value if ein_array.scalar else ein_array

    def __rmul__(self, other: "EinsteinArray") -> "EinsteinArray":
        if isinstance(
            other, (float, int)
        ):  # If we're number then just multiply every component by it.
            return EinsteinArray(
                components=other * self.components,
                indices=self.indices,
                basis=self.basis,
            )
        if other.scalar and not self.scalar:
            return EinsteinArray(
                components=other.components * self.components,
                indices=self.indices,
                basis=self.basis,
            )
        if not other.scalar and self.scalar:
            return EinsteinArray(
                components=other.components * self.components,
                indices=other.indices,
                basis=other.basis,
            )
        return self * other

    def __truediv__(self, other: "EinsteinArray") -> "EinsteinArray":
        if isinstance(
            other, (float, int)
        ):  # If we're number then just divide every component by it.
            return EinsteinArray(
                components=self.components / other,
                indices=self.indices,
                basis=self.basis,
            )
        else:
            raise ValueError(f"unsupported operand type(s) for / or __truediv__(): 'EinsteinArray' and {type(other)}")
        
    def __pow__(self, other: "EinsteinArray") -> "EinsteinArray":
        if isinstance(
            other, (float, int)
        ) and self.scalar:  # If we're number then just divide every component by it.
            return EinsteinArray(
                components=SymbolArray([self.scalar_comp_value ** other]),
                indices=self.indices,
                basis=self.basis,
            ).scalar_comp_value
        else:
            raise ValueError(f"unsupported operand type(s) for ** or pow() on non-scalar EinsteinArray and '{type(other)}'")
        
    def components_operation(self, operation: Callable):
        self.components = operation(self.components)
        return self
    
    def index_operation(self, operation: Callable):
        self.indices = operation(self.indices)
        return self

    def comps_contraction(self, other: "EinsteinArray", idcs: List[List[int]]):
        return tensor_trace_product(self.components, other.components, idcs)

    # Privates
    def __set_self_summed(self, basis = None) -> None:
        if self.indices.self_summed:
            result = self.selfsum_operation()
            self.components = result.components
            self.indices = result.indices
        else:
            pass
