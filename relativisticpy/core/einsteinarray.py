# Standard Library
from typing import List, Callable
from itertools import product

# External Modules
from relativisticpy.utils import tensor_trace_product
from relativisticpy.symengine import SymbolArray, Basic

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
        scalar: Property indicating if the tensor is a scalar.
        shape: Property to get the shape of the tensor.
        dimention: Property to get the dimension of the tensor space.
        add, subtract, multiply, etc.: Methods implementing tensor operations.

    """

    def __init__(
        self,
        indices: Indices,
        components: SymbolArray = None,
        basis: SymbolArray = None,
    ):
        """
        Initializes an instance of the EinsteinArray class.

        Args:
            indices (Indices): The indices of the tensor.
            components (SymbolArray, optional): The tensor's components. Defaults to None.
            basis (SymbolArray, optional): The basis vectors for the tensor space. Defaults to None.
        """
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
        """
        Property to get the rank of the tensor.

        Returns:
            int: The rank of the tensor.
        """
        return self.indices.rank

    @property
    def scalar(self) -> bool:
        """
        Property indicating if the tensor is a scalar.

        Returns:
            bool: True if the tensor is a scalar, False otherwise.
        """
        return self.rank == (0, 0)
    
    @property
    def scalar_comp_value(self):
        """
        Property to get the scalar component value of the tensor.

        Returns:
            Union[float, SymbolArray]: The scalar component value of the tensor.
        """
        if self.scalar:
            return list(self.components)[0]
        else:
            return self.components

    @property
    def shape(self):
        """
        Property to get the shape of the tensor.

        Returns:
            Tuple[int]: The shape of the tensor.
        """
        return self.indices.shape

    @property
    def basis(self): return self.abasis

    @property
    def dimention(self):
        """
        Property to get the dimension of the tensor space.

        Returns:
            int: The dimension of the tensor space.
        """
        return len(self.basis)

    @property
    def subcomponents(self):
        """
        Property to get the subcomponents derived from the tensor components.

        Returns:
            SymbolArray: The subcomponents derived from the tensor components.
        """
        return self._subcomponents

    @subcomponents.setter
    def subcomponents(self, value: SymbolArray):
        """
        Setter for the subcomponents property.

        Args:
            value (SymbolArray): The subcomponents derived from the tensor components.
        """
        self._subcomponents = value

    @basis.setter
    def basis(self, value: SymbolArray) -> None:
        """
        Setter for the basis property.

        Args:
            value (SymbolArray): The basis vectors for the tensor space.
        """
        self.basis = value
        self.indices.basis = value

    def get_subcomponents(self, indices: Indices):
        """
        Gets the subcomponents derived from the tensor components.

        Args:
            indices (Indices): The indices of the tensor.

        Returns:
            SymbolArray: The subcomponents derived from the tensor components.
        """
        self._subcomponents = self.components[indices.__index__()]
        return self._subcomponents

    def reshape_tensor_components(self, indices: Indices):
        """
        Reshapes the tensor components based on the given indices.

        Args:
            indices (Indices): The indices of the tensor.

        Returns:
            EinsteinArray: The reshaped tensor.
        """
        reshape_tuple_order = self.indices.get_reshape(indices)
        indices.basis = self.basis
        new_components = self.rearrange_components(reshape_tuple_order)
        return type(self)(indices, new_components, indices.basis)
    
    def rearrange_components(self, new_order: List[int]):
        """
        Rearranges the components of the tensor based on the given order.

        Args:
            new_order (List[int]): The new order of the components.

        Returns:
            SymbolArray: The tensor components with rearranged order.
        """
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

    def components_operation(self, operation: Callable):
        """
        Performs an operation on the tensor components.

        Args:
            operation (Callable): The operation to be performed on the components.

        Returns:
            EinsteinArray: The tensor with the operation applied to its components.
        """
        self.components = operation(self.components)
        return self
    
    def index_operation(self, operation: Callable):
        """
        Performs an operation on the tensor indices.

        Args:
            operation (Callable): The operation to be performed on the indices.

        Returns:
            EinsteinArray: The tensor with the operation applied to its indices.
        """
        self.indices = operation(self.indices)
        return self

    def comps_contraction(self, other: "EinsteinArray", idcs: List[List[int]]):
        """
        Performs contraction of tensor components with another tensor.

        Args:
            other (EinsteinArray): The tensor to be contracted with.
            idcs (List[List[int]]): The indices to be contracted.

        Returns:
            SymbolArray: The contracted tensor components.
        """
        return tensor_trace_product(self.components, other.components, idcs)

    # Dunders
    def __post_init__(self, basis = None) -> None:
        """
        Performs post-initialization operations.

        Args:
            basis (SymbolArray, optional): The basis vectors for the tensor space. Defaults to None.
        """
        self.__set_self_summed(basis)  # After __init__ -> check and perform self-sum i.e. G_{a}^{a}_{b}_{c}

    def __neg__(self):
        """
        Negates the tensor components.

        Returns:
            EinsteinArray: The negated tensor.
        """
        self.components = -self.components
        return self

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
        if not isinstance(other, (float, int, Basic, EinsteinArray)):
            raise TypeError(f"Unsupported operand type(s) for *: 'EinsteinArray' and '{type(other).__name__}'")

        if isinstance(other, (float, int, Basic)):
            return EinsteinArray(
                components=other * self.components,
                indices=self.indices,
                basis=self.basis,
            )

        if self.scalar or other.scalar:
            return EinsteinArray(
                components=other.components * self.components,
                indices=other.indices if self.scalar else self.indices,
                basis=other.basis if self.scalar else self.basis,
            )

        operation = lambda a, b: a * b
        result = self.einsum_operation(other, operation)
        ein_array = EinsteinArray(components=result.components, indices=result.indices, basis=self.basis)
        return ein_array.scalar_comp_value if ein_array.scalar else ein_array

    def __rmul__(self, other: "EinsteinArray") -> "EinsteinArray":
        if not isinstance(other, (float, int, Basic, EinsteinArray)):
            raise TypeError(f"Unsupported operand type(s) for *: 'EinsteinArray' and '{type(other).__name__}'")
    
        if isinstance(
            other, (float, int, Basic)
        ):  # If we're number then just multiply every component by it.
            return EinsteinArray(
                components=other * self.components,
                indices=self.indices,
                basis=self.basis,
            )
        if self.scalar or other.scalar:
            return EinsteinArray(
                components=other.components * self.components,
                indices=other.indices if self.scalar else self.indices,
                basis=other.basis if self.scalar else self.basis,
            )
    
        return self * other

    def __truediv__(self, other: "EinsteinArray") -> "EinsteinArray":
        if not isinstance(other, (float, int, Basic)):
            raise TypeError(f"unsupported operand type(s) for / or __truediv__(): 'EinsteinArray' and {type(other).__name__}")
        elif other == 0:
            raise ZeroDivisionError("division by zero")
        else:
            return EinsteinArray(
                components=self.components / other,
                indices=self.indices,
                basis=self.basis,
            )
        
    def __pow__(self, other: "EinsteinArray") -> "EinsteinArray":
        if not isinstance(other, (int, float, Basic)):
            raise TypeError(f"Unsupported operand type(s) for **: 'EinsteinArray' and '{type(other).__name__}'")
        elif not self.scalar:
            raise ValueError(f"unsupported operand type(s) for ** or pow() on non-scalar EinsteinArray")
        else:
            return EinsteinArray(
                components=SymbolArray([self.scalar_comp_value ** other]),
                indices=self.indices,
                basis=self.basis,
            ).scalar_comp_value

    # Privates
    def __set_self_summed(self, basis = None) -> None:
        """
        Sets the tensor to be self-summed if necessary.

        Args:
            basis (SymbolArray, optional): The basis vectors for the tensor space. Defaults to None.
        """
        if self.indices.self_summed:
            result = self.selfsum_operation()
            self.components = result.components
            self.indices = result.indices
        else:
            pass
    