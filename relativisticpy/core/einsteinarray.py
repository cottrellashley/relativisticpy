# Standard Library
from typing import List

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

        if indices.anyrunnig: # Need a better solution (EinArray should not know indices implementation.)
            if basis != None:
                indices.basis = basis
                self._subcomponents = self.get_subcomponents(indices)
                self.indices = indices.get_non_running()
            else:
                raise ValueError(
                    f"Basis parameter must be provided to initialize {self} with non-running indices."
                )

    @property
    def rank(self):
        return self.indices.rank

    @property
    def scalar(self):
        return self.rank == (0, 0)

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
    def __post_init__(self) -> None:
        self.__set_self_summed()  # After __init__ -> check and perform self-sum i.e. G_{a}^{a}_{b}_{c}

    def __neg__(self):
        self.components = -self.components
        return self

    def get_subcomponents(self, indices: Indices):
        self._subcomponents = self.components[indices.__index__()]
        return self._subcomponents

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
        operation = lambda a, b: a * b
        result = self.einsum_operation(other, operation)
        return EinsteinArray(
            components=result.components, indices=result.indices, basis=self.basis
        )

    def __rmul__(self, other: "EinsteinArray") -> "EinsteinArray":
        if isinstance(
            other, (float, int)
        ):  # If we're number then just multiply every component by it.
            return EinsteinArray(
                components=other * self.components,
                indices=self.indices,
                basis=self.basis,
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
            raise ValueError("Cannot divide with anything other than int or float.")

    def comps_contraction(self, other: "EinsteinArray", idcs: List[List[int]]):
        return tensor_trace_product(self.components, other.components, idcs)

    # Privates
    def __set_self_summed(self) -> None:
        if self.indices.self_summed:
            result = self.selfsum_operation()
            self.components = result.components
            self.indices = result.indices
            self.basis = result.basis
        else:
            pass
