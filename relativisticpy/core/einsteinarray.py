# Standard Library
from typing import List, Callable, Optional, Union
from itertools import product

# External Modules
from relativisticpy.utils import tensor_trace_product
from relativisticpy.symengine import SymbolArray, Basic

# This Module
from relativisticpy.core.indices import Indices, Idx
from relativisticpy.core.metric import Metric
from relativisticpy.core.einsum_array import _EinsumArray

class EinsteinArray(_EinsumArray):
    """
    A class representing arrays that operate under the Einstein summation convention.
    It supports operations like addition, subtraction, multiplication, and division,
    following the rules of tensor algebra. Additionally, it opperates under metric spaces, 
    meaning it we can raise and lower indices by either defining a metric or passing one as parameter.
    """

    def __init__(
        self,
        indices: Indices,
        components: SymbolArray,
        basis: SymbolArray = None,
        metric: Metric = None
    ):
        """
        Initializes an instance of the EinsteinArray class.

        Args:
            indices (Indices): The indices of the tensor.
            components (SymbolArray, optional): The tensor's components. Defaults to None.
            basis (SymbolArray, optional): The basis vectors for the tensor space. Defaults to None.
        """
        super().__init__(indices, components)
        self.__basis = basis
        self._subcomponents = None
        self.__metric = metric

        # I think this is trying to first remove an array 
        if indices.anyrunnig: # Need a better solution (EinArray should not know indices implementation.)
            self._subcomponents = self.get_subcomponents(indices)
            self.indices = indices.get_non_running()

        self.__post_init__()
    
    def __post_init__(self) -> None:
        """
        Performs post-initialization operations.

        Args:
            basis (SymbolArray, optional): The basis vectors for the tensor space. Defaults to None.
        """
        if self.indices.self_summed:
            result = self.trace()
            self.components = result.components
            self.indices = result.indices
        else:
            pass # After __init__ -> check and perform self-sum i.e. G_{a}^{a}_{b}_{c}

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
    def basis(self): return self.__basis

    @property
    def subcomponents(self):
        """
        Property to get the subcomponents derived from the tensor components.

        Returns:
            SymbolArray: The subcomponents derived from the tensor components.
        """
        return self._subcomponents or self.components

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
    
    @property
    def metric(self) -> Metric: return self.__metric or Metric.default(self.dimention)

    @metric.setter
    def metric(self, value: 'EinsteinArray') -> None: self.__metric = value

    def get_subcomponents(self, indices: Indices):
        """
        Gets the subcomponents derived from the tensor components.

        Args:
            indices (Indices): The indices of the tensor.

        Returns:
            SymbolArray: The subcomponents derived from the tensor components.
        """
        if not isinstance(indices, Indices):
            raise TypeError(f"Expected Indices, got {type(indices).__name__}")
        if not self.indices.symbol_covariance_eq(indices):
            raise ValueError(f"New indices must have same symbol and covariance as old indices, but in any different order.")
    
        self._subcomponents = self.components[indices.__index__()]
        return self._subcomponents

    def reshape(self, new_indices: Indices):
        """
        Reshapes the tensor components based on the new given indices. 
        New indices must have same symbol and covariance as old indices, but in any different order.

        Parameters
        ----------
        indices : Indices
            The indices of the tensor.

        Returns
        -------
        EinsteinArray
            The tensor with re-shaped components and new indices object, re-shaped from old indices.

        Raises
        ------
        TypeError
            when indices arg not of Indices type.
        ValueError
            when indices arg does not have same symbol and covariance as current indices.
        """
        if not isinstance(new_indices, Indices):
            raise TypeError(f"Expected Indices, got {type(new_indices).__name__}")
        if not self.indices.symbol_covariance_eq(new_indices):
            raise ValueError(f"New indices must have same symbol and covariance as old indices, but in any different order.")

        reshape_tuple_order = self.indices.get_reshape(new_indices)
        new_indices.basis = self.basis
        new_components = self.reshape_components(reshape_tuple_order)
        return type(self)(new_indices, new_components, self.basis)
    
    def tensor_from_new_indices(self, indices: Indices, metric: Metric = None):
        """
        Returns new .

        Parameters
        ----------
        indices : Indices
            The indices of the tensor.

        Returns
        -------
        EinsteinArray
            The tensor with re-shaped components and new indices object, re-shaped from old indices.

        Raises
        ------
        TypeError
            when indices arg not of Indices type.
        ValueError
            when indices arg does not have same symbol and covariance as current indices.
        """
        if not isinstance(indices, Indices):
            raise TypeError(f"Expected Indices, got {type(indices).__name__}")
        if len(self.indices.indices) != len(indices.indices):
            raise ValueError(f"New indices must have same number of indices as current indices.")
        if metric:
            self.__metric = metric
        if not self.metric:
            raise ValueError(f"Cannot perform operation on EinsteinArray objects without Metric defined.")

        for index in indices:
            if index.covariant:
                result *= type(self.metric)(self.indices, self.components, self.basis)
        reshape_tuple_order = self.indices.get_reshape(indices)
        indices.basis = self.basis
        new_components = self.reshape_components(reshape_tuple_order)
        return type(self)(indices, new_components, indices.basis)

    def __neg__(self):
        """
        Negates the tensor components.

        Returns:
            EinsteinArray: The negated tensor.
        """
        self.components = -self.components
        return self

    def __add__(self, other: "EinsteinArray") -> "EinsteinArray":
        if not isinstance(other, _EinsumArray):
            raise TypeError(f"Unsupported operand type(s) for +: 'EinsteinArray' and '{type(other).__name__}'")
        if self.indices.is_einsum_product(other.indices):
            raise ValueError(f"Cannot perform addition of EinsteinArray's with the indices: {str(self.indices)} and {str(other.indices)} ")
    
        product = self.prod(other, lambda a, b: a + b)
        return EinsteinArray(
            indices=product.indices, components=product.components, basis=self.basis
        )

    def __sub__(self, other: "EinsteinArray") -> "EinsteinArray":
        if not isinstance(other, _EinsumArray):
            raise TypeError(f"Unsupported operand type(s) for -: 'EinsteinArray' and '{type(other).__name__}'")
        if self.indices.is_einsum_product(other.indices):
            raise ValueError(f"Cannot perform addition of EinsteinArray's with the indices: {str(self.indices)} and {str(other.indices)} ")
    
        product = self.prod(other, lambda a, b: a - b)
        return EinsteinArray(
            indices=product.indices, components=product.components, basis=self.basis
        )

    def __mul__(self, other: "EinsteinArray") -> "EinsteinArray":
        if not isinstance(other, (float, int, Basic, EinsteinArray)):
            raise TypeError(f"Unsupported operand type(s) for *: 'EinsteinArray' and '{type(other).__name__}'")
        if not self.indices.is_einsum_product(other.indices):
            raise ValueError(f"Cannot perform addition of EinsteinArray's with the indices: {str(self.indices)} and {str(other.indices)} ")

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

        product = self.prod(other, lambda a, b: a * b)
        ein_array = EinsteinArray(
            indices=product.indices, components=product.components, basis=self.basis
        )
        return ein_array.scalar_comp_value if ein_array.scalar else ein_array

    def __rmul__(self, other: "EinsteinArray") -> "EinsteinArray":
        if not isinstance(other, (float, int, Basic, EinsteinArray)):
            raise TypeError(f"Unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'")
        if not self.indices.is_einsum_product(other.indices):
            raise ValueError(f"Cannot perform addition of EinsteinArray's with the indices: {str(self.indices)} and {str(other.indices)} ")
    
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
