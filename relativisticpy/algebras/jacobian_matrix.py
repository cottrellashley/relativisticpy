from typing import Union, Callable, List, Tuple
from itertools import product
# External Modules
from relativisticpy.symengine import SymbolArray, Basic, diff
from relativisticpy.algebras import EinsumArray, Indices, Idx

# Pattern. A lot of errors can be caused by public methods not being called with the correct arguments.
# To avoid this we implement a simple patter:
# 1. Public methods all have explicit type hints and argument names.
# 2. Public methods all have a docstring that explains what the method does.
# 3. Public methods all perform argument validation and raise exceptions if the arguments are invalid.
# 4. If all arguments are valid, the public method calls a protected method that performs the actual operation.
# 5. The protected method does not perform argument validation, as it is assumed that the public method has already done this.

class CoordinateMap:
    
    def __init__(self):
        self.valid = False

    def add_map(self, symbol: Tuple[Basic], expression: Tuple[Basic]):
        """
        Add a transformation to the coordinate map.

        Args:
            symbol (Basic): The symbol to transform.
            transformation (Callable[[Basic], Basic]): The transformation function.
        """
        pass

class Jacobian(EinsumArray):

    def __init__(self, indices: Indices, components: SymbolArray):
        """
        Initializes an instance of the EinsteinArray class.

        Args:
            indices (Indices): The indices of the tensor.
            components (SymbolArray, optional): The tensor's components. Defaults to None.
        """
        super().__init__(indices, components)

    def set_indices(self, indices: Indices): self.indices = indices
    
    @classmethod
    def from_transformations(cls, indices: Indices, transformations: List[Tuple[Basic, Basic, Basic]]):
        """
        Create a Jacobian matrix from a list of transformation functions.

        Args:
            indices (Indices): The indices of the tensor.
            transformations (list): A list of transformation functions.

        Returns:
            Jacobian: A Jacobian matrix.
        """
        components = cls.compute_jacobian(transformations)
        return cls(indices, components)
    
    @staticmethod
    def compute_jacobian(transformations: List[Tuple[Basic, Basic, Basic]]) -> SymbolArray:
        """
        Compute the Jacobian matrix from a list of transformation functions.

        Args:
            transformations List[Tuple[Basic, Basic, Basic]]: A list of transformation tuples.
            Each tuple contains the following elements in the following order:
            1. The symbol to transform from.
            2. The symbol to transform to.
            3. The transformation function in terms of the symbols to transform to.

        Returns:
            SymbolArray: The Jacobian matrix.
        """
        dim = len(transformations)
        components = SymbolArray.zeros(dim, dim)
        for i, j in product(range(dim), range(dim)):
            expr = transformations[i][2]
            wrt = transformations[j][1]
            comp = diff(expr, wrt)

            # Set the component in the Jacobian matrix
            components[i, j] = comp
        return components

