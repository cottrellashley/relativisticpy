from typing import Union
# External Modules
from relativisticpy.diffgeom.manifold import CoordIndices
from relativisticpy.symengine import SymbolArray, Basic, simplify, trigsimp
from relativisticpy.algebras import EinsumArray, Indices, Idx
from relativisticpy.algebras.jacobian_matrix import Jacobian


# Pattern. A lot of errors can be caused by public methods not being called with the correct arguments.
# To avoid this we implement a simple patter:
# 1. Public methods all have explicit type hints and argument names.
# 2. Public methods all have a docstring that explains what the method does.
# 3. Public methods all perform argument validation and raise exceptions if the arguments are invalid.
# 4. If all arguments are valid, the public method calls a protected method that performs the actual operation.
# 5. The protected method does not perform argument validation, as it is assumed that the public method has already done this.


class Tensor(EinsumArray):
    """
        Tensor algebra is a mathematical framework that generalizes scalars, vectors, and matrices to higher-dimensional structures. This algebra deals with objects called tensors, which can have multiple indices, and the operations that can be applied to them. Here's a concise summary of the fundamental rules and operations of tensor algebra:

        **1. Tensors**

        - **Definition**: A tensor is a mathematical object that can be characterized by its order (or rank), which indicates the number of indices required to uniquely specify each of its components. For example:
            - A scalar is a tensor of rank 0 (no indices).
            - A vector is a tensor of rank 1 (one index).
            - A matrix is a tensor of rank 2 (two indices), and so on.
        - **Components**: The elements of a tensor can be scalars from any field, typically real or complex numbers, which are indexed according to the tensor's rank.

        **2. Tensor Operations**

        - **Addition and Subtraction**: Tensors of the same rank and shape can be added or subtracted component-wise.
        - **Scalar Multiplication**: Each component of a tensor can be multiplied by a scalar.
        - **Tensor Product (Outer Product)**: Given two tensors, their tensor product is a new tensor whose rank is the sum of the ranks of the two original tensors. The components of the resulting tensor are the products of the components of the original tensors, indexed by the combined indices.
        - **Contraction (Inner Product)**: A contraction of a tensor involves summing over the indices of one or more pairs of matching indices, typically reducing the overall rank. For example, contracting a rank-2 tensor (matrix) over one index pair yields a scalar (the trace of the matrix).

        **3. Einstein Summation Convention**

        - This is a notational simplification used in tensor operations where repeated indices in a term imply summation over all the values of that index. For example, in the expression \( a_i b^i \), the repeated index \( i \) means to sum over it, simplifying the description of dot products or more complex contractions.

        **4. Index Notation**

        - **Covariant and Contravariant Indices**: In the context of tensors, indices can be either covariant (subscripts) or contravariant (superscripts), which reflect different types of geometric transformations. Covariant indices typically correspond to row vectors and contravariant to column vectors in matrix terms.
        - **Raising and Lowering Indices**: Using a metric tensor, indices of a tensor can be "raised" (converted from covariant to contravariant) or "lowered" (vice versa), changing the way the tensor components transform under coordinate changes.

        **6. (TODO) Symmetry and Skew-Symmetry**

        - Some tensors have symmetric or skew-symmetric properties with respect to certain pairs of indices. For instance, a tensor \( T_{ij} \) is symmetric if \( T_{ij} = T_{ji} \) and skew-symmetric if \( T_{ij} = -T_{ji} \).

        **7. (TODO) Tensor Decomposition**
        - Tensors can often be decomposed into simpler, constituent tensors, which can be useful for various applications like principal component analysis (PCA) in statistics or spectral decomposition in linear algebra.

    """

    def __init__(
            self,
            indices: CoordIndices,
            components: SymbolArray
    ):
        """
        Initializes an instance of the EinsteinArray class.

        Args:
            indices (Indices): The indices of the tensor.
            components (SymbolArray, optional): The tensor's components. Defaults to None.
        """
        super().__init__(indices, components)

    @property
    def args(self):
        return [self.indices, self.components]

    @classmethod
    def component_equations(cls):
        return (
            (SymbolArray, lambda arg: arg)
        )

    @classmethod
    def from_equation(cls, indices: CoordIndices, *args, **kwargs) -> 'Tensor':
        """Dynamic constructor for the inheriting classes."""
        components = None

        # Categorize positional arguments
        for arg in args:
            if isinstance(arg, SymbolArray):
                components = arg
            elif isinstance(arg, cls):
                components = arg.reshape(indices).components
            else:
                for equation_type, equation_func in cls.component_equations():
                    if isinstance(arg, equation_type):
                        components = equation_func(arg)
                        break

        # Categorize keyword arguments
        for key, value in kwargs.items():
            if isinstance(value, SymbolArray):
                components = value
            elif isinstance(value, cls):
                components = value.reshape(indices).components
            else:
                for equation_type, equation_func in cls.component_equations():
                    if isinstance(value, equation_type):
                        components = equation_func(value)
                        break

        return cls(indices, components)

    # ### 2. Tensor Operations - **Addition and Subtraction**: Tensors of the same rank and shape can be added or
    # subtracted component-wise. - **Scalar Multiplication**: Each component of a tensor can be multiplied by a
    # scalar. - **Tensor Product (Outer Product)**: Given two tensors, their tensor product is a new tensor whose
    # rank is the sum of the ranks of the two original tensors. The components of the resulting tensor are the
    # products of the components of the original tensors, indexed by the combined indices.
    # - **Contraction (Inner Product)**:
    # A contraction of a tensor involves summing over the indices of one or more pairs of matching
    # indices, typically reducing the overall rank. For example, contracting a rank-2 tensor (matrix) over one index
    # pair yields a scalar (the trace of the matrix).

    def __add__(self, other: 'Tensor'):
        return self.add(other, Tensor)

    def __sub__(self, other: 'Tensor'):
        return self.sub(other, Tensor)

    def __mul__(self, other: 'Tensor'):
        if isinstance(other, (int, float, Basic)):
            return self.mul(other, type(self))
        elif isinstance(other, Tensor):
            # type(other) will most often be Tensor or child object. This is temporary until we have a better
            # solution. When connections multiply with tensors, the result is not necessarily a tensor. We make it so
            # to not lose tensor functionality.
            return self.mul(other, Tensor)
        else:
            raise TypeError(f"Expected int, float, or EinsumArray, got {type(other).__name__}")

    def __truediv__(self, other: 'Tensor'):
        return self.div(other, type(self))

    def __pow__(self, other: 'Tensor'):
        return self.pow(other, type(self))

    def coordinate_transformation(self, indices: Indices, jacobian: Jacobian):
        """
        Transforms the tensor's components under a change of coordinates.

        Args:
            jacobian (Jacobian): The Jacobian matrix representing the coordinate transformation.
        """
        jacobians = []
        for old, new in zip(self.indices.indices, indices.indices):
            if old.contravariant != new.contravariant:
                raise ValueError(
                    f"Index {old} is contravariant while index {new} is covariant. This is not allowed within a transformation.")
            if old.contravariant:
                jacobians.append(Jacobian(Indices(new, -old), jacobian))
            else:
                jacobians.append(Jacobian(Indices(-old, new), jacobian))

        new_tensor = self
        for jacobian in jacobians:
            new_tensor = new_tensor.mul(jacobian, Tensor)
        new_tensor.components = simplify(trigsimp(new_tensor.components))

        return new_tensor

    def decompose(self, algorithm: str) -> Union['Tensor', None]:
        """
        Decomposes the tensor into simpler constituent tensors using a specified decomposition function.

        Args:
            decomposition (Callable): The decomposition function to apply to the tensor.
        """
        try:
            decomposed_tensor = getattr(self, algorithm)()
            return decomposed_tensor
        except Exception as e:
            print(f"failed to decompose tensor, following error: {e}")
            return None

    def amd(self):
        """
        Decomposes the tensor using the Arnowitt-Deser-Misner (ADM) decomposition.
        """
        pass

    def spherical_symmetry(self):
        """
        Decomposes the tensor under spherical symmetry into radial and angular parts.
        """
        pass

    def petrov_classification(self):
        """
        Decomposes the tensor using the Petrov classification for the Weyl tensor.
        """
        pass

    def newman_penrose(self):
        """
        Decomposes the tensor using the Newman-Penrose formalism.
        """
        pass
