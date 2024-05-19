from typing import Union, Callable
# External Modules
from relativisticpy.symengine import SymbolArray, Basic
from relativisticpy.algebras import EinsumArray, Indices, Idx

# Pattern. A lot of errors can be caused by public methods not being called with the correct arguments.
# To avoid this we implement a simple patter:
# 1. Public methods all have explicit type hints and argument names.
# 2. Public methods all have a docstring that explains what the method does.
# 3. Public methods all perform argument validation and raise exceptions if the arguments are invalid.
# 4. If all arguments are valid, the public method calls a protected method that performs the actual operation.
# 5. The protected method does not perform argument validation, as it is assumed that the public method has already done this.

class GrTensor(EinsumArray):
    """
    A GrTensor is a tensor which represents either a physical or geometric quantity/object which is defined on a manifold.
    Being defined on a manifold, this tensor must have a metric associated with the manifild the tensor quantity lives on.
    Practically, this simply means we can raise and lower indices of the tensor using the metric which was set.
    """

    def __init__(
        self,
        indices: Indices,
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
    def args(self): return [self.indices, self.components, self.metric]

    def rerank(self, new_indices: Indices):
        if not isinstance(new_indices, Indices):
            raise TypeError(f"Expected Indices, got {type(new_indices).__name__}")
        if not self.indices.symbol_order_eq(new_indices):
            raise ValueError(f"New indices must have same symbol and covariance as old indices, but in any different order.")
        self._rerank(new_indices)

        for index in new_indices.indices:
            if new_in_old := self.indices.get_same_symbol(index):
                if new_in_old.covariant != index.covariant:
                    self.lower_index(new_in_old) if new_in_old.contravariant else self.raise_index(new_in_old)
                else:
                    continue
            else:
                continue

    def reshape_rerank(self, new_indices: Indices):
        if not isinstance(new_indices, Indices):
            raise TypeError(f"Expected Indices, got {type(new_indices).__name__}")
        if len(self.indices.indices) != len(new_indices.indices):
            raise ValueError(f"New indices must have same number of indices as current indices.")
        if not self.indices.symbol_eq(new_indices):
            raise ValueError(f"New indices must have same symbols as current indices.")
        
        # DIFFERENT ORDERS - IGNORING COVERIANCE
        # _{a b c} -> _{b a c}
        # _{a b}^{c} -> ^{c}_{b a}
        if self.indices.symbol_eq(new_indices):
            self._reshape(new_indices, ignore_covariance=True)
        
        # SAME SYMBOLS & ORDERS - DIFFERENT COVARIANCE
        # _{a b c} -> ^{a}_{b c}
        # ^{a}_{b c} -> _{a b}^{c}
        if self.indices.symbol_order_eq(new_indices):
            for index in new_indices.indices:
                if new_in_old := self.indices.get_same_symbol(index):
                    if new_in_old.covariant != index.covariant:
                        self.lower_index(new_in_old) if new_in_old.contravariant else self.raise_index(new_in_old)
                    else:
                        continue
                else:
                    continue

    def lower_index(self, index: Idx):
        """ Protected method to lower an index of the tensor. """
        if not isinstance(index, Idx):
            raise TypeError(f"Expected Idx, got {type(index).__name__}")
        if not self.indices.has_index(index):
            raise ValueError(f"Index {index} not found in indices {self.indices}.")
        if index.covariant:
            raise ValueError(f"Cannot raise index {index} as it is already covariant.")
        if not self.metric:
            raise ValueError(f"Cannot perform operation on EinsteinArray objects without Metric defined.")
        self.__new_index(index, MetricIndices.lower_indices)

    def raise_index(self, index: Idx) -> 'GrTensor':
        """ Protected method to raise an index of the tensor. """
        if not isinstance(index, Idx):
            raise TypeError(f"Expected Idx, got {type(index).__name__}")
        if not self.indices.has_index(index):
            raise ValueError(f"Index {index} not found in indices {self.indices}.")
        if index.contravariant:
            raise ValueError(f"Cannot raise index {index} as it is already contravariant.")
        if not self.metric:
            raise ValueError(f"Cannot perform operation on EinsteinArray objects without Metric defined.")
        self.__new_index(index, MetricIndices.raise_indices)

    def _rerank(self, new_indices: Indices):
        """ Protected method to rerank the tensor in same rank as the new_indices. """
        if self.indices.symbol_order_eq(new_indices):
            for index in new_indices.indices:
                if new_in_old := self.indices.get_same_symbol(index):
                    if new_in_old.covariant != index.covariant:
                        self.lower_index(new_in_old) if new_in_old.contravariant else self.raise_index(new_in_old)
                    else:
                        continue
                else:
                    continue
        else:
            for self_idx, new_idx in list(zip(self.indices.indices, new_indices.indices)):
                if self_idx.covariant != new_idx.covariant:
                    if self_idx.covariant:
                        self.lower_index(self_idx)
                    else:
                        self.raise_index(self_idx)
                else:
                    continue
            self.indices = new_indices
            new_indices.dimention = self.dimention
    
    def __new_index(self, index: Idx, dummy_indices_generator: Callable):
        """ Protected method to raise an index of the tensor. """
        
        # The most efficient way is to directly manipulate the underlying components of the tensor.
        # For the time being, we perform the expensive operation by generate two dummy tensors objects and multiplying them.
        metric_dummy_indices, tensor_dummy_indices = dummy_indices_generator(index, self.indices)

        tensor_args = [tensor_dummy_indices] + self.args[1:]
        dummy_metric = Metric.from_metric(self.metric, metric_dummy_indices)
        dummy_tensor = type(self)(*tensor_args)
        result = dummy_metric.mul(dummy_tensor, type(self))

        # Set new properties
        self.components = result.components
        self.indices = result.indices
