# Standard Library
from typing import Union, Callable, Tuple
from itertools import product
from operator import itemgetter

# External Modules
from relativisticpy.algebras import EinsumArray, Indices, Idx, Tensor
from relativisticpy.diffgeom.manifold import CoordinatePatch
from relativisticpy.symengine import SymbolArray, diff, simplify, tensorproduct, Symbol
from relativisticpy.utils import tensor_trace_product, transpose_list


class MetricIndices(Indices):
    # We can allow users to initiate the metric via the __setitem__ method: if user inits the Metric without the comps => they mapp the components

    def __init__(self, *args: Idx):
        super().__init__(*args)

    def _get_einsum_metric_result(
        self: "MetricIndices", other: Union["Indices", "MetricIndices"]
    ) -> (
        "Indices"
    ):  # G_{a}_{b} * T^{c}^{d}^{a}^{f} => T^{c}^{d}_{b}^{f} (metric indices) != T_{b}^{c}^{d}^{f} (base indices)
        other_indices = list(other.indices)
        metric_indices = list(self.indices)
        for (
            idx
        ) in (
            metric_indices
        ):  # iterate on metric indices => if summed with other indices => replace other summed idx with second metric idx
            if idx.is_summed_wrt_indices(other_indices):
                other_indices[other_indices.index(-idx)] = metric_indices[
                    metric_indices.index(idx) - 1
                ]
                break
        return Indices(*other_indices)

    @classmethod
    def raise_indices(
        cls, idx: Idx, indices: Indices
    ) -> Tuple["MetricIndices", Indices]:
        temp_indices = indices.replace_symbol(idx.symbol, "__dummy__")
        temp_metric = cls(-Idx(idx.symbol), -Idx("__dummy__"))
        temp_indices.dimention = indices.dimention
        temp_metric.dimention = indices.dimention
        return temp_metric, temp_indices

    @classmethod
    def lower_indices(
        cls, idx: Idx, indices: Indices
    ) -> Tuple["MetricIndices", Indices]:
        temp_indices = indices.replace_symbol(idx.symbol, "__dummy__")
        temp_metric = cls(Idx(idx.symbol), Idx("__dummy__"))
        return temp_metric, temp_indices

    def einsum_product(self, other: "Indices") -> "Indices":
        summed_index_locations = transpose_list(self._get_all_summed_locations(other))
        all = (
            [
                (IndexA, IndexB)
                for (IndexA, IndexB) in list(product(self, other))
                if itemgetter(*summed_index_locations[0])(IndexA)
                == itemgetter(*summed_index_locations[1])(IndexB)
            ]
            if len(summed_index_locations) > 0
            else [(IndexA, IndexB) for (IndexA, IndexB) in list(product(self, other))]
        )
        res = self._get_einsum_metric_result(other)
        result_indices_in_A = [
            i[0] for i in res._get_all_repeated_location(self) if len(i) > 0
        ]
        result_indices_in_B = [
            i[0] for i in res._get_all_repeated_location(other) if len(i) > 0
        ]
        A_indices_not_summed = [
            i[0] for i in self._get_all_repeated_location(res) if len(i) > 0
        ]
        B_indices_not_summed = [
            i[0] for i in other._get_all_repeated_location(res) if len(i) > 0
        ]

        def generator(
            idx,
        ):  # Possible Abstraction => create a method attribute which takes in the function and its arguments as input and structures the if statements in list compr in acordance with what is not an empty array --> apply itemgetter.
            if not res.scalar and idx != None:
                if (
                    len(A_indices_not_summed) != 0 and len(B_indices_not_summed) != 0
                ):  # e.g. A_{i}_{j}_{s} * B^{i}^{j}_{k}
                    return [
                        (IndicesA, IndicesB)
                        for (IndicesA, IndicesB) in all
                        if itemgetter(*A_indices_not_summed)(IndicesA)
                        == itemgetter(*result_indices_in_A)(idx)
                        and itemgetter(*B_indices_not_summed)(IndicesB)
                        == itemgetter(*result_indices_in_B)(idx)
                    ]
                elif (
                    len(A_indices_not_summed) == 0 and len(B_indices_not_summed) != 0
                ):  # e.g. A_{i}_{j} * B^{i}^{j}_{k}
                    return [
                        (IndicesA, IndicesB)
                        for (IndicesA, IndicesB) in all
                        if itemgetter(*B_indices_not_summed)(IndicesB)
                        == itemgetter(*result_indices_in_B)(idx)
                    ]
                elif (
                    len(B_indices_not_summed) == 0 and len(A_indices_not_summed) != 0
                ):  # e.g. A_{i}_{j}_{k} * B^{i}^{j}
                    return [
                        (IndicesA, IndicesB)
                        for (IndicesA, IndicesB) in all
                        if itemgetter(*A_indices_not_summed)(IndicesA)
                        == itemgetter(*result_indices_in_A)(idx)
                    ]
            else:
                return all

        res.generator = generator
        res.generator_implementor = self.EINSUM_GENERATOR
        res.dimention = self.dimention
        return res


class Metric(EinsumArray):
    cron_delta = (1, 1)
    contravariant = (0, 2)
    covariant = (2, 0)
    indices_type = MetricIndices

    def __init__(
        self,
        indices: MetricIndices,
        components: SymbolArray,
        # coordinate_patch: CoordinatePatch, TODO: Should be within the indices object
    ):
        super().__init__(indices=indices, components=components)

    @property
    def args(self) -> Tuple[Indices, SymbolArray]:
        return [self.indices, self.components, self.coordinate_patch]

    @property
    def uu_components(self) -> SymbolArray:
        return (
            SymbolArray(self.components.tomatrix().inv())
            if self.rank == (0, 2)
            else self.components
        )

    @property
    def ll_components(self) -> SymbolArray:
        return self.components if self.rank == (0, 2) else self.uu_components

    @property
    def covariant(self):
        if self.rank == Metric.contravariant:
            comp = self.components
            ind = self.indices
        else:
            comp = SymbolArray(self.components.tomatrix().inv())
            ind = MetricIndices(*[-j for j in self.indices.indices])
        return Metric(
            indices=ind, components=comp
        )

    @property
    def contravariant(self):
        if self.rank == Metric.covariant:
            comp = self.components
            ind = self.indices
        else:
            comp = SymbolArray(self.components.tomatrix().inv())
            ind = MetricIndices(*[-j for j in self.indices.indices])
        return Metric(
            indices=ind, components=comp
        )

    def __pow__(self, other):
        if other == -1:
            return self.covariant if self.rank == (0, 2) else self.contravariant
        else:
            raise ValueError(
                "Cannot raise Tensor to power. Only combatible with taking the inverse by taking the pow of value -1."
            )

    def new_metric(self, indices: MetricIndices):
        return self[indices]

    def __getitem__(self: "Metric", idcs: MetricIndices):
        return (
            self.covariant.components
            if idcs.rank == (0, 2)
            else self.contravariant.components
        )

    @classmethod
    def default(cls, dim: int, indices_symbols: list[str] = ["mu", "nu"]):
        return cls(
            cls.indices_type(Idx(indices_symbols[0]), Idx(indices_symbols[1])),
            SymbolArray([[1 if i == j else 0 for j in range(dim)] for i in range(dim)]),
            SymbolArray([Symbol(f"x{i}") for i in range(dim)]),
        )

    @classmethod
    def from_metric(cls, metric: "Metric", indices: MetricIndices):
        components = (
            metric.ll_components if indices.rank == (0, 2) else metric.uu_components
        )
        return cls(indices, components)

    # This needs drastic change.
    def __add__(self, other: EinsumArray):
        return self.add(other, type(other))

    def __sub__(self, other: EinsumArray):
        return self.sub(other, type(other))

    def __mul__(self, other: EinsumArray):
        return self.mul(other, type(other))

    def __truediv__(self, other: EinsumArray):
        return self.div(other, type(self))

    def __pow__(self, other: EinsumArray):
        return self.pow(other, type(self))

    def rerank_a_tensor_idx(
        self, tensor: Tensor, index: Idx, dummy_indices_generator: Callable
    ):
        """Protected method to raise an index of the tensor."""

        # The most efficient way is to directly manipulate the underlying components of the tensor.
        # For the time being, we perform the expensive operation by generate two dummy tensors objects and multiplying them.
        metric_dummy_indices, tensor_dummy_indices = dummy_indices_generator(
            index, self.indices
        )
        # Following the Args convention of the Tensor classes: indices, components, other *args, **kwargs
        # TODO: Create a class method in the base class which can be inherited by all the tensor classes which init_new_args(args) which
        # will return the new args for the tensor class only changing the args passes in, everyhing else remains the same.
        tensor_args = [tensor_dummy_indices] + tensor.args[1:]
        dummy_metric = Metric(metric_dummy_indices, self.components)
        dummy_tensor = type(tensor)(*tensor_args)
        result = dummy_metric.mul(dummy_tensor, type(tensor))

        # Set new properties
        tensor.components = result.components
        tensor.indices = result.indices

    def raise_index(self, index: Idx, dummy_indices_generator):
        """Protected method to raise an index of the tensor."""

        # The most efficient way is to directly manipulate the underlying components of the tensor.
        # For the time being, we perform the expensive operation by generate two dummy tensors objects and multiplying them.
        metric_dummy_indices, tensor_dummy_indices = dummy_indices_generator(
            index, self.indices
        )

        tensor_args = [tensor_dummy_indices] + self.args[1:]
        dummy_metric = Metric.from_metric(self.metric, metric_dummy_indices)
        dummy_tensor = type(self)(*tensor_args)
        result = dummy_metric.mul(dummy_tensor, type(self))

        # Set new properties
        self.components = result.components
        self.indices = result.indices

    def rerank(self, tensor: Tensor, new_indices: Indices):
        if not isinstance(new_indices, Indices):
            raise TypeError(f"Expected Indices, got {type(new_indices).__name__}")
        if not self.indices.symbol_order_eq(new_indices):
            raise ValueError(
                f"New indices must have same symbol and covariance as old indices, but in any different order."
            )
        self._rerank(new_indices)

        for index in new_indices.indices:
            if new_in_old := self.indices.get_same_symbol(index):
                if new_in_old.covariant != index.covariant:
                    (
                        self.lower_index(new_in_old)
                        if new_in_old.contravariant
                        else self.raise_index(new_in_old)
                    )
                else:
                    continue
            else:
                continue

    def reshape_rerank(self, new_indices: Indices):
        if not isinstance(new_indices, Indices):
            raise TypeError(f"Expected Indices, got {type(new_indices).__name__}")
        if len(self.indices.indices) != len(new_indices.indices):
            raise ValueError(
                f"New indices must have same number of indices as current indices."
            )
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
                        (
                            self.lower_index(new_in_old)
                            if new_in_old.contravariant
                            else self.raise_index(new_in_old)
                        )
                    else:
                        continue
                else:
                    continue

    def lower_index(self, index: Idx):
        """Protected method to lower an index of the tensor."""
        if not isinstance(index, Idx):
            raise TypeError(f"Expected Idx, got {type(index).__name__}")
        if not self.indices.has_index(index):
            raise ValueError(f"Index {index} not found in indices {self.indices}.")
        if index.covariant:
            raise ValueError(f"Cannot raise index {index} as it is already covariant.")
        if not self.metric:
            raise ValueError(
                f"Cannot perform operation on EinsteinArray objects without Metric defined."
            )
        self.__new_index(index, MetricIndices.lower_indices)

    def raise_index(self, index: Idx) -> "GrTensor":
        """Protected method to raise an index of the tensor."""
        if not isinstance(index, Idx):
            raise TypeError(f"Expected Idx, got {type(index).__name__}")
        if not self.indices.has_index(index):
            raise ValueError(f"Index {index} not found in indices {self.indices}.")
        if index.contravariant:
            raise ValueError(
                f"Cannot raise index {index} as it is already contravariant."
            )
        if not self.metric:
            raise ValueError(
                f"Cannot perform operation on EinsteinArray objects without Metric defined."
            )
        self.__new_index(index, MetricIndices.raise_indices)

    def _rerank(self, new_indices: Indices):
        """Protected method to rerank the tensor in same rank as the new_indices."""
        if self.indices.symbol_order_eq(new_indices):
            for index in new_indices.indices:
                if new_in_old := self.indices.get_same_symbol(index):
                    if new_in_old.covariant != index.covariant:
                        (
                            self.lower_index(new_in_old)
                            if new_in_old.contravariant
                            else self.raise_index(new_in_old)
                        )
                    else:
                        continue
                else:
                    continue
        else:
            for self_idx, new_idx in list(
                zip(self.indices.indices, new_indices.indices)
            ):
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
        """Protected method to raise an index of the tensor."""

        # The most efficient way is to directly manipulate the underlying components of the tensor.
        # For the time being, we perform the expensive operation by generate two dummy tensors objects and multiplying them.
        metric_dummy_indices, tensor_dummy_indices = dummy_indices_generator(
            index, self.indices
        )

        tensor_args = [tensor_dummy_indices] + self.args[1:]
        dummy_metric = Metric.from_metric(self.metric, metric_dummy_indices)
        dummy_tensor = type(self)(*tensor_args)
        result = dummy_metric.mul(dummy_tensor, type(self))

        # Set new properties
        self.components = result.components
        self.indices = result.indices
