# NOTE: This file contains the absolute minimum logic for all of its children to be able to perform the Einstein
# Summation Convention.

# Standard Library
import re
import copy
from functools import singledispatchmethod
from itertools import product, combinations
from operator import itemgetter
from typing import List, Callable, Literal, Tuple, Union, Optional, Any

from loguru import logger
from sympy import NDimArray, MutableDenseNDimArray

# External Modules
from relativisticpy.symengine import SymbolArray, Basic
from relativisticpy.utils import transpose_list


# TODO: Add exception handling, check if the settable properties have correct values and types, and throw exceptions
#  if they do not. TODO: Exceptions from this file should not surface to the user, unless the error message is
#   meaningfull. If it is not meaningful, catch and re-write a meningful error message at the workbook layer. TODO:
#    Add a copy method to the Indices class. This will be useful for the einsum implementation.

# TODO IDEA: Index object should be able to have a inherited type, which takes a CoordinatePatch(Patch(Manifold()),
#  CoordinateSystem()) -> This way users can defined and switch coordinate systems from simply defining indices
#  within the tensors:

# User defined the indices in a coordinate system and can then call the components of the tensor in a different
# coordinate system. This will be useful for the einsum implementation. T_{mu nu} -> {mu nu alpha beta} :=
# Schwarzschild T_{sigma phi} -> {sigma phi} := Eddington-Finkelstein

class Idx:
    """
        IMPORTANT: This class is not for instantiation for use. The Indices class auto-initiates this class and sets all relevant properties.
    """

    def __init__(self,
                 symbol: str,
                 order: Optional[int] = None,
                 values: Optional[Union[List, int]] = None,
                 covariant: Optional[bool] = True
                 ):
        self.symbol: str = symbol
        self.order: Optional[int] = order
        self.values: Optional[Union[List, int]] = values
        self.covariant: Optional[bool] = covariant

    @property
    def running(self) -> bool:
        return not isinstance(self.values, int)

    @property
    def contravariant(self) -> bool:
        return not self.covariant

    @property
    def dimention(self) -> int:
        return self.__dimention

    @dimention.setter
    def dimention(self, dimention: int) -> None:
        self.__dimention = dimention

    def set_order(self, order: int) -> 'Idx':
        return Idx(self.symbol, order, self.values, self.covariant)

    def non_running(self) -> 'Idx':
        self.order = [i for i in range(self.dimention)]
        return self

    # Publics (Index - Index operations)
    def is_identical_to(self, other: 'Idx') -> bool:
        return self == other  # and id(self) != id(other) <-- Still undicided

    def is_contracted_with(self, other: 'Idx') -> bool:
        return self.symbol == other.symbol and self.covariant != other.covariant  # and id(self) != id(other) <-- Still undicided

    # Publics (Index - Indices operations)
    def is_summed_wrt_indices(self, indices: 'Indices') -> bool:
        return any([self.is_contracted_with(index) for index in indices])

    def is_repeated_wrt_indices(self, indices: 'Indices') -> bool:
        return any([self.is_identical_to(index) for index in indices])

    def get_summed_location(self, indices: 'Indices') -> List[int]:
        return [self.order for index in indices if self.is_contracted_with(index)]

    def get_repeated_location(self, indices: 'Indices') -> List[int]:
        return [self.order for index in indices if self.is_identical_to(index)]

    def get_summed_locations(self, indices: 'Indices') -> list[tuple[int | None, Any]]:
        return [(self.order, index.order) for index in indices if self.is_contracted_with(index)]

    def get_repeated_locations(self, indices: 'Indices') -> list[tuple[int | None, Any]]:
        return [(self.order, index.order) for index in indices if self.is_identical_to(index)]

    def get_symbol_eq_location(self, indices: 'Indices') -> int:
        return [index.order for index in indices.indices if self.symbol == index.symbol][0] if len(
            [index for index in indices.indices if self.symbol == index.symbol]) > 0 else None

    # Idx - Idx Comparitors
    def symbol_eq(self, other: 'Idx') -> bool:
        return self.symbol == other.symbol

    def covariance_eq(self, other: 'Idx') -> bool:
        return self.covariant == other.covariant

    def order_match(self, other: 'Idx') -> bool:
        return self.order == other.order and self.symbol == other.symbol

    # Idx - Indices Comparitors
    def symbol_in_indices(self, other: 'Indices') -> bool:
        return any([self.symbol == other_idx.symbol for other_idx in other.indices])

    def symbol_covariance_eq(self, other: 'Indices') -> bool:
        return any(
            [self.symbol == other_idx.symbol and self.covariant == other_idx.covariant for other_idx in other.indices])

    def symbol_in_indices_and_order(self, other: 'Indices') -> bool:
        return any([self.symbol == other_idx.symbol and self.order == other_idx.order for other_idx in other.indices])

    def rank_match_in_indices(self, other: 'Indices') -> bool:
        return any(
            [self.order == other_idx.order and self.covariant == other_idx.covariant for other_idx in other.indices])

    # Dunders
    def __neg__(self) -> 'Idx':
        return Idx(self.symbol, self.order, self.values, not self.covariant)

    def __len__(self) -> int:
        return self.dimention

    def __eq__(self, other: 'Idx') -> bool:
        return self.covariant == other.covariant if self.symbol == other.symbol else False

    def __repr__(self) -> str:
        return f"""Indices('{self.symbol}',{self.order},{self.values},{self.covariant}) """

    def __str__(self) -> str:
        return f"""{'_' if self.covariant else '^'}{{{self.symbol}}}"""  # <== Note when we have multiple str representation of indices, this need to be dynamic.

    def __iter__(self):
        start = self.values[0] if isinstance(self.values, list) else self.values
        end = self.values[-1] if isinstance(self.values, list) else self.values
        self.first_index_value = 0 if self.running and not isinstance(self.values, list) else start
        self.last_index_value = self.dimention - 1 if self.running and not isinstance(self.values, list) else end
        return self

    def __next__(self):
        if self.first_index_value <= self.last_index_value:
            x = self.first_index_value
            self.first_index_value += 1
            return x
        else:
            raise StopIteration


class Indices:
    """ Representation of Tensor Indices. Initialized as a list of Idx objecs. """

    EINSUM_GENERATOR = "EINSUM"
    SUMMATION_GENERATOR = "SUMMATION"
    SELFSUM_GENERATOR = "SELFSUM"

    def __init__(self, *args: Idx):
        self.indices: Tuple[Idx, ...] = tuple([index.set_order(order) for order, index in enumerate([*args])])
        self.generator = lambda: None  # monkey patch product implementations of index depending on mul or add products
        self.generator_implementor = None  # Curretly only used for unit tests => for use to know which
        # implementation the generator is in currently.

    @property
    def anyrunnig(self) -> bool:
        return any([not idx.running for idx in self.indices])

    @property
    def scalar(self) -> bool:
        return self.rank == (0, 0)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple([i.dimention for i in self.indices])

    @property
    def rank(self) -> Tuple[int, ...]:
        return tuple([len([i for i in self.indices if not i.covariant]), len([i for i in self.indices if i.covariant])])

    @property
    def self_summed(self) -> bool:
        return len([[i.order, j.order] for i, j in combinations(self.indices, r=2) if i.is_contracted_with(j)]) > 0

    @property
    def dimention(self) -> int:
        return self.__dimention

    @dimention.setter
    def dimention(self, dimention: int) -> None:
        for index in self.indices:
            index.dimention = dimention
        self.__dimention = dimention

    # Dunders
    def __index__(self) -> Union[Tuple[int, ...], Tuple[slice, ...]]:
        return tuple([int(i.values) if not i.running else slice(None) for i in self.indices])

    def __len__(self) -> int:
        return len(self.indices)

    def __eq__(self, other: Union['Indices', Any]) -> bool:
        return [i == j for (i, j) in list(product(self.indices, other.indices))].count(True) == len(self)

    def __mul__(self, other: 'Indices') -> 'Indices':
        return self.einsum_product(other)

    def __add__(self, other: 'Indices') -> 'Indices':
        return self.additive_product(other)

    def __sub__(self, other: 'Indices') -> 'Indices':
        return self.additive_product(other)

    def __getitem__(self, index: Idx) -> List[Idx]:
        return [idx for idx in self.indices if idx.symbol == index.symbol and idx.covariant == index.covariant]

    def __setitem__(self, key: Idx, new: Idx) -> 'Indices':
        return Indices(
            *[new if idx.symbol == key.symbol and idx.covariant == key.covariant else idx for idx in self.indices])

    def __str__(self) -> str:
        return "".join([str(index) for index in self.indices])

    def __iter__(self):
        self.__length = int(len(self._indices_iterator()))
        self.__i = 1
        return self

    def __next__(self):
        if self.__i <= self.__length:
            x = self.__i
            self.__i += 1
            return self._indices_iterator()[x - 1]
        else:
            raise StopIteration

    # Publics
    def zeros_array(self):
        return SymbolArray.zeros(*self.shape)

    def replace(self, old: Idx, new: Idx) -> 'Indices':
        """ Replaces an index with another index, matching only on the symbol. """
        indices = [new if index.symbol == old.symbol else index for index in self.indices]
        return Indices(*indices)

    def replace_symbol(self, old: str, new: str) -> 'Indices':
        """ Replaces an index symbol with another symbol, matching only on the symbol. """
        return Indices(
            *[index if index.symbol != old else Idx(new, covariant=index.covariant) for index in self.indices])

    def get_symbol_only_reshape(self, other: 'Indices') -> Union[Tuple, None]:
        return tuple([index.get_symbol_eq_location(self) for index in other.indices]) if self.symbol_eq(other) else None

    def init_reshape_symbol_only(self, other: 'Indices') -> 'Indices':
        return Indices(*[index for index in other.indices if index.symbol_in_indices(self)])

    # [this[index][0].order for index in other.indices if index in this.indices]
    def get_reshape(self, other: 'Indices') -> Union[Tuple, None]:
        return tuple([self[index][0].order for index in other.indices if
                      index in self.indices]) if self.symbol_and_symbol_rank_eq(other) else None

    def find(self, key: Idx) -> int:
        return [idx.order for idx in self.indices if idx.symbol == key.symbol and idx.covariant == key.covariant][
            0] if len(
            [idx for idx in self.indices if idx.symbol == key.symbol and idx.covariant == key.covariant]) > 0 else None

    def has_index(self, idx: Idx) -> bool:
        return any([idx.symbol == index.symbol and idx.covariant == index.covariant for index in self.indices])

    def get_same_symbol(self, idx: Idx) -> Idx:
        lst = [index for index in self.indices if index.symbol == idx.symbol];
        return lst[0] if len(lst) == 1 else None

    def covariance_delta(self, other: 'Indices') -> list[tuple[str | Any, ...]]:
        return [tuple(['rs', i.order]) if i.covariant else tuple(['lw', i.order]) for i, j in
                product(self.indices, other.indices) if i.order == j.order and i.covariant != j.covariant]

    def get_non_running(self) -> 'Indices':
        for index in self.indices:
            index.values = None
        return self

    # Types of equality
    def order_delta(self, other: 'Indices') -> tuple[Any, ...] | None:
        return tuple([j.order for i, j in product(self.indices, other.indices) if
                      i.symbol == j.symbol and i.covariant == j.covariant]) if self.symbol_eq(other) else None

    def rank_eq(self, other: 'Indices') -> bool:
        return all([idx.rank_match_in_indices(other) for idx in self.indices])

    def symbol_covariance_eq(self, other: 'Indices') -> bool:
        return all([idx.symbol_covariance_eq(other) for idx in self.indices]) and len(self.indices) == len(
            other.indices)

    def symbol_eq(self, other: 'Indices') -> bool:
        return all([idx.symbol_in_indices(other) for idx in self.indices]) and len(self.indices) == len(other.indices)

    def symbol_and_symbol_rank_eq(self, other: 'Indices') -> bool:
        return all([idx in other.indices for idx in self.indices])

    def symbol_order_eq(self, other: 'Indices') -> bool:
        return all([idx.symbol_in_indices_and_order(other) for idx in self.indices])

    def symbol_order_rank_eq(self, other: 'Indices') -> bool:
        return all([i[0] == i[1] for i in zip(self.indices, other.indices)]) if len(self.indices) == len(
            other.indices) else False

    def einsum_product(self, other: 'Indices') -> 'Indices':
        summed_index_locations = transpose_list(self._get_all_summed_locations(other))
        all = [(IndexA, IndexB) for (IndexA, IndexB) in list(product(self, other)) if
               itemgetter(*summed_index_locations[0])(IndexA) == itemgetter(*summed_index_locations[1])(IndexB)] if len(
            summed_index_locations) > 0 else [(IndexA, IndexB) for (IndexA, IndexB) in list(product(self, other))]
        res = self._get_einsum_result(other)
        result_indices_in_A = [i[0] for i in res._get_all_repeated_location(self) if len(i) > 0]
        result_indices_in_B = [i[0] for i in res._get_all_repeated_location(other) if len(i) > 0]
        A_indices_not_summed = [i[0] for i in self._get_all_repeated_location(res) if len(i) > 0]
        B_indices_not_summed = [i[0] for i in other._get_all_repeated_location(res) if len(i) > 0]

        def generator(
                idx):  # Possible Abstraction => create a method attribute which takes in the function and its arguments as input and structures the if statements in list compr in acordance with what is not an empty array --> apply itemgetter.
            if not res.scalar and idx != None:
                if len(A_indices_not_summed) != 0 and len(
                        B_indices_not_summed) != 0:  # e.g. A_{i}_{j}_{s} * B^{i}^{j}_{k} : No exausted indices
                    return [(IndicesA, IndicesB) for (IndicesA, IndicesB) in all if
                            itemgetter(*A_indices_not_summed)(IndicesA) == itemgetter(*result_indices_in_A)(
                                idx) and itemgetter(*B_indices_not_summed)(IndicesB) == itemgetter(
                                *result_indices_in_B)(idx)]
                elif len(A_indices_not_summed) == 0 and len(
                        B_indices_not_summed) != 0:  # e.g. A_{i}_{j} * B^{i}^{j}_{k} : self.indices exausted -> all summed
                    return [(IndicesA, IndicesB) for (IndicesA, IndicesB) in all if
                            itemgetter(*B_indices_not_summed)(IndicesB) == itemgetter(*result_indices_in_B)(idx)]
                elif len(B_indices_not_summed) == 0 and len(
                        A_indices_not_summed) != 0:  # e.g. A_{i}_{j}_{k} * B^{i}^{j} : other.indices exausted -> all summed
                    return [(IndicesA, IndicesB) for (IndicesA, IndicesB) in all if
                            itemgetter(*A_indices_not_summed)(IndicesA) == itemgetter(*result_indices_in_A)(idx)]
            else:
                return all

        res.generator = generator
        res.generator_implementor = self.EINSUM_GENERATOR
        res.dimention = self.dimention
        return res

    def is_einsum_product(self, other: 'Indices') -> bool:
        """
        Returns True if indices structure is valid einstein summation convention structure.
        Returns False if indices are additive structure.
        """
        additive_product = True
        if len(self.indices) != len(other.indices):
            additive_product = False
        else:
            for i in self.indices:
                if not any([i.symbol == idx.symbol and i.covariant == idx.covariant for idx in other.indices]):
                    additive_product = False

        return not additive_product

    def additive_product(self, other: 'Indices') -> 'Indices':
        repeated_index_locations = transpose_list(self._get_all_repeated_locations(other))
        all_ = [(IndexA, IndexB) for (IndexA, IndexB) in list(product(self, other)) if
                itemgetter(*repeated_index_locations[0])(IndexA) == itemgetter(*repeated_index_locations[1])(IndexB)]
        res = self._get_additive_result()
        result_in_old = [i[0] for i in res._get_all_repeated_location(self) if len(i) > 0]

        def generator(idx):
            lst = []
            if not res.scalar and idx != None:
                for (IndicesA, IndicesB) in all_:
                    test = itemgetter(*result_in_old)(IndicesA)

                    if isinstance(test, int):
                        test = (test,)

                    if test == tuple(idx):
                        lst.append((IndicesA, IndicesB))

                return lst
            else:
                return all_

        res.generator = generator
        res.generator_implementor = self.SUMMATION_GENERATOR
        res.dimention = self.dimention
        return res

    def self_product(self):
        ne = [[i.order, j.order] for i, j in combinations(self.indices, r=2) if i.is_contracted_with(j)]
        repeated_index_locations = transpose_list(ne)
        all_ = [indices for indices in list(self) if
                itemgetter(*repeated_index_locations[0])(indices) == itemgetter(*repeated_index_locations[1])(indices)]
        res = self._get_selfsum_result()
        old_indices_not_self_summed = [i[0] for i in self._get_all_repeated_location(res) if len(i) > 0]

        def generator(idx=None):
            if res.scalar or idx == None:
                return all_

            return_list = []
            for indices in all_:
                item = itemgetter(*old_indices_not_self_summed)(indices)
                item = (item,) if isinstance(item, int) else item
                index = tuple(idx)
                if item == index:
                    return_list.append(indices)

            return return_list

        # res.generator = lambda idx : [indices for indices in all_ if itemgetter(*old_indices_not_self_summed)(indices) == tuple(idx)] if not res.scalar and idx != None else all_
        res.generator_implementor = self.SELFSUM_GENERATOR
        res.generator = generator
        res.dimention = self.dimention
        return res

    # Private Helpers
    def _indices_iterator(self):
        return list(product(*[x for x in self.indices]))

    def _is_all_summed_with(self, other: 'Indices') -> 'Indices':
        return all([idx.is_summed_wrt_indices(other.indices) for idx in self.indices])

    def _get_einsum_result(self, other: 'Indices') -> 'Indices':
        lst = [idx for idx in self.indices if not idx.is_summed_wrt_indices(other.indices)] + [idx for idx in
                                                                                               other.indices if
                                                                                               not idx.is_summed_wrt_indices(
                                                                                                   self.indices)];
        return Indices(
            *lst)

    def _get_selfsum_result(self) -> 'Indices':
        return Indices(*[idx for idx in self.indices if not idx.is_summed_wrt_indices(self.indices)])

    def _get_additive_result(self) -> 'Indices':
        return Indices(*[idx for idx in self.indices])  # Need to add commutation & anti-commutation rules

    def _get_all_summed_locations(self, other: 'Indices') -> List[Tuple[int, int]]:
        return [index.get_summed_locations(other.indices)[0] for index in self.indices if
                len(index.get_summed_locations(other.indices)) > 0]

    def _get_all_repeated_locations(self, other: 'Indices') -> List[Tuple[int, int]]:
        return [index.get_repeated_locations(other.indices)[0] for index in self.indices if
                len(index.get_repeated_locations(other.indices)) > 0]

    def _get_all_repeated_location(self, other: 'Indices') -> List[List[int]]:
        return [index.get_repeated_location(other.indices) for index in self.indices if
                len(index.get_repeated_location(other.indices)) > 0]


class _IdxAlgebraNCubeArray:
    """
    A private module class, intended for inheritance, representing logic for a multi-index algebraic array-like
    object. It serves as a base object for all algebraic logic of multi-indices/component objects. Indices can be
    covariant and contravariant. Product operations follow the Einstein Summation Convention, while addition follows
    normal index addition rules.
        
        Ensures:
        - The object remains in a valid state after any manipulations.
        - Manipulations change the state of the object rather than returning new objects.
        
        Usage: - Logic of this object is to be interfaced outside the Algebra Module via the public methods of the
        inheriting class.

        Args:
            indices (Indices): The indices object representing the multi-index structure.
            components (SymbolArray): The array of components associated with the indices.

        Raises: TypeError: If indices or components are not of the expected type. ValueError: If the components'
        shape does not match the indices or is invalid. i.e. not a shape Hypercube Array shape: (N, N, N, ...).
    """

    def __init__(self, indices: Indices, components: SymbolArray):
        self.__validate(indices, components)
        self.indices = indices
        if self.indices.scalar:
            if hasattr(self.indices, "dim"):
                self.indices.dimention = self.indices.dim
        else:
            self.indices.dimention = components.shape[0]
        self.__components = components

    @property
    def components(self) -> SymbolArray:
        return self.__components

    @components.setter
    def components(self, components: SymbolArray) -> None:
        self.__components = components

    @property
    def rank(self):
        return self.indices.rank

    @property
    def args(self) -> List[Indices | SymbolArray]:
        """
        Any object which inherits from this class should have a property called 'args' defined in sunch a way
        that Cls(*args) initializes the object.
        """
        return [self.indices, self.components]

    @property
    def scalar(self) -> bool:
        return self.rank == (0, 0)

    @property
    def shape(self) -> tuple:
        return self.components.shape

    @property
    def dimention(self) -> int:
        return self.shape[0]

    @property
    def subcomponents(self) -> SymbolArray:
        return self.components[self.indices.__index__()]

    @property
    def scalar_component(self) -> Union[int, float, Basic]:
        if self.scalar:
            return list(self.components)[0]
        else:
            raise ValueError("Not a scalar object.")

    # All methods which are callable by the inheriting classes have this _pattern.
    def _components_operation(self, operation: Callable) -> None:
        self.components = operation(self.components)

    def _index_operation(self, operation: Callable) -> None:
        self.indices = operation(self.indices)

    def _product_copy(self,
                      other: Union['_IdxAlgebraNCubeArray', int, float, Basic],
                      binary_op: Callable,
                      idx_op: Callable = None,
                      new_type_cls=None
                      ) -> '_IdxAlgebraNCubeArray':
        new_obj = copy.deepcopy(self)
        new_obj.__product(other, binary_op, idx_op)
        return new_type_cls(new_obj.indices, new_obj.components) if new_type_cls else type(self)(new_obj.indices,
                                                                                                 new_obj.components)

    def _trace(self) -> None:
        resulting_indices = self.indices.self_product()
        rarray = resulting_indices.zeros_array()
        for i in resulting_indices:
            rarray[i] = sum(
                [self.components[Indices] for Indices in resulting_indices.generator(i)]
            )
        self.indices = resulting_indices
        self.components = rarray

    def __scalar_product(self,
                         other: Union[int, float, Basic],
                         binary_op: Callable = lambda a, b: a * b
                         ) -> None:
        self.indices.dimention = self.components.shape[0]
        rarray = self.indices.zeros_array()
        if isinstance(other, (int, float, Basic)):
            for i in self.indices:
                # Vectorizing the result dynamically.
                # We build a read from the resulting indices.generate function which is monkey patched into this.
                rarray[i] = binary_op(self.components[i], other)
        self.components = rarray

    def __product(self,
                  other: Union['_IdxAlgebraNCubeArray', int, float, Basic],
                  binary_op: Callable,
                  idx_op: Union[Literal['EINSUM'], Literal['SUMMATION'], Literal['SELFSUM']] = None
                  ) -> None:
        """
            Performs a binary operation between all elements of two array-like objects.
            The order and/or shape in which the two arrays are combined depends on the indices object.
        """

        if binary_op == None:
            binary_op = lambda a, b: a * b
        if isinstance(other, (int, float, Basic)):
            self.__scalar_product(other, binary_op)
        elif other.scalar:
            self.__scalar_product(other.scalar_component, binary_op)
        elif self.scalar and not other.scalar:
            other.__scalar_product(self.scalar_component, binary_op)
            self.indices = other.indices
            self.components = other.components
        elif self.scalar and other.scalar:
            self.components = SymbolArray([binary_op(self.scalar_component, other.scalar_component)])
        else:
            if self.components.shape[0] != other.components.shape[0]:
                raise ValueError("Incompatible shapes for einstein array multiplication")

            res_indices = self.__indices_product(other.indices, idx_op)

            generator: Callable = res_indices.generator
            rarray = res_indices.zeros_array()

            for i in res_indices:
                # Vectorizing the result dynamically.
                # We build a read from the resulting indices.generate function which is monkey patched into this.
                rarray[i] = sum(
                    [
                        binary_op(self.components[idx0], other.components[idx1])
                        for idx0, idx1 in generator(i)
                    ]
                )
            self.indices = res_indices
            self.components = rarray

    def __indices_product(self, indices: Indices,
                          idx_op: Union[Literal['EINSUM'], Literal['SUMMATION'], Literal['SELFSUM']] = None) -> Indices:
        if not self.indices.dimention:
            self.indices.dimention = self.components.shape[0]
        if not indices.dimention:
            indices.dimention = self.components.shape[0]

        if idx_op:
            if idx_op == Indices.EINSUM_GENERATOR:
                res_indices = self.indices.einsum_product(indices)
            elif idx_op == Indices.SUMMATION_GENERATOR:
                res_indices = self.indices.additive_product(indices)
            elif idx_op == Indices.SELFSUM_GENERATOR:
                res_indices = self.indices.self_product()
        elif not self.indices.is_einsum_product(indices):
            res_indices = self.indices.additive_product(indices)
        else:
            res_indices = self.indices.einsum_product(indices)
        return res_indices

    def _reshape(self, new_indices: Indices, ignore_covariance: bool = False) -> None:
        if not ignore_covariance:
            reshape_tuple_order = self.indices.get_reshape(new_indices)
            self._reshape_components(reshape_tuple_order)
        else:
            reshape_tuple_order = self.indices.get_symbol_only_reshape(new_indices)
            self._reshape_components(reshape_tuple_order)
            # We return indices with same rank and symbols as our old original indices, but in the symbol order of
            # the new indices.
            self.indices = Indices(*[self.indices.indices[i] for i in reshape_tuple_order])
        self.indices.dimention = self.dimention

    def _reshape_components(self, new_order: List[int]) -> None:
        """
        Rearranges the components of the tensor based on the given order.

        Args:
            new_order (List[int]): The new order of the components.
            e.g. [1,0] => always reshapped from old_order as [0,1]

        Returns:
            None i.e. The self.components is updated within the objects state.
        """
        if len(new_order) != len(self.shape):
            raise ValueError(f"Invalid Argument: {new_order}: cannot reshape when arg is a different than shape.")
        elif any([len(self.shape) - 1 < i for i in new_order]):
            raise ValueError(f"Invalid Argument: 'new_order'.")

        new_array = SymbolArray.zeros(*self.shape)
        for new_index in product(*[range(s) for s in self.shape]):
            original_index = tuple(new_index[new_order.index(i)] for i in range(len(new_order)))
            new_array[new_index] = self.components[original_index]

        self.components = new_array

    @staticmethod
    def __validate(indices: Indices, components: SymbolArray):
        """
        Validation of all arguments passed to the constructor.
        This method is being called in the constructor for all children of this class.

        """
        if not isinstance(indices, Indices):
            raise TypeError(f"Expected Indices, got {type(indices).__name__}")
        if indices.scalar:
            if isinstance(components, (NDimArray, MutableDenseNDimArray, SymbolArray)):
                if components.shape != ():
                    raise ValueError("Invalid Argument: components argument must be a scalar.")
            elif not isinstance(components, (int, float, Basic)):
                raise ValueError("Invalid Argument: components argument must be a scalar.")

        elif not indices.scalar:
            if not isinstance(components, (NDimArray, MutableDenseNDimArray, SymbolArray)):
                raise TypeError(f"Expected SymbolArray, got {type(components).__name__}")

            if not hasattr(components, 'shape'):
                raise ValueError("Invalid Argument: components argument must have attribute 'shape'.")

            shape: tuple = components.shape

            if len(shape) > 1:
                if not all([i == shape[0] for i in shape[1:]]):
                    raise ValueError("Incompatible shapes for einstein array multiplication")

            if len(shape) == 0:
                raise ValueError("Incompatible shapes for einstein array multiplication")

            if len(shape) != len(indices.indices):
                raise ValueError("Components shape does not match indices structure.")

    def __getitem__(self, index):
        if self.scalar:
            return self.scalar_component
        else:
            return self.components[index]


class EinsumArray(_IdxAlgebraNCubeArray):
    """Encodes Einstein summation convention for array like objects. (This does not represent a Tensor.)"""

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
        if self.indices.self_summed: self._trace()  # G^{a}_{a}_{c} = G_{c} after self-summing (trace) over indices a

    def reshape(self, new_indices: Indices, ignore_covariance: bool = False):
        """
        Reshapes the tensor components according to the new indices provided.

        args: new_indices (Indices): The new indices to reshape the tensor to.
        args: ignore_covariance (bool, optional): Whether to ignore covariance when reshaping. Defaults to False.

        ignore_covariance: If True, the symbols is matched to re-shape, but the covariance is ignored. 
                           We create a new indices object with the same sumbols and rank, but different oders.

        ignore_covariance: If False, the symbols and covariance must match exactly, but not the order.

        Returns:
            GrTensor: The reshaped tensor.
        """
        if not isinstance(new_indices, Indices):
            raise TypeError(f"Expected Indices, got {type(new_indices).__name__}")
        if not ignore_covariance and not self.indices.symbol_covariance_eq(new_indices):
            raise ValueError(
                f"New indices must have same symbol and covariance as old indices, but in any different order.")
        if ignore_covariance and not self.indices.symbol_eq(new_indices):
            raise ValueError(f"New indices must have same symbols as old indices.")
        self._reshape(new_indices, ignore_covariance)
        return type(self)(*self.args)

    def add(self, other: "EinsumArray", result_cls=None) -> "EinsumArray":
        if not isinstance(other, EinsumArray):
            raise TypeError(f"Unsupported operand type(s) for +: 'EinsteinArray' and '{type(other).__name__}'")
        if self.indices.is_einsum_product(other.indices):
            raise ValueError(
                f"Cannot perform addition of EinsteinArray's with the indices: {str(self.indices)} and {str(other.indices)} ")
        return self._product_copy(
            other=other,
            binary_op=lambda a, b: a + b,
            idx_op=Indices.SUMMATION_GENERATOR,
            new_type_cls=result_cls
        )

    def sub(self, other: "EinsumArray", result_cls=None) -> "EinsumArray":
        if not isinstance(other, EinsumArray):
            raise TypeError(f"Unsupported operand type(s) for -: '{EinsumArray.__name__}' and '{type(other).__name__}'")
        if self.indices.is_einsum_product(other.indices):
            raise ValueError(
                f"Cannot perform addition of EinsteinArray's with the indices: {str(self.indices)} and {str(other.indices)} ")
        return self._product_copy(
            other=other,
            binary_op=lambda a, b: a - b,
            idx_op=Indices.SUMMATION_GENERATOR,
            new_type_cls=result_cls
        )

    def mul(self, other: Union['EinsumArray', float, int, Basic], result_cls=None) -> "EinsumArray":
        if not isinstance(other, (float, int, Basic, EinsumArray)):
            raise TypeError(f"Unsupported operand type(s) for *: '{EinsumArray.__name__}' and '{type(other).__name__}'")
        if isinstance(other, EinsumArray):
            if not self.indices.is_einsum_product(other.indices):
                raise ValueError(
                    f"Cannot perform summation of {EinsumArray.__name__}'s with the indices: {str(self.indices)} and {str(other.indices)} ")
        return self._product_copy(
            other=other,
            binary_op=lambda a, b: a * b,
            idx_op=Indices.EINSUM_GENERATOR,
            new_type_cls=result_cls
        )

    def div(self, other: Union[float, int, Basic], result_cls: object = None) -> "EinsumArray":
        if not isinstance(other, (float, int, Basic)):
            raise TypeError(
                f"Unsupported operand type(s) for / or __truediv__(): 'EinsteinArray' and {type(other).__name__}")
        elif other == 0:
            raise ZeroDivisionError("Divide by zero exception: Cannot divide by zero.")
        else:
            return self._product_copy(
                other=other,
                binary_op=lambda a, b: a / b,
                new_type_cls=result_cls
            )

    def pow(self, other: Union[float, int, Basic], result_cls=None) -> "EinsumArray":
        if not isinstance(other, (int, float, Basic)):
            raise TypeError(f"Unsupported operand type(s) for **: 'EinsteinArray' and '{type(other).__name__}'")
        elif not self.scalar:
            raise ValueError(f"unsupported operand type(s) for ** or pow() on non-scalar EinsteinArray")
        else:
            return self._product_copy(
                other=other,
                binary_op=lambda a, b: a ** b,
                new_type_cls=result_cls
            )

    @classmethod
    def component_equations(cls):
        return [
            (SymbolArray, lambda arg: arg),
            (EinsumArray, lambda arg: arg.components),
        ]

    @classmethod
    def new(cls, indices: Indices, operand: Union[SymbolArray, 'EinsumArray']) -> 'EinsumArray':
        """
        Dynamic constructor for child classes. This returns instances of tensors, based on the input arguments.

        Args:
            indices (Indices): The indices of the tensor.
            operand (Union[SymbolArray, 'EinsumArray']): The operand, used to apply and compute the components of the
             new tensor.
        """
        # If the operand is just an array, we build the tensor, taking the components given to be the components of the
        # new tensor.
        if isinstance(operand, SymbolArray):
            return cls(indices, operand)

        # If the operand is a tensor, but the same exact type of tensor. Then we can just reshape the tensor to the new
        # indices provided and return the new tensor instance.
        elif isinstance(operand, cls):
            if operand.indices.symbol_and_symbol_rank_eq(indices):
                return cls(indices, operand.reshape(indices, ignore_covariance=True).components)
            elif operand.indices.rank_eq(indices):
                return cls(indices, operand.components)
        else:
            return cls._new(operand, indices)

    @singledispatchmethod
    @classmethod
    def _new(cls, operand, indices):
        logger.debug(f"[EinsumArray] Handling init: {operand.__class__.__name__}")

    # Dunders (Still unsure if these should even be implemented since the class is not a tensor itself.)
    def __add__(self, other):
        return self.add(other, type(self))

    def __radd__(self, other):
        return self.add(other, type(self))

    def __sub__(self, other):
        return self.sub(other, type(self))

    def __rsub__(self, other):
        return self.sub(other, type(self))

    def __mul__(self, other):
        return self.mul(other, type(self))

    def __rmul__(self, other):
        return self.mul(other, type(self))

    def __truediv__(self, other):
        return self.div(other, type(self))

    def __pow__(self, other):
        return self.pow(other, type(self))

    def __neg__(self):
        self.components = -self.components;
        return self
