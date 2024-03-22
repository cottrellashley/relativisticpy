# NOTE: This file contains the absolute minimum logic for all of it's childrent to be able to perform the Einstein Summation Convention.

# Standard Library
from operator import itemgetter
from itertools import product, combinations
from itertools import product
from typing import Tuple, List, Union, Optional, Any, Callable
from relativisticpy.core.tensor_equality_types import TensorEqualityType

# External Modules
from relativisticpy.utils import transpose_list
from relativisticpy.symengine import SymbolArray

class _Idx:
    """
        IMPORTANT: This class is not for instantiation for use. The _Indices class auto-initiates this class and sets all relevant properties.
    """

    def __init__(self, symbol: str, order: Optional[int] = None, values: Optional[Union[List, int]] = None, covariant: Optional[bool] = True):
        self.symbol: str = symbol
        self.order: Optional[int] = order
        self.values: Optional[Union[List, int]] = values
        self.covariant: Optional[bool] = covariant

    @property
    def running(self) -> bool: return not isinstance(self.values, int)
    @property
    def dimention(self) -> int: return self.__dimention
    @dimention.setter
    def dimention(self, dimention: int) -> None: self.__dimention = dimention

    def set_order(self, order: int) -> '_Idx': return _Idx(self.symbol, order, self.values, self.covariant)
    
    # Publics (Index - Index operations)
    def is_identical_to(self, other: '_Idx') -> bool: return self == other # and id(self) != id(other) <-- Still undicided
    def is_contracted_with(self, other: '_Idx') -> bool: return self.symbol == other.symbol and self.covariant != other.covariant # and id(self) != id(other) <-- Still undicided

    # Publics (Index - _Indices operations)
    def is_summed_wrt_indices(self, indices: '_Indices') -> bool: return any([self.is_contracted_with(index) for index in indices])
    def get_repeated_location(self, indices: '_Indices') -> List[int]: return [self.order for index in indices if self.is_identical_to(index)]
    def get_summed_locations(self, indices: '_Indices') -> List[Tuple[int]]: return [(self.order, index.order) for index in indices if self.is_contracted_with(index)]
    def get_repeated_locations(self, indices: '_Indices') -> List[Tuple[int]]: return [(self.order, index.order) for index in indices if self.is_identical_to(index)]

    # Dunders
    def __neg__(self) -> '_Idx': return _Idx(self.symbol, self.order, self.values, not self.covariant)
    def __len__(self) -> int: return self.dimention
    def __eq__(self, other: '_Idx') -> bool: return self.covariant == other.covariant if self.symbol == other.symbol else False
    def __repr__(self) -> str: return f"""_Idx('{self.symbol}', {self.dimention}, {self.order}, {self.values}, {self.covariant}) """

    ################################# Index Iteration Logic #################################
    # We implement the iteration logic with the Einsein Summation logic in mind. 
    # The index should iterate over the whole dimention from 0 to Dim - 1 
    # If the user enters values within the index object, then we iterate over the values.
    #########################################################################################

    def __iter__(self):
        if not self.dimention:
            raise AttributeError("No dimention attribute set.")
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

class _Indices:
    """ Representation of Tensor _Indices. Initialized as a list of _Idx objecs. """
    
    EINSUM_GENERATOR = "EINSUM"
    SUMMATION_GENERATOR = "SUMMATION"
    SELFSUM_GENERATOR = "SELFSUM"

    def __init__(self, *args: _Idx):
        self.indices: Union[List[_Idx], Tuple[_Idx]] = tuple([index.set_order(order) for order, index in enumerate([*args])])
        self.generator = lambda: None # mokey patch product implementations of index depending on mul or add products
        self.generator_implementor = None # Curretly only used for unit tests => for use to know which implementation the generator is in currently.

    # Properties
    @property
    def anyrunnig(self) -> bool: return any([not idx.running for idx in self.indices])
    @property
    def scalar(self) -> bool: return self.rank == (0,0)
    @property
    def shape(self) -> Tuple[int, ...]: return tuple([i.dimention for i in self.indices])
    @property
    def rank(self) -> Tuple[int, ...]: return tuple([len([i for i in self.indices if not i.covariant]), len([i for i in self.indices if i.covariant])])
    @property
    def self_summed(self) -> bool: return len([[i.order, j.order] for i, j in combinations(self.indices, r=2) if i.is_contracted_with(j)]) > 0
    @property
    def dimention(self) -> int: return self.__dimention
    @dimention.setter
    def dimention(self, dimention: int) -> None:
        for index in self.indices:
            index.dimention = dimention
        self.__dimention = dimention

    # Dunders
    def __index__(self) -> Union[Tuple[int, ...], Tuple[slice, ...]]: return tuple([int(i.values) if not i.running else slice(None) for i in self.indices])
    def __len__(self) -> int: return len(self.indices)
    def __eq__(self, other: Union['_Indices', Any]) -> bool: return [i==j for (i, j) in list(product(self.indices, other.indices))].count(True) == len(self)
    def __mul__(self, other: '_Indices') -> '_Indices': return self.einsum_product(other)
    def __add__(self, other: '_Indices') -> '_Indices': return self.additive_product(other)
    def __sub__(self, other: '_Indices') -> '_Indices': return self.additive_product(other)
    def __getitem__(self, index: _Idx) -> List[_Idx]: return [idx for idx in self.indices if idx.symbol == index.symbol and idx.covariant == index.covariant]
    def __setitem__(self, key: _Idx, new: _Idx) -> '_Indices': return _Indices(*[new if idx.symbol == key.symbol and idx.covariant == key.covariant else idx for idx in self.indices])
    def __str__(self) -> str: return "".join([str(index) for index in self.indices])
    def __iter__(self):
        self.__length = int(len(self._indices_iterator()))
        self.__i = 1
        return self

    def __next__(self):
        if self.__i <= self.__length:
            x = self.__i
            self.__i += 1
            return self._indices_iterator()[x-1]
        else:
            raise StopIteration
 
    # Publics
    def zeros_array(self): return SymbolArray.zeros(*self.shape)
    
    # [this[index][0].order for index in other.indices if index in this.indices]
    def find(self, key: _Idx) -> int: return [idx.order for idx in self.indices if idx.symbol == key.symbol and idx.covariant == key.covariant][0] if len([idx for idx in self.indices if idx.symbol == key.symbol and idx.covariant == key.covariant]) > 0 else None
    def covariance_delta(self, other: '_Indices') -> List[Tuple[int, str]]: return [tuple(['rs', i.order]) if i.covariant else tuple(['lw', i.order]) for i, j in product(self.indices, other.indices) if i.order == j.order and i.covariant != j.covariant]

    def einsum_product(self, other: '_Indices') -> '_Indices':
        summed_index_locations = transpose_list(self._get_all_summed_locations(other))
        all = [(IndexA, IndexB) for (IndexA, IndexB) in list(product(self, other)) if itemgetter(*summed_index_locations[0])(IndexA) == itemgetter(*summed_index_locations[1])(IndexB)] if len(summed_index_locations) > 0 else [(IndexA, IndexB) for (IndexA, IndexB) in list(product(self, other))]
        res = self._get_einsum_result(other)
        result_indices_in_A = [i[0] for i in res._get_all_repeated_location(self) if len(i) > 0]
        result_indices_in_B = [i[0] for i in res._get_all_repeated_location(other) if len(i) > 0]
        A_indices_not_summed = [i[0] for i in self._get_all_repeated_location(res) if len(i) > 0]
        B_indices_not_summed = [i[0] for i in other._get_all_repeated_location(res) if len(i) > 0]
        
        def generator(idx): # Possible Abstraction => create a method attribute which takes in the function and its arguments as input and structures the if statements in list compr in acordance with what is not an empty array --> apply itemgetter.
            if not res.scalar and idx != None:
                if len(A_indices_not_summed) != 0 and len(B_indices_not_summed) != 0: # e.g. A_{i}_{j}_{s} * B^{i}^{j}_{k} : No exausted indices
                    return [(IndicesA, IndicesB) for (IndicesA, IndicesB) in all if itemgetter(*A_indices_not_summed)(IndicesA) == itemgetter(*result_indices_in_A)(idx) and itemgetter(*B_indices_not_summed)(IndicesB) == itemgetter(*result_indices_in_B)(idx)]
                elif len(A_indices_not_summed) == 0 and len(B_indices_not_summed) != 0: # e.g. A_{i}_{j} * B^{i}^{j}_{k} : self.indices exausted -> all summed
                    return [(IndicesA, IndicesB) for (IndicesA, IndicesB) in all if itemgetter(*B_indices_not_summed)(IndicesB) == itemgetter(*result_indices_in_B)(idx)]
                elif len(B_indices_not_summed) == 0 and len(A_indices_not_summed) != 0: # e.g. A_{i}_{j}_{k} * B^{i}^{j} : other.indices exausted -> all summed
                    return [(IndicesA, IndicesB) for (IndicesA, IndicesB) in all if itemgetter(*A_indices_not_summed)(IndicesA) == itemgetter(*result_indices_in_A)(idx)]
            else:
                return all

        res.generator = generator
        res.generator_implementor = self.EINSUM_GENERATOR
        res.dimention = self.dimention
        return res
    
    def is_einsum_product(self, other: '_Indices') -> bool:
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

    def additive_product(self, other: '_Indices') -> '_Indices':
        repeated_index_locations = transpose_list(self._get_all_repeated_locations(other))
        all_ = [(IndexA, IndexB) for (IndexA, IndexB) in list(product(self, other)) if itemgetter(*repeated_index_locations[0])(IndexA) == itemgetter(*repeated_index_locations[1])(IndexB)]
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
        all_ = [indices for indices in list(self) if itemgetter(*repeated_index_locations[0])(indices) == itemgetter(*repeated_index_locations[1])(indices)]
        res = self._get_selfsum_result()
        old_indices_not_self_summed = [i[0] for i in self._get_all_repeated_location(res) if len(i) > 0]

        def generator(idx = None):
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
    def _indices_iterator(self): return list(product(*[x for x in self.indices]))
    def _is_all_summed_with(self, other: '_Indices') -> '_Indices': return all([idx.is_summed_wrt_indices(other.indices) for idx in self.indices])
    def _get_einsum_result(self, other: '_Indices') -> '_Indices': lst = [idx for idx in self.indices if not idx.is_summed_wrt_indices(other.indices)] + [idx for idx in other.indices if not idx.is_summed_wrt_indices(self.indices)]; return _Indices(*lst)
    def _get_selfsum_result(self) -> '_Indices': return _Indices(*[idx for idx in self.indices if not idx.is_summed_wrt_indices(self.indices)])
    def _get_additive_result(self) -> '_Indices': return _Indices(*[idx for idx in self.indices]) # Need to add commutation & anti-commutation rules
    def _get_all_summed_locations(self, other: '_Indices') -> List[Tuple[int, int]]: return [index.get_summed_locations(other.indices)[0] for index in self.indices if len(index.get_summed_locations(other.indices)) > 0]
    def _get_all_repeated_locations(self, other: '_Indices') -> List[Tuple[int, int]]: return [index.get_repeated_locations(other.indices)[0] for index in self.indices if len(index.get_repeated_locations(other.indices)) > 0 ]
    def _get_all_repeated_location(self, other: '_Indices') -> List[Tuple[int, int]]: return [index.get_repeated_location(other.indices) for index in self.indices if len(index.get_repeated_location(other.indices)) > 0 ]

class _EinsumArray:
    """
    A class representing Multi-Index Algebraic Array-Like object.
    Base object for all Algebraic Logic of multi-indices/component objects. Indices can be coveriant and contravariant.
    Indices Product follows Einstein Summation Convention Logic. Indices Addition follows normal indices addition rules.
    """

    def __init__(self, indices: _Indices, components: SymbolArray):
        self.__validate_components(indices, components)
        self.indices = indices
        self.components = components

    @property
    def rank(self): return self.indices.rank

    @property
    def scalar(self) -> bool: return self.rank == (0, 0)

    @property
    def shape(self) -> tuple: return self.components.shape

    @property
    def dimention(self) -> int: return self.shape[0]

    def prod(self, other: '_EinsumArray', binary_op: Callable = lambda a, b : a * b, einsum_op : bool = True):
        """
        Performs a binary operation between all elements of two array-like objects. 
        The order and/or shape in which the two arrays are combined depends on the indices object.
        """

        if self.components.shape[0] != other.components.shape[0]:
            raise ValueError("Incompatible shapes for einstein array multiplication")
        
        self.indices.dimention = self.components.shape[0]
        other.indices.dimention = other.components.shape[0]
        
        if not self.indices.is_einsum_product(other.indices):
            res_indices = self.indices.additive_product(other.indices)
        else:
            res_indices = self.indices.einsum_product(other.indices)

        generator : Callable = res_indices.generator
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
        return _EinsumArray(res_indices, rarray)
    
    def trace(self):
        resulting_indices = self.indices.self_product()
        rarray = resulting_indices.zeros_array()
        for i in resulting_indices:
            rarray[i] = sum(
                [self.components[Indices] for Indices in resulting_indices.generator(i)]
            )
        return _EinsumArray(components=rarray, indices=resulting_indices)
    
    def reshape_components(self, new_order: List[int]):
        """
        Rearranges the components of the tensor based on the given order.

        Args:
            new_order (List[int]): The new order of the components.
            e.g. [1,0] => always reshapped from old_order as [0,1]

        Returns:
            SymbolArray: The tensor components with rearranged order.
        """
        if len(new_order) != len(self.shape):
            raise ValueError(f"Invalid Argument: {new_order}: cannot reshape when arg is a different than shape.")
        elif any([len(self.shape) - 1  < i for i in new_order]):
            raise ValueError(f"Invalid Argument: 'new_order'.")
    
        # Create a new array with the same data but new shape
        new_array = SymbolArray.zeros(*self.shape)

        # Iterate over each possible index in the new array
        for new_index in product(*[range(s) for s in self.shape]):
            # Map the new index to the corresponding index in the original array
            original_index = tuple(new_index[new_order.index(i)] for i in range(len(new_order)))
            # Assign the value from the original array to the new index in the new array
            new_array[new_index] = self.components[original_index]

        return new_array
    
    def components_operation(self, operation: Callable): self.components = operation(self.components); return self
    def index_operation(self, operation: Callable): self.indices = operation(self.indices); return self

    def __validate_components(self, indices: _Indices, components: SymbolArray):
        if not hasattr(components, 'shape'):
            raise ValueError("Invalid Argument: components argument must have attribute 'shape'.")
        shape : tuple = components.shape
        if len(shape) > 1:
            if not all([i == shape[0] for i in shape[1:]]):
                raise ValueError("Incompatible shapes for einstein array multiplication")
        if len(shape) == 0:
            raise ValueError("Incompatible shapes for einstein array multiplication")
        
        if len(shape) != len(indices.indices):
            raise ValueError("Components shape does not match indices structure.")
        
    def __getitem__(self, index):
        return self.components[index]