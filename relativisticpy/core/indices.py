# Standard Library
import re
from operator import itemgetter
from itertools import product, combinations
from itertools import product
from typing import Callable, Tuple, List, Union, Optional, Any
from relativisticpy.core.string_to_tensor import deserialisable_indices
from relativisticpy.core.tensor_equality_types import TensorEqualityType

# External Modules
from relativisticpy.providers import SymbolArray, transpose_list, symbols

class Idx:

    cartesian_basis_2d = SymbolArray([symbols('x'), symbols('y')])
    polar_basis_2d = SymbolArray([symbols('t'), symbols('theta')])
    default_basis = cartesian_basis_2d

    def __init__(self, symbol: str, order: Optional[int] = None, values: Optional[Union[list, int]] = None, covariant: Optional[bool] = True):
        self.symbol: str = symbol
        self.order: Optional[int] = order
        self.values: Optional[Union[list, int]] = values
        self.covariant: Optional[bool] = covariant
        self._basis = Idx.default_basis

    @property
    def running(self) -> bool: return not isinstance(self.values, int)
    @property
    def dimention(self) -> int: return len(self.basis)
    @property
    def basis(self) -> SymbolArray: return self._basis
    @basis.setter
    def basis(self, value: SymbolArray) -> None: self._basis = value
    
    def set_order(self, order: int) -> 'Idx': return Idx(self.symbol, order, self.values, self.covariant)
    def non_running(self) -> 'Idx': return Idx(self.symbol, order = self.order, values = [i for i in range(self.dimention)], covariant = self.covariant) if self.dimention != None else None

    # Publics (Index - Index operations)
    def is_identical_to(self, other: 'Idx') -> bool: return self == other # and id(self) != id(other) <-- Still undicided
    def is_contracted_with(self, other: 'Idx') -> bool: return self.symbol == other.symbol and self.covariant != other.covariant # and id(self) != id(other) <-- Still undicided

    # Publics (Index - Indices operations)
    def is_summed_wrt_indices(self, indices: 'Indices') -> bool: return any([self.is_contracted_with(index) for index in indices])
    def is_repeated_wrt_indices(self, indices: 'Indices') -> bool: return any([self.is_identical_to(index) for index in indices])
    def get_summed_location(self, indices: 'Indices') -> List[int]: return [self.order for index in indices if self.is_contracted_with(index)]
    def get_repeated_location(self, indices: 'Indices') -> List[int]: return [self.order for index in indices if self.is_identical_to(index)]
    def get_summed_locations(self, indices: 'Indices') -> List[Tuple[int]]: return [(self.order, index.order) for index in indices if self.is_contracted_with(index)]
    def get_repeated_locations(self, indices: 'Indices') -> List[Tuple[int]]: return [(self.order, index.order) for index in indices if self.is_identical_to(index)]

    # Equality comparitors
    def symbol_eq(self, other: 'Idx') -> bool: return self.symbol == other.symbol
    def symbol_in_indices(self, other: 'Indices') -> bool: return any([self.symbol == other_idx.symbol for other_idx in other.indices])
    def symbol_in_indices_and_order(self, other: 'Indices') -> bool: return any([self.symbol == other_idx.symbol and self.order == other_idx.order for other_idx in other.indices])
    def covariance_eq(self, other: 'Idx') -> bool: return self.covariant == other.covariant
    def rank_match_in_indices(self, other: 'Indices') -> bool: return any([self.order == other_idx.order and self.covariant == other_idx.covariant for other_idx in other.indices])
    def order_match(self, other: 'Idx') -> bool: return self.order == other.order and self.symbol == other.symbol

    # Dunders
    def __neg__(self) -> 'Idx': return Idx(self.symbol, self.order, self.values, not self.covariant)
    def __len__(self) -> int: return self.dimention
    def __eq__(self, other: 'Idx') -> bool: return self.covariant == other.covariant if self.symbol == other.symbol else False
    def __repr__(self) -> str: return f"""{'_' if self.covariant else '^'}{self.symbol}  """
    def __str__(self) -> str: return self.__repr__()

    def __iter__(self):
        self.first_index_value = 0 if self.running else self.values
        self.last_index_value = self.dimention - 1 if self.running else self.values
        return self

    def __next__(self):
        if self.first_index_value <= self.last_index_value:
            x = self.first_index_value
            self.first_index_value += 1
            return x
        else:
            raise StopIteration

@deserialisable_indices
class Indices:
    """ Representation of Tensor Indices. Initialized as a list of Idx objecs. """
    _cls_idx = Idx

    def __init__(self, *args: Idx):
        self.indices: Union[List[Idx], Tuple[Idx]] = tuple([index.set_order(order) for order, index in enumerate([*args])])
        self.generator = lambda: None # mokey patch product implementations of index depending on mul or add products
        self._basis = None

    # Properties
    @property 
    def basis(self) -> SymbolArray: return self._basis
    @property
    def anyrunnig(self) -> bool: return any([not idx.running for idx in self.indices])
    @property
    def dimention(self) -> int: return self.indices[0].dimention
    @property
    def scalar(self) -> bool: return self.rank == (0,0)
    @property
    def shape(self) -> Tuple[int, ...]: return tuple([i.dimention for i in self.indices])
    @property
    def rank(self) -> Tuple[int, ...]: return tuple([len([i for i in self.indices if not i.covariant]), len([i for i in self.indices if i.covariant])])
    @property
    def self_summed(self) -> bool: return len([[i.order, j.order] for i, j in combinations(self.indices, r=2) if i.is_contracted_with(j)]) > 0

    @basis.setter
    def basis(self, value: SymbolArray) -> None:
        self._basis = value
        for idx in self.indices:
            idx.basis = value

    # Dunders
    def __index__(self) -> Union[Tuple[int, ...], Tuple[slice, ...]]: return tuple([int(i.values) if not i.running else slice(None) for i in self.indices])
    def __len__(self) -> int: return len(self.indices)
    def __eq__(self, other: Union['Indices', Any]) -> bool: return [i==j for (i, j) in list(product(self.indices, other.indices))].count(True) == len(self)
    def __mul__(self, other: 'Indices') -> 'Indices': return self.einsum_product(other)
    def __add__(self, other: 'Indices') -> 'Indices': return self.additive_product(other)
    def __sub__(self, other: 'Indices') -> 'Indices': return self.additive_product(other)
    def __getitem__(self, index: Idx) -> List[Idx]: return [idx for idx in self.indices if idx.symbol == index.symbol and idx.covariant == index.covariant]
    def __setitem__(self, key: Idx, new: Idx) -> 'Indices': return Indices(*[new if idx.symbol == key.symbol and idx.covariant == key.covariant else idx for idx in self.indices])

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
    def find(self, key: Idx) -> int: return [idx.order for idx in self.indices if idx.symbol == key.symbol and idx.covariant == key.covariant][0] if len([idx for idx in self.indices if idx.symbol == key.symbol and idx.covariant == key.covariant]) > 0 else None
    def covariance_delta(self, other: 'Indices') -> List[Tuple[int, str]]: return [tuple(['rs', i.order]) if i.covariant else tuple(['lw', i.order]) for i, j in product(self.indices, other.indices) if i.order == j.order and i.covariant != j.covariant]
    def get_non_running(self) -> 'Indices': return Indices(*[idx.non_running() for idx in self.indices])

    # Types of equality
    def order_delta(self, other: 'Indices') -> Tuple[int]: return tuple([j.order for i, j in product(self.indices, other.indices) if i.symbol == j.symbol and i.covariant == j.covariant]) if self.symbol_eq(other) else None
    def rank_eq(self, other: 'Indices') -> bool:  return all([idx.rank_match_in_indices(other) for idx in self.indices])
    def symbol_eq(self, other: 'Indices') -> bool:  return all([idx.symbol_in_indices(other) for idx in self.indices])
    def symbol_order_eq(self, other: 'Indices') -> bool:  return all([idx.symbol_in_indices_and_order(other) for idx in self.indices])
    def symbol_order_rank_eq(self, other: 'Indices') -> bool:  return all([i[0] == i[1] for i in zip(self.indices, other.indices)]) if len(self.indices) == len(other.indices) else False

    # Mapper for equality types
    def get_equality_type_callback(self, equality_type: TensorEqualityType) -> Union['Indices', None]:
        equality_map = {
            TensorEqualityType.RankEquality : self.rank_eq,
            TensorEqualityType.IndicesSymbolEquality : self.symbol_eq,
            TensorEqualityType.IndicesOrderEquality : self.symbol_order_eq,
            TensorEqualityType.RankSymbolOrderEquality : self.symbol_order_rank_eq
        }
        return equality_map[equality_type] if equality_type in equality_map else None

    def einsum_product(self, other: 'Indices') -> 'Indices':
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
        return res

    def additive_product(self, other: 'Indices') -> 'Indices':
        repeated_index_locations = transpose_list(self._get_all_repeated_locations(other))
        all = [(IndexA, IndexB) for (IndexA, IndexB) in list(product(self, other)) if itemgetter(*repeated_index_locations[0])(IndexA) == itemgetter(*repeated_index_locations[1])(IndexB)]
        res = self._get_additive_result()
        result_in_old = [i[0] for i in res._get_all_repeated_location(self) if len(i) > 0]
        res.generator = lambda idx : [(IndicesA, IndicesB) for (IndicesA, IndicesB) in all if itemgetter(*result_in_old)(IndicesA) == tuple(idx)] if not res.scalar and idx != None else all
        return res

    def self_product(self):
        ne = [[i.order, j.order] for i, j in combinations(self.indices, r=2) if i.is_contracted_with(j)]
        repeated_index_locations = transpose_list(ne)
        all = [indices for indices in list(self) if itemgetter(*repeated_index_locations[0])(indices) == itemgetter(*repeated_index_locations[1])(indices)]
        res = self._get_selfsum_result()
        old_indices_not_self_summed = [i[0] for i in self._get_all_repeated_location(res) if len(i) > 0]
        res.generator = lambda idx : [indices for indices in all if itemgetter(*old_indices_not_self_summed)(indices) == tuple(idx)] if not res.scalar and idx != None else all
        return res

    # Privates
    def _indices_iterator(self): return list(product(*[x for x in self.indices]))
    def _is_all_summed_with(self, other: 'Indices') -> 'Indices': return all([idx.is_summed_wrt_indices(other.indices) for idx in self.indices])
    def _get_einsum_result(self, other: 'Indices') -> 'Indices': lst = [idx for idx in self.indices if not idx.is_summed_wrt_indices(other.indices)] + [idx for idx in other.indices if not idx.is_summed_wrt_indices(self.indices)]; return Indices(*lst)
    def _get_selfsum_result(self) -> 'Indices': return Indices(*[idx for idx in self.indices if not idx.is_summed_wrt_indices(self.indices)])
    def _get_additive_result(self) -> 'Indices': return Indices(*[idx for idx in self.indices]) # Need to add commutation & anti-commutation rules
    def _get_all_summed_locations(self, other: 'Indices') -> List[Tuple[int, int]]: return [index.get_summed_locations(other.indices)[0] for index in self.indices if len(index.get_summed_locations(other.indices)) > 0]
    def _get_all_repeated_locations(self, other: 'Indices') -> List[Tuple[int, int]]: return [index.get_repeated_locations(other.indices)[0] for index in self.indices if len(index.get_repeated_locations(other.indices)) > 0 ]
    def _get_all_repeated_location(self, other: 'Indices') -> List[Tuple[int, int]]: return [index.get_repeated_location(other.indices) for index in self.indices if len(index.get_repeated_location(other.indices)) > 0 ]
