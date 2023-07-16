from typing import Tuple
from operator import itemgetter
from itertools import product
from relativisticpy.core import MultiIndexObject
from relativisticpy.providers import transpose_list

class TensorProduct:

    # ((1,2),(3,0))

    def __init__(self, t1: MultiIndexObject, t2: MultiIndexObject, tr: Tuple[Tuple[int, int]]):
        self.t1 = t1
        self.t2 = t2
        self.tr = tr

    def einsum_product(self, tr):
        summed_index_locations = transpose_list(tr)
        all = [(IndexA, IndexB) for (IndexA, IndexB) in list(product(self.t1.indices, self.t2.indices)) if itemgetter(*summed_index_locations[0])(IndexA) == itemgetter(*summed_index_locations[1])(IndexB)] if len(summed_index_locations) > 0 else [(IndexA, IndexB) for (IndexA, IndexB) in list(product(self.t1.indices, self.t2.indices))]

        c1 = len(summed_index_locations[0]) == len(self.t1.shape)
        c2 = len(summed_index_locations[1]) == len(self.t2.shape)

        def generator(idx): # Possible Abstraction => create a method attribute which takes in the function and its arguments as input and structures the if statements in list compr in acordance with what is not an empty array --> apply itemgetter.
            if not c1 and not c2: # e.g. A_{i}_{j}_{s} * B^{i}^{j}_{k} : No exausted indices
                return [(IndicesA, IndicesB) for (IndicesA, IndicesB) in all if itemgetter(*A_indices_not_summed)(IndicesA) == itemgetter(*result_indices_in_A)(idx) and itemgetter(*B_indices_not_summed)(IndicesB) == itemgetter(*result_indices_in_B)(idx)]
            elif len(A_indices_not_summed) == 0 and len(B_indices_not_summed) != 0: # e.g. A_{i}_{j} * B^{i}^{j}_{k} : self.indices exausted -> all summed
                return [(IndicesA, IndicesB) for (IndicesA, IndicesB) in all if itemgetter(*B_indices_not_summed)(IndicesB) == itemgetter(*result_indices_in_B)(idx)]
            elif len(B_indices_not_summed) == 0 and len(A_indices_not_summed) != 0: # e.g. A_{i}_{j}_{k} * B^{i}^{j} : other.indices exausted -> all summed
                return [(IndicesA, IndicesB) for (IndicesA, IndicesB) in all if itemgetter(*A_indices_not_summed)(IndicesA) == itemgetter(*result_indices_in_A)(idx)]
        pass
