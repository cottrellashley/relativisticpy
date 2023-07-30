from typing import Tuple
from operator import itemgetter
from itertools import product
from typing import List
from sympy import tensorproduct
from sympy import MutableDenseNDimArray as SymbolArray

def transpose_list(l):
    """
    This will transpose a list.

    Input : [[1,2],
             [3,4],
             [5,6]]

    Output: [[1,3,5],
             [2,4,6]]
    """
    return list(map(list, zip(*[i for i in l if i != None])))


def tensor_trace_product(a: SymbolArray, b: SymbolArray, trace: List[List[int]]):
    """
    Performs the tensor product 
    """
    if len(trace) == 0:
        return tensorproduct(a, b)

    shape_a = a.shape
    shape_b = b.shape
    new_shape = [shape_a[0] for i in range(len(shape_b) + len(shape_a) - 2*len(trace))]
    summed_index_locations = transpose_list(trace)
    all = [(idxs[:len(a.shape)],idxs[len(a.shape):]) for idxs in list(product(*[range(i) for i in a.shape + b.shape])) if itemgetter(*summed_index_locations[0])(idxs[:len(a.shape)]) == itemgetter(*summed_index_locations[1])(idxs[len(a.shape):])] if len(summed_index_locations) > 0 else [(idxs[:len(a.shape)],idxs[len(a.shape):]) for idxs in list(product(*[range(i) for i in a.shape + b.shape]))]

    result_indices = [i for i in range(len(shape_a) + len(shape_b) - 2*len(trace))]
    res_a_indices = [i for i in range(len(shape_a)) if i not in summed_index_locations[0]]
    res_b_indices = [i for i in range(len(shape_b)) if i not in summed_index_locations[1]]

    def generator(idx):
        if not len(res_a_indices) == 0 and not len(res_b_indices) == 0:
            return [(IndicesA, IndicesB) for (IndicesA, IndicesB) in all if itemgetter(*res_a_indices)(IndicesA) == itemgetter(*result_indices[:len(res_a_indices)])(idx) and itemgetter(*res_b_indices)(IndicesB) == itemgetter(*result_indices[len(res_a_indices):])(idx)]
        elif len(res_a_indices) == 0 and not len(res_b_indices) == 0: 
            return [(IndicesA, IndicesB) for (IndicesA, IndicesB) in all if itemgetter(*res_b_indices)(IndicesB) == itemgetter(*result_indices[:len(res_b_indices)])(idx)]
        elif len(res_b_indices) == 0 and not len(res_a_indices) == 0:
            return [(IndicesA, IndicesB) for (IndicesA, IndicesB) in all if itemgetter(*res_a_indices)(IndicesA) == itemgetter(*result_indices[:len(res_a_indices)])(idx)]
        else:
            return [(IndicesA, IndicesB) for IndicesA, IndicesB in all]

    func = lambda idcs : sum([a[i]*b[j] for i, j in generator(idcs)])
    zeros = SymbolArray.zeros(*new_shape)

    for i in list(product(*[range(i) for i in new_shape])):
        zeros[i] = func(i)

    return zeros