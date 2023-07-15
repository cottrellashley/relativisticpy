from sympy import MutableDenseNDimArray, zeros

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

def empty_sympy_array(dim, shape): return MutableDenseNDimArray(zeros(dim**len(shape)),shape)