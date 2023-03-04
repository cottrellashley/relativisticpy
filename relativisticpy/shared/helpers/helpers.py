
def transpose_list(l):
    """
    This will transpose a list.

    Input : [[1,2],[3,4],[5,6]]
    Output: [[1,3,5],[2,4,6]]
    """
    return list(map(list, zip(*l)))