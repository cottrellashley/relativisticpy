import ctypes

class Node:
    def __init__(self, item):
        self.item = item
        self.left = None
        self.right = None

    def __str__(self):  
        # For demonstration purposes, we can print the node's item
        return str(self.item)


class CNode(ctypes.Structure):
    pass

CNode._fields_ = [("item", ctypes.c_int),
                 ("left", ctypes.POINTER(CNode)),
                 ("right", ctypes.POINTER(CNode))]

# Load the shared library
libtree = ctypes.CDLL('./libtree.so')  # On Linux
# libtree = ctypes.CDLL('./tree.dll')  # On Windows

libtree.create_sample_tree.restype = ctypes.POINTER(CNode)

def convert_c_tree_to_python(cnode):
    if not cnode:
        return None
    
    node = Node(cnode.contents.item)
    node.left = convert_c_tree_to_python(cnode.contents.left)
    node.right = convert_c_tree_to_python(cnode.contents.right)
    return node

if __name__ == '__main__':
    c_root = libtree.create_sample_tree()
    root = convert_c_tree_to_python(c_root)
    print(root.left.left)


    # Now you can use the Python `root` object with your Python tree traversal functions.
