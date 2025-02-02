import os
import ast

class TreeNode:
    def __init__(self, object=None, internal=False, parents=None, logical_parent=None):
        self.children = []
        self.internal = internal
        self.object = object
        self.parents = parents
        self.logical_parent = logical_parent

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parents = self

    def print_tree(self, level=0, indent="  "):
        prefix = indent * level + ("- " if level > 0 else "")
        if self.internal:
            print(f"{prefix}[Dir] {self.object}")
        else:
            print(f"{prefix}{self.object}")

        if self.logical_parent:
            print(f"{indent * (level + 1)}(Inherits from: {', '.join(self.logical_parent)})")

        for child in self.children:
            child.print_tree(level + 1, indent)

def parse_python_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        node = ast.parse(file.read(), filename=file_path)
    
    classes = []
    for item in node.body:
        if isinstance(item, ast.ClassDef):
            base_names = [base.id for base in item.bases if isinstance(base, ast.Name)]
            classes.append((item.name, base_names))
    return classes

def parse_python_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        node = ast.parse(file.read(), filename=file_path)
    
    classes = []
    for item in node.body:
        if isinstance(item, ast.ClassDef):
            base_names = [base.id for base in item.bases if isinstance(base, ast.Name)]
            classes.append((item.name, base_names))
    return classes

def build_tree():
    directory = os.path.join(os.path.dirname(__file__), "relativisticpy")
    root = TreeNode(object=directory, internal=True)
    path_to_node = {directory: root}  # Initialize the dictionary with the root directory
    
    for dirpath, dirnames, filenames in os.walk(directory):
        for dirname in dirnames:
            dir_path = os.path.join(dirpath, dirname)
            node = TreeNode(object=dirname, internal=True, parents=path_to_node.get(dirpath, root))
            path_to_node[dir_path] = node
            path_to_node.get(dirpath, root).children.append(node)
        
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(dirpath, filename)
                file_node = TreeNode(object=filename, internal=False, parents=path_to_node.get(dirpath, root))
                path_to_node.get(dirpath, root).children.append(file_node)

                # Parse the file and create nodes for each class
                classes = parse_python_file(file_path)
                for class_name, bases in classes:
                    class_node = TreeNode(object=class_name, internal=False, parents=file_node, logical_parent=bases)
                    file_node.children.append(class_node)
    
    return root
