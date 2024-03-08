from typing import Protocol, Union, Dict

# Protocol typing class allows for very flexible and powerful type hints, 
# enabling you to specify exactly what behavior is required from an object, 
# without specifying the exact class of the object. 
# This is especially useful in cases where you want to allow multiple different classes to be used, as long as they provide certain methods.

class Scope:
    def __init__(self):
        self.variables = {}
        self.function_variables = {}
        self.tensor_variables = {}

    @property
    def variables(self) -> Dict:
        ...

    @property
    def function_variables(self) -> Dict:
        ...

    @property
    def tensor_variables(self) -> Dict:
        ...

    def check_variable(self, var: str) -> bool:
        ...
    
    def check_function(self, function: str) -> bool:
        ...
    
    def check_tensor(self, tensor_key: str) -> bool:
        ...

class State(Protocol):
    """ All external classes which are implementing the Nodes generated by the interpreter module MUST match this protocol. Implementing all it's methods. """

    @property
    def is_global(self) -> bool:
        ...

    @property
    def current_scope(self) -> Scope:
        "Return the current scope"
        ...
    
    @property
    def global_scope(self) -> Scope:
        "Return the global scope"
        ...

    def reset(self) -> None:
        "Resets the full state."
        ...

    def push_scope(self) -> None:
        "Enter a new function call (or scope) by adding a new dictionary to the stack"
        ...

    def pop_scope(self) -> None:
        "Exit the current function call (or scope) by popping the top dictionary from the stack"
        ...

    def set_variable(self, name, value) -> None:
        "Set a variable in the current scope (the top dictionary on the stack)"
        ...

    def get_variable(self, name) -> Union[any, None]:
        "Search for a variable in the stack, starting from the top"
        ...

    def set_tensor(self, name, value) -> None:
        "Set a variable in the current scope (the top dictionary on the stack)"
        ...

    def get_tensor(self, name) -> Union[any, None]:
        "Search for a variable in the stack, starting from the top"
        ...

    def set_function(self, name, value) -> None:
        "Set a variable in the current scope (the top dictionary on the stack)"
        ...

    def get_function(self, name) -> Union[any, None]:
        "Search for a variable in the stack, starting from the top"
        ...