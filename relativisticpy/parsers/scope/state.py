from typing import Union, Dict


class Scope:
    def __init__(self):
        self.variables = {}
        self.function_variables = {}
        self.tensor_variables = {}

class ScopedState:

    def __init__(self):
        # Initialize the stack with a global scope

        self.stack = [Scope()]

    def reset(self):
        del self.stack
        self.stack = []
        self.stack.append(Scope())

    @property
    def is_global(self): return len(self.stack) == 1

    def push_scope(self):
        "Enter a new function call (or scope) by adding a new dictionary to the stack"
        self.stack.append(Scope())

    def pop_scope(self):
        "Exit the current function call (or scope) by popping the top dictionary from the stack"
        if len(self.stack) > 1:
            self.stack.pop()
        else:
            raise Exception("Cannot pop the global scope")

    def set_variable(self, name, value):
        "Set a variable in the current scope (the top dictionary on the stack)"
        self.stack[-1].variables[name] = value

    def get_variable(self, name) -> Union[any, None]:
        "Search for a variable in the stack, starting from the top"
        for scope in reversed(self.stack):
            if name in scope.variables:
                return scope.variables[name]
        return None

    def set_tensor(self, name, value):
        "Set a variable in the current scope (the top dictionary on the stack)"
        self.stack[-1].tensor_variables[name] = value

    def get_tensor(self, name) -> Union[any, None]:
        "Search for a variable in the stack, starting from the top"
        for scope in reversed(self.stack):
            if name in scope.tensor_variables:
                return scope.tensor_variables[name]
        return None

    def set_function(self, name, value):
        "Set a variable in the current scope (the top dictionary on the stack)"
        self.stack[-1].function_variables[name] = value

    def get_function(self, name) -> Union[any, None]:
        "Search for a variable in the stack, starting from the top"
        for scope in reversed(self.stack):
            if name in scope.function_variables:
                return scope.function_variables[name]
        return None

    @property
    def current_scope(self):
        "Return the current scope"
        return self.stack[-1]
    
    @property
    def global_scope(self):
        "Return the global scope"
        return self.stack[0]
