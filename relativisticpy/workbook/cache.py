from relativisticpy.workbook.node import AstNode

class RelPyCache:
    def __init__(self):
        self.store = {}

    def set_variable(self, name, value):
        self.store[name] = value

    def get_variable(self, name):
        return self.store.get(name, None)

    def has_variable(self, name):
        return name in self.store

    def metric(self):
        pass