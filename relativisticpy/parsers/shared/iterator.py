from typing import Iterable


class Iterator:
    def __init__(self, object: Iterable):
        self.object = object
        if isinstance(self.object, Iterable):
            self.iterable_object = iter(object)
            self.current_item_location = -1

    def advance(self):
        self.current_item = next(self.iterable_object, None)
        if self.current_item != None:
            self.current_item_location += 1

    def peek(self, n: int = 1, default=None):
        i = self.current_item_location
        if i + n < len(self.object):
            return self.object[i + n]
        else:
            return default

    def __len__(self):
        return len(self.object)

    def current(self):
        return self.current_item
    
    @property
    def original(self) -> str: 
        if isinstance(self.object , str):
            return str(self.object)
        else:
            raise AttributeError("Cannot call this atttribute when the iterable arg is not a string type.")