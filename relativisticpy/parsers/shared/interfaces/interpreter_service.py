from abc import ABC, abstractmethod


class IInterpreterService(ABC):
    @abstractmethod
    def interpret_ast(self):
        pass
