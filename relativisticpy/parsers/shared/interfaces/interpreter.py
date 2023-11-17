from abc import ABC, abstractmethod


class IInterpreter(ABC):
    @abstractmethod
    def interpret(self):
        pass
