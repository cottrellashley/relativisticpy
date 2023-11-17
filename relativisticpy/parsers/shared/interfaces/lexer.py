from abc import ABC, abstractmethod


class ILexer(ABC):
    @abstractmethod
    def tokenize():
        pass
