from abc import ABC, abstractmethod, abstractproperty


class IParser(ABC):
    @abstractmethod
    def parse() -> dict:
        pass
