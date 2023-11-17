from abc import ABC, abstractmethod, abstractproperty


class IIterator(ABC):
    @abstractmethod
    def current(self) -> any:
        pass

    @abstractmethod
    def advance(self) -> None:
        pass

    @abstractmethod
    def peek(self) -> any:
        pass
