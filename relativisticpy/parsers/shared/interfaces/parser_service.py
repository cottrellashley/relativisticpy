from abc import ABC, abstractmethod


class IParserService(ABC):
    @abstractmethod
    def parse_string(self):
        pass

    @abstractmethod
    def tokenize_string(self):
        pass
