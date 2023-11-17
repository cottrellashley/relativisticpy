from abc import ABC, abstractmethod


class NodeHandler(ABC):
    @abstractmethod
    def execute(self):
        pass
