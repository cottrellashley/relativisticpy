from abc import ABC, abstractmethod


class INodeProvider(ABC):
    @abstractmethod
    def new_node(self, node_type: str, *args):
        pass

    @abstractmethod
    def set_matcher(self, node_type: str, *args):
        pass
