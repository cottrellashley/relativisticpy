from abc import abstractmethod
from relativisticpy.parsers.shared.models.semantic_analyzer_node import SANode


class ISemanticAnalyzer:

    @abstractmethod
    def analyse_tree(self) -> SANode:
        pass