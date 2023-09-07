from dataclasses import dataclass
from typing import List
from relativisticpy.relparser.shared.models.basic_nodes import NodeConfigurationModel
from relativisticpy.relparser.shared.models.node_handler import NodeHandler
from relativisticpy.relparser.shared.models.object_configuration import ObjectConfigurationModel

@dataclass
class ConfigurationModels:
    node_configurations: List[NodeConfigurationModel]
    objs_configurations: List[ObjectConfigurationModel]

    def get_node_handlers(self) -> dict[str, NodeHandler]:
        return { i.node_key : i.node_handler for i in self.node_configurations }