from dataclasses import dataclass


@dataclass
class NodeConfigurationModel:
    node: str
    handler: str
