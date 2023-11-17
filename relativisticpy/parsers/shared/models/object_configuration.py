from dataclasses import dataclass


@dataclass
class ObjectConfigurationModel:
    node: str
    node_key: str
    string_matcher_callback: object
