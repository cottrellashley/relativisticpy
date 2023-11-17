from relativisticpy.parsers.shared.interfaces.node_provider import INodeProvider
from relativisticpy.parsers.shared.models.token import Token
from relativisticpy.parsers.shared.models.node_keys import ConfigurationModels
from relativisticpy.parsers.shared.constants import NodeKeys, NodeType


class BaseNode:
    def __init__(self, node_type: NodeType):
        self.node_type = node_type

    def node(self, a, b, c):
        return {
            NodeKeys.Node.value: a,
            NodeKeys.Handler.value: b,
            NodeKeys.Arguments.value: c,
        }

    def set_node_configuration(self, node_configuration: ConfigurationModels):
        self.node_configuration = node_configuration

    def get_node_handler_name(self):
        for key_provider in self.node_configuration.node_configurations:
            if self.node_type.value == key_provider.node:
                return key_provider.handler
        return "undefined"


class InternalNode(BaseNode):
    def __init__(self, node_type: NodeType):
        super().__init__(node_type)
        self.type = self.node_type.value

    def new(self, child_nodes: list):
        self.child_nodes = child_nodes
        if self.node_type == NodeType.FUNCTION:
            return self.node("function", self.child_nodes[0], self.child_nodes[1])
        return self.node(self.type, self.get_node_handler_name(), self.child_nodes)


class LeafNode(BaseNode):
    def __init__(self, node_type: NodeType):
        super().__init__(node_type)
        self.type = self.node_type.value

    def new(self, token: Token):
        self.token = token
        if self.node_type == NodeType.OBJECT:
            return self.node(
                "object", self.match_on_object_node(self.token.value), self.token.value
            )
        return self.node(self.type, self.get_node_handler_name(), self.token.value)

    def match_on_object_node(self, object_string: str):
        for variable_key_provider in self.node_configuration.objs_configurations:
            if variable_key_provider.string_matcher_callback(object_string):
                return variable_key_provider.node_key
        return NodeType.OBJECT.value


class NodeProvider(INodeProvider):
    def set_matcher(self, node_configuration):
        self.node_configuration = node_configuration
        self.internal_node = InternalNode
        self.leaf_node = LeafNode

    def new_node(self, node_type: NodeType, args):
        if isinstance(args, list):
            internal_node = self.internal_node(node_type)
            internal_node.set_node_configuration(self.node_configuration)
            return internal_node.new(args)

        if isinstance(args, Token):
            leaf_node = self.leaf_node(node_type)
            leaf_node.set_node_configuration(self.node_configuration)
            return leaf_node.new(args)
