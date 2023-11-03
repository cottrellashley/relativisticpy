# Interface between relativisticpy.gr and workbook
# Can depend and know about Workbook objects and use them to initialise gr tensors which users define via strings in workbook module.

from relativisticpy.core.indices import Indices
from relativisticpy.core.metric import Metric, MetricIndices
from relativisticpy.gr import Derivative, Ricci, Riemann
from relativisticpy.providers.regex import tensor_index_running
from relativisticpy.workbook.cache import RelPyCache, TensorReference
from relativisticpy.workbook.constants import METRIC
from relativisticpy.workbook.node import AstNode


class RelPyError:
    pass

class TensorNode:
    """ Responsible for completly handling tensor nodes, by mediating with cache to initiate or call new and existing tensors. """

    def __init__(self, cache: RelPyCache):
        self.cache = cache

    def handle(self, node: AstNode):

        tensor_ref = TensorReference(''.join(node.args))
        is_same_indices = tensor_ref.indices.symbol_eq
        has_same_rank = tensor_ref.indices.rank_eq

        if not self.cache.has_tensor(tensor_ref):
            return self.generate_tensor(tensor_ref)

        if is_same_indices:
            tensor = self.cache.get_tensor_instance(tensor_ref) # If we already have an instance, return that.

            if tensor == None: #  => cache does not have an instance.
                tensor = self.cache.get_tensor_with_eq_indices(tensor_ref)

                if tensor == None: # => Last resort, for whatever reason, just return a new generated tensor.
                    return self.generate_tensor(tensor_ref)
    
                return type(tensor)(tensor_ref.indices, tensor[tensor_ref.indices], tensor.basis)
            
            return tensor
        
        if has_same_rank:
            tensor = self.cache.get_tensor_with_eq_rank(tensor_ref)
            if tensor == None:
                return tensor
            return type(tensor)(tensor_ref.indices, tensor.components, tensor.basis)
        
        # last thing we do is generate a new instance of the tensor.
        return self.generate_tensor(tensor_ref)

    def metric(self, tensor_ref: TensorReference):
        indices = MetricIndices.from_string(tensor_ref)
        return self.cache.get_variable(Metric.NAME)[indices] if not tensor_index_running(tensor_ref) else self.cache.get_variable(Metric.NAME)

    def ricci(self, tensor_ref: TensorReference):

        if not self.cache.has_variable(Ricci.NAME):
            ricci = Ricci(tensor_ref.indices, self.cache.get_variable(Metric.NAME))
            self.cache.set_tensor(tensor_ref, ricci)

        return ricci[tensor_ref.indices] if tensor_ref.indices.anyrunnig else ricci

    def riemann(self, tensor_ref: TensorReference):

        if not self.cache.has_variable(Riemann.NAME):
            ricci = Riemann(tensor_ref.indices, self.cache.get_variable(Metric.NAME))
            self.cache.set_tensor(tensor_ref, ricci)

        return ricci[tensor_ref.indices] if tensor_ref.indices.anyrunnig else ricci

    def generate_tensor(self, tensor_ref: TensorReference):
        str_key = tensor_ref.id
        metric_defined = self.cache.has_variable(Metric.NAME)

        if not metric_defined or str_key not in self.cache:
            return RelPyError()

        if str_key == self.cache.get_variable(Metric.SYMBOL) and metric_defined:
            return self.metric(tensor_ref)
        elif str_key == self.cache.get_variable(Derivative.SYMBOL) and metric_defined:
            return Derivative(tensor_ref)
        elif str_key == self.cache.get_variable(Riemann.SYMBOL) and metric_defined:
            return self.riemann(tensor_ref)
        elif str_key == self.cache.get_variable(Ricci.SYMBOL) and metric_defined:
            return self.ricci(tensor_ref)

class TensorKeyNode:

    def handle(self, node: AstNode):
        self.str_indices = ''.join(node.args)
        return self

class TensorDefinitionNode:

    def __init__(self, cache: RelPyCache):
        self.cache = cache

    def handle(self, node: AstNode):
        tensor_key: TensorKeyNode = node.args[0]
        tensor_comps = node.args[1]
        metric_symbol = self.cache.get_variable(Metric.SYMBOL) if self.cache.has_variable(Metric.SYMBOL) else 'G'

        metric = Metric(
                        MetricIndices.from_string(tensor_key.str_indices),
                        tensor_comps,
                        self.cache.get_variable('Coordinates')
                    )

        self.cache.set_variable(Metric.NAME,  metric)