# Interface between relativisticpy.gr and workbook
# Can depend and know about Workbook objects and use them to initialise gr tensors which users define via strings in workbook module.

from relativisticpy.core.indices import Indices
from relativisticpy.core.metric import Metric, MetricIndices
from relativisticpy.core import MultiIndexObject
from relativisticpy.gr import Derivative, Ricci, Riemann
from relativisticpy.providers.regex import tensor_index_running
from relativisticpy.providers import permutedims, SymbolArray
from relativisticpy.workbook.cache import RelPyCache, TensorReference
from relativisticpy.workbook.node import AstNode
from relativisticpy.core.tensor_equality_types import TensorEqualityType
from itertools import product
from relativisticpy.workbook.constants import WorkbookConstants

class RelPyError:
    pass

class TensorNode:
    """ Responsible for completly handling tensor nodes, by mediating with cache to initiate or call new and existing tensors. """

    def __init__(self, cache: RelPyCache):
        self.cache = cache

    def handle(self, node: AstNode):
        tensor_ref = TensorReference(''.join(node.args))

        if not self.cache.has_tensor(tensor_ref.id): # If not cached => skip to generate imediatly.
            return self.generate_tensor(tensor_ref)

        is_same_indices = self.cache.match_on_cached_tensors(TensorEqualityType.IndicesSymbolEquality, tensor_ref)

        if is_same_indices != None:
            tensor = self.cache.match_on_cached_tensors(TensorEqualityType.RankSymbolOrderEquality, tensor_ref) # If we already have an instance, return that.

            if tensor == None: #  => cache does not have an instance.

                # Reset tensor to point to tensor we know not to be Null
                tensor = is_same_indices
                components = tensor[tensor_ref.indices]

                # Handling any changes in order of indices.
                diff_order = tensor_ref.indices.order_delta(tensor.indices)
                if diff_order == None or len(diff_order) == 0:
                    # No order changes => just init new instance with new indices.
                    return type(tensor)(tensor_ref.indices, components, tensor.basis)

                return type(tensor)(tensor_ref.indices, permutedims(components, diff_order), tensor.basis) 
                
            
            return tensor
    
        has_same_rank = self.cache.match_on_cached_tensors(TensorEqualityType.RankEquality, tensor_ref)
        if has_same_rank != None:
            tensor = type(has_same_rank)(tensor_ref.indices, has_same_rank.components, has_same_rank.basis)
            self.cache.set_tensor(tensor_ref, tensor)
            return tensor
        
        # last thing we do is generate a new instance of the tensor.
        return self.generate_tensor(tensor_ref)

    def metric(self, tensor_ref: TensorReference):
        metric = self.cache.get_tensors(tensor_ref.id)
        self.cache.set_tensor(tensor_ref, metric)
        return metric[tensor_ref.indices] if tensor_ref.indices.anyrunnig else metric


    def ricci(self, tensor_ref: TensorReference):

        if not self.cache.has_tensor(tensor_ref.id):
            ricci = Ricci(tensor_ref.indices, self.cache.get_metric())
            self.cache.set_tensor(tensor_ref, ricci)

        return ricci[tensor_ref.indices] if tensor_ref.indices.anyrunnig else ricci

    def riemann(self, tensor_ref: TensorReference):

        if not self.cache.has_tensor(tensor_ref.id):
            ricci = Riemann(tensor_ref.indices, self.cache.get_metric())
            self.cache.set_tensor(tensor_ref, ricci)

        return ricci[tensor_ref.indices] if tensor_ref.indices.anyrunnig else ricci

    def generate_tensor(self, tensor_ref: TensorReference):
        str_key = tensor_ref.id

        metric_defined = self.cache.has_metric()
        if not metric_defined: # Without metric, we cannot generate anything.
            return RelPyError()

        if str_key == self.cache.get_variable(WorkbookConstants.METRICSYMBOL.value): return self.metric(tensor_ref)
        elif str_key == self.cache.get_variable(WorkbookConstants.DERIVATIVESYMBOL.value): return Derivative(tensor_ref)
        elif str_key == self.cache.get_variable(WorkbookConstants.RIEMANNSYMBOL.value): return self.riemann(tensor_ref)
        elif str_key == self.cache.get_variable(WorkbookConstants.RICCISYMBOL.value): return self.ricci(tensor_ref)

class TensorKeyNode:

    def handle(self, node: AstNode):
        self.str_indices = ''.join(node.args)
        return self

class TensorDefinitionNode:

    def __init__(self, cache: RelPyCache):
        self.cache = cache

    def handle(self, node: AstNode):
        tensor_key: TensorKeyNode = node.args[0]
        tref = TensorReference(tensor_key.str_indices)
        tensor_comps = node.args[1]

        # Check if the tensor being called is the metric
        metric_symbol = self.cache.get_variable(WorkbookConstants.METRICSYMBOL.value) if self.cache.has_variable(WorkbookConstants.METRICSYMBOL.value) else 'G'
        if metric_symbol == tref.id:
            new_tensor = Metric(
                            MetricIndices.from_string(tref.indices_repr),
                            tensor_comps,
                            self.cache.coordinates
                            )
        else:
            new_tensor = MultiIndexObject(
                            Indices.from_string(tensor_key.str_indices),
                            tensor_comps,
                            self.cache.coordinates
                            )

        self.cache.set_tensor(tref, new_tensor)

class TensorDiagBuilder:

    def handle(self, node: AstNode):
        # Determine n from the length of diag_values
        n = len(node.args)

        # Create an NxN MutableDenseNDimArray with zeros
        ndarray = SymbolArray.zeros(n, n)

        # Set the diagonal values
        for i in range(n):
            ndarray[i, i] = node.args[i]

        return ndarray
    
class DefinitionNode:

    def __init__(self, cache: RelPyCache):
        self.cache = cache

    def handle(self, node: AstNode):
        if isinstance(node.args[0], str): key = node.args[0]
        else: raise ValueError('AstNode is not a string.')

        if key == WorkbookConstants.COORDINATES.value:
            self.cache.set_coordinates(node.args[1])
            self.cache.set_variable(key, node.args[1])
        self.cache.set_variable(key, node.args[1])