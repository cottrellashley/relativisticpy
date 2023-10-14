# Interface between relativisticpy.gr and workbook
# Can depend and know about Workbook objects and use them to initialise gr tensors which users define via strings in workbook module.
from relativisticpy.core.indices import Indices
from relativisticpy.core.metric import Metric, MetricIndices
from relativisticpy.gr import Derivative, Ricci, Riemann
from relativisticpy.providers.regex import extract_tensor_symbol, extract_tensor_indices, tensor_index_running
from relativisticpy.workbook.cache import RelPyCache
from relativisticpy.workbook.node import AstNode

class TensorNode:
    """
      Handles nodes of the string types: T_{a}_{b}...
    """

    # Cache rules:
    def tensor_symbol_already_cached(self) -> bool:
        """ 
        Is the tensor type (symbol) which user called already within the cache?
            true -> condition2()
            false -> condition3()

         """

        res = 'test'
        if res: return self.are_indices_of_called_tensor_same_as_a_cached_tensor()
        else: return self.has_the_tensor_symbol_been_defined_by_user()

    def are_indices_of_called_tensor_same_as_a_cached_tensor(self) -> bool:
        """ 
        Is the tensor type (symbol) which user called already within the cache?
            true -> condition2()
            false -> condition3()

         """

        res = 'test'
        if res: return self.condition2()
        else: return self.condition3()

    def has_the_tensor_symbol_been_defined_by_user(self):
        """ 
        Is the tensor type (symbol) which user called already within the cache?
            true -> condition2()
            false -> condition3()

         """

        res = 'test'
        if res: return self.condition2()
        else: return self.condition3()



    # is_tensor_called_symbol_in_cache() -> true  : are_indices_of_called_tensor_same_as_a_cached_tensor()
    # is_tensor_called_symbol_in_cache() -> false : has_the_tensor_symbol_been_defined()

    # are_indices_of_called_tensor_same_as_a_cached_tensor() -> true : indices_symbol_and_structure_of_cached_tensor_same_as_called_tensor()
    # are_indices_of_called_tensor_same_as_a_cached_tensor() -> false: 

    # indices_symbol_and_structure_of_cached_tensor_same_as_called_tensor() -> true : are_tensor_symbols_in_same_order()
    # indices_symbol_and_structure_of_cached_tensor_same_as_called_tensor() -> false:

    # same_tensor_symbols_in_same_order() -> true  : Return cached tensor
    # same_tensor_symbols_in_same_order() -> false : Build new tensor with components in new indices order + new indices order => cache tensor

    # has_the_tensor_symbol_been_defined() -> true : is_the_tensor_symbol_metric_dependendent()

    # is_the_tensor_symbol_metric_dependendent() -> true : has_the_metric_been_defined()

    # has_the_metric_been_defined() -> true : 
    

    def __init__(self, cache: RelPyCache):
        self.cache = cache

    def handle(self, node: AstNode):
        tensor_string_repr = ''.join(node.args)
        str_key = extract_tensor_symbol(tensor_string_repr)
        str_tensor = tensor_string_repr
        str_indices = extract_tensor_indices(tensor_string_repr)
        indices_obj = Indices.from_string(str_indices)
        metric_defined = self.cache.has_variable('Metric')

        if str_key == self.cache.get_variable('MetricSymbol') and metric_defined:
            return self.metric(str_indices)
        elif str_key == self.cache.get_variable('DiffSymbol') and metric_defined:
            return Derivative(indices_obj)
        elif str_key == self.cache.get_variable('RiemannSymbol') and metric_defined:
            return self.riemann(str_indices)
        elif str_key == self.cache.get_variable('RicciSymbol') and metric_defined:
            return self.ricci(str_indices)
        else:
            return str_tensor

    def metric(self, indices_str: str):
        indices = MetricIndices.from_string(indices_str)
        return self.cache.get_variable('Metric')[indices] if not tensor_index_running(indices_str) else self.cache.get_variable('Metric')

    def ricci(self, indices_str: str):
        indices = Indices.from_string(indices_str)

        if not self.cache.has_variable('Ricci'):
            self.cache.set_variable('Ricci', Ricci(Indices.from_string(indices_str if tensor_index_running(indices_str) else '_{a}_{b}'), self.cache.get_variable('Metric')))

        return self.cache.get_variable('Ricci')[indices] if not tensor_index_running(indices_str) else self.cache.get_variable('Ricci')

    def riemann(self, indices_str: str):
        indices = Indices.from_string(indices_str)

        if not self.cache.has_variable('Riemann'):
            self.cache.set_variable('Riemann', Riemann(Indices.from_string(indices_str if tensor_index_running(indices_str) else '_{a}_{b}_{c}_{d}'), self.cache.get_variable('Metric')))

        return self.cache.get_variable('Riemann')[indices] if not tensor_index_running(indices_str) else self.cache.get_variable('Riemann')

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
        metric_symbol = self.cache.get_variable('MetricSymbol') if self.cache.has_variable('MetricSymbol') else 'G'

        metric = Metric(
                        MetricIndices.from_string(tensor_key.str_indices),
                        tensor_comps,
                        self.cache.get_variable('Coordinates')
                    )

        self.cache.set_variable('Metric',  metric)