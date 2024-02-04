# Interface between relativisticpy.gr and workbook
# Can depend and know about Workbook objects and use them to initialise gr tensors which users define via strings in workbook module.

from relativisticpy.core import (
    EinsteinArray,
    Metric,
    MetricIndices,
    Indices,
    TensorEqualityType,
)
from relativisticpy.gr import Derivative, Ricci, Riemann, Connection, EinsteinTensor
from relativisticpy.symengine import SymbolArray, permutedims
from relativisticpy.workbook.state import WorkbookState, TensorReference
from relativisticpy.workbook.constants import WorkbookConstants

from relativisticpy.parsers.types.gr_nodes import TensorNode

class RelPyError:
    pass


class TensorHandler: #If there is an issue with tensor in Workbook --> This is where you should begin looking.
    """Responsible for completly handling tensor nodes, by mediating with state to initiate or call new and existing tensors."""

    def __init__(self, state: WorkbookState):
        self.state = state

    def handle(self, tensor_ref: TensorReference):

        tensor_ref.indices.basis = (
            self.state.coordinates
        )  # Error handling needed => if no coordinates defined cannot continue

        if not self.state.has_tensor(
            tensor_ref.id
        ):  # If not stated => skip to generate imediatly.
            generated_tensor : EinsteinArray = self.generate_tensor(tensor_ref)
            return generated_tensor.subcomponents if tensor_ref.is_calling_tensor_subcomponent else generated_tensor

        is_same_indices = self.state.match_on_tensors(
            TensorEqualityType.IndicesSymbolEquality, tensor_ref
        )

        if is_same_indices != None:
            tensor = self.state.match_on_tensors(
                TensorEqualityType.RankSymbolOrderEquality, tensor_ref
            )  # If we already have an instance, return that.

            if tensor == None:  #  => state does not have an instance.
                # Reset tensor to point to tensor we know not to be Null
                tensor = is_same_indices
                components = tensor[tensor_ref.indices]

                # Handling any changes in order of indices.
                diff_order = tensor.indices.get_reshape(tensor_ref.indices)
                if diff_order == None:
                    # No order changes => just init new instance with new indices.
                    new_tensor : EinsteinArray = type(tensor)(tensor_ref.indices, components, tensor.basis)
                    return new_tensor.subcomponents if tensor_ref.is_calling_tensor_subcomponent else new_tensor

                new_tensor = tensor.reshape_tensor_components(tensor_ref.indices)
                return new_tensor.subcomponents if tensor_ref.is_calling_tensor_subcomponent else new_tensor
            
            # We need to generate the new subcomponents if user is calling new subcomponents
            tensor = type(tensor)(tensor_ref.indices, tensor.components, tensor.basis)
            if tensor_ref.is_calling_tensor_subcomponent and tensor_ref.indices.anyrunnig:
                return tensor.subcomponents

            return (
                tensor[tensor_ref.indices] if tensor_ref.indices.anyrunnig or tensor_ref.is_calling_tensor_subcomponent else tensor
            )
        

        has_same_rank = self.state.match_on_tensors(
            TensorEqualityType.RankEquality, tensor_ref
        )
        if has_same_rank != None:
            new_tensor : EinsteinArray = type(has_same_rank)(
                tensor_ref.indices, has_same_rank.components, has_same_rank.basis
            )
            self.state.set_tensor(tensor_ref, new_tensor)
            return new_tensor.subcomponents if tensor_ref.is_calling_tensor_subcomponent else new_tensor

        # last thing we do is generate a new instance of the tensor.
        return self.generate_tensor(tensor_ref)

    def metric(self, tensor_ref: TensorReference):
        metric_definition = self.state.get_metric()
        metric_components = metric_definition[tensor_ref.indices.get_non_running()]
        metric = Metric(tensor_ref.indices, metric_components, metric_definition.basis)
        self.state.set_tensor(tensor_ref, metric)
        return metric

    def ricci(self, tensor_ref: TensorReference):
        ricci = Ricci(tensor_ref.indices, self.state.get_metric())
        self.state.set_tensor(tensor_ref, ricci)
        return ricci

    def riemann(self, tensor_ref: TensorReference):
        riemann = Riemann(tensor_ref.indices, self.state.get_metric())
        self.state.set_tensor(tensor_ref, riemann)
        return riemann

    def connection(self, tensor_ref: TensorReference):
        connection = Connection(indices=tensor_ref.indices, symbols=self.state.get_metric())
        self.state.set_tensor(tensor_ref, connection)
        return connection

    def einstein_tensor(self, tensor_ref: TensorReference):
        tensor = EinsteinTensor(tensor_ref.indices, self.state.get_metric())
        self.state.set_tensor(tensor_ref, tensor)
        return tensor

    def generate_tensor(self, tensor_ref: TensorReference):
        str_key = tensor_ref.id

        metric_defined = self.state.has_metric()
        if not metric_defined:  # Without metric, we cannot generate anything.
            return RelPyError()

        if str_key == self.state.metric_symbol:
            return self.metric(tensor_ref)
        elif str_key == self.state.derivative_symbol:
            return Derivative(tensor_ref.indices, self.state.coordinates)
        elif str_key == self.state.reimann_symbol:
            return self.riemann(tensor_ref)
        elif str_key == self.state.ricci_symbol:
            return self.ricci(tensor_ref)
        elif str_key == self.state.connection_symbol:
            return self.connection(tensor_ref)
        elif str_key == self.state.einstein_tensor_symbol:
            return self.einstein_tensor(tensor_ref)

class TensorAssignmentHandler:
    def __init__(self, state: WorkbookState):
        self.state = state

    def handle(self, tref: TensorReference):

        tref.indices.basis = (
            self.state.coordinates
        )  # Error handling needed => if no coordinates defined cannot continue


        if tref.is_metric and tref.tensor.components_definition_type == 'array':
            new_tensor = self.init_metric(tref)
        elif tref.tensor.components_definition_type == 'array':
            new_tensor = self.init_tensor(tref)
        elif tref.tensor.components_definition_type == 'tensor':
            einstein_array_obj : EinsteinArray = tref.tensor.component_ast_result
            new_tensor = einstein_array_obj.reshape_tensor_components(tref.indices)

        self.state.set_tensor(tref, new_tensor)


    def init_metric(self, tref: TensorReference):
        return Metric(
            tref.indices,
            tref.tensor.component_ast_result,
            self.state.coordinates,
        )
        

    def init_tensor(self, tref: TensorReference):
        return EinsteinArray(
                tref.indices,
                tref.tensor.component_ast_result,
                self.state.coordinates,
            )
        

class InitTensorFromComponentsNode: # TODO: NEEEDS CAREFUL THOUGHT AND RE-IMPLEMENTATION
    
    def __init__(self, state: WorkbookState):
        self.state = state

    def handle(self, node: TensorNode):
        tref : TensorReference = TensorReference(node)
        rhs = node.args[1]

        tref.indices.basis = (
            self.state.coordinates
        ) 

        self.state.set_tensor(tref, EinsteinArray(tref.indices, rhs, self.state.coordinates))


class DefinitionNode:
    def __init__(self, state: WorkbookState):
        self.state = state

    def handle(self, node: TensorNode):
        key = node.args[0]
        coordinates = node.args[1]

        self.state.set_coordinates(coordinates)
        self.state.set_variable(key, coordinates)

