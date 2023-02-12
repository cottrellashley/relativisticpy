from src.relativisticpy.tensor.data_structure import TensorObject
from src.relativisticpy.indices.pattern_identifier import TensorRepresentationIdentifier
from src.relativisticpy.indices.representations import IndexRepresentationA
from sympy import MutableDenseNDimArray

class GrTensor(TensorObject):
    def __init__(self,
                 components,
                 indices,
                 basis,
                 name : str = None
                 ):
        tensor = TensorRepresentationIdentifier(components, indices, basis)
        TensorObject.__init__(self,
                                components  = tensor.get_components(),
                                basis       = tensor.get_basis(),
                                indices     = tensor.get_indices(),
                                dimention   = tensor.get_dimention(),
                                shape       = tensor.get_shape(),
                                rank        = tensor.get_shape(),
                                scalar      = tensor.get_scalar_bool()
        )
        self.name = name

    def get_specified_components(self):
        return self.components[tuple(self.indices.slice)]

    def gr_tensor_as_dict(self):
        return {
                'name' : self.name,
                'components' : str(self.components),
                'basis' : str(self.basis),
                'dimention' : self.dimention,
                'indices' : IndexRepresentationA(self.indices).indices_as_dict()
                }
    
    def return_tensor_ans_with_numbers(self, number):
        D = self.dimention
        A = int(self.rank)
        Shape = self.return_tensor_ans_shape()
        return MutableDenseNDimArray(smp.ones(D**A)*number, Shape)
