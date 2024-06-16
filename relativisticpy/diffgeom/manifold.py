# Matifild class
# Patch class
# Coordinate System class
#   - Coordinate trasnformation
#   - Coordinate point -> point on manifild (Same on all coord systems)

# Manifold
# Patch
# Coordinate System/Patch
# Transformations between patches

from relativisticpy.algebras.einstein_summation import Indices
from relativisticpy.symengine import Symbol, Basic
from relativisticpy.symengine import SymbolArray

class Manifold:

    def __init__(self, name: str, dim: int):
        self.__name = name
        self.__dim = dim

    @property
    def name(self) -> str: return self.__name

    @property
    def dim(self) -> int: return self.__dim

class Patch:

    def __init__(self, name: str, manifold: Manifold):
        self.__name = name
        self.__manifold = manifold

    @property
    def name(self) -> str: return self.__name

    @property
    def manifold(self) -> Manifold: return self.__manifold

    @property
    def dim(self) -> int: return self.__manifold.dim

class CoordinatePatch:

    def __init__(self, patch: Patch, symbols: tuple[Symbol], bounds: tuple[Basic] = None):
        if len(symbols) != patch.dim:
            raise ValueError("The number of symbols must be equal to the dimension of the patch.")
        self.__patch = patch
        self.__symbols = symbols
        self.__transformations = {}
    
    @property
    def symbols(self) -> tuple[Symbol]: return self.__symbols

    @property
    def patch(self) -> Patch: return self.__patch

    @property
    def name(self) -> str: return self.__patch.name

    @property
    def dim(self) -> int: return self.__patch.dim

    @property
    def manifold(self) -> Manifold: return self.__patch.manifold

    def add_transformation(self, cord_patch: 'CoordinatePatch'):
        pass

    def transformation(self, to_cord_patch: 'CoordinatePatch'):
        pass

class CoordIndices(Indices):

    def __init__(self, *indices: Symbol, coord_patch: CoordinatePatch = None):
        super().__init__(*indices)
        self.__coord_patch = coord_patch
        self.__basis = coord_patch.symbols if coord_patch else None

    @property
    def basis(self) -> SymbolArray: return self.__basis

    @basis.setter
    def basis(self, basis: SymbolArray):
        self.__basis = basis

    @property
    def coord_symbols(self) -> CoordinatePatch: return self.__coord_patch.symbols

    @property
    def coord_patch(self) -> CoordinatePatch: return self.__coord_patch

    @property
    def manifold(self) -> Manifold: return self.__coord_patch.manifold

    @property
    def patch(self) -> Patch: return self.__coord_patch.patch

    @property
    def dim(self) -> int: return self.__coord_patch.dim
    


    # def get_transformed_metric(self, transformation):
    #     ############ STEP ONE: #################

    #     # - Get components from metric
    #     metric_components = self.components

    #     # - If component is a sympy object, substitute metric components from given transformation definitions.
    #     sub = (
    #         lambda i: i.subs(transformation.transformation.as_dict)
    #         if issubclass(type(i), Symbol)
    #         else i
    #     )

    #     # - New components is substituted components
    #     self.components = SymbolArray([[sub(i) for i in j] for j in metric_components])

    #     ############# STEP TWO: #################

    #     # - Get the jacobian matrix from transformation given.
    #     jacobian = lambda t, b: SymbolArray([[diff(j, i) for j in t] for i in b])
    #     components = jacobian(
    #         [i[1] for i in list(transformation.transformation.as_dict.items())],
    #         transformation.new_basis.as_symbol,
    #     )

    #     # - Acquire indices from metric given.
    #     a, b = [i.symbol for i in self.indices.indices]

    #     # - Generate a new index of the derivatives and new matric by na and nb (new a index and new b index)
    #     indices = "_{n" + a + "}^{" + a + "}_{n" + b + "}^{" + b + "}"

    #     # - Finally we multiply the metric with new substituted components with a tensor product of the jacobian.
    #     resulting_GRtensor = self * EinsumArray(
    #         tensorproduct(components, components),
    #         indices
    #     )

    #     new_components = simplify(resulting_GRtensor.components)
    #     new_indices = "_{n" + a + "}_{n" + b + "}"

    #     # Return simplified components
    #     return Metric(new_components, new_indices, transformation.new_basis.as_string)