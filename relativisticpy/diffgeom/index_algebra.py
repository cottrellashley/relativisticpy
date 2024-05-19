from algebras import Idx, Indices
from diffgeom.manifold import CoordinatePatch

class GrIndices(Indices):

    def __init__(self, coord_sys: CoordinatePatch, *args: Idx):
        self.coord_sys = coord_sys
        super().__init__(*args)
