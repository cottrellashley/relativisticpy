from algebras import Idx, Indices
from diffgeom.manifold import CoordinatePatch

class CoordIdx(Idx):
    def __init__(self, coord: CoordinatePatch, *args, **kwargs):
        self.coord = coord  
        super().__init__(*args, **kwargs)

class GrIndices(Indices):

    def __init__(self, *args: Idx):
        super().__init__(*args)
 
 