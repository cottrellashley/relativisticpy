from relativisticpy.diffgeom.geotensor import GrTensor
from relativisticpy.diffgeom.metric import Metric
from relativisticpy.diffgeom.connection import LeviCivitaConnection

def type_map(type_a: str, type_b: str):
    {
        type(GrTensor).__name__: { type(GrTensor).__name__: GrTensor, type(Metric).__name__: GrTensor, type(LeviCivitaConnection).__name__: GrTensor },
    }