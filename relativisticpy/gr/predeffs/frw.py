from relativisticpy.algebras import Idx
from relativisticpy.diffgeom import Metric, MetricIndices, CoordinatePatch, Patch, Manifold
from relativisticpy.symengine import sin, Function, symbols
from relativisticpy.symengine.sympy import SymbolArray


class FRWMetric(Metric):
    def __init__(self, a, k, basis):
        t, r, theta, phi = symbols('t r theta phi')  # Define the symbols
        k = symbols('k')  # Spatial curvature constant
        a = Function('a')(t)  # 'a' is the scale factor, assumed to be a function of time t

        g_tt = -1
        g_rr = a ** 2 / (1 - k * r ** 2)
        g_thetatheta = a ** 2 * r ** 2
        g_phiphi = a ** 2 * r ** 2 * sin(theta) ** 2

        components = SymbolArray([
            [g_tt, 0, 0, 0],
            [0, g_rr, 0, 0],
            [0, 0, g_thetatheta, 0],
            [0, 0, 0, g_phiphi]
        ])

        coordinate_patch = CoordinatePatch(
            patch=Patch(
                name='frw',
                manifold=Manifold('universe', 4)
            ),
            symbols=SymbolArray([t, r, theta, phi])
        )

        indices = MetricIndices(Idx('a'), Idx('b'), coord_patch=coordinate_patch)

        super().__init__(indices, components)
