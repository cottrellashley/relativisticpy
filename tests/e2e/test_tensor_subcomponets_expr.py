import pytest
import sympy as smp
from relativisticpy.workbook.workbook import Workbook
from relativisticpy.diffgeom import (
    Ricci,
    Metric,
    RicciScalar,
    Riemann,
    KScalar,
    MetricScalar,
    LeviCivitaConnection,
)
from relativisticpy.gr.einstein import EinsteinTensor


# TODO: <<< PUT THIS FUNCTION SOMEWHERE ELSE + SIMPLIFY IT AS ITs IMPLEMENTATION LOOKS HORIBLE >>>>>>

def test_metric_sub_vector_getter():
    wb = Workbook()

    res = wb.exe(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                g_{a:0}_{b:0}
                g_{a:1}_{b:1}
                g_{a:2}_{b:2}
                g_{a:3}_{b:3}
                g_{a:1}_{b:0}
                g_{a:0}_{b:1}
                g_{a:0}_{b}
                g_{a}_{b:2}
    """
    )
    assert str(res[0]) == '2*G*M/r - 1'
    assert str(res[1]) == '1/(-2*G*M/r + 1)'
    assert str(res[2]) == 'r**2'
    assert str(res[3]) == 'r**2*sin(theta)**2'
    assert str(res[4]) == '0'
    assert str(res[5]) == '0'
    assert str(res[6]) == '[2*G*M/r - 1, 0, 0, 0]'
    assert str(res[7]) == '[0, 0, r**2, 0]'


def test_connection_sub_components():
    wb = Workbook()

    res = wb.exe(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                C^{a:0}_{b:0}_{c:0}
                C^{a}_{b:0}_{c:0}
                C^{a:0}_{b}_{c:0}
                C^{a:0}_{b:0}_{c}
    """
    )
    assert str(res[0]) == "0"
    assert str(res[1]) == "[0, G*M*(-2*G*M + r)/r**3, 0, 0]"
    assert str(res[2]) == "[0, G*M/(r*(-2*G*M + r)), 0, 0]"
    assert str(res[3]) == "[0, G*M/(r*(-2*G*M + r)), 0, 0]"


def test_ricci_sub_components():
    wb = Workbook()

    res = wb.exe(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                Ric_{a:0}_{b:0}
                Ric_{a:1}_{b}
    """
    )
    assert str(res[0]) == "0"
    assert str(res[1]) == "[0, 0, 0, 0]"


def test_riemann_sub_components():
    wb = Workbook()
    res = wb.exe(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                R^{a:0}_{b:0}_{c:0}_{n:0}
                R^{a}_{b:0}_{c:0}_{n:0}
                R^{a:0}_{b:1}_{c:0}_{n}
                R^{a:1}_{b}_{c}_{n:3}
        """
    )
    assert str(res[0]) == "0"
    assert str(res[1]) == "[0, 0, 0, 0]"
    assert str(res[2]) == "[0, 2*G*M/(r**2*(-2*G*M + r)), 0, 0]"
    assert str(res[3]) == "[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, -G*M*sin(theta)**2/r, 0, 0]]"


def test_riemann_sub_components_latex():
    wb = Workbook()

    res = wb.exe(
        """
                Coordinates := \\begin{bmatrix} 
                                    t & r & theta & phi 
                                \\end{bmatrix}
                \\vspace{5mm}
                g_{\\mu\\nu} := \\begin{bmatrix}
                                                -(1 - \\frac{2 \\cdot G \\cdot M}{r}) & 0 & 0 & 0 \\
                                                0 & \\frac{1}{1 - \\frac{2 \\cdot G \\cdot M}{r}} & 0 & 0 \\
                                                0 & 0 & r^2 & 0 \\
                                                0 & 0 & 0 & r^2 \sin(\\theta)^2
                                 \\end{bmatrix}
                \\newline
                R^{a=0}_{b=0 c=0 n=0}
                \\vspace{5mm}
                R^{a}_{b:0 c=0 n=0}
                \\linebreak[4]
                R^{a=0}_{b=1 c=0 n}
                \\vspace{5mm}
                R^{a=1}_{b c n=3}
        """
    )
    assert str(res[0]) == "riemann_components[0, 0, 0, 0]"
    assert str(res[1]) == "riemann_components[:, 0, 0, 0]"
    assert str(res[2]) == "riemann_components[0, 1, 0, :]"
    assert str(res[3]) == "riemann_components[1, :, :, 3]"
