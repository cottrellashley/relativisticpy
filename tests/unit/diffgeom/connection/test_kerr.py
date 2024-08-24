import pytest
import sympy as sp

from relativisticpy.algebras import Indices, Idx
from relativisticpy.diffgeom.connection import \
    LeviCivitaConnection  # Adjust the import according to your project structure

# Define the symbols
# Define necessary symbols
G, M, r, theta, phi, a = sp.symbols('G M r theta phi a')

import sympy as sp


def expected_christoffels_kerr():
    # Auxiliary quantities
    rho_squared = r ** 2 + a ** 2 * sp.cos(theta) ** 2
    Delta = r ** 2 - 2 * G * M * r + a ** 2

    # Non-zero Christoffel symbols for the Kerr metric
    gamma_t_tr = (r ** 2 - a ** 2 * sp.cos(theta) ** 2) / (Delta * rho_squared)
    gamma_t_ttheta = -2 * a * r * sp.cos(theta) * sp.sin(theta) / rho_squared
    gamma_r_tt = (Delta / rho_squared ** 3) * (r * (r ** 2 + a ** 2) - 2 * a ** 2 * r * sp.sin(theta) ** 2)
    gamma_r_rr = (r - M) / Delta
    gamma_r_thetatheta = -Delta / rho_squared
    gamma_r_phiphi = -(sp.sin(theta) ** 2 / rho_squared ** 3) * (
                (r ** 3 + r * a ** 2 - 2 * M * a ** 2 * r) * sp.sin(theta) ** 2 - a ** 2 * sp.sin(theta) ** 4 * (r - M))
    gamma_theta_rtheta = 1 / r
    gamma_theta_phiphi = -sp.sin(theta) * sp.cos(theta) * (1 + 2 * M * r * (r ** 2 + a ** 2) / rho_squared ** 2)
    gamma_phi_tr = a * (r - M) / (Delta * rho_squared)
    gamma_phi_ttheta = 2 * a * r * sp.cos(theta) / rho_squared ** 2

    # Dictionary to store non-zero Christoffel symbols, initially filled with known non-zero values
    non_zero = {
        (0, 1, 0): gamma_t_tr,
        (0, 0, 1): gamma_t_tr,
        (0, 0, 2): gamma_t_ttheta,
        (0, 2, 0): gamma_t_ttheta,
        (1, 0, 0): gamma_r_tt,
        (1, 1, 1): gamma_r_rr,
        (1, 2, 2): gamma_r_thetatheta,
        (1, 3, 3): gamma_r_phiphi,
        (2, 1, 2): gamma_theta_rtheta,
        (2, 2, 1): gamma_theta_rtheta,
        (2, 3, 3): gamma_theta_phiphi,
        (3, 1, 3): gamma_phi_tr,
        (3, 0, 1): gamma_phi_tr,
        (3, 0, 2): gamma_phi_ttheta,
        (3, 2, 0): gamma_phi_ttheta
    }

    # Fill in zero for all unspecified components
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if (i, j, k) not in non_zero:
                    non_zero[(i, j, k)] = 0
    return non_zero


@pytest.mark.skip(reason="no way of currently testing this")
def test_levi_civita_connection(kerr_metric):
    metric = kerr_metric
    connection = LeviCivitaConnection.components_from_metric(metric)

    exp = expected_christoffels_kerr()

    for key, value in exp.items():
        calculated = sp.sympify(connection[key])
        expected = sp.sympify(value)
        assert sp.simplify(calculated - expected) == 0, f"Mismatch in Christoffel symbol at index {key}"

