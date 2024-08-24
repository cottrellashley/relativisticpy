import pytest
import sympy as sp

from relativisticpy.algebras import Indices, Idx
from relativisticpy.diffgeom.connection import \
    LeviCivitaConnection  # Adjust the import according to your project structure

# Define the symbols
t, r, theta, phi = sp.symbols('t r theta phi')
G, M = sp.symbols('G M')


# Expected Christoffel symbols based on the Schwarzschild metric
# These should be simplified versions or directly input exact values for known indices
def expected_christoffels():
    # Define the Christoffel symbols using SymPy
    gamma_t_rt = gamma_t_tr = G * M / (r ** 2 * (1 - 2 * G * M / r))
    gamma_r_tt = G * M * (1 - 2 * G * M / r) / r ** 2
    gamma_r_rr = -G * M / (r ** 2 * (1 - 2 * G * M / r))
    gamma_r_thetatheta = -r * (1 - 2 * G * M / r)
    gamma_r_phiphi = -r * sp.sin(theta) ** 2 * (1 - 2 * G * M / r)
    gamma_theta_rtheta = gamma_theta_thetar = 1 / r
    gamma_theta_phiphi = -sp.sin(theta) * sp.cos(theta)
    gamma_phi_rphi = gamma_phi_phir = 1 / r
    gamma_phi_thetaphi = gamma_phi_phitheta = 1/sp.tan(theta)

    # Create a dictionary to store non-zero Christoffel symbols
    non_zero = {
        (0, 1, 0): gamma_t_rt,
        (0, 0, 1): gamma_t_tr,
        (1, 0, 0): gamma_r_tt,
        (1, 1, 1): gamma_r_rr,
        (2, 1, 2): gamma_theta_rtheta,  # Correction: Symmetry and using defined symbol
        (2, 2, 1): gamma_theta_thetar,  # Correction: Symmetry and using defined symbol
        (3, 1, 3): gamma_phi_rphi,  # Correction: Symmetry and using defined symbol
        (3, 3, 1): gamma_phi_phir,  # Correction: Symmetry and using defined symbol
        (1, 2, 2): gamma_r_thetatheta,
        (1, 3, 3): gamma_r_phiphi,
        (2, 3, 3): gamma_theta_phiphi,
        (3, 2, 3): gamma_phi_thetaphi,  # Correct use of already defined symbol
        (3, 3, 2): gamma_phi_phitheta  # Correct use of already defined symbol
    }

    for i in range(4):
        for j in range(4):
            for k in range(4):
                if (i, j, k) not in non_zero:
                    non_zero[(i, j, k)] = 0
    return non_zero


def test_levi_civita_connection(schwarzschild_metric):
    metric = schwarzschild_metric
    connection = LeviCivitaConnection.components_from_metric(metric)

    exp = expected_christoffels()

    for key, value in exp.items():
        calculated = sp.sympify(connection[key])
        expected = sp.sympify(value)
        assert sp.simplify(calculated - expected) == 0, f"Mismatch in Christoffel symbol at index {key}"

