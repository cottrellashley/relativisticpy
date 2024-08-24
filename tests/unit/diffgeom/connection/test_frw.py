import pytest
import sympy as sp

from relativisticpy.diffgeom import LeviCivitaConnection


# Function to define expected Christoffel symbols for FRW metric based on your corrections
def expected_frw_christoffels():
    # Define necessary symbols for FRW metric
    t, r, theta, phi, k = sp.symbols('t r theta phi k')
    a = sp.Function('a')(t)  # Scale factor as a function of time
    dot_a = sp.diff(a, t)  # Time derivative of the scale factor
    # Christoffel symbols from the uploaded image
    gamma_t_rr = a * dot_a / (1 - k * r**2)
    gamma_t_thetatheta = a * dot_a * r**2
    gamma_t_phiphi = a * dot_a * r**2 * sp.sin(theta)**2
    gamma_r_tr = gamma_r_rt = dot_a / a
    gamma_phi_phit = gamma_phi_tphi = dot_a / a
    gamma_r_thetatheta = -r * (1 - k * r**2)
    gamma_r_phiphi = -r * (1 - k * r**2) * sp.sin(theta)**2
    gamma_theta_rtheta = gamma_theta_thetar = 1 / r
    gamma_theta_ttheta = gamma_theta_thetat = dot_a / a
    gamma_theta_phiphi = -sp.sin(theta) * sp.cos(theta)
    gamma_phi_rphi = gamma_phi_phir = 1 / r
    gamma_phi_thetaphi = gamma_phi_phitheta = sp.cot(theta)
    gamma_r_rr = k*r / (1 - k * r**2)  # Additional term provided in your image

    non_zero = {
        (0, 1, 1): gamma_t_rr,
        (0, 2, 2): gamma_t_thetatheta,
        (0, 3, 3): gamma_t_phiphi,
        (1, 0, 1): gamma_r_tr,
        (1, 1, 0): gamma_r_rt,
        (1, 1, 1): gamma_r_rr,
        (1, 2, 2): gamma_r_thetatheta,
        (1, 3, 3): gamma_r_phiphi,
        (2, 1, 2): gamma_theta_rtheta,
        (2, 2, 1): gamma_theta_thetar,
        (2, 3, 3): gamma_theta_phiphi,
        (3, 1, 3): gamma_phi_rphi,
        (3, 3, 1): gamma_phi_phir,
        (3, 2, 3): gamma_phi_thetaphi,
        (3, 3, 2): gamma_phi_phitheta,
        (1, 0, 1): gamma_r_tr,
        (2, 0, 2): gamma_theta_ttheta,
        (2, 2, 0): gamma_theta_thetat,
        (3, 0, 3): gamma_phi_tphi,
        (3, 3, 0): gamma_phi_phit,
    }

    # Include zeros for all unspecified components
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if (i, j, k) not in non_zero:
                    non_zero[(i, j, k)] = 0
    return non_zero

# Pytest to test the FRW metric Christoffel symbols
def test_frw_levi_civita_connection(frw_metric):
    metric = frw_metric
    connection = LeviCivitaConnection.components_from_metric(metric)

    expected = expected_frw_christoffels()

    for key, value in expected.items():
        calculated = sp.sympify(connection[key])
        expected_value = sp.sympify(value)
        print(f"Calculated Value: {calculated}", f"Calculated Value: {expected_value}")
        assert sp.simplify(calculated - expected_value) == 0, f"Mismatch in Christoffel symbol at index {key}"
