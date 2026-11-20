"""Light near a black hole.

Rays of light pass a Schwarzschild black hole of mass M (with G = c = 1). A ray
passing far away is bent by 4M/b, Einstein's deflection of starlight; a ray
passing closer than b = 3√3 M is captured. Both numbers come out of the
geodesic equation, integrated numerically.

    python examples/light.py

This needs NumPy and SciPy: pip install "relativisticpy[numeric]".
"""

import numpy as np

from relativisticpy import spacetimes
from relativisticpy.numeric import four_velocity, geodesic

M = 1.0
SCHWARZSCHILD = spacetimes.schwarzschild()


def ray(b, r0, length):
    """Light sent inwards from radius r0 with impact parameter b, until it falls in
    (reaches r = 2.5M, inside the photon sphere) or has travelled ``length``."""
    start = [0.0, r0, np.pi / 2, 0.0]  # t, r, θ, φ
    inward = -np.sqrt(1 - (1 - 2 * M / r0) * b**2 / r0**2)
    u = four_velocity(SCHWARZSCHILD, start, [inward, 0.0, b / r0**2], kind="null", M=M)
    return geodesic(SCHWARZSCHILD, start, u, (0.0, length), M=M, stop=lambda x, _: x[1] - 2.5 * M)


def deflection(b, r0=10_000.0):
    """How far a ray passing at impact parameter b is bent from a straight line."""
    path = ray(b, r0, length=2 * r0)
    r, phi = path["r"][-1], path["phi"][-1]
    straight = np.pi - np.arcsin(b / r0) - np.arcsin(b / r)  # the angle a straight line sweeps
    return phi - straight


def critical_impact_parameter(low=4.0, high=7.0, steps=30):
    """The impact parameter dividing captured rays from escaping ones, by bisection."""
    for _ in range(steps):
        middle = (low + high) / 2
        if ray(middle, r0=1000.0, length=3000.0).stopped:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def main():
    b = 100 * M
    einstein = 4 * M / b
    next_order = einstein + 15 * np.pi / 4 * (M / b) ** 2
    print(f"Light passing at b = {b:g}M is bent by {deflection(b):.5f} rad.")
    print(f"  Einstein's 4M/b:                      {einstein:.5f} rad")
    print(f"  with the next order, + 15π/4 (M/b)²:  {next_order:.5f} rad")

    critical = critical_impact_parameter()
    print(f"Light is captured below b = {critical:.5f}M,")
    print(f"  as the photon sphere predicts: 3√3 M = {3 * np.sqrt(3) * M:.5f}M.")


if __name__ == "__main__":
    main()
