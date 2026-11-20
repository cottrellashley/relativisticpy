"""Shared helpers for the end-to-end tests."""

import pytest

from relativisticpy import Workbook

SCHWARZSCHILD = r"""
    Coordinates := [t, r, theta, phi]
    g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],
                    [0, 1 / (1 - (2 * G * M) / (r)), 0, 0],
                    [0, 0, r**2, 0],
                    [0, 0, 0, r**2 * sin(theta) ** 2]]
"""


@pytest.fixture
def wb():
    with Workbook() as workbook:
        yield workbook


@pytest.fixture
def schwarzschild(wb):
    wb.exe(SCHWARZSCHILD)
    return wb
