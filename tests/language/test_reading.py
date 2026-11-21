r"""Reading as a person means it: what the front end makes of notation, run end to end.

The parser's own tests state structure; these state what a user sees when they
write the notation, through a :class:`~relativisticpy.Workbook`.
"""

import numpy as np
import pytest

from relativisticpy import Workbook

SCHWARZSCHILD = "Coordinates := [t, r, \\theta, \\phi]\ng_{\\mu\\nu} := Schwarzschild(M)"


@pytest.fixture
def schwarzschild():
    with Workbook() as wb:
        wb.exe(SCHWARZSCHILD)
        yield wb


class TestScripts:
    @pytest.mark.parametrize(
        ("unbraced", "braced"),
        [
            ("R^a_b", "R^{a}_{b}"),
            ("G^t_t", "G^{t}_{t}"),
            ("R^t_r", "R^{t}_{r}"),
            ("g^a_b", "g^{a}_{b}"),
            (r"\delta^a_b", r"\delta^{a}_{b}"),
            (r"\Gamma^r_{tt}", r"\Gamma^{r}_{tt}"),
            ("x^a_b", "x^{a}_{b}"),
        ],
    )
    def test_a_script_without_braces_is_one_letter(self, schwarzschild, unbraced, braced):
        # R^a_b has two indices, a up and b down, as TeX sets it: not one, named a_b
        assert str(schwarzschild.expr(unbraced)) == str(schwarzschild.expr(braced))

    def test_data_bound_with_mixed_indices_reads_back(self):
        with Workbook() as wb:
            wb["A^i_j"] = np.array([[1.0, 2.0], [3.0, 4.0]])
            assert wb.expr("A^i_i") == 5.0
            assert wb.expr("A^i_j").to_numpy().tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_names_written_with_an_underscore_stay_names(self):
        with Workbook() as wb:
            assert str(wb.expr("r_s := 2 G M\nx_0 := 3\n1 - r_s/r + x_0^2")) == "-2*G*M/r + 10"
