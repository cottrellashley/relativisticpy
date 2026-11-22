r"""Grids and sums over an index of data: a range written beside a definition declares the
space of its label, ``x_n := n h, \quad n = 0, \ldots, N-1``, and the definition is made for
each value of n; a sum whose body writes its variable on data, or on a list, ranges over the
whole of it, from its lower limit."""

import numpy as np
import pytest
import sympy

from relativisticpy import CompileError, IndexNotationError, Workbook
from relativisticpy.language import parse, to_sexpr
from relativisticpy.tensors import Space

GRID = "N := 8\nh := \\frac{1}{N}\nx_n := n\\,h, \\quad n = 0, \\ldots, N-1"


class TestRanges:
    def test_a_range_beside_a_definition_belongs_to_it(self):
        program = parse(r"x_n := n\,h, \quad n = 0, \ldots, N-1")
        (statement,) = program.statements
        assert to_sexpr(statement) == "(:= x_n (* n h) (∈ n (tuple 0 ⋯ (- N 1))))"
        assert statement.span.text == r"x_n := n\,h, \quad n = 0, \ldots, N-1"

    def test_a_range_for_what_the_definition_is_not_written_with_stays_apart(self):
        program = parse(r"a := 2, \quad n = 0, 1")
        assert [type(s).__name__ for s in program.statements] == ["Define", "Evaluate"]

    def test_a_grid_is_a_vector_over_its_range(self, wb):
        wb.exe(GRID)
        x = wb["x"]
        assert x.spaces == (Space("n = 0, …, 7", 8, tuple(map(str, range(8)))),)
        assert [sympy.Rational(v) for v in x.components] == [sympy.Rational(n, 8) for n in range(8)]
        assert wb.expr(r"\sum_n x_n") == sympy.Rational(7, 2)
        assert wb.expr("x_{3}") == sympy.Rational(3, 8)  # the element labelled 3

    def test_powers_quotients_and_constants_are_taken_for_each_value(self, wb):
        wb.exe("N := 4")
        for source, values in [
            (r"y_n := n^2 + 1, \quad n = 0, \ldots, N-1", [1, 2, 5, 10]),
            (r"y_n := \frac{1}{n + 1}, \quad n = 0, \ldots, N-1", [1, 1 / 2, 1 / 3, 1 / 4]),
            (r"y_n := 2^n, \quad n = 0, \ldots, N-1", [1, 2, 4, 8]),
            (r"y_n := 5, \quad n = 0, \ldots, N-1", [5, 5, 5, 5]),
        ]:
            wb.exe(source)
            assert np.allclose(np.asarray(wb["y"].to_numpy(), dtype=float), values), source

    def test_a_grid_of_many_points_is_numbers(self, wb):
        wb.exe(r"N := 1000" "\n" r"t_n := \frac{n}{N}, \quad n = 0, \ldots, N-1")
        assert np.allclose(wb["t"].to_numpy(), np.arange(1000) / 1000)
        wb.exe(r"f_n := \sin(2\pi t_n)")
        assert np.allclose(wb["f"].to_numpy(), np.sin(2 * np.pi * np.arange(1000) / 1000))

    def test_a_range_on_its_own_line_declares_the_label_for_the_program(self, wb):
        wb.exe("n = 0, \\ldots, 3\nx_n := n^2")
        assert list(wb["x"].components) == [0, 1, 4, 9]

    def test_a_range_of_values_one_apart_only(self, wb):
        with pytest.raises(CompileError, match="a range of values one apart is supported"):
            wb.exe(r"x_n := n, \quad n = 0, 2, \ldots, 8")

    def test_the_end_of_a_range_is_known_before_anything_is_computed(self, wb):
        with pytest.raises(CompileError, match="by what is not known before anything"):
            wb.exe(r"x_n := n, \quad n = 0, \ldots, M")

    def test_a_range_written_without_an_end_is_read_as_before(self, wb):
        with pytest.raises(CompileError, match="read but not computed"):
            wb.exe(r"n = 0, 1, 2, \ldots")


class TestSumsOverAnIndexOfData:
    def test_a_sum_over_data_is_a_contraction_over_its_space(self, wb):
        wb["x_t"] = np.array([1.0, 2.0, 3.0])
        assert wb.expr(r"\sum_t x_t") == 6.0
        assert wb.expr(r"\sum_{t=0}^{2} t\, x_t") == 8.0  # t counts from the lower limit
        assert wb.expr(r"\sum_{t=1}^{3} t\, x_t") == 14.0

    def test_a_sum_over_a_part_of_the_space_is_not_supported_yet(self, wb):
        wb["x_t"] = np.array([1.0, 2.0, 3.0])
        with pytest.raises(CompileError, match="a sum over a part of its space is not supported"):
            wb.expr(r"\sum_{t=0}^{1} x_t")

    def test_a_range_fixes_where_its_values_start(self, wb):
        wb.exe(GRID)
        with pytest.raises(CompileError, match="ranges over 0, …, 7, but \\\\sum counts it from 1"):
            wb.expr(r"\sum_{n=1}^{8} x_n")

    def test_the_other_indices_are_kept(self, wb):
        wb["R_{ta}"] = np.arange(6.0).reshape(3, 2)
        assert np.allclose(wb.expr(r"\sum_t R_{ta}").to_numpy(), [6.0, 9.0])

    def test_a_list_summed_over_whole_counts_from_the_lower_limit(self, wb):
        wb.exe("x := (2, 4, 6)")
        assert wb.expr(r"\sum_{i=1}^{3} i\, x_i") == 28
        assert wb.expr(r"\sum_{i=0}^{2} i\, x_i") == 16

    def test_a_sum_whose_body_has_no_data_is_as_before(self, wb):
        assert wb.expr(r"\sum_{n=1}^{4} n^2") == 30

    def test_what_is_summed_must_carry_the_index(self, wb):
        wb["x_t"] = np.array([1.0, 2.0, 3.0])
        with pytest.raises(IndexNotationError, match="sums what has no index"):
            wb.expr(r"\sum_t \sum_t x_t")


def test_the_grid_of_the_design(wb: Workbook):
    """``design/integrations.md``: the range of an index declares its space."""
    wb.exe("N := 4\nh := 2\n" r"x_n := n\,h, \quad n = 0, \ldots, N-1")
    assert list(wb["x"].components) == [0, 2, 4, 6]
