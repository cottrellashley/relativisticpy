r"""Python objects read by name: ``\texttt{prices}`` is the object called prices in the scope
a workbook is given, read by a grammar of its own, never run as Python, and what it becomes
is decided by the object integrations: a number a scalar, an array data, a table labelled
data, a SymPy expression an expression, a list a vector or a matrix, a function called."""

import fractions
import math
import types

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import sympy

from relativisticpy import Workbook
from relativisticpy.documents import Document
from relativisticpy.errors import (
    CompileError,
    DataError,
    EvaluationError,
    IndexNotationError,
    UsageError,
)
from relativisticpy.integrations import _objects as objects
from relativisticpy.tensors import Tensor

x = sympy.Symbol("x")


@pytest.fixture
def prices():
    dates = pd.Index(["2026-11-16", "2026-11-17", "2026-11-18"], name="date")
    columns = {
        "AAPL": [230.0, 232.0, 234.0],
        "MSFT": [410.0, 412.0, 417.0],
        "NVDA": [140.0, 146.0, 143.0],
    }
    return pd.DataFrame(columns, index=dates)


def hann(n, N):
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))


# -- the scope --------------------------------------------------------------------------------


def test_a_name_is_read_from_a_mapping_a_module_or_an_object():
    assert Workbook(scope={"G": 2}).expr(r"\texttt{G} + 1") == 3
    module = types.ModuleType("constants")
    module.c = 3
    assert Workbook(scope=module).expr(r"\texttt{c}^2") == 9
    assert Workbook(scope=types.SimpleNamespace(h=5)).expr(r"2 \texttt{h}") == 10


def test_a_scope_is_a_mapping_or_what_has_attributes():
    for wrong in ("globals", 3, [1, 2]):
        with pytest.raises(UsageError, match="a scope is a mapping, as globals"):
            Workbook(scope=wrong)


def test_attributes_keys_and_positions_follow(prices):
    table = types.SimpleNamespace(x=[1.0, 2.0], rows={"first": 7})
    wb = Workbook(scope={"prices": prices, "table": table, "data": [10, 20, 30]})
    assert wb.expr(r"\texttt{data[1]} + \texttt{data[-1]}") == 50
    assert wb.expr(r"\texttt{table.rows['first']}") == 7
    wb.exe(r"y_t := \texttt{prices[" + '"NVDA"' + r"]}")
    assert wb["y"].to_numpy().tolist() == [140.0, 146.0, 143.0]


def test_an_escaped_underscore_is_read_as_one():
    wb = Workbook(scope={"M_sun": 2})
    assert wb.expr(r"\texttt{M\_sun} + \texttt{M_sun}") == 4


def test_a_reference_is_never_python_to_run():
    wb = Workbook(scope={"prices": {"a": 1}, "os": math})
    for written in ("prices.keys()", "prices['a'] + 1", "1 + 1", "prices[a]", "my prices"):
        with pytest.raises(CompileError, match="is not a reference to a Python object") as raised:
            wb.expr(r"\texttt{" + written + "}")
        assert r"\texttt{prices}" in raised.value.hint
        assert "never run as Python" in raised.value.hint
    with pytest.raises(CompileError, match="reads __class__, which is Python's own"):
        wb.expr(r"\texttt{prices.__class__}")


def test_what_cannot_be_found_is_said_with_what_there_is(prices):
    wb = Workbook(scope={"prices": prices})
    with pytest.raises(
        CompileError, match=r"names price, which is not in the Python scope"
    ) as raised:
        wb.expr(r"\texttt{price}")
    assert raised.value.notes == ["did you mean prices?"]
    assert "show(source, price=price)" in raised.value.hint
    with pytest.raises(CompileError, match="prices has no attribute colums") as raised:
        wb.expr(r"\texttt{prices.colums}")
    assert raised.value.notes == ["did you mean prices.columns?"]
    with pytest.raises(CompileError, match=r'prices has nothing at \["TSLA"\]') as raised:
        wb.expr(r"\texttt{prices['TSLA']}")
    assert raised.value.notes == ["its keys are AAPL, MSFT, NVDA"]
    assert raised.value.span.text == r"\texttt{prices['TSLA']}"


def test_without_a_scope_a_program_reaches_nothing_in_python():
    with pytest.raises(CompileError, match="no Python scope was given") as raised:
        Workbook().expr(r"\texttt{os} + 1")
    assert raised.value.hint.startswith("give the workbook one, as Workbook(scope=globals())")
    assert "relativisticpy check" in raised.value.notes[0]


def test_a_document_checked_has_no_scope():
    report = Workbook().check(
        Document("\\begin{equation}\nx = \\texttt{prices}\n\\end{equation}\n")
    )
    (finding,) = report.findings
    assert finding.verdict.value == "not understood"
    assert "no Python scope was given" in finding.detail
    assert "Workbook(scope=globals())" in report.describe(all=True)


# -- what an object becomes -------------------------------------------------------------------


def test_a_number_is_a_scalar_exact_where_pythons_is():
    wb = Workbook(scope={"third": fractions.Fraction(1, 3), "G": 6.6743e-11, "yes": True})
    assert wb.expr(r"\texttt{third} + \frac{2}{3}") == 1
    assert wb.expr(r"\texttt{G} \cdot 10^{11}") == pytest.approx(6.6743)
    assert wb.expr(r"\texttt{yes}") is sympy.true


def test_an_array_is_data_over_spaces_of_its_size():
    wb = Workbook(scope={"signal": np.array([1.0, 2.0, 3.0, 6.0]), "grid": np.ones((2, 3))})
    signal = wb.expr(r"\texttt{signal}")
    assert isinstance(signal, Tensor)
    assert [space.name for space in signal.spaces] == ["the entries of signal"]
    grid = wb.expr(r"\texttt{grid}")
    assert [(space.name, space.dim) for space in grid.spaces] == [
        ("the rows of grid", 2),
        ("the columns of grid", 3),
    ]
    wb.exe(r"x_t := \texttt{signal}")
    assert wb.expr(r"\mathbb{E}_t[x_t]") == 3.0
    assert Workbook(scope={"half": np.float64(0.5)}).expr(r"2 \texttt{half}") == 1.0


def test_a_table_is_bound_as_wb_binds_it(prices):
    read = Workbook(scope={"prices": prices})
    read.exe(r"P_{ta} := \texttt{prices}")
    bound = Workbook()
    bound["P_{ta}"] = prices
    assert read["P"].spaces == bound["P"].spaces
    assert read["P"].to_numpy().tolist() == bound["P"].to_numpy().tolist()
    assert read.expr(r"\mathbb{E}_t[P_{ta}]").to_numpy().tolist() == [232.0, 413.0, 143.0]
    assert read.expr(r"P_{t=2\, a:NVDA}") == 143.0


def test_data_from_two_objects_meets_by_label_never_by_position(prices):
    weights = pd.Series([0.2, 0.5, 0.3], index=pd.Index(["NVDA", "AAPL", "MSFT"], name="asset"))
    wb = Workbook(scope={"prices": prices, "weights": weights})
    wb.exe(r"P_{ta} := \texttt{prices}; w^a := \texttt{weights}")
    expected = prices.to_numpy() @ weights.reindex(prices.columns).to_numpy()
    assert wb.expr(r"P_{ta} w^a").to_numpy() == pytest.approx(expected)
    assert wb["w"].spaces == (wb["P"].spaces[1],)
    other = pd.Series([1.0, 2.0, 3.0], index=["AAPL", "MSFT", "TSLA"])
    wb = Workbook(scope={"prices": prices, "other": other})
    wb.exe(r"P_{ta} := \texttt{prices}; v^a := \texttt{other}")
    with pytest.raises(IndexNotationError, match="the index a ranges over"):
        wb.expr(r"P_{ta} v^a")


def test_labelled_data_bound_from_python_meets_by_label_too(prices):
    wb = Workbook()
    wb["P_{ta}"] = prices
    wb["w^a"] = pd.Series([0.5, 0.5, 0.0], index=pd.Index(["MSFT", "AAPL", "NVDA"], name="asset"))
    assert wb.expr(r"P_{ta} w^a").to_numpy().tolist() == [320.0, 322.0, 325.5]


def test_data_takes_the_indices_written_on_it(prices):
    weights = pd.Series([0.2, 0.5, 0.3], index=pd.Index(["NVDA", "AAPL", "MSFT"], name="asset"))
    wb = Workbook(scope={"signal": np.array([1.0, 2.0, 6.0]), "grid": np.eye(2), "c": 3})
    assert wb.expr(r"\mathbb{E}_t[\texttt{signal}_t]") == 3.0
    assert wb.expr(r"\texttt{grid}_{ii}") == 2.0  # an index written twice is summed
    assert wb.expr(r"\texttt{grid}_{ij}").spaces[0] == wb.expr(r"\texttt{grid}_{ij}").spaces[1]
    market = Workbook(scope={"prices": prices, "weights": weights})
    expected = prices.to_numpy() @ weights.reindex(prices.columns).to_numpy()
    assert market.expr(r"\texttt{prices}_{ta} \texttt{weights}^a").to_numpy() == pytest.approx(
        expected
    )
    with pytest.raises(CompileError, match="is a number, which has no indices"):
        wb.expr(r"\texttt{c}_1")
    with pytest.raises(DataError, match=r"has 1 index, but the data has 2 axes"):
        wb.expr(r"\texttt{grid}_i")


def test_an_object_is_never_taken_for_a_name_of_the_languages():
    signal = np.array([1.0, 2.0, 6.0])
    assets = pd.Index(["A", "B"], name="asset")
    scope = {
        "Var": signal,
        "E": signal,
        "x": signal,
        "cov": pd.DataFrame([[2.0, 0.0], [0.0, 3.0]], index=assets, columns=assets),
        "w": pd.Series([1.0, 1.0], index=assets),
        "tr": 2,
        "J": lambda n: n + 100,
    }
    wb = Workbook(scope=scope)
    assert wb.expr(r"\texttt{Var}_t \texttt{x}_t") == 41.0  # a contraction, not a variance
    assert wb.expr(r"\texttt{E}_t[\texttt{x}_t]") == 41.0  # not the expectation
    assert wb.expr(r"\texttt{cov}_{ab} \texttt{w}^a \texttt{w}^b") == 5.0
    assert wb.expr(r"\texttt{tr} \cdot 3") == 6  # not the trace, a book name
    assert wb.expr(r"\texttt{J}(0)") == 100  # not the Bessel function
    steps = list(wb.each(r"\texttt{cov}_{ab} \texttt{w}^a \texttt{w}^b"))
    assert steps[0].notes == []
    with pytest.raises(CompileError, match=r"what follows a Python object's name is not part"):
        wb.expr(r"\texttt{J}_0(3)")


def test_a_prime_or_an_accent_on_an_object_is_not_read():
    wb = Workbook(scope={"f": sympy.sin(x), "c": 3})
    for written in (r"\texttt{f}'", r"\texttt{f}'(x)", r"\hat{\texttt{c}}", r"\dot{\texttt{c}}"):
        with pytest.raises(CompileError, match="a prime or an accent on a Python object"):
            wb.expr(written)


def test_data_read_is_a_copy_and_read_anew_each_time():
    signal = np.array([1.0, 2.0, 3.0])
    wb = Workbook(scope={"signal": signal})
    wb.exe(r"x_t := \texttt{signal}")
    signal[0] = 100.0
    assert wb["x"].to_numpy().tolist() == [1.0, 2.0, 3.0]
    wb.exe(r"x_t := \texttt{signal}")
    assert wb["x"].to_numpy().tolist() == [100.0, 2.0, 3.0]


def test_a_sparse_matrix_stays_sparse():
    adjacency = scipy.sparse.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    wb = Workbook(scope={"adjacency": adjacency})
    wb.exe(r"A_{ij} := \texttt{adjacency}")
    assert scipy.sparse.issparse(wb["A"].components)
    assert wb.expr(r"A_{ij} 1^j").to_numpy().tolist() == [1.0, 1.0]


def test_a_sympy_expression_is_an_expression_in_the_programs_symbols():
    positive = sympy.Symbol("x", positive=True)
    wb = Workbook(scope={"f": sympy.sin(x) ** 2, "g": positive**3})
    assert wb.expr(r"\frac{d}{dx} \texttt{f}") == 2 * sympy.sin(x) * sympy.cos(x)
    assert wb.expr(r"\frac{d}{dx} \texttt{g}") == 3 * x**2  # the program's x, by name
    declared = Workbook(scope={"h": sympy.Symbol("y") ** 2})
    declared.exe("Positive := [y]")
    assert declared.expr(r"\sqrt{\texttt{h}}") == sympy.Symbol("y", positive=True)


def test_what_makes_itself_sympy_is_an_expression():
    class Quantity:
        def _sympy_(self):
            return 2 * x

    assert Workbook(scope={"q": Quantity()}).expr(r"\texttt{q} + x") == 3 * x


def test_lists_and_sympy_matrices_are_vectors_and_matrices():
    wb = Workbook(scope={"A": [[1, 2], [3, 4]], "M": sympy.Matrix([[x, 1], [1, x]]), "v": (1, 2)})
    assert wb.expr(r"\det(\texttt{A})") == -2
    assert wb.expr(r"\texttt{A}^{-1}").tolist() == [
        [-2, 1],
        [sympy.Rational(3, 2), -sympy.Rational(1, 2)],
    ]
    assert wb.expr(r"\det(\texttt{M})") == x**2 - 1
    wb.exe(r"w^a := \texttt{v}")
    assert wb["w"].to_numpy().tolist() == [1.0, 2.0]


def test_a_negative_power_of_singular_data_is_refused_as_its_exact_value_is():
    # SciPy's fractional power of a singular matrix is a matrix of nan, shown as a value
    wb = Workbook(scope={"M": np.array([[1.0, 2.0], [2.0, 4.0]])})
    wb.exe(r"A^{a}{}_{b} := \texttt{M}")
    for power in ("-1/2", "-1"):
        with pytest.raises(EvaluationError, match="this matrix has no inverse"):
            wb.exe(f"matrix_power(A^{{a}}{{}}_{{b}}, {power})")


def test_a_python_list_is_a_list_to_the_reader_as_the_languages_own_are():
    """Finding 21 of the review of the reading stack: a Python list or matrix bound from
    the scope was a constant array, which nothing that reads a list recognised: a sum
    over its elements was ``3 x_i``, ``x_1`` a new symbol, ``|v|`` and ``v \\cdot v``
    refused, and ``A^T`` a power. It is lowered as the lists a program writes are."""
    scope = {"x": [1.0, 2.0, 3.0], "v": [3.0, 4.0], "L": [[1.0, 2.0], [3.0, 4.0]]}
    wb = Workbook(scope=scope)
    wb.exe(r"x := \texttt{x}" "\n" r"v := \texttt{v}" "\n" r"A := \texttt{L}")
    assert float(wb.expr(r"\sum_{i=1}^{3} x_i")) == 6.0
    with pytest.raises(CompileError, match="x is a list"):
        wb.expr("x_1")  # as of a list the program writes: its positions have no start
    assert float(wb.expr("|v|")) == 5.0
    assert float(wb.expr(r"v \cdot v")) == 25.0
    transpose = [[float(entry) for entry in row] for row in wb.expr("A^T").tolist()]
    assert transpose == [[1.0, 3.0], [2.0, 4.0]]
    assert "core.list" in wb.ir(r"\texttt{x}")
    own = Workbook()
    own.exe("x := [1.0, 2.0, 3.0]\nv := [3.0, 4.0]\nA := [[1.0, 2.0], [3.0, 4.0]]")
    for written in (r"\sum_{i=1}^{3} x_i", "|v|", r"v \cdot v"):  # the two read alike
        assert wb.expr(written) == own.expr(written)
    assert wb.expr("A^T").tolist() == own.expr("A^T").tolist()


def test_data_whose_indices_no_one_wrote_names_none_in_its_hint():
    wb = Workbook(scope={"M": np.array([[1.0, 2.0], [3.0, 4.0]])})
    wb.exe(r"B := \texttt{M}")
    with pytest.raises(IndexNotationError) as raised:
        wb.expr("B^T")
    assert raised.value.hint == "for example B_{ab}"  # was B_{#i}_{#j}


def test_lists_that_are_no_vector_are_refused():
    with pytest.raises(DataError, match=r"\\texttt\{v\}: the object is ragged: entry 2 has 1"):
        Workbook(scope={"v": [[1, 2], [3]]}).expr(r"\texttt{v}")
    with pytest.raises(DataError, match="holds text, 'x'"):
        Workbook(scope={"v": [1, "x"]}).expr(r"\texttt{v}")


def test_a_type_that_is_not_read_is_refused_naming_what_is():
    with pytest.raises(
        DataError, match=r"\\texttt\{weights\} is a dict, which RelativisticPy"
    ) as raised:
        Workbook(scope={"weights": {"a": 1}}).expr(r"\texttt{weights}")
    (note,) = raised.value.notes
    for kind in ("numpy.ndarray", "pandas.DataFrame", "sympy.Basic", "list", "anything callable"):
        assert kind in note
    assert r'\texttt{weights["key"]}' in raised.value.hint
    with pytest.raises(DataError, match="is a str") as raised:
        Workbook(scope={"name": "x^2"}).expr(r"\texttt{name}")
    assert "text is not a value" in raised.value.hint


# -- functions --------------------------------------------------------------------------------


def test_a_function_is_called_on_numbers():
    wb = Workbook(scope={"hann": hann, "cube": lambda n: n**3})
    assert wb.expr(r"\texttt{hann}(3, 8)") == pytest.approx(hann(3, 8))
    assert wb.expr(r"\frac{d}{dx} \texttt{cube}(x)") == 3 * x**2
    assert wb.expr(r"\sum_{n=0}^{3} \texttt{cube}(n)") == 36


def test_a_function_applies_elementwise_to_data():
    signal = np.array([0.0, 1.0, 2.0, 3.0])
    wb = Workbook(scope={"hann": hann, "signal": signal, "floor": math.floor})
    wb.exe(r"x_t := \texttt{signal}")
    assert wb.expr(r"\texttt{hann}(x_t, 8)").to_numpy() == pytest.approx(hann(signal, 8))
    assert wb.expr(r"\texttt{floor}(x_t / 2)").to_numpy().tolist() == [0.0, 0.0, 1.0, 1.0]
    assert wb.expr(r"\texttt{hann}(x_t, 8)").spaces == wb["x"].spaces


def test_a_function_applies_to_each_symbolic_component():
    wb = Workbook(scope={"square": lambda value: value**2})
    wb.exe(r"\mathcal{V} := \mathbb{R}^{2}; a \in \mathcal{V}; v^{a} := [x, 2]")
    assert wb.expr(r"\texttt{square}(v^a)").components.tolist() == [x**2, 4]


def test_the_tensors_a_function_is_given_have_the_same_indices():
    wb = Workbook(scope={"add": lambda a, b: a + b})
    wb["x_t"] = np.array([1.0, 2.0])
    wb["y_s"] = np.array([1.0, 2.0])
    with pytest.raises(IndexNotationError, match=r"the free indices of the arguments of"):
        wb.expr(r"\texttt{add}(x_t, y_s)")


def test_a_name_defined_as_a_function_is_called_as_it_is():
    wb = Workbook(scope={"hann": hann})
    wb.exe(r"w := \texttt{hann}")
    assert wb.expr("w(3, 8)") == pytest.approx(hann(3, 8))
    with pytest.raises(CompileError, match=r"is a Python function, so it is called"):
        wb.expr(r"\texttt{hann} + 1")


def test_a_number_or_expression_followed_by_brackets_is_no_call():
    wb = Workbook(scope={"G": 2, "f": sympy.sin(x)})
    assert wb.expr(r"\texttt{G}(3 + 1)") == 8
    assert wb.expr(r"\texttt{f}(x)") == sympy.sin(x)
    with pytest.raises(CompileError, match="is data of 2 numbers, not a function"):
        Workbook(scope={"signal": np.ones(2)}).expr(r"\texttt{signal}(1)")


def test_what_a_function_does_wrong_is_its_own():
    def broken(n):
        raise ZeroDivisionError("division by zero")

    wb = Workbook(scope={"broken": broken, "listed": lambda n: [n, n]})
    with pytest.raises(EvaluationError, match=r"\\texttt\{broken\}\(1\) raised ZeroDivisionError"):
        wb.expr(r"\texttt{broken}(1)")
    with pytest.raises(EvaluationError, match="gave a list, not a number"):
        wb.expr(r"\texttt{listed}(1)")


# -- reading, explained ------------------------------------------------------------------------


def test_explain_says_what_each_object_became(prices):
    wb = Workbook(scope={"prices": prices, "G": 6.6743e-11})
    text = str(wb.explain(r"P_{ta} := \texttt{prices}" "\n" r"\texttt{G} + 1"))
    assert r"\texttt{prices}: the Python object prices, data of 3 × 3 numbers" in text
    assert r"\texttt{G}: the Python object G, a number" in text
    assert "P" not in wb


def test_a_copy_and_a_reset_keep_the_scope():
    wb = Workbook(scope={"G": 2})
    assert wb.copy().expr(r"\texttt{G}") == 2
    wb.reset()
    assert wb.expr(r"\texttt{G}") == 2


def test_references_are_parsed_by_their_own_grammar():
    read = objects.reference(r'prices["NVDA"].values[0]')
    assert (read.name, read.steps) == (
        "prices",
        (("key", "NVDA"), ("attribute", "values"), ("key", 0)),
    )
    assert objects.reference(r"M\_sun").name == "M_sun"
    assert objects.reference("data [ -1 ]").steps == (("key", -1),)
