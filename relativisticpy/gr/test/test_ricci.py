import pytest
from relativisticpy.symengine import Symbol, Function, sin, cos, diff, SymbolArray
from relativisticpy.gr.tensors.ricci import Ricci
from relativisticpy.core import Metric, MetricIndices, Indices


@pytest.fixture
def schild_setup():
    # setup
    M = Symbol("M")
    r = Symbol("r")
    theta = Symbol("theta")
    t = Symbol("t")
    phi = Symbol("phi")

    # init
    indices = MetricIndices.from_string('_{a}_{b}')
    components = SymbolArray([[-(1 - M/r), 0, 0, 0], [0, 1/(1 - M/r), 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2*sin(theta)**2]])
    basis = [t, r, theta, phi]
    metric = Metric(indices, components, basis)

    # return objs
    return metric, basis


@pytest.fixture
def indices_a_b():
    return Indices.from_string('_{a}_{b}')


@pytest.fixture
def schild_ricci_result():
    x = Symbol("x")
    y = Symbol("y")
    result = SymbolArray([[x, 0], [0, y]])
    return result


def test_ricci_init_parameters(schild_setup, indices_a_b):
    indices = indices_a_b
    metric, basis = schild_setup
    ricci = Ricci(indices = indices, arg = metric, basis = basis)
    assert metric.components == ricci.metric.components
    assert ricci.dimention == 4
    assert ricci.basis == basis
    assert ricci.rank == (0, 2)

@pytest.mark.skip(reason="Test not implemented.")
def test_schild_ricci_symetries(schild_setup):
    pass

