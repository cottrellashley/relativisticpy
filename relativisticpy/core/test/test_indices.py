from itertools import product
import pytest
from relativisticpy.core.indices import Idx, Indices
from relativisticpy.core.metric import MetricIndices
from relativisticpy.deserializers import Mathify


@pytest.fixture
def indices_setup():
    idx1 = Idx("a", order=1, values=5, covariant=True)
    idx2 = Idx("b", order=2, values=4, covariant=True)
    indices = Indices(idx1, idx2)
    return indices, idx1, idx2


@pytest.fixture
def indices_product_setup():
    # Init indices.
    a1_f0_c1 = Indices(-Idx("a"), Idx("f"), -Idx("c"))
    a0_b0_c0 = Indices(Idx("a"), Idx("b"), Idx("c"))
    mu0_nu0 = Indices(Idx("mu"), Idx("nu"))
    mu1_nu1 = Indices(-Idx("mu"), -Idx("nu"))
    metric_a1_nu1 = MetricIndices(-Idx("a"), -Idx("nu"))
    metric_a0_nu0 = MetricIndices(Idx("a"), Idx("nu"))

    # Set basis for space.
    basis2D = Mathify("[t, r]")
    basis3D = Mathify("[t, r, theta]")

    a1_f0_c1.basis = basis3D
    a0_b0_c0.basis = basis3D
    mu0_nu0.basis = basis2D
    mu1_nu1.basis = basis2D
    metric_a0_nu0.basis = basis2D
    metric_a1_nu1.basis = basis2D

    # return objects
    return a1_f0_c1, a0_b0_c0, mu0_nu0, mu1_nu1, metric_a0_nu0, metric_a1_nu1


@pytest.fixture
def const_indices_product_setup():
    # Init indices.
    a1_f0_c1 = Indices(-Idx("a"), Idx("f", values=2), -Idx("c"))
    a0_b0_c0 = Indices(Idx("a"), Idx("b"), Idx("c", values=1))
    mu0_nu0 = Indices(Idx("mu", values=0), Idx("nu"))
    mu1_nu1 = Indices(-Idx("mu"), -Idx("nu", values=1))

    # Set basis for space.
    basis2D = Mathify("[t, r]")
    basis3D = Mathify("[t, r, theta]")

    a1_f0_c1.basis = basis3D
    a0_b0_c0.basis = basis3D
    mu0_nu0.basis = basis2D
    mu1_nu1.basis = basis2D

    # return objects
    return a1_f0_c1, a0_b0_c0, mu0_nu0, mu1_nu1


def test_len(indices_setup):
    indices, _, _ = indices_setup
    assert len(indices) == 2


def test_eq(indices_setup):
    indices, idx1, idx2 = indices_setup
    other_indices = Indices(idx1, idx2)
    assert indices == other_indices


def test_mul(indices_setup):
    a0_b0_c0, idx1, idx2 = indices_setup
    mu0_nu0 = Indices(idx1, -idx2)
    product_indices = a0_b0_c0 * mu0_nu0
    assert isinstance(product_indices, Indices)


def test_iteration(indices_setup):
    indices, _, _ = indices_setup
    for idx in indices.indices:
        assert isinstance(idx, Idx)


def test_iteration(indices_setup):
    indices, _, _ = indices_setup
    for idx in indices:
        assert isinstance(idx, tuple)


def test_getitem(indices_setup):
    indices, idx1, _ = indices_setup
    assert indices[idx1] == [
        idx
        for idx in indices.indices
        if idx.symbol == idx1.symbol and idx.covariant == idx1.covariant
    ]


def test_from_string(indices_setup):
    a0_b0_c0, idx1, idx2 = indices_setup
    indices_from_string_init = Indices.from_string("_{a}_{b}")

    assert indices_from_string_init.indices[0] == idx1
    assert indices_from_string_init.indices[1] == idx2
    assert indices_from_string_init == a0_b0_c0


# Testing Indices Product - (this is the core compute feature of index/indices object. => they were designed to around the simplification of this compute.)
# when we perform opertations between indices, a few things happen:
# 1. We return another index object which represents the einstein summed index as a result of the two indices that were summed.
# 2. We inject a method implementation into the callback method "generator" => which generates the indices which are to be summed based on the resulting indices comps.
# 1 and 2 together means we can iterate through the resulting indices object, which returns all which it needs to be iterated, and pass its own iteration in as an argument
# to the "generator" callback, which will take those components and workout the components which are to be summed.


def test_all_running_indices_iteration(indices_product_setup):
    a1_f0_c1, a0_b0_c0, mu0_nu0, mu1_nu1, _, _ = indices_product_setup

    # Easy one for visual explenantion
    mu0_nu0_iter = [i for i in mu0_nu0]  # mu0_nu0 = _{mu}_{nu} ; mu0_nu0.basis = [t, r]
    mu0_nu0_answer = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]  # Should be all combinations of (mu, nu) where mu, nu range over: len(basis) ==> 2

    mu1_nu1_iter = [i for i in mu1_nu1]  # mu1_nu1 = ^{mu}^{nu} ; mu1_nu1.basis = [t, r]
    mu1_nu1_answer = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]  # Should be all combinations of (mu, nu) where mu, nu range over: len(basis) ==> 2

    # longer ones using itertools.product
    a1_f0_c1_iter = [
        i for i in a1_f0_c1
    ]  # a1_f0_c1 = ^{a}_{f}^{c} ; a1_f0_c1.basis = [t, r, theta]
    a1_f0_c1_answer = [
        i for i in product((0, 1, 2), (0, 1, 2), (0, 1, 2))
    ]  # Should be all combinations of (a, f, c) where a, f, c range over: len(basis) ==> 3

    a0_b0_c0_iter = [
        i for i in a0_b0_c0
    ]  # a0_b0_c0 =  _{a}_{b}_{c} ; a0_b0_c0.basis = [t, r, theta]
    a0_b0_c0_answer = [
        i for i in product((0, 1, 2), (0, 1, 2), (0, 1, 2))
    ]  # Should be all combinations of (a, b, c) where a, b, c range over: len(basis) ==> 3

    assert set(a1_f0_c1_iter) == set(
        a1_f0_c1_answer
    )  # set => order indices should not matter
    assert set(a0_b0_c0_iter) == set(
        a0_b0_c0_answer
    )  # set => order indices should not matter
    assert set(mu0_nu0_iter) == set(
        mu0_nu0_answer
    )  # set => order indices should not matter
    assert set(mu1_nu1_iter) == set(
        mu1_nu1_answer
    )  # set => order indices should not matter


def test_constant_valued_indices_iteration(const_indices_product_setup):
    a1_f0_c1, a0_b0_c0, mu0_nu0, mu1_nu1 = const_indices_product_setup

    # Easy one for visual explenantion
    mu0_nu0_iter = [
        i for i in mu0_nu0
    ]  # mu0_nu0 = _{mu:0}_{nu} ; mu0_nu0.basis = [t, r]
    mu0_nu0_answer = [
        (0, 0),
        (0, 1),
    ]  # Should be all combinations of (mu, nu) where mu =0 and nu range over: len(basis) ==> 2

    mu1_nu1_iter = [
        i for i in mu1_nu1
    ]  # mu1_nu1 = ^{mu}^{nu:1} ; mu1_nu1.basis = [t, r]
    mu1_nu1_answer = [
        (0, 1),
        (1, 1),
    ]  # Should be all combinations of (mu, nu) where mu range over: len(basis) ==> 2 and nu = 1

    # longer ones using itertools.product
    a1_f0_c1_iter = [
        i for i in a1_f0_c1
    ]  # a1_f0_c1 = ^{a}_{f:2}^{c} ; a1_f0_c1.basis = [t, r, theta]
    a1_f0_c1_answer = [
        i for i in product((0, 1, 2), (0, 1, 2), (0, 1, 2)) if i[1] == 2
    ]  # Should be all combinations of (a, f, c) where a, c range over: len(basis) ==> 3 and f = 2

    a0_b0_c0_iter = [
        i for i in a0_b0_c0
    ]  # a0_b0_c0 =  _{a}_{b}_{c:1} ; a0_b0_c0.basis = [t, r, theta]
    a0_b0_c0_answer = [
        i for i in product((0, 1, 2), (0, 1, 2), (0, 1, 2)) if i[2] == 1
    ]  # Should be all combinations of (a, b, c) where a, b range over: len(basis) ==> 3 and c = 1

    assert set(a1_f0_c1_iter) == set(
        a1_f0_c1_answer
    )  # set => order indices should not matter
    assert set(a0_b0_c0_iter) == set(
        a0_b0_c0_answer
    )  # set => order indices should not matter
    assert set(mu0_nu0_iter) == set(
        mu0_nu0_answer
    )  # set => order indices should not matter
    assert set(mu1_nu1_iter) == set(
        mu1_nu1_answer
    )  # set => order indices should not matter


def test_einsum_product_result_iteration(indices_product_setup):
    # Here we test that the resulting indices has been generated correctly.
    a1_f0_c1, a0_b0_c0, mu0_nu0, mu1_nu1, _, _ = indices_product_setup

    result0 = mu0_nu0.einsum_product(mu1_nu1)  # should return a rank 0 indices.
    result1 = a1_f0_c1.einsum_product(a0_b0_c0)  # should return a rank 2 indices.

    result0_answer = [()]
    result1_answer = [i for i in product((0, 1, 2), (0, 1, 2))]

    # Einstein summation result is again another Index
    assert isinstance(result1, Indices)
    assert isinstance(result0, Indices)

    # Result should iterate just as any other Index object.
    assert set([i for i in result1]) == set(result1_answer)
    assert set([i for i in result0]) == set(result0_answer)

    # After Einstein summation convention applied -> generator implementor tag must be EINSUM
    assert result0.generator_implementor == result0.EINSUM_GENERATOR
    assert result1.generator_implementor == result1.EINSUM_GENERATOR

    assert str(result1) == "_{f}_{b}"
    assert str(result1.indices[0]) == "_{f}"
    assert str(result1.indices[1]) == "_{b}"


def test_einsum_product_with_metric_indices(indices_product_setup):
    # Here we test that the resulting indices has been generated correctly.
    a1_f0_c1, a0_b0_c0, mu0_nu0, mu1_nu1, metric, metric_inv = indices_product_setup

    result0 = mu0_nu0.einsum_product(mu1_nu1)  # should return a rank 0 indices.
    result1 = a1_f0_c1.einsum_product(a0_b0_c0)  # should return a rank 2 indices.

    result0_answer = [()]
    result1_answer = [i for i in product((0, 1, 2), (0, 1, 2))]

    # Einstein summation result is again another Index
    assert isinstance(result1, Indices)
    assert isinstance(result0, Indices)

    # Result should iterate just as any other Index object.
    assert set([i for i in result1]) == set(result1_answer)
    assert set([i for i in result0]) == set(result0_answer)

    # After Einstein summation convention applied -> generator implementor tag must be EINSUM
    assert result0.generator_implementor == result0.EINSUM_GENERATOR
    assert result1.generator_implementor == result1.EINSUM_GENERATOR

    assert str(result1) == "_{f}_{b}"
    assert str(result1.indices[0]) == "_{f}"
    assert str(result1.indices[1]) == "_{b}"


# TODO: We should be testing more cases than just these two.
def test_einsum_product(indices_product_setup):
    # Here we test that the resulting indices has been generated correctly.
    a1_f0_c1, a0_b0_c0, mu0_nu0, mu1_nu1, _, _ = indices_product_setup

    result0 = mu0_nu0.einsum_product(mu1_nu1)  # should return a rank 0 indices.
    result1 = a1_f0_c1.einsum_product(a0_b0_c0)  # should return a rank 2 indices.

    # Result is a scalar -> derived from two rank 2 tensors -> should be all combinations of the two tensors where matching indices match in numbers.
    assert set(result0.generator(())) == set(
        [
            ((0, 0), (0, 0)),
            ((0, 1), (0, 1)),
            ((1, 0), (1, 0)),
            ((1, 1), (1, 1)),
        ]
    )

    # Result is a rank 2 -> derived from two rank 3 tensors -> should be all combinations of the two tensors where matching indices match in numbers.
    # for _{f = 0} and _{b = 0} we should see the result of all combinations where the 2nd components of both = 0
    assert set(result1.generator((0, 0))) == set(
        [
            ((0, 0, 0), (0, 0, 0)),
            ((0, 0, 1), (0, 0, 1)),
            ((0, 0, 2), (0, 0, 2)),
            ((1, 0, 0), (1, 0, 0)),
            ((1, 0, 1), (1, 0, 1)),
            ((1, 0, 2), (1, 0, 2)),
            ((2, 0, 0), (2, 0, 0)),
            ((2, 0, 1), (2, 0, 1)),
            ((2, 0, 2), (2, 0, 2)),
        ]
    )

    assert set(result1.generator((1, 0))) == set(
        [
            ((0, 1, 0), (0, 0, 0)),
            ((0, 1, 1), (0, 0, 1)),
            ((0, 1, 2), (0, 0, 2)),
            ((1, 1, 0), (1, 0, 0)),
            ((1, 1, 1), (1, 0, 1)),
            ((1, 1, 2), (1, 0, 2)),
            ((2, 1, 0), (2, 0, 0)),
            ((2, 1, 1), (2, 0, 1)),
            ((2, 1, 2), (2, 0, 2)),
        ]
    )

    assert set(result1.generator((2, 2))) == set(
        [
            ((0, 2, 0), (0, 2, 0)),
            ((0, 2, 1), (0, 2, 1)),
            ((0, 2, 2), (0, 2, 2)),
            ((1, 2, 0), (1, 2, 0)),
            ((1, 2, 1), (1, 2, 1)),
            ((1, 2, 2), (1, 2, 2)),
            ((2, 2, 0), (2, 2, 0)),
            ((2, 2, 1), (2, 2, 1)),
            ((2, 2, 2), (2, 2, 2)),
        ]
    )


def test_einsum_product_with_metric_indices(indices_product_setup):
    # Here we test that the resulting indices has been generated correctly.
    _, _, mu0_nu0, mu1_nu1, metric_a0_nu0, metric_a1_nu1 = indices_product_setup

    result0 = metric_a1_nu1.einsum_product(mu0_nu0)  # should return a rank 2 indices.
    result1 = metric_a0_nu0.einsum_product(mu1_nu1)  # should return a rank 2 indices.

    result0_answer = [i for i in product((0, 1), (0, 1))]
    result1_answer = [i for i in product((0, 1), (0, 1))]

    # Einstein summation result is again another Index
    assert isinstance(result1, Indices)
    assert isinstance(result0, Indices)

    # Result should iterate just as any other Index object.
    assert set([i for i in result1]) == set(result1_answer)
    assert set([i for i in result0]) == set(result0_answer)

    # After Einstein summation convention applied -> generator implementor tag must be EINSUM
    assert result0.generator_implementor == result0.EINSUM_GENERATOR
    assert result1.generator_implementor == result1.EINSUM_GENERATOR

    # The result of G^{a}^{nu}*T_{mu}_{nu} => T_{mu}^{a}
    assert str(result0) == "_{mu}^{a}"
    assert str(result0.indices[0]) == "_{mu}"
    assert str(result0.indices[1]) == "^{a}"

    # The result of G_{a}_{nu}*T^{mu}^{nu} => T^{mu}_{a}
    assert str(result1) == "^{mu}_{a}"
    assert str(result1.indices[0]) == "^{mu}"
    assert str(result1.indices[1]) == "_{a}"

    assert result0.rank == (1, 1)
    assert result1.rank == (1, 1)

    assert result0.dimention == 2
    assert result1.dimention == 2

    # The result of G^{a}^{nu}*T_{mu}_{nu} => T_{mu}^{a}
    # This means the resulting indices will be _{mu}^{a}
    # (0,0) -> ((0, iterating),(0, iterating))
    assert set(result0.generator((0, 0))) == set([((0, 0), (0, 0)), ((0, 1), (0, 1))])

    # i.e. it is a good test as the resulting order will be swapped (1,0) -> ((0, iterating),(1, iterating))
    assert set(result0.generator((0, 1))) == set([((1, 0), (0, 0)), ((1, 1), (0, 1))])

    assert set(result1.generator((0, 0))) == set([((0, 0), (0, 0)), ((0, 1), (0, 1))])

    # i.e. it is a good test as the resulting order will be swapped (1,0) -> ((0, iterating),(1, iterating))
    assert set(result1.generator((0, 1))) == set([((1, 0), (0, 0)), ((1, 1), (0, 1))])


def test_selfsum_product_from_initiation():
    indices_a0_b0_a1 = Indices(Idx("a"), Idx("b"), -Idx("a"))
    indices_a0_a1 = Indices(Idx("a"), -Idx("a"))

    basis3D = Mathify("[t, r, theta]")
    indices_a0_b0_a1.basis = basis3D
    indices_a0_a1.basis = basis3D

    assert (
        indices_a0_b0_a1.self_summed
    )  # The self summed property should be true as _{a}_{b}^{a} -> _{b} : summed on _{a} and ^{a}
    assert indices_a0_a1.self_summed

    result0 = indices_a0_b0_a1.self_product()
    result1 = indices_a0_a1.self_product()

    assert str(result0) == "_{b}"
    assert str(result1) == ""

    assert set([i for i in result0]) == set([i for i in product((0, 1, 2))])
    assert set([i for i in result1]) == set([()])

    assert isinstance(result0, Indices)
    assert isinstance(result1, Indices)

    assert result0.generator_implementor == result0.SELFSUM_GENERATOR
    assert result1.generator_implementor == result1.SELFSUM_GENERATOR

    assert result0.rank == (0, 1)
    assert result1.rank == (0, 0)

    assert result0.dimention == 3
    assert result1.dimention == 3

    assert set(result0.generator((0,))) == set([(0, 0, 0), (1, 0, 1), (2, 0, 2)])
    assert set(result0.generator((1,))) == set([(0, 1, 0), (1, 1, 1), (2, 1, 2)])
    assert set(result0.generator((2,))) == set([(0, 2, 0), (1, 2, 1), (2, 2, 2)])

    assert set(result1.generator()) == set([(0, 0), (1, 1), (2, 2)])

def test_selfsum_product_from_metric_einsum_result():
    # Conprehensive test on an important computation path:
    # Metric*Indices -> Indices.self_product() -> Indices

    indices_a0_b0_c0 = Indices(Idx("a"), Idx("b"), Idx("c"))
    indices_a0_b0 = Indices(Idx("a"), Idx("b"))
    metric_a1_b1 = MetricIndices(-Idx("a"), -Idx("b"))
    basis3D = Mathify("[t, r, theta]")

    indices_a0_b0_c0.basis = basis3D
    indices_a0_b0.basis = basis3D
    metric_a1_b1.basis = basis3D

    assert not indices_a0_b0_c0.self_summed
    assert not indices_a0_b0.self_summed
    assert not metric_a1_b1.self_summed

    result0 = metric_a1_b1.einsum_product(indices_a0_b0)
    result1 = metric_a1_b1.einsum_product(indices_a0_b0_c0)

    assert result1.generator_implementor == result1.EINSUM_GENERATOR
    assert result0.generator_implementor == result0.EINSUM_GENERATOR

    assert isinstance(result0, Indices)
    assert isinstance(result1, Indices)

    assert result0.rank == (1,1)
    assert result1.rank == (1,2)

    # We have not called the self sum function on the results yet, so they should just have the self_summed boolean tag so far.
    assert result0.self_summed
    assert result1.self_summed

    assert str(result0) == "^{b}_{b}"
    assert str(result1) == "^{b}_{b}_{c}"

    # Call the self_product method 
    self_summed0 = result0.self_product()
    self_summed1 = result1.self_product()

    assert self_summed0.rank == (0, 0)
    assert self_summed1.rank == (0, 1)

    assert str(self_summed0) == ""
    assert str(self_summed1) == "_{c}"

    assert set([i for i in self_summed1]) == set([i for i in product((0, 1, 2))])
    assert set([i for i in self_summed0]) == set([()])

    assert isinstance(self_summed1, Indices)
    assert isinstance(self_summed0, Indices)

    assert self_summed1.generator_implementor == self_summed1.SELFSUM_GENERATOR
    assert self_summed0.generator_implementor == self_summed0.SELFSUM_GENERATOR

    assert self_summed1.dimention == 3
    assert self_summed0.dimention == 3

    assert set(self_summed1.generator((0,))) == set([(0, 0, 0), (1, 1, 0), (2, 2, 0)])
    assert set(self_summed1.generator((1,))) == set([(0, 0, 1), (1, 1, 1), (2, 2, 1)])
    assert set(self_summed1.generator((2,))) == set([(0, 0, 2), (1, 1, 2), (2, 2, 2)])

    assert set(self_summed0.generator()) == set([(0, 0), (1, 1), (2, 2)])

    # After all the Indices transformations => the basis should have been persisted 
    assert self_summed0.basis == basis3D
    assert self_summed1.basis == basis3D
    