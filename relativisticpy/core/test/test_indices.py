from itertools import product
import pytest
from relativisticpy.core.indices import Idx, Indices
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
    indices0 = Indices(-Idx("a"), Idx("f"), -Idx("c"))
    indices1 = Indices(Idx("a"), Idx("b"), Idx("c"))
    indices2 = Indices(Idx("mu"), Idx("nu"))
    indices3 = Indices(-Idx("mu"), -Idx("nu"))

    # Set basis for space.
    basis2D = Mathify("[t, r]")
    basis3D = Mathify("[t, r, theta]")

    indices0.basis = basis3D
    indices1.basis = basis3D
    indices2.basis = basis2D
    indices3.basis = basis2D

    # return objects
    return indices0, indices1, indices2, indices3


@pytest.fixture
def const_indices_product_setup():
    # Init indices.
    indices0 = Indices(-Idx("a"), Idx("f", values=2), -Idx("c"))
    indices1 = Indices(Idx("a"), Idx("b"), Idx("c", values=1))
    indices2 = Indices(Idx("mu", values=0), Idx("nu"))
    indices3 = Indices(-Idx("mu"), -Idx("nu", values=1))

    # Set basis for space.
    basis2D = Mathify("[t, r]")
    basis3D = Mathify("[t, r, theta]")

    indices0.basis = basis3D
    indices1.basis = basis3D
    indices2.basis = basis2D
    indices3.basis = basis2D

    # return objects
    return indices0, indices1, indices2, indices3


def test_len(indices_setup):
    indices, _, _ = indices_setup
    assert len(indices) == 2


def test_eq(indices_setup):
    indices, idx1, idx2 = indices_setup
    other_indices = Indices(idx1, idx2)
    assert indices == other_indices


def test_mul(indices_setup):
    indices1, idx1, idx2 = indices_setup
    indices2 = Indices(idx1, -idx2)
    product_indices = indices1 * indices2
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
    indices1, idx1, idx2 = indices_setup
    indices_from_string_init = Indices.from_string("_{a}_{b}")

    assert indices_from_string_init.indices[0] == idx1
    assert indices_from_string_init.indices[1] == idx2
    assert indices_from_string_init == indices1


# Testing Indices Product - (this is the core compute feature of index/indices object. => they were designed to around the simplification of this compute.)
# when we perform opertations between indices, a few things happen:
# 1. We return another index object which represents the einstein summed index as a result of the two indices that were summed.
# 2. We inject a method implementation into the callback method "generator" => which generates the indices which are to be summed based on the resulting indices comps.
# 1 and 2 together means we can iterate through the resulting indices object, which returns all which it needs to be iterated, and pass its own iteration in as an argument
# to the "generator" callback, which will take those components and workout the components which are to be summed.


def test_all_running_indices_iteration(indices_product_setup):
    indices0, indices1, indices2, indices3 = indices_product_setup

    # Easy one for visual explenantion
    indices2_iter = [
        i for i in indices2
    ]  # indices2 = _{mu}_{nu} ; indices2.basis = [t, r]
    indices2_answer = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]  # Should be all combinations of (mu, nu) where mu, nu range over: len(basis) ==> 2

    indices3_iter = [
        i for i in indices3
    ]  # indices3 = ^{mu}^{nu} ; indices3.basis = [t, r]
    indices3_answer = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]  # Should be all combinations of (mu, nu) where mu, nu range over: len(basis) ==> 2

    # longer ones using itertools.product
    indices0_iter = [
        i for i in indices0
    ]  # indices0 = ^{a}_{f}^{c} ; indices0.basis = [t, r, theta]
    indices0_answer = [
        i for i in product((0, 1, 2), (0, 1, 2), (0, 1, 2))
    ]  # Should be all combinations of (a, f, c) where a, f, c range over: len(basis) ==> 3

    indices1_iter = [
        i for i in indices1
    ]  # indices1 =  _{a}_{b}_{c} ; indices1.basis = [t, r, theta]
    indices1_answer = [
        i for i in product((0, 1, 2), (0, 1, 2), (0, 1, 2))
    ]  # Should be all combinations of (a, b, c) where a, b, c range over: len(basis) ==> 3

    assert set(indices0_iter) == set(
        indices0_answer
    )  # set => order indices should not matter
    assert set(indices1_iter) == set(
        indices1_answer
    )  # set => order indices should not matter
    assert set(indices2_iter) == set(
        indices2_answer
    )  # set => order indices should not matter
    assert set(indices3_iter) == set(
        indices3_answer
    )  # set => order indices should not matter


def test_constant_valued_indices_iteration(const_indices_product_setup):
    indices0, indices1, indices2, indices3 = const_indices_product_setup

    # Easy one for visual explenantion
    indices2_iter = [
        i for i in indices2
    ]  # indices2 = _{mu:0}_{nu} ; indices2.basis = [t, r]
    indices2_answer = [
        (0, 0),
        (0, 1),
    ]  # Should be all combinations of (mu, nu) where mu =0 and nu range over: len(basis) ==> 2

    indices3_iter = [
        i for i in indices3
    ]  # indices3 = ^{mu}^{nu:1} ; indices3.basis = [t, r]
    indices3_answer = [
        (0, 1),
        (1, 1),
    ]  # Should be all combinations of (mu, nu) where mu range over: len(basis) ==> 2 and nu = 1

    # longer ones using itertools.product
    indices0_iter = [
        i for i in indices0
    ]  # indices0 = ^{a}_{f:2}^{c} ; indices0.basis = [t, r, theta]
    indices0_answer = [
        i for i in product((0, 1, 2), (0, 1, 2), (0, 1, 2)) if i[1] == 2
    ]  # Should be all combinations of (a, f, c) where a, c range over: len(basis) ==> 3 and f = 2

    indices1_iter = [
        i for i in indices1
    ]  # indices1 =  _{a}_{b}_{c:1} ; indices1.basis = [t, r, theta]
    indices1_answer = [
        i for i in product((0, 1, 2), (0, 1, 2), (0, 1, 2)) if i[2] == 1
    ]  # Should be all combinations of (a, b, c) where a, b range over: len(basis) ==> 3 and c = 1

    assert set(indices0_iter) == set(
        indices0_answer
    )  # set => order indices should not matter
    assert set(indices1_iter) == set(
        indices1_answer
    )  # set => order indices should not matter
    assert set(indices2_iter) == set(
        indices2_answer
    )  # set => order indices should not matter
    assert set(indices3_iter) == set(
        indices3_answer
    )  # set => order indices should not matter


def test_einsum_product_result_iteration(indices_product_setup):
    # Here we test that the resulting indices has been generated correctly.
    indices0, indices1, indices2, indices3 = indices_product_setup

    result0 = indices2.einsum_product(indices3)  # should return a rank 0 indices.
    result1 = indices0.einsum_product(indices1)  # should return a rank 2 indices.

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


# TODO: SUMMATION and SELFSUM

# TODO: We should be testing more cases than just these two.
def test_einsum_product(indices_product_setup):
    # Here we test that the resulting indices has been generated correctly.
    indices0, indices1, indices2, indices3 = indices_product_setup

    result0 = indices2.einsum_product(indices3)  # should return a rank 0 indices.
    result1 = indices0.einsum_product(indices1)  # should return a rank 2 indices.

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


# TODO: Test of Metric Indices interation with normal indices.