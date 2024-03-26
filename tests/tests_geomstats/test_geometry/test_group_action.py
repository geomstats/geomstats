import random

import pytest

from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.group_action import (
    CongruenceAction,
    PermutationAction,
    RowPermutationAction,
    permutation_matrix_from_vector,
)
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test.test_case import TestCase
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.group_action import (
    GroupActionTestCase,
    PermutationGroup,
)

from .data.group_action import (
    GroupActionTestData,
    PermutationMatrixFromVectorTestData,
    RowPermutationAction4TestData,
)


class TestGLCongruenceActionOnSPD(GroupActionTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(4, 6)
    space = SPDMatrices(_n, equip=False)
    group = GeneralLinear(_n, equip=False)
    group_action = CongruenceAction()

    testing_data = GroupActionTestData()


class TestPermutationAction(GroupActionTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(4, 6)
    space = Matrices(_n, _n, equip=False)
    group = PermutationGroup(_n)
    group_action = PermutationAction()

    testing_data = GroupActionTestData()


class TestRowPermutationAction(GroupActionTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(4, 6)
    space = Matrices(_n, _n, equip=False)
    group = PermutationGroup(_n)
    group_action = RowPermutationAction()

    testing_data = GroupActionTestData()


@pytest.mark.smoke
class TestRowPermutationAction4(GroupActionTestCase, metaclass=DataBasedParametrizer):
    _n = 4
    space = Matrices(_n, _n, equip=False)
    group = PermutationGroup(_n)
    group_action = RowPermutationAction()

    testing_data = RowPermutationAction4TestData()


class TestPermutationMatrixFromVector(TestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(4, 6)
    _group = PermutationGroup(_n)
    data_generator = RandomDataGenerator(_group)
    testing_data = PermutationMatrixFromVectorTestData()

    def test_permutation_matrix_from_vector(self, group_elem, expected, atol):
        perm_mat = permutation_matrix_from_vector(group_elem, dtype=expected.dtype)
        self.assertAllClose(perm_mat, expected, atol=atol)

    @pytest.mark.vec
    def test_permutation_matrix_from_vector_vec(self, n_reps, atol):
        group_elem = self.data_generator.random_point()

        expected = permutation_matrix_from_vector(group_elem)

        vec_data = generate_vectorization_data(
            data=[dict(group_elem=group_elem, expected=expected, atol=atol)],
            arg_names=["group_elem"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)
