import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

import geomstats.backend as gs
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.group_action import (
    CongruenceAction,
    PermutationAction,
    RowPermutationAction,
    permutation_matrix_from_vector,
)
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.test.test_case import TestCase
from geomstats.test_cases.geometry.group_action import (
    GroupActionTestCase,
    PermutationGroup,
)

from .data.group_action import (
    GroupActionTestData,
    PermutationMatrixFromVectorTestData,
    RowPermutationAction4TestData,
)


class TestGLCongruenceActionOnSPD(
    GroupActionTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n = random.randint(4, 6)
    space = SPDMatrices(_n, equip=False)
    group = GeneralLinear(_n, equip=False)
    group_action = CongruenceAction()

    testing_data = GroupActionTestData(space, group, group_action)


class TestPermutationAction(
    GroupActionTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n = random.randint(4, 6)
    space = Matrices(_n, _n, equip=False)
    group = PermutationGroup(_n)
    group_action = PermutationAction()

    testing_data = GroupActionTestData(space, group, group_action)


class TestRowPermutationAction(
    GroupActionTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n = random.randint(4, 6)
    space = Matrices(_n, _n, equip=False)
    group = PermutationGroup(_n)
    group_action = RowPermutationAction()

    testing_data = GroupActionTestData(space, group, group_action)


@pytest.mark.smoke
class TestRowPermutationAction4(GroupActionTestCase, metaclass=DataBasedParametrizer):
    group = PermutationGroup(4)
    group_action = RowPermutationAction()

    testing_data = RowPermutationAction4TestData()


class TestPermutationMatrixFromVector(
    TestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = PermutationMatrixFromVectorTestData(
        PermutationGroup(random.randint(4, 6))
    )

    def test_permutation_matrix_from_vector(self, group_elem, expected, atol=gs.atol):
        """Check permutation matrix from vector.

        Parameters
        ----------
        group_elem : array-like, shape=[..., n]
            Permutation, as a vector of indices.
        expected : array-like, shape=[..., n, n]
            Expected permutation matrix.
        atol : float
            Absolute tolerance.
        """
        perm_mat = permutation_matrix_from_vector(group_elem, dtype=expected.dtype)
        self.assertAllClose(perm_mat, expected, atol=atol)
