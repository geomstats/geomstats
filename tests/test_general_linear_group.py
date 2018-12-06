"""
Unit tests for General Linear group.
"""

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper

from geomstats.general_linear_group import GeneralLinearGroup
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup

RTOL = 1e-5


class TestGeneralLinearGroup(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)
        n = 3
        self.group = GeneralLinearGroup(n=n)
        # We generate invertible matrices using so3_group
        self.so3_group = SpecialOrthogonalGroup(n=n)

    def test_belongs(self):
        """
        A rotation matrix belongs to the matrix Lie group
        of invertible matrices.
        """
        rot_vec = gs.array([0.2, -0.1, 0.1])
        rot_mat = self.so3_group.matrix_from_rotation_vector(rot_vec)
        result = self.group.belongs(rot_mat)
        expected = gs.array([True])

        with self.session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_compose(self):
        # 1. Composition by identity, on the right
        # Expect the original transformation
        rot_vec = gs.array([0.2, -0.1, 0.1])
        mat = self.so3_group.matrix_from_rotation_vector(rot_vec)

        result = self.group.compose(mat, self.group.identity)
        expected = mat
        expected = helper.to_matrix(mat)

        with self.session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

        # 2. Composition by identity, on the left
        # Expect the original transformation
        rot_vec = gs.array([0.2, 0.1, -0.1])
        mat = self.so3_group.matrix_from_rotation_vector(rot_vec)

        result = self.group.compose(self.group.identity, mat)
        expected = mat

        with self.session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_inverse(self):
        mat = gs.array([
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 10.]])
        result = self.group.inverse(mat)
        expected = 1. / 3. * gs.array([
            [-2., -4., 3.],
            [-2., 11., -6.],
            [3., -6., 3.]])
        expected = helper.to_matrix(expected)

        with self.session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_compose_and_inverse(self):
        # 1. Compose transformation by its inverse on the right
        # Expect the group identity
        rot_vec = gs.array([0.2, 0.1, 0.1])
        mat = self.so3_group.matrix_from_rotation_vector(rot_vec)
        inv_mat = self.group.inverse(mat)

        result = self.group.compose(mat, inv_mat)
        expected = self.group.identity
        expected = helper.to_matrix(expected)

        with self.session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

        # 2. Compose transformation by its inverse on the left
        # Expect the group identity
        rot_vec = gs.array([0.7, 0.1, 0.1])
        mat = self.so3_group.matrix_from_rotation_vector(rot_vec)
        inv_mat = self.group.inverse(mat)

        result = self.group.compose(inv_mat, mat)
        expected = self.group.identity
        expected = helper.to_matrix(expected)

        with self.session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))


if __name__ == '__main__':
    geomstats.test.main()
