"""Unit tests for special euclidean group in matrix representation."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices


class TestSpecialEuclideanMethods(geomstats.tests.TestCase):
    def setUp(self):
        self.n = 2
        self.group = SpecialEuclidean(n=self.n)
        self.n_samples = 4

    def test_belongs(self):
        theta = gs.pi / 3
        point_1 = gs.array([[gs.cos(theta), - gs.sin(theta), 2.],
                          [gs.sin(theta), gs.cos(theta), 3.],
                          [0., 0., 1.]])
        result = self.group.belongs(point_1)
        expected = True
        self.assertAllClose(result, expected)

        point_2 = gs.array([[gs.cos(theta), - gs.sin(theta), 2.],
                          [gs.sin(theta), gs.cos(theta), 3.],
                          [0., 0., 0.]])
        result = self.group.belongs(point_2)
        expected = False
        self.assertAllClose(result, expected)

        point = gs.array([point_1, point_2])
        expected = gs.array([True, False])
        result = self.group.belongs(point)
        self.assertAllClose(result, expected)

    def test_random_uniform_and_belongs(self):
        point = self.group.random_uniform()
        result = self.group.belongs(point)
        expected = True
        self.assertAllClose(result, expected)

        point = self.group.random_uniform(self.n_samples)
        result = self.group.belongs(point)
        expected = gs.array([True] * self.n_samples)
        self.assertAllClose(result, expected)

    def test_identity(self):
        result = self.group.identity
        expected = gs.eye(self.n + 1)
        self.assertAllClose(result, expected)

    def test_is_in_lie_algebra(self):
        theta = gs.pi / 3
        vec_1 = gs.array([[0., - theta, 2.],
                           [theta, 0., 3.],
                           [0., 0., 0.]])
        result = self.group._is_in_lie_algebra(vec_1)
        expected = True
        self.assertAllClose(result, expected)

        vec_2 = gs.array([[0., - theta, 2.],
                           [theta, 0., 3.],
                           [0., 0., 1.]])
        result = self.group._is_in_lie_algebra(vec_2)
        expected = False
        self.assertAllClose(result, expected)

        vec = gs.array([vec_1, vec_2])
        expected = gs.array([True, False])
        result = self.group._is_in_lie_algebra(vec)
        self.assertAllClose(result, expected)

    def test_to_tangent_vec_vectorization(self):
        n = self.group.n
        tangent_vecs = gs.arange(self.n_samples * (n + 1) ** 2)
        tangent_vecs = gs.cast(tangent_vecs, gs.float32)
        tangent_vecs = gs.reshape(
            tangent_vecs, (self.n_samples,) + (n + 1,) * 2)
        point = self.group.random_uniform(self.n_samples)
        tangent_vecs = self.group.compose(point, tangent_vecs)
        regularized = self.group.to_tangent(tangent_vecs, point)
        result = self.group.compose(
            self.group.transpose(point), regularized) + \
            self.group.compose(self.group.transpose(regularized), point)
        result = result[:, :n, :n]
        expected = gs.zeros_like(result)
        self.assertAllClose(result, expected)

    def test_compose_and_inverse_matrix_form(self):
        point = self.group.random_uniform()
        inv_point = self.group.inverse(point)
        result = self.group.compose(point, inv_point)
        expected = self.group.identity
        self.assertAllClose(result, expected)

        if not geomstats.tests.tf_backend():
            result = self.group.compose(inv_point, point)
            expected = self.group.identity
            self.assertAllClose(result, expected)
