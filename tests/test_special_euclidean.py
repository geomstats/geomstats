"""Unit tests for special euclidean group in matrix representation."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.special_euclidean import SpecialEuclidean


class TestSpecialEuclidean(geomstats.tests.TestCase):
    def setUp(self):
        self.n = 2
        self.group = SpecialEuclidean(n=self.n)
        self.n_samples = 4

    def test_belongs(self):
        theta = gs.pi / 3
        point_1 = gs.array([
            [gs.cos(theta), - gs.sin(theta), 2.],
            [gs.sin(theta), gs.cos(theta), 3.],
            [0., 0., 1.]])
        result = self.group.belongs(point_1)
        expected = True
        self.assertAllClose(result, expected)

        point_2 = gs.array([
            [gs.cos(theta), - gs.sin(theta), 2.],
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
        vec_1 = gs.array([
            [0., - theta, 2.],
            [theta, 0., 3.],
            [0., 0., 0.]])
        result = self.group.is_tangent(vec_1)
        expected = True
        self.assertAllClose(result, expected)

        vec_2 = gs.array([
            [0., - theta, 2.],
            [theta, 0., 3.],
            [0., 0., 1.]])
        result = self.group.is_tangent(vec_2)
        expected = False
        self.assertAllClose(result, expected)

        vec = gs.array([vec_1, vec_2])
        expected = gs.array([True, False])
        result = self.group.is_tangent(vec)
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

    def test_compose_vectorization(self):
        n_samples = self.n_samples
        n_points_a = self.group.random_uniform(n_samples=n_samples)
        n_points_b = self.group.random_uniform(n_samples=n_samples)
        one_point = self.group.random_uniform(n_samples=1)

        result = self.group.compose(one_point, n_points_a)
        self.assertAllClose(
            gs.shape(result), (n_samples,) + (self.group.n + 1,) * 2)

        result = self.group.compose(n_points_a, one_point)

        if not geomstats.tests.tf_backend():
            self.assertAllClose(
                gs.shape(result), (n_samples,) + (self.group.n + 1,) * 2)

            result = self.group.compose(n_points_a, n_points_b)
            self.assertAllClose(
                gs.shape(result), (n_samples,) + (self.group.n + 1,) * 2)

    def test_inverse_vectorization(self):
        n_samples = self.n_samples
        points = self.group.random_uniform(n_samples=n_samples)
        result = self.group.inverse(points)
        self.assertAllClose(
            gs.shape(result), (n_samples,) + (self.group.n + 1,) * 2)

    def test_compose_matrix_form(self):
        point = self.group.random_uniform()
        result = self.group.compose(point, self.group.identity)
        expected = point
        self.assertAllClose(result, expected)

        if not geomstats.tests.tf_backend():
            # Composition by identity, on the left
            # Expect the original transformation
            result = self.group.compose(self.group.identity, point)
            expected = point
            self.assertAllClose(result, expected)

            # Composition of translations (no rotational part)
            # Expect the sum of the translations
            point_a = gs.array([[1., 0., 1.],
                                [0., 1., 1.5],
                                [0., 0., 1.]])
            point_b = gs.array([[1., 0., 2.],
                                [0., 1., 2.5],
                                [0., 0., 1.]])

            result = self.group.compose(point_a, point_b)
            last_line_0 = gs.array_from_sparse(
                [(0, 2), (1, 2)], [1., 1.], (3, 3))
            expected = point_a + point_b * last_line_0
            self.assertAllClose(result, expected)
