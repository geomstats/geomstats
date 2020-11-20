"""Unit tests for special orthogonal group SO(n)."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


class TestSpecialOrthogonal(geomstats.tests.TestCase):
    def setUp(self):
        self.n = 2
        self.group = SpecialOrthogonal(n=self.n)
        self.n_samples = 4

    def test_belongs(self):
        theta = gs.pi / 3
        point_1 = gs.array([[gs.cos(theta), - gs.sin(theta)],
                            [gs.sin(theta), gs.cos(theta)]])
        result = self.group.belongs(point_1)
        expected = True
        self.assertAllClose(result, expected)

        point_2 = gs.array([[gs.cos(theta), gs.sin(theta)],
                            [gs.sin(theta), gs.cos(theta)]])
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
        expected = gs.eye(self.n)
        self.assertAllClose(result, expected)

    def test_is_in_lie_algebra(self):
        theta = gs.pi / 3
        vec_1 = gs.array([[0., - theta],
                         [theta, 0.]])
        result = self.group.is_tangent(vec_1)
        expected = True
        self.assertAllClose(result, expected)

        vec_2 = gs.array([[0., - theta],
                         [theta, 1.]])
        result = self.group.is_tangent(vec_2)
        expected = False
        self.assertAllClose(result, expected)

        vec = gs.array([vec_1, vec_2])
        expected = gs.array([True, False])
        result = self.group.is_tangent(vec)
        self.assertAllClose(result, expected)

    def test_is_tangent(self):
        point = self.group.random_uniform()
        theta = 1.
        vec_1 = gs.array([[0., - theta],
                         [theta, 0.]])
        vec_1 = self.group.compose(point, vec_1)
        result = self.group.is_tangent(vec_1, point, atol=1e-6)
        expected = True
        self.assertAllClose(result, expected)

        vec_2 = gs.array([[0., - theta],
                         [theta, 1.]])
        vec_2 = self.group.compose(point, vec_2)
        result = self.group.is_tangent(vec_2, point, atol=1e-6)
        expected = False
        self.assertAllClose(result, expected)

        vec = gs.array([vec_1, vec_2])
        point = gs.array([point, point])
        expected = gs.array([True, False])
        result = self.group.is_tangent(vec, point, atol=1e-6)
        self.assertAllClose(result, expected)

    def test_to_tangent(self):
        theta = 1.
        vec_1 = gs.array([[0., - theta],
                         [theta, 0.]])
        result = self.group.to_tangent(vec_1)
        expected = vec_1
        self.assertAllClose(result, expected)

        n_samples = self.n_samples
        base_points = self.group.random_uniform(n_samples=n_samples)
        tangent_vecs = self.group.compose(base_points, vec_1)
        result = self.group.to_tangent(tangent_vecs, base_points)
        expected = tangent_vecs
        self.assertAllClose(result, expected)

    def test_projection_and_belongs(self):
        gs.random.seed(3)
        group = SpecialOrthogonal(n=4)
        mat = gs.random.rand(4, 4)
        point = group.projection(mat)
        result = group.belongs(point, atol=1e-5)
        self.assertTrue(result)

        mat = gs.random.rand(2, 4, 4)
        point = group.projection(mat)
        result = group.belongs(point, atol=1e-4)
        self.assertTrue(gs.all(result))
