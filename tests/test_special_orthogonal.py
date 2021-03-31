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

    @geomstats.tests.np_and_pytorch_only
    def test_rotation_from_angle(self):
        """Test rotation_from_angle."""
        theta = [gs.pi / 6, gs.pi / 3, gs.pi / 2]
        points = self.group.rotation_from_angle(theta)
        expected_points = gs.array([[[gs.sqrt(3) / 2, -.5],
                                     [.5, gs.sqrt(3) / 2]],
                                    [[.5, -gs.sqrt(3) / 2],
                                     [gs.sqrt(3) / 2, .5]],
                                    [[0, -1],
                                     [1, 0]]])
        self.assertAllClose(points, expected_points)

    @geomstats.tests.np_and_pytorch_only
    def test_angle_of_rot2(self):
        """Test angle_of_rot2."""
        theta = gs.array([gs.pi / 4, 3 * gs.pi / 4])
        point_1 = self.group.rotation_from_angle(theta)
        theta_result = self.group.angle_of_rot2(point_1)
        # self.assertAllClose((theta - theta_result) % (2 * gs.pi), 0)
        self.assertAllClose(gs.abs(theta - theta_result) % (2 * gs.pi), 0)

    @geomstats.tests.np_and_pytorch_only
    def test_multiply_angle_of_rot2(self):
        """Test multiply_angle_of_rot2."""
        theta = gs.array([3 * gs.pi / 4])
        point = self.group.rotation_from_angle(theta)
        mul_factor = gs.array([1 / 3])
        point_new = self.group.multiply_angle_of_rot2(point, mul_factor)
        theta_new = self.group.angle_of_rot2(point_new)
        theta_new_expected = theta * mul_factor
        self.assertAllClose((theta_new - theta_new_expected) % (2 * gs.pi), 0)

    @geomstats.tests.np_and_pytorch_only
    def test_random_gaussian(self):
        """Test random_gaussian."""
        n_samples = 4
        mean = self.group.random_uniform(n_samples=n_samples)
        var = gs.array([1.] * n_samples)
        points = self.group.random_gaussian(
            mean=mean, var=var, n_samples=n_samples)
        result = self.group.belongs(points)
        expected = gs.array([True] * n_samples)
        return self.assertAllClose(result, expected)
