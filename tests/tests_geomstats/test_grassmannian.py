"""Unit tests for the Grassmannian."""

import tests.helper as helper

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.grassmannian import Grassmannian
from geomstats.geometry.grassmannian import GrassmannianCanonicalMetric
from geomstats.geometry.matrices import Matrices

p_xy = gs.array([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 0.]])
p_yz = gs.array([
    [0., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.]])
p_xz = gs.array([
    [1., 0., 0.],
    [0., 0., 0.],
    [0., 0., 1.]])

r_y = gs.array([
    [0., 0., 1.],
    [0., 0., 0.],
    [-1., 0., 0.]])
r_z = gs.array([
    [0., -1., 0.],
    [1., 0., 0.],
    [0., 0., 0.]])
pi_2 = gs.pi / 2
pi_4 = gs.pi / 4


class TestGrassmannian(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)

        self.n = 3
        self.k = 2
        self.space = Grassmannian(self.n, self.k)
        self.metric = GrassmannianCanonicalMetric(self.n, self.k)

    def test_exp_np(self):
        vec = Matrices.bracket(pi_2 * r_y, gs.array([p_xy, p_yz]))
        result = self.metric.exp(
            vec, gs.array([p_xy, p_yz]))
        expected = gs.array([p_yz, p_xy])
        self.assertAllClose(result, expected)

        vec = Matrices.bracket(
            pi_2 * gs.array([r_y, r_z]), gs.array([p_xy, p_yz]))
        result = self.metric.exp(
            vec, gs.array([p_xy, p_yz]))
        expected = gs.array([p_yz, p_xz])
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_log(self):
        expected = Matrices.bracket(pi_4 * r_y, p_xy)
        result = self.metric.log(
            self.metric.exp(expected, p_xy), p_xy)
        self.assertTrue(self.space.is_tangent(result, p_xy))
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_log_vectorized(self):
        tangent_vecs = pi_4 * gs.array([r_y, r_z])
        base_points = gs.array([p_xy, p_xz])
        points = self.metric.exp(tangent_vecs, base_points)
        result = self.metric.log(points, base_points)
        expected = tangent_vecs
        self.assertAllClose(result, expected)

    def test_belongs(self):
        point = p_xy
        result = self.space.belongs(point)
        self.assertTrue(result)

        point = gs.array([p_yz, p_xz])
        result = self.space.belongs(point)
        self.assertTrue(gs.all(result))

        not_a_point = gs.random.rand(3, 2)
        result = self.space.belongs(not_a_point)
        self.assertTrue(~result)

        not_a_point = gs.random.rand(3, 3)
        result = self.space.belongs(not_a_point)
        self.assertTrue(~result)

        point = gs.array([p_xy, not_a_point])
        result = self.space.belongs(point)
        expected = gs.array([True, False])
        self.assertAllClose(result, expected)

    def test_random_and_belongs(self):
        point = self.space.random_uniform()
        result = self.space.belongs(point)
        self.assertTrue(result)

        expected = (self.n,) * 2
        result_shape = point.shape
        self.assertAllClose(result_shape, expected)

        n_samples = 5
        points = self.space.random_uniform(n_samples)
        result = gs.all(self.space.belongs(points))
        self.assertTrue(result)

        expected = (n_samples,) + (self.n,) * 2
        result_shape = points.shape
        self.assertAllClose(result_shape, expected)

    def test_is_to_tangent(self):
        base_point = self.space.random_uniform()
        vector = gs.random.rand(self.n, self.n)
        tangent_vec = self.space.to_tangent(vector, base_point)
        result = self.space.is_tangent(tangent_vec, base_point)
        self.assertTrue(result)

        reprojected = self.space.to_tangent(tangent_vec, base_point)
        self.assertAllClose(tangent_vec, reprojected)

    def test_projection_and_belongs(self):
        shape = (2, self.n, self.n)
        result = helper.test_projection_and_belongs(self.space, shape)
        for res in result:
            self.assertTrue(res)

    def test_parallel_transport(self):
        space = self.space
        metric = self.metric
        shape = (2, space.n, space.n)

        result = helper.test_parallel_transport(space, metric, shape)
        for res in result:
            self.assertTrue(res)
