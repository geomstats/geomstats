"""
Unit tests for the affine connections.
"""

import geomstats.backend as gs
import geomstats.tests

from geomstats.geometry.connection import Connection, LeviCivitaConnection
from geomstats.geometry.euclidean_space import EuclideanMetric
from geomstats.geometry.hypersphere import Hypersphere


class TestConnectionMethods(geomstats.tests.TestCase):
    def setUp(self):
        self.dimension = 4
        self.euc_metric = EuclideanMetric(dimension=self.dimension)
        self.lc_connection = LeviCivitaConnection(self.euc_metric)

        self.connection = Connection(dimension=2)
        self.hypersphere = Hypersphere(dimension=2)

    def test_metric_matrix(self):
        base_point = gs.array([0., 1., 0., 0.])

        result = self.lc_connection.metric_matrix(base_point)
        expected = gs.array([gs.eye(self.dimension)])

        with self.session():
            self.assertAllClose(result, expected)

    def test_cometric_matrix(self):
        base_point = gs.array([0., 1., 0., 0.])

        result = self.lc_connection.cometric_matrix(base_point)
        expected = gs.array([gs.eye(self.dimension)])

        with self.session():
            self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_metric_derivative(self):
        base_point = gs.array([0., 1., 0., 0.])

        result = self.lc_connection.metric_derivative(base_point)
        expected = gs.zeros((1,) + (self.dimension, ) * 3)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_christoffels(self):
        base_point = gs.array([0., 1., 0., 0.])

        result = self.lc_connection.christoffels(base_point)
        expected = gs.zeros((1,) + (self.dimension, ) * 3)

        self.assertAllClose(result, expected)

    def test_parallel_transport(self):
        sphere = Hypersphere(dimension=2)
        connection = LeviCivitaConnection(sphere.metric)
        n_samples = 10
        base_point = sphere.random_uniform(n_samples)
        tan_vec_a = sphere.projection_to_tangent_space(
            gs.random.rand(n_samples, 3), base_point)
        tan_vec_b = sphere.projection_to_tangent_space(
            gs.random.rand(n_samples, 3), base_point)
        expected = sphere.metric.parallel_transport(
            tan_vec_a, tan_vec_b, base_point)
        result = connection.pole_ladder_parallel_transport(
            tan_vec_a, tan_vec_b, base_point)

        self.assertAllClose(result, expected, rtol=1e-7, atol=1e-5)

    @geomstats.tests.np_only
    def test_exp(self):
        p = gs.array([[gs.pi / 2, 0], [gs.pi / 6, gs.pi / 4]])
        vec = gs.array([[0.25, 0.5], [0.30, 0.2]])
        point_ext = self.hypersphere.spherical_to_extrinsic(p)
        vector_ext = self.hypersphere.tangent_spherical_to_extrinsic(vec, p)
        self.connection.christoffels = self.hypersphere.metric.christoffels
        expected = self.hypersphere.metric.exp(vector_ext, point_ext)
        result_spherical = self.connection.exp(vec, p, n_steps=1000)
        result = self.hypersphere.spherical_to_extrinsic(result_spherical)

        self.assertAllClose(result, expected, rtol=1e-3)

    @geomstats.tests.np_only
    def test_log(self):
        p = gs.array([[gs.pi / 3, gs.pi / 4], [gs.pi / 2, gs.pi / 4]])
        q = gs.array([[1.0, gs.pi / 2], [gs.pi / 6, gs.pi / 3]])
        self.connection.christoffels = self.hypersphere.metric.christoffels
        v = self.connection.log(point=q, base_point=p, n_steps=300)
        result = self.hypersphere.tangent_spherical_to_extrinsic(v, p)
        p_ext = self.hypersphere.spherical_to_extrinsic(p)
        q_ext = self.hypersphere.spherical_to_extrinsic(q)
        expected = self.hypersphere.metric.log(base_point=p_ext, point=q_ext)

        self.assertAllClose(result, expected, rtol=1e-2)


if __name__ == '__main__':
    geomstats.tests.main()
