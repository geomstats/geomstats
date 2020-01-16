"""
Unit tests for the affine connections.
"""

import geomstats.backend as gs
import geomstats.tests

from geomstats.geometry.connection import LeviCivitaConnection
from geomstats.geometry.euclidean_space import EuclideanMetric
from geomstats.geometry.hypersphere import Hypersphere


class TestConnectionMethods(geomstats.tests.TestCase):
    def setUp(self):
        self.dimension = 4
        self.metric = EuclideanMetric(dimension=self.dimension)
        self.connection = LeviCivitaConnection(self.metric)

    def test_metric_matrix(self):
        base_point = gs.array([0., 1., 0., 0.])

        result = self.connection.metric_matrix(base_point)
        expected = gs.array([gs.eye(self.dimension)])

        with self.session():
            self.assertAllClose(result, expected)

    def test_cometric_matrix(self):
        base_point = gs.array([0., 1., 0., 0.])

        result = self.connection.cometric_matrix(base_point)
        expected = gs.array([gs.eye(self.dimension)])

        with self.session():
            self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_metric_derivative(self):
        base_point = gs.array([0., 1., 0., 0.])

        result = self.connection.metric_derivative(base_point)
        expected = gs.zeros((1,) + (self.dimension, ) * 3)

        gs.testing.assert_allclose(result, expected)

    @geomstats.tests.np_only
    def test_christoffel_symbols(self):
        base_point = gs.array([0., 1., 0., 0.])

        result = self.connection.christoffel_symbols(base_point)
        expected = gs.zeros((1,) + (self.dimension, ) * 3)

        gs.testing.assert_allclose(result, expected)

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
        gs.testing.assert_allclose(result, expected)


if __name__ == '__main__':
        geomstats.tests.main()

