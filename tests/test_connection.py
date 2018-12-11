"""
Unit tests for the affine connections.
"""

import geomstats.tests
import geomstats.backend as gs

from geomstats.connection import LeviCivitaConnection
from geomstats.euclidean_space import EuclideanMetric


class TestConnectionMethods(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.dimension = 4
        self.metric = EuclideanMetric(dimension=self.dimension)
        self.connection = LeviCivitaConnection(self.metric)

    def test_metric_matrix(self):
        base_point = gs.array([0., 1., 0., 0.])

        result = self.connection.metric_matrix(base_point)
        expected = gs.array([gs.eye(self.dimension)])

        with self.session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_cometric_matrix(self):
        base_point = gs.array([0., 1., 0., 0.])

        result = self.connection.cometric_matrix(base_point)
        expected = gs.array([gs.eye(self.dimension)])

        with self.session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

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


if __name__ == '__main__':
        geomstats.tests.main()
