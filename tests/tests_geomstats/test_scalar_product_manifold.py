"""Tests of ScalarProductMetric for errors.

This should be rewritten to test all methods available in the PoincareBall and
PoincareHalfSpace, as these are the ones where scaling is already implemented. Some
failures are likely but this may reflect errors in the other classes."""

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from geomstats.geometry.scalar_product_metric import ScalarProductMetric
import tests.conftest


class TestScalarProductMetrics(tests.conftest.TestCase):

    def test_exp(self):
        space = Euclidean(2)
        new_metric = 2 * space.metric
        point = gs.array([0.0, 0.0])
        tan_vec = gs.array([1.0, 0.0])
        result = new_metric.exp(point, tan_vec)
        expected = tan_vec
        self.assertAllClose(result, expected)

    def test_shape(self):
        space = Euclidean(2)
        new_metric = 2 * space.metric
        result = new_metric.shape
        expected = (2,)
        assert (result == expected)

    def test_equal_distances(self):
        space_1 = PoincareHalfSpace(dim=2, scale=1)
        rescaled_metric = space_1.metric * 2

        space_2 = PoincareHalfSpace(dim=2, scale=2)

        p = space_1.random_point()
        q = space_1.random_point()

        result = rescaled_metric.dist(p, q)
        expected = space_2.metric.dist(p, q)

        self.assertAllClose(result, expected)
