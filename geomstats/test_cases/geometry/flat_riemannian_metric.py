import pytest

import geomstats.backend as gs
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class FlatRiemannianMetricTestCase(RiemannianMetricTestCase):
    @pytest.mark.random
    def test_inner_product_derivative_matrix_is_zeros(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        res = self.space.metric.inner_product_derivative_matrix(base_point)

        batch_shape = (n_points,) if n_points > 1 else ()
        expected = gs.zeros(batch_shape + 3 * (self.space.dim,))
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_christoffels_are_zeros(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        res = self.space.metric.christoffels(base_point)

        batch_shape = (n_points,) if n_points > 1 else ()
        expected = gs.zeros(batch_shape + 3 * (self.space.dim,))
        self.assertAllClose(res, expected, atol=atol)
