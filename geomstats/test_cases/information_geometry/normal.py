import pytest

import geomstats.backend as gs
from geomstats.test_cases.geometry.pullback_metric import PullbackDiffeoMetricTestCase


def univariate_fisher_rao_dist(point_a, point_b):
    mean_a = point_a[..., 0]
    mean_b = point_b[..., 0]
    sigma_a = point_a[..., 1]
    sigma_b = point_b[..., 1]

    sdiff_mean = (mean_a - mean_b) ** 2
    sdiff_sigma = (sigma_a - sigma_b) ** 2
    ssum_sigma = (sigma_a + sigma_b) ** 2

    return (
        2
        * gs.sqrt(2)
        * gs.arctanh(
            gs.sqrt((sdiff_mean + 2 * sdiff_sigma) / (sdiff_mean + 2 * ssum_sigma))
        )
    )


class UnivariateNormalMetricTestCase(PullbackDiffeoMetricTestCase):
    @pytest.mark.random
    def test_dist_against_closed_form(self, n_points, atol):
        point_a = self.data_generator.random_point(n_points)
        point_b = self.data_generator.random_point(n_points)

        res = self.space.metric.dist(point_a, point_b)
        expected = univariate_fisher_rao_dist(point_a, point_b)
        self.assertAllClose(res, expected, atol=atol)
