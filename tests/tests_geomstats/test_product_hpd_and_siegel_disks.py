"""Unit tests for the ProductHPDMatricesAndSiegelDisks manifold."""

import geomstats.backend as gs
from geomstats.geometry.hpd_matrices import HPDMatrices
from geomstats.geometry.siegel import Siegel
from tests.conftest import Parametrizer, TestCase
from tests.data.product_hpd_and_siegel_disks_data import (
    ProductHPDMatricesAndSiegelDisksMetricTestData,
    ProductHPDMatricesAndSiegelDisksTestData,
)
from tests.geometry_test_cases import OpenSetTestCase


class TestProductHPDMatricesAndSiegelDisks(OpenSetTestCase, metaclass=Parametrizer):
    skip_test_to_tangent_is_tangent_in_embedding_space = True
    skip_test_to_tangent_is_tangent = True

    testing_data = ProductHPDMatricesAndSiegelDisksTestData()

    def test_dimension(self, n_manifolds, n, expected):
        space = self.Space(n_manifolds, n)
        self.assertAllClose(space.dim, expected)


class TestProductHPDMatricesAndSiegelDisksMetric(TestCase, metaclass=Parametrizer):
    testing_data = ProductHPDMatricesAndSiegelDisksMetricTestData()
    Metric = testing_data.Metric

    def test_signature(self, space, expected):
        space.equip_with_metric(self.Metric)
        self.assertAllClose(space.metric.signature, expected)

    def test_squared_dist(self, space):
        space.equip_with_metric(self.Metric)
        n_manifolds = space.n_manifolds
        n = space.n

        def _get_random_point():
            point_hpd = hpd.random_point()
            point_siegel = siegel.random_point()

            rep_point_siegel = gs.repeat(
                gs.expand_dims(point_siegel, axis=0), n_manifolds - 1, axis=0
            )

            point = gs.vstack([gs.expand_dims(point_hpd, axis=0), rep_point_siegel])

            return point, point_hpd, point_siegel

        siegel = Siegel(n)
        hpd = HPDMatrices(n)

        point_a, point_hpd_a, point_siegel_a = _get_random_point()
        point_b, point_hpd_b, point_siegel_b = _get_random_point()

        sq_dist_hpd = hpd.metric.squared_dist(point_hpd_a, point_hpd_b)
        sq_dist_siegel = siegel.metric.squared_dist(point_siegel_a, point_siegel_b)

        sq_dist_prod = space.metric.squared_dist(point_a, point_b)

        sq_dist_expected = (
            n_manifolds * sq_dist_hpd
            + (n_manifolds - 1) * n_manifolds / 2 * sq_dist_siegel
        )
        self.assertAllClose(sq_dist_prod, sq_dist_expected)
