"""Unit tests for the ProductPositiveRealsAndComplexPoincareDisks manifold."""

import geomstats.backend as gs
from geomstats.geometry.complex_poincare_disk import ComplexPoincareDisk
from geomstats.geometry.positive_reals import PositiveReals
from tests.conftest import Parametrizer, TestCase
from tests.data.product_positive_reals_and_poincare_disks_data import (
    ProductPositiveRealsAndComplexPoincareDisksMetricTestData,
    ProductPositiveRealsAndComplexPoincareDisksTestData,
)
from tests.geometry_test_cases import OpenSetTestCase


class TestProductPositiveRealsAndComplexPoincareDisks(
    OpenSetTestCase, metaclass=Parametrizer
):
    skip_test_to_tangent_is_tangent_in_embedding_space = True

    testing_data = ProductPositiveRealsAndComplexPoincareDisksTestData()

    def test_dimension(self, n_manifolds, expected):
        space = self.Space(n_manifolds)
        self.assertAllClose(space.dim, expected)


class TestProductPositiveRealsAndComplexPoincareDisksMetric(
    TestCase, metaclass=Parametrizer
):
    testing_data = ProductPositiveRealsAndComplexPoincareDisksMetricTestData()
    Metric = testing_data.Metric

    def test_signature(self, space, expected):
        space.equip_with_metric(self.Metric)
        self.assertAllClose(space.metric.signature, expected)

    def test_squared_dist(self, space):
        space.equip_with_metric(self.Metric)
        n_manifolds = space.n_manifolds

        def _get_random_point():
            point_positive_reals = positive_reals.random_point()
            point_complex_poincare_disk = complex_poincare_disk.random_point()

            rep_point_complex_poincare_disk = gs.repeat(
                gs.expand_dims(point_complex_poincare_disk, axis=0),
                n_manifolds - 1,
                axis=0,
            )

            point = gs.vstack(
                [
                    gs.expand_dims(point_positive_reals, axis=0),
                    rep_point_complex_poincare_disk,
                ]
            )

            return point, point_positive_reals, point_complex_poincare_disk

        complex_poincare_disk = ComplexPoincareDisk()
        positive_reals = PositiveReals()

        (
            point_a,
            point_positive_reals_a,
            point_complex_poincare_disk_a,
        ) = _get_random_point()
        (
            point_b,
            point_positive_reals_b,
            point_complex_poincare_disk_b,
        ) = _get_random_point()

        sq_dist_positive_reals = positive_reals.metric.squared_dist(
            point_positive_reals_a, point_positive_reals_b
        )
        sq_dist_complex_poincare_disk = complex_poincare_disk.metric.squared_dist(
            point_complex_poincare_disk_a, point_complex_poincare_disk_b
        )

        sq_dist_prod = space.metric.squared_dist(point_a, point_b)

        sq_dist_expected = (
            n_manifolds * sq_dist_positive_reals
            + (n_manifolds - 1) * n_manifolds / 2 * sq_dist_complex_poincare_disk
        )
        self.assertAllClose(sq_dist_prod, sq_dist_expected)
        assert not gs.is_complex(sq_dist_prod)
