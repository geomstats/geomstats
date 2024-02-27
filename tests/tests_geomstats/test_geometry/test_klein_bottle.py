import pytest

import geomstats.backend as gs
from geomstats.geometry.klein_bottle import KleinBottle
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.klein_bottle import KleinBottleMetricTestData, KleinBottleTestData


class TestKleinBottle(ManifoldTestCase, metaclass=DataBasedParametrizer):
    space = KleinBottle(equip=False)
    testing_data = KleinBottleTestData()

    def test_equivalent(self, point_a, point_b, expected, atol):
        is_equivalent = self.space.equivalent(point_a, point_b, atol=atol)
        self.assertAllEqual(is_equivalent, expected)

    @pytest.mark.random
    def test_regularize_correct_domain(self, n_points):
        points = self.data_generator.random_point(n_points)
        regularized_computed = self.space.regularize(points)
        greater_zero = gs.all(regularized_computed >= 0)
        smaller_one = gs.all(regularized_computed < 1)
        self.assertTrue(greater_zero and smaller_one)

    @pytest.mark.random
    def test_intrinsic_to_extrinsic_coords(self):
        point_intrinsic = self.data_generator.random_point(1)
        point_extrinsic = self.intrinsic_to_extrinsic_coords(point_intrinsic)
        self.assertTrue(gs.len(point_extrinsic) == 4)

    @pytest.mark.random
    def test_intrinsic_to_bagel_coords(self):
        point_intrinsic = self.data_generator.random_point(1)
        point_bagel = self.intrinsic_to_bagel_coords(point_intrinsic)
        self.assertTrue(gs.len(point_bagel) == 3)

    @pytest.mark.random
    def test_intrinsic_to_bottle_coords(self):
        point_intrinsic = self.data_generator.random_point(1)
        point_bottle = self.intrinsic_to_bottle_coords(point_intrinsic)
        self.assertTrue(gs.len(point_bottle) == 3)


class TestKleinBottleMetric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = KleinBottle()
    data_generator = RandomDataGenerator(space, amplitude=10.0)
    testing_data = KleinBottleMetricTestData()
