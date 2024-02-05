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


class TestKleinBottleMetric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = KleinBottle()
    data_generator = RandomDataGenerator(space, amplitude=10.0)
    testing_data = KleinBottleMetricTestData()
