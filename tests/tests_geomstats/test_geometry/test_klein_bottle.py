from polpo.testing.parametrizers import GeometricDataBasedParametrizer

import geomstats.backend as gs
from geomstats.geometry.klein_bottle import KleinBottle
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.klein_bottle import KleinBottleMetricTestData, KleinBottleTestData


class TestKleinBottle(ManifoldTestCase, metaclass=GeometricDataBasedParametrizer):
    space = KleinBottle(equip=False)
    testing_data = KleinBottleTestData(space)

    def test_equivalent(self, point_a, point_b, expected, atol=gs.atol):
        """Check equivalence of representatives.

        Parameters
        ----------
        point_a : array-like, shape=[..., 2]
            Point.
        point_b : array-like, shape=[..., 2]
            Point.
        expected : array-like, shape=[...]
            Boolean evaluating if `point_a` and `point_b` represent the same
            point of the Klein bottle.
        atol : float
            Absolute tolerance.
        """
        is_equivalent = self.space.equivalent(point_a, point_b, atol=atol)
        self.assertAllEqual(is_equivalent, expected)

    def test_to_coords(self, point, coords_type, expected, atol=gs.atol):
        """Check coordinates change.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point.
        coords_type : str
            Coordinates type.
        expected : array-like, shape=[..., k]
            Expected point in the given coordinates.
        atol : float
            Absolute tolerance.
        """
        transformed_point = self.space.to_coords(point, coords_type)
        self.assertAllClose(transformed_point, expected, atol=atol)

    def test_regularize_correct_domain(self, point):
        """Check regularized point is in the fundamental domain.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point.
        """
        regularized_point = self.space.regularize(point)
        greater_zero = gs.all(regularized_point >= 0)
        smaller_one = gs.all(regularized_point < 1)
        self.assertTrue(greater_zero and smaller_one)


class TestKleinBottleMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = KleinBottle()
    testing_data = KleinBottleMetricTestData(
        space, generator_kwargs={"amplitude": 10.0}
    )
