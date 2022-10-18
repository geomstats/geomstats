from tests.conftest import Parametrizer
from tests.data.klein_bottle_data import KleinBottleTestData
from tests.geometry_test_cases import ManifoldTestCase


class TestKleinBottle(ManifoldTestCase, metaclass=Parametrizer):

    testing_data = KleinBottleTestData()

    def test_projection_belongs(self, space_args, point, atol):
        # no projection for intrinsic coordinates
        return True

    def test_equivalent(self, point1, point2, expected):
        space = self.Space()
        is_equivalent = space.equivalent(point1, point2)
        self.assertAllEqual(is_equivalent, expected)

    def test_regularize(self, point, regularized):
        space = self.Space()
        regularized_computed = space.regularize(point)
        self.assertAllClose(regularized_computed, regularized)
