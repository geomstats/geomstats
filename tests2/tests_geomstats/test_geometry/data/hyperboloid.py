import geomstats.backend as gs
from geomstats.test.data import TestData

from .base import LevelSetTestData
from .hyperbolic import HyperbolicMetricTestData


class HyperboloidTestData(LevelSetTestData):
    tolerances = {"projection_belongs": {"atol": 1e-8}}


class Hyperboloid2TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(
                point=gs.array([0.5, 7, 3.0]),
                expected=False,
            ),
        ]
        return self.generate_tests(data)


class Hyperboloid3TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(
                point=gs.array([1.0, 0.0, 0.0, 0.0]),
                expected=True,
            ),
        ]
        return self.generate_tests(data)


class HyperboloidMetricTestData(HyperbolicMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    skips = ("metric_matrix_is_spd",)

    def inner_product_is_minkowski_inner_product_test_data(self):
        return self.generate_random_data()
