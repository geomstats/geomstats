import geomstats.backend as gs
from geomstats.test.data import TestData

from ...test_geometry.data.base import LevelSetTestData
from ...test_geometry.data.riemannian_metric import RiemannianMetricTestData
from .base import InformationManifoldMixinTestData


class MultinomialDistributionsTestData(
    InformationManifoldMixinTestData, LevelSetTestData
):
    fail_for_not_implemented_errors = False


class MultinomialDistributions2TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([-1.0, 0.3]), expected=False),
        ]
        return self.generate_tests(data)


class MultinomialDistributions3TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([0.1, 0.2, 0.3, 0.4]), expected=True),
            dict(point=gs.array([0.0, 1.0, 0.3, 0.4]), expected=False),
        ]
        return self.generate_tests(data)


class MultinomialMetricTestData(RiemannianMetricTestData):
    trials = 5
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    tolerances = {
        "log_after_exp": {"atol": 1e-3},
    }
    xfails = ("log_after_exp",)

    def sectional_curvature_is_positive_test_data(self):
        return self.generate_random_data()

    def dist_against_closed_form_test_data(self):
        return self.generate_random_data()
