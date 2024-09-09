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

    def simplex_to_sphere_vec_test_data(self):
        return self.generate_vec_data()

    def simplex_to_sphere_belongs_test_data(self):
        return self.generate_random_data()

    def sphere_to_simplex_vec_test_data(self):
        return self.generate_vec_data()

    def sphere_to_simplex_belongs_test_data(self):
        return self.generate_random_data()

    def sphere_to_simplex_after_simplex_to_sphere_test_data(self):
        return self.generate_random_data()

    def simplex_to_sphere_after_sphere_to_simplex_test_data(self):
        return self.generate_random_data()

    def tangent_simplex_to_sphere_vec_test_data(self):
        return self.generate_vec_data()

    def tangent_simplex_to_sphere_is_tangent_test_data(self):
        return self.generate_random_data()

    def tangent_sphere_to_simplex_vec_test_data(self):
        return self.generate_vec_data()

    def tangent_sphere_to_simplex_is_tangent_test_data(self):
        return self.generate_random_data()

    def tangent_sphere_to_simplex_after_tangent_simplex_to_sphere_test_data(self):
        return self.generate_random_data()

    def tangent_simplex_to_sphere_after_tangent_sphere_to_simplex_test_data(self):
        return self.generate_random_data()

    def sectional_curvature_is_positive_test_data(self):
        return self.generate_random_data()

    def dist_against_closed_form_test_data(self):
        return self.generate_random_data()
