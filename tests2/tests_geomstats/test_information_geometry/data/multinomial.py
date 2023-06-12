from tests2.tests_geomstats.test_geometry.data.base import LevelSetTestData
from tests2.tests_geomstats.test_geometry.data.riemannian_metric import (
    RiemannianMetricTestData,
)
from tests2.tests_geomstats.test_information_geometry.data.base import (
    InformationManifoldMixinTestData,
)


class MultinomialDistributionsTestData(
    InformationManifoldMixinTestData, LevelSetTestData
):
    pass


class MultinomialMetricTestData(RiemannianMetricTestData):
    trials = 3
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    tolerances = {
        "log_after_exp": {"atol": 1e-3},
    }

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
