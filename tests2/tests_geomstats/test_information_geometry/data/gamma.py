from tests2.tests_geomstats.test_geometry.data.base import OpenSetTestData
from tests2.tests_geomstats.test_geometry.data.riemannian_metric import (
    RiemannianMetricTestData,
)
from tests2.tests_geomstats.test_information_geometry.data.base import (
    InformationManifoldMixinTestData,
)


class GammaDistributionsTestData(InformationManifoldMixinTestData, OpenSetTestData):
    def natural_to_standard_vec_test_data(self):
        return self.generate_vec_data()

    def standard_to_natural_vec_test_data(self):
        return self.generate_vec_data()

    def standard_to_natural_after_natural_to_standard_test_data(self):
        return self.generate_random_data()

    def tangent_natural_to_standard_vec_test_data(self):
        return self.generate_vec_data()

    def tangent_standard_to_natural_vec_test_data(self):
        return self.generate_vec_data()

    def tangent_standard_to_natural_after_tangent_natural_to_standard_test_data(self):
        return self.generate_random_data()


class GammaMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    tolerances = {
        "dist_is_symmetric": {"atol": 1e-3},
        "log_after_exp": {"atol": 1e-2},
        "exp_after_log": {"atol": 1e-2},
        "squared_dist_is_symmetric": {"atol": 1e-3},
    }

    xfails = (
        # fail often, usually not by far
        "dist_is_symmetric",
        "squared_dist_is_symmetric",
        "exp_after_log",
        # rarely fail, but can fail by noisely
        "exp_geodesic_ivp",
    )

    def jacobian_christoffels_vec_test_data(self):
        return self.generate_vec_data()
