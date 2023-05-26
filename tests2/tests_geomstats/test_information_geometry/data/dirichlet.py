from tests2.tests_geomstats.test_geometry.data.base import OpenSetTestData
from tests2.tests_geomstats.test_geometry.data.riemannian_metric import (
    RiemannianMetricTestData,
)
from tests2.tests_geomstats.test_information_geometry.data.base import (
    InformationManifoldMixinTestData,
)


class DirichletDistributionsTestData(InformationManifoldMixinTestData, OpenSetTestData):
    pass


class DirichletMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    tolerances = {
        "dist_is_symmetric": {"atol": 1e-4},
        "log_after_exp": {"atol": 1e-2},
        "exp_after_log": {"atol": 1e-2},
        "squared_dist_is_symmetric": {"atol": 1e-3},
    }

    def jacobian_christoffels_vec_test_data(self):
        return self.generate_vec_data()
