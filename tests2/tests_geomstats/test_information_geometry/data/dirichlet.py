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
    trials = 3
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    tolerances = {
        "dist_is_symmetric": {"atol": 1e-3},
        "log_after_exp": {"atol": 1e-3},
        "exp_after_log": {"atol": 1e-2},
        "squared_dist_is_symmetric": {"atol": 1e-3},
    }

    xfails = (
        # rarely fail, but can fail by far
        "dist_is_symmetric",
        "squared_dist_is_symmetric",
        # fail often, usually not by far
        "exp_after_log",
    )

    def jacobian_christoffels_vec_test_data(self):
        return self.generate_vec_data()
