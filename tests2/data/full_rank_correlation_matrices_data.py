from tests2.data.base_data import FiberBundleTestData, LevelSetTestData
from tests2.data.quotient_metric_data import QuotientMetricTestData
from tests2.data.spd_matrices_data import SPDMatricesTestData


class FullRankCorrelationMatricesTestData(LevelSetTestData):
    def from_covariance_belongs_test_data(self):
        return self.generate_random_data()

    def from_covariance_vec_test_data(self):
        return self.generate_vec_data()

    def diag_action_vec_test_data(self):
        return self.generate_vec_data()


class CorrelationMatricesBundleTestData(FiberBundleTestData, SPDMatricesTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    xfails = ("align_vec",)
    tolerances = {
        "log_after_align_is_horizontal": {"atol": 1e-2},
        "align_vec": {"atol": 1e-2},
    }


class FullRankCorrelationAffineQuotientMetricTestData(QuotientMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    xfails = ("log_after_exp",)
    tolerances = {
        "dist_point_to_itself_is_zero": {"atol": 1e-6},
        "geodesic_bvp_vec": {"atol": 1e-4},
        "geodesic_bvp_reverse": {"atol": 1e-6},
        "geodesic_boundary_points": {"atol": 1e-6},
        "log_after_exp": {"atol": 1e-4},
        "exp_after_log": {"atol": 1e-6},
        "log_vec": {"atol": 1e-6},
    }
