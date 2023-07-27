from .base import LevelSetTestData
from .fiber_bundle import FiberBundleTestData
from .pullback_metric import PullbackDiffeoMetricTestData
from .quotient_metric import QuotientMetricTestData


class FullRankCorrelationMatricesTestData(LevelSetTestData):
    def from_covariance_belongs_test_data(self):
        return self.generate_random_data()

    def from_covariance_vec_test_data(self):
        return self.generate_vec_data()

    def diag_action_vec_test_data(self):
        return self.generate_vec_data()


class CorrelationMatricesBundleTestData(FiberBundleTestData):
    trials = 2
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    tolerances = {
        "log_after_align_is_horizontal": {"atol": 1e-2},
        "align_vec": {"atol": 1e-2},
    }

    def horizontal_projection_is_horizontal_v2_test_data(self):
        return self.generate_random_data()


class FullRankCorrelationAffineQuotientMetricTestData(QuotientMetricTestData):
    trials = 3
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    tolerances = {
        "dist_point_to_itself_is_zero": {"atol": 1e-4},
        "geodesic_bvp_vec": {"atol": 1e-4},
        "geodesic_bvp_reverse": {"atol": 1e-4},
        "geodesic_boundary_points": {"atol": 1e-4},
        "log_after_exp": {"atol": 1e-3},
        "exp_after_log": {"atol": 1e-4},
        "log_vec": {"atol": 1e-4},
    }


class FullRankEuclideanCholeskyMetricTestData(PullbackDiffeoMetricTestData):
    trials = 3
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    # tolerances = {
    #     "dist_point_to_itself_is_zero": {"atol": 1e-4},
    #     "geodesic_bvp_vec": {"atol": 1e-4},
    #     "geodesic_bvp_reverse": {"atol": 1e-4},
    #     "geodesic_boundary_points": {"atol": 1e-4},
    #     "log_after_exp": {"atol": 1e-3},
    #     "exp_after_log": {"atol": 1e-4},
    #     "log_vec": {"atol": 1e-4},
    # }
