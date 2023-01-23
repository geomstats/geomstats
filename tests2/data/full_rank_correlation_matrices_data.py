from tests2.data.base_data import FiberBundleTestData, LevelSetTestData
from tests2.data.spd_matrices_data import SPDMatricesTestData


class FullRankCorrelationMatricesTestData(LevelSetTestData):
    def from_covariance_belongs_test_data(self):
        return self.generate_random_data()

    def from_covariance_vec_test_data(self):
        return self.generate_vec_data()

    def diag_action_vec_test_data(self):
        return self.generate_vec_data()


class CorrelationMatricesBundleTestData(SPDMatricesTestData, FiberBundleTestData):
    skips = (
        "integrability_tensor_vec",
        "integrability_tensor_derivative_vec",
        "align_vec",
    )
    ignores_if_not_autodiff = (
        "log_after_align_is_horizontal",
        "align_vec",
    )
    tolerances = {"log_after_align_is_horizontal": {"atol": 1e-2}}

    def log_after_align_is_horizontal_test_data(self):
        # TODO: remove after fixing align vectorization
        data = [dict(n_points=1)]
        return self.generate_tests(data)
