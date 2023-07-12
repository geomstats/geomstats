from tests2.tests_geomstats.test_geometry.data.invariant_metric import (
    InvariantMetricMatrixTestData,
)
from tests2.tests_geomstats.test_geometry.data.riemannian_metric import (
    RiemannianMetricTestData,
)

from .base import OpenSetTestData


class PositiveLowerTriangularMatricesTestData(OpenSetTestData):
    def gram_vec_test_data(self):
        return self.generate_vec_data()

    def differential_gram_vec_test_data(self):
        return self.generate_vec_data()

    def inverse_differential_gram_vec_test_data(self):
        return self.generate_vec_data()

    def gram_belongs_to_spd_matrices_test_data(self):
        return self.generate_random_data()

    def differential_gram_belongs_to_symmetric_matrices_test_data(self):
        return self.generate_random_data()

    def inverse_differential_gram_belongs_to_lower_triangular_matrices_test_data(self):
        return self.generate_random_data()


class CholeskyMetricTestData(RiemannianMetricTestData):
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False

    def diag_inner_product_vec_test_data(self):
        return self.generate_vec_data()

    def strictly_lower_inner_product_vec_test_data(self):
        return self.generate_vec_data()


class InvariantPositiveLowerTriangularMatricesMetricTestData(
    InvariantMetricMatrixTestData
):
    trials = 3
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False

    tolerances = {
        "exp_after_log_at_identity": {"atol": 1e-4},
        "geodesic_boundary_points": {"atol": 1e-4},
        "geodesic_bvp_reverse": {"atol": 1e-4},
        "log_after_exp": {"atol": 1e-4},
        "parallel_transport_bvp_norm": {"atol": 1e-4},
        "parallel_transport_ivp_norm": {"atol": 1e-4},
        "squared_dist_is_symmetric": {"atol": 1e-4},
    }
