import random

from .base import OpenSetTestData
from .riemannian_metric import RiemannianMetricTestData


class SPDMatricesMixinsTestData:
    def _generate_power_vec_data(self):
        power = [random.randint(1, 4)]
        data = []
        for power_ in power:
            data.extend(
                [dict(n_reps=n_reps, power=power_) for n_reps in self.N_VEC_REPS]
            )
        return self.generate_tests(data)

    def differential_power_vec_test_data(self):
        return self._generate_power_vec_data()

    def inverse_differential_power_vec_test_data(self):
        return self._generate_power_vec_data()

    def differential_log_vec_test_data(self):
        return self.generate_vec_data()

    def inverse_differential_log_vec_test_data(self):
        return self.generate_vec_data()

    def differential_exp_vec_test_data(self):
        return self.generate_vec_data()

    def inverse_differential_exp_vec_test_data(self):
        return self.generate_vec_data()

    def logm_vec_test_data(self):
        return self.generate_vec_data()

    def cholesky_factor_vec_test_data(self):
        return self.generate_vec_data()

    def cholesky_factor_belongs_to_positive_lower_triangular_matrices_test_data(self):
        return self.generate_random_data()

    def differential_cholesky_factor_vec_test_data(self):
        return self.generate_vec_data()

    def differential_cholesky_factor_belongs_to_positive_lower_triangular_matrices_test_data(
        self,
    ):
        return self.generate_random_data()


class SPDMatricesTestData(SPDMatricesMixinsTestData, OpenSetTestData):
    pass


class SPDAffineMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False


class SPDBuresWassersteinMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    tolerances = {
        "dist_point_to_itself_is_zero": {"atol": 1e-6},
    }

    skips = (
        "parallel_transport_transported_is_tangent",
        "parallel_transport_vec_with_direction",
        "parallel_transport_vec_with_end_point",
    )


class SPDEuclideanMetricPower1TestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    def exp_domain_vec_test_data(self):
        return self.generate_vec_data()


class SPDEuclideanMetricTestData(SPDEuclideanMetricPower1TestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False


class SPDLogEuclideanMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False
