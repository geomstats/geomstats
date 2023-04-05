import random

from tests2.data.base_data import OpenSetTestData, RiemannianMetricTestData


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
    skips = (
        # not implemented
        "christoffels_vec",
        "cometric_matrix_vec",
        "covariant_riemann_tensor_is_interchange_symmetric",
        "covariant_riemann_tensor_vec",
        "covariant_riemann_tensor_bianchi_identity",
        "covariant_riemann_tensor_is_skew_symmetric_1",
        "covariant_riemann_tensor_is_skew_symmetric_2",
        "curvature_derivative_vec",
        "directional_curvature_derivative_vec",
        "curvature_vec",
        "directional_curvature_vec",
        "inner_coproduct_vec",
        "inner_product_derivative_matrix_vec",
        "metric_matrix_is_spd",
        "metric_matrix_vec",
        "ricci_tensor_vec",
        "riemann_tensor_vec",
        "scalar_curvature_vec",
        "sectional_curvature_vec",
    )


class SPDBuresWassersteinMetricTestData(RiemannianMetricTestData):
    skips = (
        # not implemented
        "christoffels_vec",
        "cometric_matrix_vec",
        "covariant_riemann_tensor_is_interchange_symmetric",
        "covariant_riemann_tensor_vec",
        "covariant_riemann_tensor_bianchi_identity",
        "covariant_riemann_tensor_is_skew_symmetric_1",
        "covariant_riemann_tensor_is_skew_symmetric_2",
        "curvature_derivative_vec",
        "directional_curvature_derivative_vec",
        "curvature_vec",
        "directional_curvature_vec",
        "inner_coproduct_vec",
        "inner_product_derivative_matrix_vec",
        "metric_matrix_is_spd",
        "metric_matrix_vec",
        "ricci_tensor_vec",
        "riemann_tensor_vec",
        "scalar_curvature_vec",
        "sectional_curvature_vec",
    )


class SPDEuclideanMetricTestData(RiemannianMetricTestData):
    skips = (
        # not implemented
        "christoffels_vec",
        "cometric_matrix_vec",
        "covariant_riemann_tensor_is_interchange_symmetric",
        "covariant_riemann_tensor_vec",
        "covariant_riemann_tensor_bianchi_identity",
        "covariant_riemann_tensor_is_skew_symmetric_1",
        "covariant_riemann_tensor_is_skew_symmetric_2",
        "curvature_derivative_vec",
        "directional_curvature_derivative_vec",
        "curvature_vec",
        "directional_curvature_vec",
        "inner_coproduct_vec",
        "inner_product_derivative_matrix_vec",
        "metric_matrix_is_spd",
        "metric_matrix_vec",
        "ricci_tensor_vec",
        "riemann_tensor_vec",
        "scalar_curvature_vec",
        "sectional_curvature_vec",
    )

    def exp_domain_vec_test_data(self):
        return self.generate_vec_data()
