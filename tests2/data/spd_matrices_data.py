import random

from tests2.data.base_data import (
    FiberBundleTestData,
    OpenSetTestData,
    RiemannianMetricTestData,
)
from tests2.data.comparison_data import RiemannianMetricComparisonTestData
from tests2.data.general_linear_data import GeneralLinearTestData


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


class BuresWassersteinBundleTestData(FiberBundleTestData, GeneralLinearTestData):
    skips = (
        # not implemented
        "integrability_tensor_derivative_vec",
        "integrability_tensor_vec",
    )


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

    tolerances = {
        "dist_point_to_itself_is_zero": {"atol": 1e-6},
    }


class SPDBuresWassersteinQuotientMetricTestData(RiemannianMetricComparisonTestData):
    skips = (
        # not implemented
        "christoffels",
        "cometric_matrix",
        "covariant_riemann_tensor",
        "curvature_derivative",
        "directional_curvature_derivative",
        "curvature",
        "directional_curvature",
        "inner_coproduct",
        "inner_product_derivative_matrix",
        "metric_matrix",
        "ricci_tensor",
        "riemann_tensor",
        "scalar_curvature",
        "sectional_curvature",
        "parallel_transport_with_direction",
        "parallel_transport_with_end_point",
        "injectivity_radius",
    )
    ignores_if_not_autodiff = (
        "dist",
        "geodesic_bvp",
        "log",
        "squared_dist",
    )


class SPDEuclideanMetricPower1TestData(RiemannianMetricTestData):
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


class SPDEuclideanMetricTestData(SPDEuclideanMetricPower1TestData):
    skips = SPDEuclideanMetricPower1TestData.skips + (
        # not implemented
        "parallel_transport_vec_with_end_point",
        "parallel_transport_vec_with_direction",
        "parallel_transport_transported_is_tangent",
    )


class SPDLogEuclideanMetricTestData(RiemannianMetricTestData):
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
        "parallel_transport_vec_with_end_point",
        "parallel_transport_vec_with_direction",
        "parallel_transport_transported_is_tangent",
    )
