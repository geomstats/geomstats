from .connection import (
    ConnectionCmpWithPointTransformTestData,
    ConnectionCmpWithTransformTestData,
    ConnectionComparisonTestData,
    ConnectionTestData,
)


class RiemannianMetricTestData(ConnectionTestData):
    def metric_matrix_vec_test_data(self):
        return self.generate_vec_data()

    def metric_matrix_is_spd_test_data(self):
        return self.generate_random_data()

    def cometric_matrix_vec_test_data(self):
        return self.generate_vec_data()

    def inner_product_derivative_matrix_vec_test_data(self):
        return self.generate_vec_data()

    def inner_product_vec_test_data(self):
        return self.generate_vec_data()

    def inner_product_is_symmetric_test_data(self):
        return self.generate_random_data()

    def inner_coproduct_vec_test_data(self):
        return self.generate_vec_data()

    def squared_norm_vec_test_data(self):
        return self.generate_vec_data()

    def norm_vec_test_data(self):
        return self.generate_vec_data()

    def norm_is_positive_test_data(self):
        return self.generate_random_data()

    def normalize_vec_test_data(self):
        return self.generate_vec_data()

    def normalize_is_unitary_test_data(self):
        return self.generate_random_data()

    def squared_dist_vec_test_data(self):
        return self.generate_vec_data()

    def squared_dist_is_symmetric_test_data(self):
        return self.generate_random_data()

    def squared_dist_is_positive_test_data(self):
        return self.generate_random_data()

    def dist_vec_test_data(self):
        return self.generate_vec_data()

    def dist_is_symmetric_test_data(self):
        return self.generate_random_data()

    def dist_is_positive_test_data(self):
        return self.generate_random_data()

    def dist_is_log_norm_test_data(self):
        return self.generate_random_data()

    def dist_point_to_itself_is_zero_test_data(self):
        return self.generate_random_data()

    def dist_triangle_inequality_test_data(self):
        return self.generate_random_data()

    def covariant_riemann_tensor_vec_test_data(self):
        return self.generate_vec_data()

    def covariant_riemann_tensor_is_skew_symmetric_1_test_data(self):
        return self.generate_random_data()

    def covariant_riemann_tensor_is_skew_symmetric_2_test_data(self):
        return self.generate_random_data()

    def covariant_riemann_tensor_bianchi_identity_test_data(self):
        return self.generate_random_data()

    def covariant_riemann_tensor_is_interchange_symmetric_test_data(self):
        return self.generate_random_data()

    def sectional_curvature_vec_test_data(self):
        return self.generate_vec_data()

    def scalar_curvature_vec_test_data(self):
        return self.generate_vec_data()

    def parallel_transport_ivp_norm_test_data(self):
        return self.generate_random_data()

    def parallel_transport_bvp_norm_test_data(self):
        return self.generate_random_data()


class RiemannianMetricComparisonTestData(ConnectionComparisonTestData):
    def metric_matrix_random_test_data(self):
        return self.generate_random_data()

    def cometric_matrix_random_test_data(self):
        return self.generate_random_data()

    def inner_product_derivative_matrix_random_test_data(self):
        return self.generate_random_data()

    def inner_product_random_test_data(self):
        return self.generate_random_data()

    def inner_coproduct_random_test_data(self):
        return self.generate_random_data()

    def squared_norm_random_test_data(self):
        return self.generate_random_data()

    def norm_random_test_data(self):
        return self.generate_random_data()

    def normalize_random_test_data(self):
        return self.generate_random_data()

    def squared_dist_random_test_data(self):
        return self.generate_random_data()

    def dist_random_test_data(self):
        return self.generate_random_data()

    def covariant_riemann_tensor_random_test_data(self):
        return self.generate_random_data()

    def sectional_curvature_random_test_data(self):
        return self.generate_random_data()

    def scalar_curvature_random_test_data(self):
        return self.generate_random_data()


class RiemannianMetricCmpWithPointTransformTestData(
    ConnectionCmpWithPointTransformTestData
):
    def squared_dist_random_test_data(self):
        return self.generate_random_data()

    def dist_random_test_data(self):
        return self.generate_random_data()


class RiemannianMetricCmpWithTransformTestData(
    RiemannianMetricCmpWithPointTransformTestData, ConnectionCmpWithTransformTestData
):
    def inner_product_derivative_matrix_random_test_data(self):
        return self.generate_random_data()

    def inner_product_random_test_data(self):
        return self.generate_random_data()

    def inner_coproduct_random_test_data(self):
        return self.generate_random_data()

    def squared_norm_random_test_data(self):
        return self.generate_random_data()

    def sectional_curvature_random_test_data(self):
        return self.generate_random_data()

    def scalar_curvature_random_test_data(self):
        return self.generate_random_data()
