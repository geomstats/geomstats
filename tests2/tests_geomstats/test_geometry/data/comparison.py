from geomstats.test.data import TestData


class ConnectionComparisonTestData(TestData):
    def christoffels_test_data(self):
        return self.generate_random_data()

    def exp_test_data(self):
        return self.generate_random_data()

    def log_test_data(self):
        return self.generate_random_data()

    def riemann_tensor_test_data(self):
        return self.generate_random_data()

    def curvature_test_data(self):
        return self.generate_random_data()

    def ricci_tensor_test_data(self):
        return self.generate_random_data()

    def directional_curvature_test_data(self):
        return self.generate_random_data()

    def curvature_derivative_test_data(self):
        return self.generate_random_data()

    def directional_curvature_derivative_test_data(self):
        return self.generate_random_data()

    def geodesic_bvp_test_data(self):
        return self.generate_random_data_with_time()

    def geodesic_ivp_test_data(self):
        return self.generate_random_data_with_time()

    def parallel_transport_with_direction_test_data(self):
        return self.generate_random_data()

    def parallel_transport_with_end_point_test_data(self):
        return self.generate_random_data()

    def injectivity_radius_test_data(self):
        return self.generate_random_data()


class RiemannianMetricComparisonTestData(ConnectionComparisonTestData):
    def metric_matrix_test_data(self):
        return self.generate_random_data()

    def cometric_matrix_test_data(self):
        return self.generate_random_data()

    def inner_product_derivative_matrix_test_data(self):
        return self.generate_random_data()

    def inner_product_test_data(self):
        return self.generate_random_data()

    def inner_coproduct_test_data(self):
        return self.generate_random_data()

    def squared_norm_test_data(self):
        return self.generate_random_data()

    def norm_test_data(self):
        return self.generate_random_data()

    def normalize_test_data(self):
        return self.generate_random_data()

    def squared_dist_test_data(self):
        return self.generate_random_data()

    def dist_test_data(self):
        return self.generate_random_data()

    def covariant_riemann_tensor_test_data(self):
        return self.generate_random_data()

    def sectional_curvature_test_data(self):
        return self.generate_random_data()

    def scalar_curvature_test_data(self):
        return self.generate_random_data()
