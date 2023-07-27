from geomstats.test.data import TestData


class ConnectionTestData(TestData):
    def christoffels_vec_test_data(self):
        return self.generate_vec_data()

    def exp_vec_test_data(self):
        return self.generate_vec_data()

    def exp_belongs_test_data(self):
        return self.generate_random_data()

    def log_vec_test_data(self):
        return self.generate_vec_data()

    def log_is_tangent_test_data(self):
        return self.generate_random_data()

    def exp_after_log_test_data(self):
        return self.generate_random_data()

    def log_after_exp_test_data(self):
        return self.generate_random_data()

    def riemann_tensor_vec_test_data(self):
        return self.generate_vec_data()

    def curvature_vec_test_data(self):
        return self.generate_vec_data()

    def ricci_tensor_vec_test_data(self):
        return self.generate_vec_data()

    def directional_curvature_vec_test_data(self):
        return self.generate_vec_data()

    def curvature_derivative_vec_test_data(self):
        return self.generate_vec_data()

    def directional_curvature_derivative_vec_test_data(self):
        return self.generate_vec_data()

    def geodesic_bvp_vec_test_data(self):
        return self.generate_vec_data_with_time()

    def geodesic_ivp_vec_test_data(self):
        return self.generate_vec_data_with_time()

    def geodesic_boundary_points_test_data(self):
        return self.generate_random_data()

    def geodesic_bvp_reverse_test_data(self):
        return self.generate_random_data_with_time()

    def geodesic_bvp_belongs_test_data(self):
        return self.generate_random_data_with_time()

    def geodesic_ivp_belongs_test_data(self):
        return self.generate_random_data_with_time()

    def exp_geodesic_ivp_test_data(self):
        return self.generate_random_data()

    def parallel_transport_ivp_vec_test_data(self):
        return self.generate_vec_data()

    def parallel_transport_bvp_vec_test_data(self):
        return self.generate_vec_data()

    def parallel_transport_bvp_transported_is_tangent_test_data(self):
        return self.generate_random_data()

    def parallel_transport_ivp_transported_is_tangent_test_data(self):
        return self.generate_random_data()

    def injectivity_radius_vec_test_data(self):
        return self.generate_vec_data()


class ConnectionComparisonTestData(TestData):
    def christoffels_random_test_data(self):
        return self.generate_random_data()

    def exp_random_test_data(self):
        return self.generate_random_data()

    def log_random_test_data(self):
        return self.generate_random_data()

    def riemann_tensor_random_test_data(self):
        return self.generate_random_data()

    def curvature_random_test_data(self):
        return self.generate_random_data()

    def ricci_tensor_random_test_data(self):
        return self.generate_random_data()

    def directional_curvature_random_test_data(self):
        return self.generate_random_data()

    def curvature_derivative_random_test_data(self):
        return self.generate_random_data()

    def directional_curvature_derivative_random_test_data(self):
        return self.generate_random_data()

    def geodesic_bvp_random_test_data(self):
        return self.generate_random_data_with_time()

    def geodesic_ivp_random_test_data(self):
        return self.generate_random_data_with_time()

    def parallel_transport_ivp_random_test_data(self):
        return self.generate_random_data()

    def parallel_transport_bvp_random_test_data(self):
        return self.generate_random_data()

    def injectivity_radius_random_test_data(self):
        return self.generate_random_data()


class ConnectionCmpWithPointTransformTestData(TestData):
    def geodesic_bvp_random_test_data(self):
        return self.generate_random_data_with_time()


class ConnectionCmpWithTransformTestData(ConnectionCmpWithPointTransformTestData):
    def exp_random_test_data(self):
        return self.generate_random_data()

    def log_random_test_data(self):
        return self.generate_random_data()

    def curvature_random_test_data(self):
        return self.generate_random_data()

    def geodesic_ivp_random_test_data(self):
        return self.generate_random_data_with_time()

    def parallel_transport_ivp_random_test_data(self):
        return self.generate_random_data()

    def parallel_transport_bvp_random_test_data(self):
        return self.generate_random_data()
