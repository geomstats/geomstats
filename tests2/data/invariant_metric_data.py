from tests2.data.base_data import RiemannianMetricTestData


class InvariantMetricMatrixTestData(RiemannianMetricTestData):
    def inner_product_at_identity_vec_test_data(self):
        return self.generate_vec_data()

    def structure_constant_vec_test_data(self):
        return self.generate_vec_data()

    def dual_adjoint_vec_test_data(self):
        return self.generate_vec_data()

    def connection_at_identity_vec_test_data(self):
        return self.generate_vec_data()

    def connection_vec_test_data(self):
        return self.generate_vec_data()

    def curvature_at_identity_vec_test_data(self):
        return self.generate_vec_data()

    def sectional_curvature_at_identity_vec_test_data(self):
        return self.generate_vec_data()

    def curvature_derivative_at_identity_vec_test_data(self):
        return self.generate_vec_data()


class InvariantMetricVectorTestData(RiemannianMetricTestData):
    def inner_product_at_identity_vec_test_data(self):
        return self.generate_vec_data()

    def left_exp_from_identity_vec_test_data(self):
        return self.generate_vec_data()

    def exp_from_identity_vec_test_data(self):
        return self.generate_vec_data()

    def left_log_from_identity_vec_test_data(self):
        return self.generate_vec_data()

    def log_from_identity_vec_test_data(self):
        return self.generate_vec_data()
